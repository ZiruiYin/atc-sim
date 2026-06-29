"""End-to-end dsl_parser pipeline on Modal -- nothing is kept locally.

  generate (CPU, no TTS/STT)  random valid commands -> validator -> ATC script
                              -> augmented train.jsonl + (randomly-augmented)
                                 val.jsonl  -> Volume
  train    (GPU)              read train/val from the Volume, fine-tune t5-small,
                              save -> Volume

The dataset is synthetic text (no audio), generated straight on Modal and written
to the Volume `atc-dsl-parser`; it never touches local disk.

    # smoke (small gen + a few epochs, end-to-end):
    modal run dsl_parser_training/modal_app.py::smoke

    # full run:
    modal run dsl_parser_training/modal_app.py::build --n-train 8000 --n-val 1000
    modal run dsl_parser_training/modal_app.py::run_train --epochs 8
    modal run dsl_parser_training/modal_app.py::download --dest dsl_parser_training/runs/v0
"""
import os

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOL = "/vol"

app = modal.App("atc-dsl-parser")
vol = modal.Volume.from_name("atc-dsl-parser", create_if_missing=True)

# Generation image: just the sim + the (pure-regex) normalizer/augmenter + repo
# code. No Piper/Whisper/torch -- generation is plain CPU text.
gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(os.path.join(REPO, "environment"), "/root/environment")
    .add_local_file(os.path.join(REPO, "tts", "__init__.py"), "/root/tts/__init__.py")
    .add_local_file(os.path.join(REPO, "tts", "normalize.py"), "/root/tts/normalize.py")
    .add_local_dir(os.path.join(REPO, "dsl_parser_training"), "/root/dsl_parser_training")
)

# Training image: the ML stack (reads jsonl from the Volume). Pinned to a tested
# transformers/datasets combo so the Trainer API doesn't drift.
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==4.46.3", "datasets==3.0.1",
                 "accelerate==1.0.1", "sentencepiece", "numpy<2")
)

MODEL_NAME = "t5-small"
PREFIX = "parse atc: "
MAX_IN, MAX_OUT = 96, 48


@app.function(image=gen_image, volumes={VOL: vol}, cpu=4.0, timeout=60 * 60)
def generate(n_train: int = 8000, n_val: int = 1000, aug_k: int = 5, seed: int = 7):
    import sys
    sys.path.insert(0, "/root")
    from dsl_parser_training.build_dataset import generate_dataset
    counts = generate_dataset(
        n_train, n_val, aug_k, seed,
        train_path=f"{VOL}/data/train.jsonl", val_path=f"{VOL}/data/val.jsonl",
    )
    vol.commit()
    print("generated:", counts)
    return counts


@app.function(image=train_image, gpu="T4", volumes={VOL: vol}, timeout=4 * 60 * 60)
def train(epochs: float = 8, batch_size: int = 256, lr: float = 3e-4, out: str = "runs/v0"):
    import numpy as np
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )

    vol.reload()                                  # see the freshly-generated data
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    data = load_dataset("json", data_files={
        "train": f"{VOL}/data/train.jsonl",
        "validation": f"{VOL}/data/val.jsonl",
    })

    def prep(b):
        x = tok([PREFIX + t for t in b["input"]], max_length=MAX_IN, truncation=True)
        x["labels"] = tok(text_target=b["target"], max_length=MAX_OUT, truncation=True)["input_ids"]
        return x

    data = data.map(prep, batched=True, remove_columns=data["train"].column_names)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        dp = tok.batch_decode(preds, skip_special_tokens=True)
        dl = tok.batch_decode(labels, skip_special_tokens=True)
        em = sum(p.strip() == l.strip() for p, l in zip(dp, dl)) / max(1, len(dl))
        return {"exact_match": em}

    out_dir = f"{VOL}/{out}"
    # t5 is unstable in fp16; t5-small is tiny so plain fp32 on the T4 is fine.
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=lr, eval_strategy="epoch",
        # Checkpoint every epoch to the Volume and keep the best by exact-match.
        save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="exact_match",
        greater_is_better=True,
        logging_steps=20, predict_with_generate=True, generation_max_length=MAX_OUT,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        processing_class=tok, compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    vol.commit()
    print("eval:", metrics)
    return metrics


@app.function(image=train_image, volumes={VOL: vol})
def fetch(out: str = "runs/v0") -> dict:
    import os as _os
    base = f"{VOL}/{out}"
    blobs = {}
    for root, dirs, files in _os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith("checkpoint-")]  # final model only
        for fn in files:
            p = _os.path.join(root, fn)
            blobs[_os.path.relpath(p, base)] = open(p, "rb").read()
    return blobs


@app.local_entrypoint()
def smoke():
    """Small end-to-end run: synth-generate on CPU, train a few epochs on GPU."""
    print("== generating dataset on Modal CPU (synthetic, no TTS/STT) ==")
    counts = generate.remote(n_train=300, n_val=50, aug_k=4, seed=7)
    print("counts:", counts)
    print("== training on Modal GPU ==")
    metrics = train.remote(epochs=5, batch_size=16)
    print("smoke eval metrics:", metrics)


@app.local_entrypoint()
def full(n_train: int = 8000, n_val: int = 1000, aug_k: int = 5, seed: int = 7,
         epochs: float = 8, batch_size: int = 256, lr: float = 3e-4):
    """One-shot full run: synth-generate the dataset (command -> augmented script
    -> DSL target, NO TTS/STT) on CPU, then fine-tune t5-small on the GPU.

        modal run dsl_parser_training/modal_app.py::full
        modal run dsl_parser_training/modal_app.py::full --n-train 16000 --epochs 10
    """
    print("== generate (synthetic, no TTS/STT) ==")
    counts = generate.remote(n_train, n_val, aug_k, seed)
    print("counts:", counts)
    print("== train ==")
    metrics = train.remote(epochs, batch_size, lr)
    print("eval:", metrics)
    print("\ndone. pull the model with:")
    print("  modal run dsl_parser_training/modal_app.py::download")


@app.local_entrypoint()
def build(n_train: int = 8000, n_val: int = 1000, aug_k: int = 5, seed: int = 7):
    print(generate.remote(n_train, n_val, aug_k, seed))


@app.local_entrypoint()
def run_train(epochs: float = 8, batch_size: int = 256, lr: float = 3e-4):
    print(train.remote(epochs, batch_size, lr))


@app.local_entrypoint()
def download(dest: str = "dsl_parser_training/runs/v0", out: str = "runs/v0"):
    blobs = fetch.remote(out)
    for rel, data in blobs.items():
        path = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        open(path, "wb").write(data)
    print(f"downloaded {len(blobs)} files -> {dest}")
