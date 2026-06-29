"""Fine-tune a small seq2seq (t5-small) to map raw ASR text -> sim command DSL.

Model choice: **t5-small** (~60M, encoder-decoder Transformer). It's the
low-latency pick for CPU inference, and the DSL output space is tiny so it learns
fast. Training objective is standard token-level cross-entropy (teacher forced) --
no RL, no custom loss. (ByT5-small is the byte-level fallback if subword
tokenization of digits/callsigns proves brittle; it's ~5x bigger/slower.)

Reads a {input, target} jsonl from build_dataset.py. Runs on CPU but slowly; for
real runs use Modal GPU (modal_app.py).

    python -m dsl_parser_training.train --data dsl_parser_training/data/batch.jsonl \
        --out dsl_parser_training/runs/v0 --epochs 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "t5-small"
PREFIX = "parse atc: "          # T5 task prefix
MAX_IN = 96
MAX_OUT = 48


def build(data_path, out_dir, epochs, batch_size, lr):
    # Heavy deps imported lazily so the rest of the pipeline doesn't need them.
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    ds = load_dataset("json", data_files=data_path, split="train")
    ds = ds.train_test_split(test_size=0.05, seed=0)

    def prep(batch):
        x = tok([PREFIX + t for t in batch["input"]], max_length=MAX_IN,
                truncation=True, padding=False)
        y = tok(text_target=batch["target"], max_length=MAX_OUT,
                truncation=True, padding=False)
        x["labels"] = y["input_ids"]
        return x

    ds = ds.map(prep, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorForSeq2Seq(tok, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        save_total_limit=2,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=ds["train"],
        eval_dataset=ds["test"], data_collator=collator, tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print("saved ->", out_dir)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune t5-small text->DSL.")
    ap.add_argument("--data", default="dsl_parser_training/data/batch.jsonl")
    ap.add_argument("--out", default="dsl_parser_training/runs/v0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    build(a.data, a.out, a.epochs, a.batch_size, a.lr)


if __name__ == "__main__":
    main()
