"""Modal app to fine-tune Piper (VITS) on the ATC controller voices.

Pipeline (run top to bottom):
    modal run tts_training/modal_app.py::upload_data          # 1. push local data -> Modal Volume
    modal run tts_training/modal_app.py::fetch_base            # 2. pull en_US-lessac-medium base ckpt
    modal run tts_training/modal_app.py::load_dataset          # 3. (the data-loading method) verify data on Modal
    modal run tts_training/modal_app.py::train --controller controller_1
    modal run tts_training/modal_app.py::export --controller controller_1

Training stack (native cmake/cython builds + CUDA torch) lives ENTIRELY in the
image below -- nothing TTS-related needs to be installed in the local atc-sim env.
The local env only needs `modal` (to drive this) which is already present.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import modal

# --- base checkpoint we fine-tune from (English US, medium quality VITS) ---------
HF_REPO = "rhasspy/piper-checkpoints"
BASE_CKPT_REL = "en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt"
BASE_CONFIG_REL = "en/en_US/lessac/medium/config.json"
CLEAN_REL = "base_clean/lessac-medium.ckpt"   # ckpt with legacy hparams stripped
SAMPLE_RATE = 22050
ESPEAK_VOICE = "en-us"

# model.* hyper-parameters the current piper1-gpl CLI accepts; the rhasspy ckpt
# carries extra legacy keys (e.g. sample_bytes) that break Lightning's resume.
VALID_MODEL_HPARAMS = {
    "sample_rate", "num_speakers", "resblock", "resblock_kernel_sizes",
    "resblock_dilation_sizes", "upsample_rates", "upsample_initial_channel",
    "upsample_kernel_sizes", "filter_length", "hop_length", "win_length",
    "mel_channels", "mel_fmin", "mel_fmax", "inter_channels", "hidden_channels",
    "filter_channels", "n_heads", "n_layers", "kernel_size", "p_dropout",
    "n_layers_q", "use_spectral_norm", "gin_channels", "use_sdp", "segment_size",
    "learning_rate", "learning_rate_d", "betas", "betas_d", "eps", "lr_decay",
    "lr_decay_d", "init_lr_ratio", "warmup_epochs", "c_mel", "c_kl", "grad_clip",
    "vocoder_warmstart_ckpt", "dataset",
}

# --- volumes: data (inputs) kept separate from work (base ckpt, cache, outputs) --
data_vol = modal.Volume.from_name("atc-tts-data", create_if_missing=True)
work_vol = modal.Volume.from_name("atc-tts-work", create_if_missing=True)
DATA_DIR = "/data"          # /data/controller_1/{wavs, metadata.csv}
WORK_DIR = "/work"          # /work/base, /work/cache, /work/out
PIPER_DIR = "/opt/piper"

LOCAL_DATA = Path(__file__).parent / "data"

app = modal.App("atc-tts-piper")

# Build the Piper training environment once; cached as an image layer.
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake", "ninja-build", "git", "espeak-ng")
    .run_commands(
        f"git clone https://github.com/OHF-Voice/piper1-gpl.git {PIPER_DIR}",
        f"cd {PIPER_DIR} && pip install -e '.[train]'",
        f"cd {PIPER_DIR} && ./build_monotonic_align.sh",
        # setup.py imports skbuild and needs cmake>=3.26 (debian's apt cmake is
        # 3.25); pip cmake lands in /usr/local/bin ahead of /usr/bin on PATH.
        "pip install scikit-build cmake ninja",
        f"cd {PIPER_DIR} && python setup.py build_ext --inplace",
    )
    .pip_install("huggingface_hub", "onnxscript")  # onnxscript: torch.onnx export
)


# --------------------------------------------------------------------------------
# 1. DATA LOADING: local clips -> Modal Volume
# --------------------------------------------------------------------------------
@app.local_entrypoint()
def upload_data():
    """Push tts_training/data/ into the `atc-tts-data` Volume (run from the repo)."""
    if not LOCAL_DATA.exists():
        raise SystemExit(f"{LOCAL_DATA} not found -- run prepare_data.py first")
    print(f"uploading {LOCAL_DATA} -> volume atc-tts-data:/ ...")
    with data_vol.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_DATA), "/")
    print("done. contents now under /data on Modal.")


# --------------------------------------------------------------------------------
# 2. Pull the base checkpoint into the work volume (one-time)
# --------------------------------------------------------------------------------
@app.function(image=train_image, volumes={WORK_DIR: work_vol}, timeout=1800)
def fetch_base():
    from huggingface_hub import hf_hub_download

    base_root = f"{WORK_DIR}/base"
    for rel in (BASE_CKPT_REL, BASE_CONFIG_REL):
        p = hf_hub_download(repo_id=HF_REPO, repo_type="dataset",
                            filename=rel, local_dir=base_root)
        print("fetched", p)
    work_vol.commit()


# --------------------------------------------------------------------------------
# 2b. Strip legacy hyper_parameters from the base ckpt (one-time, run after fetch)
# --------------------------------------------------------------------------------
@app.function(image=train_image, volumes={WORK_DIR: work_vol}, timeout=1800)
def clean_base():
    """Rewrite the base ckpt keeping only model hparams the current model accepts.

    Lightning re-parses ckpt['hyper_parameters'] on resume; unknown legacy keys
    (e.g. sample_bytes) abort the run. We keep the weights/optimizer/epoch state
    intact and only prune the hparam dict.
    """
    import torch

    src = f"{WORK_DIR}/base/{BASE_CKPT_REL}"
    dst = f"{WORK_DIR}/{CLEAN_REL}"
    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    removed = sorted(k for k in hp if k not in VALID_MODEL_HPARAMS)
    for k in removed:
        hp.pop(k, None)
    ckpt["hyper_parameters"] = hp

    # Reset training progress so fine-tuning starts at epoch 0. The base's loop
    # state (not ckpt['epoch']) is what Lightning restores as current_epoch; left
    # intact it immediately trips max_epochs. Dropping 'loops' + zeroing the
    # counters makes Lightning warm-start the weights but run a fresh schedule.
    had_loops = "loops" in ckpt
    ckpt.pop("loops", None)
    ckpt["epoch"] = 0
    ckpt["global_step"] = 0

    print("top-level keys:", sorted(ckpt.keys()))
    print(f"had loops: {had_loops} | reset epoch/global_step -> 0")
    print("kept hparams:", sorted(hp.keys()))
    print("removed hparams:", removed)
    torch.save(ckpt, dst)
    work_vol.commit()
    print("wrote cleaned ckpt ->", dst)


# --------------------------------------------------------------------------------
# 3. THE DATA-LOADING METHOD (what training reads): mount volume, parse + validate
# --------------------------------------------------------------------------------
@app.function(image=train_image, volumes={DATA_DIR: data_vol}, timeout=600)
def load_dataset(controller: str = "controller_1") -> dict:
    """Load + validate one controller dataset from the Volume.

    Returns the (filename, text) pairs Piper will train on, and asserts every
    referenced wav exists. This is the single source of truth the trainer uses.
    """
    import csv
    import wave

    root = Path(DATA_DIR) / controller
    meta = root / "metadata.csv"
    wavs = root / "wavs"
    assert meta.exists(), f"missing {meta}"

    rows, total_dur, missing = [], 0.0, []
    with open(meta, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) < 2 or not row[0].strip() or not row[1].strip():
                continue
            fname, text = row[0].strip(), row[1].strip()
            wp = wavs / fname
            if not wp.exists():
                missing.append(fname)
                continue
            with wave.open(str(wp), "rb") as w:
                assert w.getframerate() == SAMPLE_RATE, f"{fname}: {w.getframerate()} Hz"
                total_dur += w.getnframes() / float(w.getframerate())
            rows.append((fname, text))

    summary = {
        "controller": controller,
        "clips": len(rows),
        "minutes": round(total_dur / 60.0, 2),
        "missing": missing,
        "sample_rate": SAMPLE_RATE,
    }
    print(summary)
    if missing:
        print("WARNING missing wavs:", missing)
    return summary


# --------------------------------------------------------------------------------
# 4. Fine-tune from the base checkpoint
# --------------------------------------------------------------------------------
@app.function(image=train_image, gpu="A10G",
              volumes={DATA_DIR: data_vol, WORK_DIR: work_vol}, timeout=3 * 3600)
def train(controller: str = "controller_1", max_epochs: int = 500,
          batch_size: int = 16, ckpt_every: int = 50):
    base_ckpt = f"{WORK_DIR}/{CLEAN_REL}"   # cleaned by clean_base()
    out_dir = f"{WORK_DIR}/out/{controller}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Checkpoint ladder: snapshot every `ckpt_every` epochs and KEEP THEM ALL
    # (save_top_k=-1). VITS val loss isn't a reliable quality signal, so the
    # safeguard against overfitting is picking the best-sounding snapshot by ear.
    # Providing our own ModelCheckpoint also replaces Lightning's default (which
    # kept only the latest checkpoint).
    cfg_path = f"{out_dir}/ckpt_ladder.yaml"
    with open(cfg_path, "w") as f:
        f.write(
            "trainer:\n"
            "  callbacks:\n"
            "    - class_path: lightning.pytorch.callbacks.ModelCheckpoint\n"
            "      init_args:\n"
            f"        every_n_epochs: {ckpt_every}\n"
            "        save_top_k: -1\n"
            "        save_last: true\n"
            "        filename: '{epoch}-{step}'\n"
        )

    # clean_base() zeroed the resumed epoch/loop counters, so the warm-started
    # run begins at epoch 0 and max_epochs is simply the # of fine-tune epochs.
    abs_max_epochs = max_epochs

    # torch 2.12 defaults torch.load(weights_only=True), which rejects the
    # trusted rhasspy ckpt (it pickles a pathlib.PosixPath). Inject a
    # sitecustomize that flips the default back; auto-imported via PYTHONPATH.
    helper = "/opt/torchcompat"
    os.makedirs(helper, exist_ok=True)
    with open(f"{helper}/sitecustomize.py", "w") as f:
        f.write(
            "import torch as _t\n"
            "_orig = _t.load\n"
            "def _load(*a, **k):\n"
            "    k['weights_only'] = False  # force-override Lightning's explicit True\n"
            "    return _orig(*a, **k)\n"
            "_t.load = _load\n"
        )
    env = {**os.environ, "PYTHONPATH": helper}

    cmd = [
        "python", "-m", "piper.train", "fit",
        "--config", cfg_path,
        "--data.voice_name", f"atc_{controller}",
        "--data.csv_path", f"{DATA_DIR}/{controller}/metadata.csv",
        "--data.audio_dir", f"{DATA_DIR}/{controller}/wavs",
        "--model.sample_rate", str(SAMPLE_RATE),
        "--data.espeak_voice", ESPEAK_VOICE,
        "--data.cache_dir", f"{WORK_DIR}/cache/{controller}",
        "--data.config_path", f"{out_dir}/config.json",
        "--data.batch_size", str(batch_size),
        "--ckpt_path", base_ckpt,
        "--trainer.max_epochs", str(abs_max_epochs),
        "--trainer.default_root_dir", out_dir,
    ]
    print(f"RUN (fine-tune {max_epochs} epochs, snapshot every {ckpt_every}):",
          " ".join(cmd))
    subprocess.run(cmd, cwd=PIPER_DIR, check=True, env=env)
    work_vol.commit()
    print(f"checkpoints + config.json under {out_dir}")


# --------------------------------------------------------------------------------
# 5. Export the trained checkpoint to ONNX (CPU-runtime artifact for atc-sim)
# --------------------------------------------------------------------------------
@app.function(image=train_image, volumes={WORK_DIR: work_vol}, timeout=1800)
def export(controller: str = "controller_1", ckpt: str = ""):
    out_dir = Path(f"{WORK_DIR}/out/{controller}")
    if not ckpt:
        ckpts = sorted(out_dir.rglob("*.ckpt"))
        if not ckpts:
            raise SystemExit(f"no .ckpt found under {out_dir}; pass --ckpt")
        ckpt = str(ckpts[-1])
    onnx_path = out_dir / f"atc_{controller}.onnx"

    # torch 2.12 defaults torch.onnx.export to the dynamo exporter, which fails on
    # VITS + dynamic_axes; force the legacy TorchScript exporter. Also keep the
    # weights_only=False patch since export loads the (trusted) ckpt.
    helper = "/opt/torchcompat"
    os.makedirs(helper, exist_ok=True)
    with open(f"{helper}/sitecustomize.py", "w") as f:
        f.write(
            "import torch as _t\n"
            "_ol = _t.load\n"
            "def _load(*a, **k):\n"
            "    k['weights_only'] = False\n"
            "    return _ol(*a, **k)\n"
            "_t.load = _load\n"
            "_oe = _t.onnx.export\n"
            "def _export(*a, **k):\n"
            "    k['dynamo'] = False\n"
            "    return _oe(*a, **k)\n"
            "_t.onnx.export = _export\n"
        )
    env = {**os.environ, "PYTHONPATH": helper}

    cmd = ["python", "-m", "piper.train.export_onnx",
           "--checkpoint", ckpt, "--output-file", str(onnx_path)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=PIPER_DIR, check=True, env=env)
    work_vol.commit()
    print(f"exported {onnx_path} (pair with {out_dir}/config.json)")


# --------------------------------------------------------------------------------
# 6. Synthesize a sample wav from an exported .onnx (CPU inference) and return it
# --------------------------------------------------------------------------------
@app.function(image=train_image, volumes={WORK_DIR: work_vol}, timeout=600)
def synth(controller: str = "controller_2",
          text: str = ("Cactus eight twenty six, turn right heading one niner zero, "
                       "descend and maintain three thousand.")) -> bytes:
    """Run the piper CPU runtime on the exported onnx; return wav bytes (no commit
    -> safe to run while training writes to the same volume)."""
    import shutil

    out_dir = f"{WORK_DIR}/out/{controller}"
    onnx = f"{out_dir}/atc_{controller}.onnx"
    beside = onnx + ".json"            # piper looks for <model>.json next to it
    if not Path(beside).exists():
        shutil.copyfile(f"{out_dir}/config.json", beside)
    wav = "/tmp/sample.wav"
    subprocess.run(["python", "-m", "piper", "-m", onnx, "-f", wav, "--", text],
                   cwd=PIPER_DIR, check=True)
    return Path(wav).read_bytes()


@app.local_entrypoint()
def make_sample(controller: str = "controller_2", text: str = ""):
    """Synthesize a sample on Modal and save it next to the local model copy."""
    kwargs = {"controller": controller}
    if text:
        kwargs["text"] = text
    audio = synth.remote(**kwargs)
    out = Path(__file__).parent / "models" / f"{controller}_smoke" / "sample.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio)
    print(f"wrote {out} ({len(audio)} bytes)")


# --------------------------------------------------------------------------------
# 7. Bundle one ladder rung: export its onnx + synth a sample, return the bytes
# --------------------------------------------------------------------------------
@app.function(image=train_image, volumes={WORK_DIR: work_vol}, timeout=1800)
def bundle_epoch(controller: str = "controller_2", epoch: int = 0,
                 text: str = ("Cactus eight twenty six, turn right heading one niner zero, "
                              "descend and maintain three thousand.")) -> dict:
    """Export the epoch-N ladder checkpoint to ONNX and synthesize a sample.

    Returns onnx + config + wav bytes; performs no volume writes/commits, so it's
    safe to run against the live training volume.
    """
    import glob
    import shutil

    versions = sorted(glob.glob(f"{WORK_DIR}/out/{controller}/lightning_logs/version_*"))
    ckpt = None
    for v in reversed(versions):                      # newest run first
        hits = sorted(glob.glob(f"{v}/checkpoints/epoch={epoch}-*.ckpt"))
        if hits:
            ckpt = hits[0]
            break
    if ckpt is None:
        raise SystemExit(f"no epoch={epoch} checkpoint found for {controller}")

    # weights_only (load) + legacy onnx exporter (dynamo=False), as in export().
    helper = "/opt/torchcompat"
    os.makedirs(helper, exist_ok=True)
    with open(f"{helper}/sitecustomize.py", "w") as f:
        f.write(
            "import torch as _t\n"
            "_ol = _t.load\n"
            "def _load(*a, **k):\n"
            "    k['weights_only'] = False\n"
            "    return _ol(*a, **k)\n"
            "_t.load = _load\n"
            "_oe = _t.onnx.export\n"
            "def _export(*a, **k):\n"
            "    k['dynamo'] = False\n"
            "    return _oe(*a, **k)\n"
            "_t.onnx.export = _export\n"
        )
    env = {**os.environ, "PYTHONPATH": helper}

    onnx = "/tmp/model.onnx"
    subprocess.run(["python", "-m", "piper.train.export_onnx",
                    "--checkpoint", ckpt, "--output-file", onnx],
                   cwd=PIPER_DIR, check=True, env=env)
    shutil.copyfile(f"{WORK_DIR}/out/{controller}/config.json", onnx + ".json")
    wav = "/tmp/sample.wav"
    subprocess.run(["python", "-m", "piper", "-m", onnx, "-f", wav, "--", text],
                   cwd=PIPER_DIR, check=True, env=env)
    print(f"bundled {controller} epoch={epoch} from {ckpt}")
    return {"epoch": epoch,
            "onnx": Path(onnx).read_bytes(),
            "config": Path(onnx + ".json").read_bytes(),
            "wav": Path(wav).read_bytes()}


@app.local_entrypoint()
def pull_rungs(controller: str = "controller_2", epochs: str = "", text: str = ""):
    """Export+sample the given comma-separated epochs; save each rung under
    models/<controller>/epoch_NNNN/{onnx, onnx.json, sample.wav}."""
    eps = [int(e) for e in str(epochs).split(",") if str(e).strip()]
    if not eps:
        raise SystemExit("pass --epochs, e.g. --epochs 49,99,149")
    base = Path(__file__).parent / "models" / controller
    for ep in eps:
        kw = {"controller": controller, "epoch": ep}
        if text:
            kw["text"] = text
        b = bundle_epoch.remote(**kw)
        d = base / f"epoch_{ep:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"atc_{controller}.onnx").write_bytes(b["onnx"])
        (d / f"atc_{controller}.onnx.json").write_bytes(b["config"])
        (d / "sample.wav").write_bytes(b["wav"])
        print(f"saved {d} ({len(b['onnx'])} B onnx, {len(b['wav'])} B wav)")
