"""Populate tts_training/data/ from the cleaned ATC clips in the tts_model project.

Source (already cleaned by tts_model/data_cleaning.py):
    <SRC>/dataset_controller_1/{wavs/*.wav, metadata.csv}
    <SRC>/dataset_controller_2/{wavs/*.wav, metadata.csv}

Output (Piper single-speaker layout, one dataset per controller voice):
    tts_training/data/controller_1/{wavs/*.wav, metadata.csv}
    tts_training/data/controller_2/{wavs/*.wav, metadata.csv}

metadata.csv is Piper's single-speaker format:  <filename>.wav|<transcript>

This is intentionally a thin copy + light QA filter. The clips are already
22050 Hz / mono / 16-bit PCM (Piper-native), so no resampling is needed.
Run:  python tts_training/prepare_data.py
"""
from __future__ import annotations

import argparse
import csv
import shutil
import wave
from pathlib import Path

# Light QA bounds. ATC clips were sliced from timestamps, so a few are tiny
# fragments or very long multi-aircraft turns; both hurt VITS fine-tuning.
MIN_DUR_S = 0.6
MAX_DUR_S = 13.0

REPO_DIR = Path(__file__).resolve().parent          # tts_training/
OUT_ROOT = REPO_DIR / "data"
DEFAULT_SRC = Path(r"C:\Users\zirui\OneDrive\Desktop\tts_model")


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


def load_metadata(csv_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                rows.append((row[0].strip(), row[1].strip()))
    return rows


def prepare_controller(num: int, src_root: Path) -> dict:
    src_dir = src_root / f"dataset_controller_{num}"
    src_wavs = src_dir / "wavs"
    src_meta = src_dir / "metadata.csv"
    if not src_meta.exists():
        raise FileNotFoundError(f"missing {src_meta} -- check --src path")

    out_dir = OUT_ROOT / f"controller_{num}"
    out_wavs = out_dir / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)

    kept: list[tuple[str, str]] = []
    dropped: list[tuple[str, str]] = []
    total_dur = 0.0

    for fname, text in load_metadata(src_meta):
        src_wav = src_wavs / fname
        if not src_wav.exists():
            dropped.append((fname, "missing wav"))
            continue
        dur = wav_duration(src_wav)
        if dur < MIN_DUR_S:
            dropped.append((fname, f"too short {dur:.2f}s"))
            continue
        if dur > MAX_DUR_S:
            dropped.append((fname, f"too long {dur:.2f}s"))
            continue
        shutil.copy2(src_wav, out_wavs / fname)
        kept.append((fname, text))
        total_dur += dur

    with open(out_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", lineterminator="\n")
        for fname, text in kept:
            w.writerow([fname, text])

    return {
        "num": num,
        "kept": len(kept),
        "dropped": len(dropped),
        "minutes": total_dur / 60.0,
        "dropped_detail": dropped,
        "out_dir": out_dir,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="tts_model project root holding dataset_controller_{1,2}")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for num in (1, 2):
        r = prepare_controller(num, args.src)
        print(f"controller_{num}: kept {r['kept']} clips "
              f"({r['minutes']:.1f} min), dropped {r['dropped']} -> {r['out_dir']}")
        for fname, why in r["dropped_detail"]:
            print(f"    drop {fname}: {why}")


if __name__ == "__main__":
    main()
