"""Build a REAL-audio eval set: random command -> TTS -> STT -> transcript,
paired with the ground-truth DSL target.

This is the honest test of the parser: training uses *synthetic scripts* only
(no TTS/STT), so evaluating on actual Whisper transcripts measures the
train/inference gap. Saves eval_samples.jsonl (+ the wavs under eval_audio/).

    python -m dsl_parser_training.build_eval --n 50
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.core.aircraft import Aircraft
from tts.normalize import to_spoken
from stt import transcribe, BACKEND
from dsl_parser_training.generate_commands import (
    random_callsign, random_command, random_init_state, canonicalize_target,
)
from dsl_parser_training import voices
from dsl_parser_training.build_dataset import _airport_vocab

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Build a TTS->STT eval set.")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)   # different seed from training
    ap.add_argument("--out", default=os.path.join(HERE, "eval_samples.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    coords, nmpp, waypoints, runways = _airport_vocab()
    audio_dir = os.path.join(HERE, "eval_audio")
    os.makedirs(audio_dir, exist_ok=True)
    print(f"stt backend={BACKEND}; warming voices...")
    voices.warmup()

    rows = []
    made = 0
    while made < args.n:
        cs = random_callsign(rng)
        st = random_init_state(rng)
        cmd = random_command(rng, waypoints, runways)
        ac = Aircraft(cs, 0, 0, st["heading"], st["altitude"], st["airspeed"], nmpp, coords)
        res = ac.process_command(cmd)
        if not res.get("ok"):
            continue
        atc = res["atc"]
        target = f"{cs} {canonicalize_target(cmd, atc)}"
        spoken = to_spoken(atc, is_controller=True)
        vname, voice = voices.random_voice(rng)
        wav = voices.synth_wav_bytes(voice, spoken, rng)
        transcript = transcribe(wav).strip()
        made += 1
        fn = f"{made:02d}_{cs}.wav"
        with open(os.path.join(audio_dir, fn), "wb") as f:
            f.write(wav)
        rows.append({"input": transcript, "target": target, "atc": atc, "voice": vname, "wav": fn})
        print(f"{made:2d} [{vname}] {target}\n     -> {transcript}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} eval rows -> {args.out}\n     wavs -> {audio_dir}")


if __name__ == "__main__":
    main()
