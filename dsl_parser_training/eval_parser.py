"""Evaluate the trained dsl_parser on the REAL-audio eval set (TTS->STT).

Loads the local checkpoint, parses each Whisper transcript -> DSL, and compares
to the ground-truth DSL. No validator -- this is the raw parser accuracy on real
STT output (the train/inference gap, since training used synthetic scripts only).

    python -m dsl_parser_training.eval_parser --ckpt dsl_parser_training/runs/v0

Reports exact-match and command-only match (ignoring the callsign token, which
the runtime would resolve against live traffic anyway).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "parse atc: "


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained parser on TTS->STT set.")
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(HERE), "dsl_parser", "model"))
    ap.add_argument("--eval", default=os.path.join(HERE, "eval_samples.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "eval_results.txt"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt)
    model.eval()

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    n_exact = n_cmd = 0
    lines = []
    latencies = []
    for r in rows:
        ids = tok(PREFIX + r["input"], return_tensors="pt",
                  truncation=True, max_length=96).input_ids
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, max_length=48, num_beams=4)
        latencies.append((time.perf_counter() - t0) * 1000)   # ms, single forward (batch=1)
        pred = tok.decode(out[0], skip_special_tokens=True).strip()
        tgt = r["target"].strip()

        exact = pred == tgt
        cmd_ok = " ".join(pred.split()[1:]) == " ".join(tgt.split()[1:])  # ignore callsign
        n_exact += exact
        n_cmd += cmd_ok
        mark = "OK  " if exact else ("CMD " if cmd_ok else "MISS")
        lines.append(f"[{mark}] GT  : {tgt}\n         PRED: {pred}\n         STT : {r['input']}")

    n = len(rows)
    lat = sorted(latencies)
    mean_ms = sum(lat) / len(lat)
    median_ms = lat[len(lat) // 2]
    summary = (f"\n=== dsl_parser eval on {n} TTS->STT samples ===\n"
               f"exact match (callsign+command): {n_exact}/{n} = {n_exact/n:.1%}\n"
               f"command match (ignore callsign): {n_cmd}/{n} = {n_cmd/n:.1%}\n"
               f"forward-pass latency (CPU, batch=1, beam=4): "
               f"mean {mean_ms:.0f} ms, median {median_ms:.0f} ms, "
               f"min {lat[0]:.0f} / max {lat[-1]:.0f} ms\n")
    body = "\n".join(lines) + "\n" + summary
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(body)
    print(body)


if __name__ == "__main__":
    main()
