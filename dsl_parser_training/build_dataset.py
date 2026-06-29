"""Build a fresh (input=script, target=DSL) training batch for dsl_parser.

NO TTS/STT. The training input is the *standard ATC script* generated directly
from a random valid DSL command, then expanded with cheap text augmentation:

    random valid command            (generate_commands.random_command)
      -> fresh Aircraft + process_command    [VALIDATOR FILTER: keep ok=True]
      -> ATC phraseology                      (aircraft._build_radio_messages)
      -> telephony script                     (to_script: "DL377" -> "Delta 377")
      -> canonical DSL target                 (generate_commands.canonicalize_target)
    => {"input": <script>, "target": "<CALLSIGN> <DSL>"}   (+ aug_k augmented variants)

Augmentation (dsl_parser_training.augment) supplies the surface variety the parser
must tolerate (number word-forms, commas, "correction" self-edits, ...). Real ASR
garble at inference is handled by the runtime validator/error path -- not trained.

Generation is pure CPU text (no models), so batches are fast and cheap.

Usage:
    python -m dsl_parser_training.build_dataset --n 50 --verify       # sample files
    python -m dsl_parser_training.build_dataset --n 5000 --out data/batch.jsonl
    (full train/val generation normally runs on Modal: modal_app.py)
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import SimulationEnv
from environment.core.aircraft import Aircraft
from tts.normalize import AIRLINES
from dsl_parser_training.generate_commands import (
    random_callsign, random_command, random_init_state, canonicalize_target,
)
from dsl_parser_training.augment import augment_one

HERE = os.path.dirname(os.path.abspath(__file__))


def _airport_vocab():
    """(coords, nm_per_pixel, waypoints, runways) for the SIMULATED airport."""
    sim = SimulationEnv(radar_side=800, airport_name="test", star_mode=True)
    runways = []
    for rwy in sim.data["runways"].values():
        runways += list(rwy["thresholds"].keys())
    waypoints = (list(sim.data["vor_stations"]) + list(sim.data["ndb_stations"])
                 + list(sim.data["rnav_waypoints"]))
    return sim.coords, sim.nm_per_pixel, sorted(set(waypoints)), sorted(set(runways))


def to_script(atc, callsign):
    """Convert the raw ATC string's ICAO callsign to spoken telephony + digits
    ('DL377, turn left ...' -> 'Delta 377, turn left ...'), so the script matches
    what a controller says (and what an ASR front-end would yield). The DSL target
    keeps the ICAO callsign."""
    m = re.match(r"([A-Za-z]+)(\d+)", callsign)
    if not m:
        return atc
    name = AIRLINES.get(m.group(1).upper(), m.group(1))
    return atc.replace(callsign, f"{name} {m.group(2)}", 1)


def generate_rows(n, rng, coords, nmpp, waypoints, runways, log_every=1000):
    """Yield n valid {input: script, target: DSL, atc} rows."""
    made = 0
    while made < n:
        callsign = random_callsign(rng)
        st = random_init_state(rng)
        command = random_command(rng, waypoints, runways)

        # A go-around only validates when the aircraft is cleared for ILS and is
        # realistically low/on approach -- set that up so the phraseology reads
        # like a real go-around ("go around, ..., climb and maintain ...").
        altitude = st["altitude"]
        if command.startswith("A"):
            altitude = rng.randrange(1000, 2001, 500)   # low, on approach -> A climbs above it

        # VALIDATOR FILTER: a fresh aircraft applies the command; reject if not ok.
        ac = Aircraft(callsign, 0, 0, st["heading"], altitude, st["airspeed"], nmpp, coords)
        if command.startswith("A"):
            ac.ils_runway = rng.choice(runways)
        res = ac.process_command(command)
        if not res.get("ok"):
            continue
        atc = res["atc"]
        target = f"{callsign} {canonicalize_target(command, atc)}"
        made += 1
        if log_every and made % log_every == 0:
            print(f"  {made}/{n}")
        yield {"input": to_script(atc, callsign), "target": target, "atc": atc}


def generate_dataset(n_train, n_val, aug_k, seed, train_path, val_path):
    """Write train.jsonl + val.jsonl.

      train: each base script -> the script PLUS aug_k augmented variants.
      val:   each base script -> ONE row, randomly the script OR one augmented
             variant. val draws fresh commands (rng continues), so it never shares
             base commands with train.
    """
    rng = random.Random(seed)
    coords, nmpp, waypoints, runways = _airport_vocab()
    print(f"vocab: {len(waypoints)} waypoints, {len(runways)} runways")
    os.makedirs(os.path.dirname(train_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(val_path) or ".", exist_ok=True)

    n_tr = 0
    with open(train_path, "w", encoding="utf-8") as f:
        for row in generate_rows(n_train, rng, coords, nmpp, waypoints, runways):
            f.write(json.dumps({"input": row["input"], "target": row["target"]}) + "\n")
            n_tr += 1
            for v in augment_one(row["input"], rng, aug_k):
                f.write(json.dumps({"input": v, "target": row["target"]}) + "\n")
                n_tr += 1

    n_va = 0
    with open(val_path, "w", encoding="utf-8") as f:
        for row in generate_rows(n_val, rng, coords, nmpp, waypoints, runways):
            if rng.random() < 0.5:                       # randomly augment or not
                aug = augment_one(row["input"], rng, 1)
                text = aug[0] if aug else row["input"]
            else:
                text = row["input"]
            f.write(json.dumps({"input": text, "target": row["target"]}) + "\n")
            n_va += 1

    print(f"train rows: {n_tr}  val rows: {n_va}")
    return {"train": n_tr, "val": n_va}


def main():
    ap = argparse.ArgumentParser(description="Build a script->DSL batch (no TTS/STT).")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "batch.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verify", action="store_true",
                    help="write sample_commands.txt + sample_pairs.txt for eyeballing")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    coords, nmpp, waypoints, runways = _airport_vocab()
    rows = list(generate_rows(args.n, rng, coords, nmpp, waypoints, runways))

    if args.verify:
        cmd_path = os.path.join(HERE, "sample_commands.txt")
        pair_path = os.path.join(HERE, "sample_pairs.txt")
        with open(cmd_path, "w", encoding="utf-8") as f:
            f.write("\n".join(r["target"] for r in rows) + "\n")
        with open(pair_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"{r['target']}  ->  {r['input']}\n")
        print(f"verify files:\n  {cmd_path}\n  {pair_path}")
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"input": r["input"], "target": r["target"]}) + "\n")
        print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
