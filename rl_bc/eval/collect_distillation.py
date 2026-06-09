"""Distillation rollout collector.

Runs the bc_fm_single policy (or any BC checkpoint) STAR-by-STAR and keeps
rolling until N LANDED trajectories are collected for each of the 6 STARs.
Each successful rollout is logged through the same `HumanDataRecorder` used
by the human-play recorder, and ALL successful rollouts get appended into
ONE concatenated CSV at `human_data/bc_fm_single_distillation/<ts>_distill.csv`
(same single-file layout as `human_data/single_plane/`). Drop the folder
into training and the loader picks it up like any other human episode.

Failed rollouts (TIMEOUT, IMPROPER_EXIT, CRASHED) are discarded; we just
keep drawing new seeds until each STAR's target is met.

The model's inference path is NOT touched — `_RUNTIME.tick(sim, armed)` is
the same code path watch / runner use.

Usage (single command, defaults already do what you want):
    python -m rl_bc.eval.collect_distillation
    python -m rl_bc.eval.collect_distillation --target-per-star 50
    python -m rl_bc.eval.collect_distillation --config bc_gmm_single \
        --out-dir human_data/bc_gmm_single_distillation
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


STARS = ['NORTH1', 'NORTH2', 'NORTH3', 'SOUTH1', 'SOUTH2', 'SOUTH3']


# --------------------------------------------------------------------------- #
# Worker — same runtime-load + BLAS-pin pattern as eval/runner.py.
# --------------------------------------------------------------------------- #


_RUNTIME = None


def _load_runtime(ckpt_path: str):
    import torch
    peek = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    keys = list(peek['model_state'].keys())
    if any(k.startswith('head_logits.') for k in keys):
        from rl_bc.bc_gmm.rollout import Runtime
        return Runtime(ckpt_path)
    from rl_bc.bc_fm.rollout import Runtime
    return Runtime(ckpt_path)


def _init_worker(ckpt_path: str) -> None:
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    global _RUNTIME
    _RUNTIME = _load_runtime(ckpt_path)


def _run_case_logged(job: tuple) -> dict:
    """Roll one (star, seed) with a HumanDataRecorder attached. Returns
    the outcome + the recorded CSV text (only populated on LANDED)."""
    import random as rnd
    import traceback
    import numpy as np

    star_name, seed, max_steps, warmup_wpts, runway, airport = job
    rnd.seed(seed)
    np.random.seed(seed)

    callsign: str | None = None
    recorder = None
    try:
        from environment import SimulationEnv
        from environment.core.human_data_logger import HumanDataRecorder

        runtime = _RUNTIME
        if runtime is None:
            return {'star': star_name, 'seed': seed, 'outcome': 'INIT_FAILED',
                    'csv': ''}

        runtime.reset()
        recorder = HumanDataRecorder(spawn_single=True, in_memory=True)
        recorder.start()

        sim = SimulationEnv(airport_name=airport, spawn_single=True,
                            star_mode=True, recorder=recorder)
        if star_name not in sim.spawner.procedures:
            return {'star': star_name, 'seed': seed, 'outcome': 'STAR_MISSING',
                    'csv': ''}
        sim.spawner.procedures = {star_name: sim.spawner.procedures[star_name]}
        sim.spawner.last_spawned_star = None

        sim.step(1.0)
        if not sim.aircraft_list:
            return {'star': star_name, 'seed': seed, 'outcome': 'NO_SPAWN',
                    'csv': ''}

        callsign = next(iter(sim.aircraft_list.keys()))
        armed: set[str] = set()
        initial_star_len: dict[str, int] = {}
        landed_before = sim.num_landed
        improper_before = sim.improper_exits

        for step in range(max_steps):
            for cs, ac in sim.aircraft_list.items():
                if cs in armed:
                    continue
                if cs not in initial_star_len:
                    initial_star_len[cs] = len(ac.star) if ac.star else 0
                    continue
                initial = initial_star_len[cs]
                current = len(ac.star) if ac.star else 0
                popped = initial - current
                threshold = min(warmup_wpts, initial) if initial > 0 else 0
                if initial == 0 or popped >= threshold:
                    armed.add(cs)
                    if ac.star is not None or ac.target_wpt is not None:
                        ac.star = None
                        ac.star_name = None
                        ac.target_wpt = None
                    res = sim.command(cs, f"L {runway}")
                    if res.get('ok'):
                        runtime.state_for(cs).cleared = True

            runtime.tick(sim, armed=armed)
            sim.step(1.0)

            if callsign not in sim.aircraft_list:
                if sim.num_landed > landed_before:
                    outcome = 'LANDED'
                elif sim.improper_exits > improper_before:
                    outcome = 'IMPROPER_EXIT'
                else:
                    outcome = 'REMOVED_UNKNOWN'
                csv_text = recorder.to_csv() if outcome == 'LANDED' else ''
                recorder.close()
                return {'star': star_name, 'seed': seed, 'outcome': outcome,
                        'callsign': callsign, 'steps_used': step + 1,
                        'csv': csv_text}

        recorder.close()
        return {'star': star_name, 'seed': seed, 'outcome': 'TIMEOUT',
                'callsign': callsign, 'steps_used': max_steps, 'csv': ''}

    except Exception as e:
        try:
            if recorder is not None:
                recorder.close()
        except Exception:
            pass
        return {'star': star_name, 'seed': seed, 'outcome': 'CRASHED',
                'callsign': callsign or '?',
                'error_type': type(e).__name__, 'error': str(e),
                'traceback': traceback.format_exc(limit=3),
                'csv': ''}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _resolve_ckpt(ckpt: Optional[str], config: str,
                  run: Optional[int]) -> Path:
    if ckpt:
        return Path(ckpt)
    run_dir_root = Path(f'rl_bc/runs/{config}')
    if not run_dir_root.exists():
        raise SystemExit(f"no runs under {run_dir_root}/ — train first")
    if run is not None:
        path = run_dir_root / f'run_{run}' / 'best.pt'
        if not path.exists():
            raise SystemExit(f"checkpoint not found: {path}")
        return path
    runs = []
    for sub in run_dir_root.glob('run_*'):
        if (sub / 'best.pt').exists():
            try:
                runs.append((int(sub.name.split('_')[1]), sub))
            except (ValueError, IndexError):
                continue
    if not runs:
        raise SystemExit(f"no run_*/best.pt under {run_dir_root}")
    runs.sort()
    return runs[-1][1] / 'best.pt'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', type=str, default='bc_fm_single')
    ap.add_argument('--ckpt', type=str, default=None,
                    help='Explicit checkpoint path. Overrides --run / --config.')
    ap.add_argument('--run', type=int, default=None)
    ap.add_argument('--target-per-star', type=int, default=30,
                    help='Number of LANDED trajectories required per STAR.')
    ap.add_argument('--max-steps', type=int, default=1500)
    ap.add_argument('--warmup-wpts', type=int, default=2)
    ap.add_argument('--runway', type=str, default='27')
    ap.add_argument('--airport', type=str, default='test')
    ap.add_argument('--workers', type=int,
                    default=max(1, mp.cpu_count() // 2))
    ap.add_argument('--seed-base', type=int, default=0)
    ap.add_argument('--out-dir', type=str,
                    default='human_data/bc_fm_single_distillation',
                    help='Folder containing the single concatenated CSV.')
    ap.add_argument('--out-name', type=str, default=None,
                    help='Filename for the concatenated CSV inside --out-dir. '
                         'Defaults to <YYYYMMDD_HHMMSS>_distill.csv.')
    args = ap.parse_args()

    ckpt_path = _resolve_ckpt(args.ckpt, args.config, args.run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or (
        datetime.now().strftime('%Y%m%d_%H%M%S') + '_distill.csv')
    out_path = out_dir / out_name

    target = args.target_per_star
    print(f"Distillation rollout collection")
    print(f"  ckpt        : {ckpt_path}")
    print(f"  target      : {target}/STAR × {len(STARS)} STARs = "
          f"{target * len(STARS)} successful trajectories")
    print(f"  out_path    : {out_path}")
    print(f"  workers     : {args.workers}")
    print(f"  max_steps   : {args.max_steps}")
    print(f"  warmup_wpts : {args.warmup_wpts}\n", flush=True)

    pool_start = time.time()
    landed_per_star: dict[str, int] = {s: 0 for s in STARS}
    attempted_per_star: dict[str, int] = {s: 0 for s in STARS}

    # Single concatenated CSV: write the schema header once, then append the
    # body (header-stripped) of each successful rollout. Flushed per rollout
    # so a mid-run crash still leaves a parseable partial file.
    from environment.core.human_data_logger import FIELDNAMES
    out_file = out_path.open('w', newline='', encoding='utf-8')
    out_file.write(','.join(FIELDNAMES) + '\n')
    out_file.flush()

    try:
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(str(ckpt_path),)) as pool:
            print(f"  pool ready in {time.time() - pool_start:.1f}s\n",
                  flush=True)
            for star in STARS:
                seed = args.seed_base
                t_star = time.time()
                print(f"=== {star}: collecting {target} successful "
                      f"trajectories ===", flush=True)
                while landed_per_star[star] < target:
                    # Headroom: assume ~30% success → batch sized to land
                    # what's still needed in roughly one round, plus a buffer.
                    still_need = target - landed_per_star[star]
                    batch = max(20, min(150, int(still_need / 0.30) + 5))
                    jobs = [(star, seed + i, args.max_steps, args.warmup_wpts,
                             args.runway, args.airport) for i in range(batch)]
                    seed += batch
                    for r in pool.imap_unordered(_run_case_logged, jobs,
                                                 chunksize=2):
                        attempted_per_star[star] += 1
                        outcome = r.get('outcome', '?')
                        if outcome == 'LANDED' and landed_per_star[star] < target:
                            landed_per_star[star] += 1
                            # Strip the per-rollout header before appending.
                            csv_text = r['csv']
                            body = csv_text.split('\n', 1)[1] if '\n' in csv_text else ''
                            if body and not body.endswith('\n'):
                                body += '\n'
                            out_file.write(body)
                            out_file.flush()
                            succ_pct = (landed_per_star[star]
                                        / attempted_per_star[star] * 100)
                            n_rows = body.count('\n')
                            print(f"  [{star}] {landed_per_star[star]:>2}/{target}  "
                                  f"seed={r['seed']:<5}  "
                                  f"steps={r.get('steps_used', 0):>4}  "
                                  f"rows={n_rows:>4}  "
                                  f"attempts={attempted_per_star[star]:<4}  "
                                  f"succ%={succ_pct:>5.1f}",
                                  flush=True)
                            if landed_per_star[star] >= target:
                                break
                        elif outcome == 'CRASHED':
                            print(f"  [{star}] CRASHED seed={r['seed']}: "
                                  f"{r.get('error_type', '?')}: "
                                  f"{r.get('error', '?')[:60]}", flush=True)
                elapsed = time.time() - t_star
                print(f"=== {star} done: {landed_per_star[star]}/{target} "
                      f"landed in {attempted_per_star[star]} attempts, "
                      f"{elapsed:.0f}s ===\n", flush=True)
    finally:
        out_file.close()

    total_landed = sum(landed_per_star.values())
    total_attempts = sum(attempted_per_star.values())
    overall_pct = (total_landed / total_attempts * 100) if total_attempts else 0
    print(f"\nAll done.")
    print(f"  wrote {total_landed} trajectories → {out_path}")
    print(f"  total attempts: {total_attempts}  "
          f"(overall success: {overall_pct:.1f}%)")
    for s in STARS:
        a = attempted_per_star[s]
        pct = (landed_per_star[s] / a * 100) if a else 0
        print(f"    {s}: {landed_per_star[s]}/{target}  "
              f"({a} attempts, {pct:.1f}% success)")


if __name__ == '__main__':
    main()
