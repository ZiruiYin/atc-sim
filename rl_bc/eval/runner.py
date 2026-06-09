"""Parallel BC evaluation.

Spawns `--cases` aircraft (default 500) on EACH of the 6 STAR procedures
(NORTH1/2/3, SOUTH1/2/3), runs the BC actor on each with the usual
STAR-warmup → BC-takeover flow, and terminates the rollout the moment the
aircraft establishes on the localizer. Per-STAR step caps (see
`STAR_MAX_STEPS`) bound each rollout. Each rollout returns one of:

    LOC_BELOW_GS    — LOC intercepted with altitude at/below the 3°
                      glide projection → success (aircraft is in the
                      capture window; autoland is deterministic from here)
    LOC_ABOVE_GS    — LOC intercepted but altitude above glide → failure
                      (overshot the glidepath; can't capture from above)
    IMPROPER_EXIT   — aircraft left the radar before LOC capture
    TIMEOUT         — still in the air at the STAR's step cap
    CRASHED         — sim raised inside the rollout

Per-STAR step caps (NORTH1/2 + SOUTH1/2 are long approaches, NORTH3 +
SOUTH3 are short ones):

    NORTH1, NORTH2, SOUTH1, SOUTH2 → 1200 steps
    NORTH3, SOUTH3                 →  500 steps

`--max-steps` is an optional global override that, when > 0, replaces
all six caps.

All five outputs land under `eval/<config>/` with FIXED filenames and
are overwritten on re-run:

    eval/<config>/rollouts.csv                  — per-rollout outcomes (streamed)
    eval/<config>/summary.json                  — per-STAR + overall aggregates
    eval/<config>/raw.json                      — per-rollout result dicts (no positions)
    eval/<config>/trajectories.npz              — per-tick (a,c) positions
    eval/<config>/successful_trajectories.png   — LOC_BELOW_GS rollouts only
    eval/<config>/unsuccessful_trajectories.png — everything else

Run:
    python -m rl_bc.eval.runner --config bc_fm_single
    python -m rl_bc.eval.runner --config bc_gmm_single --cases 200
    python -m rl_bc.eval.runner --ckpt rl_bc/runs/bc_fm_single/run_2/best.pt

To re-render just the two PNGs from the cached trajectories.npz
(no re-roll), use:

    python -m rl_bc.eval.viz_trajectories --eval-dir eval/bc_fm_single
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


STARS = ['NORTH1', 'NORTH2', 'NORTH3', 'SOUTH1', 'SOUTH2', 'SOUTH3']

# Per-STAR step caps (sim-seconds). NORTH3/SOUTH3 are short approaches
# that should establish on LOC in under ~500s; the others need ~1200s
# for the longer routing. A non-zero --max-steps CLI arg overrides all.
STAR_MAX_STEPS = {
    'NORTH1': 1200, 'NORTH2': 1200,
    'SOUTH1': 1200, 'SOUTH2': 1200,
    'NORTH3':  500, 'SOUTH3':  500,
}

# Outcome buckets. SUCCESS is the single positive bucket; the others are
# all forms of failure. LANDED is kept for backward compat with legacy
# CSVs but the runner no longer produces it (we terminate at LOC capture
# before autoland completes). LOC_BEHIND_THR catches a sim quirk where
# an aircraft that overshoots the runway can still "intercept" the
# localizer from the back side — physically nonsense, never a success.
SUCCESS_OUTCOMES = ('LOC_BELOW_GS', 'LANDED')
FAILURE_OUTCOMES = ('LOC_ABOVE_GS', 'LOC_BEHIND_THR',
                    'IMPROPER_EXIT', 'TIMEOUT', 'CRASHED')


# --------------------------------------------------------------------------- #
# Worker — one process loads the model once via initializer, runs many cases.
# --------------------------------------------------------------------------- #


_RUNTIME = None
_FAMILY: Optional[str] = None


def _load_runtime(ckpt_path: str):
    """Pick the right Runtime class based on state-dict keys."""
    import torch
    peek = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    keys = list(peek['model_state'].keys())
    if any(k.startswith('head_logits.') for k in keys):
        from rl_bc.bc_gmm.rollout import Runtime
        return Runtime(ckpt_path), 'bc_gmm'
    from rl_bc.bc_fm.rollout import Runtime
    return Runtime(ckpt_path), 'bc_fm'


def _init_worker(ckpt_path: str) -> None:
    """Pool-initializer: load the model once per worker process.

    Pin BLAS / OMP to 1 thread per worker. With N workers each grabbing
    M BLAS threads we'd have N*M threads competing for cores → thermal
    throttling and cache thrash. Our model is tiny (~10k params) so
    threaded BLAS is anti-productive even per-process.
    """
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)

    global _RUNTIME, _FAMILY
    _RUNTIME, _FAMILY = _load_runtime(ckpt_path)


def _run_case(job: tuple) -> dict:
    """Run one (star, seed) rollout. Returns the outcome dict.

    Wraps the simulator + runtime loop in try/except so that a sim crash
    (e.g. occasional `math.asin` domain errors in `_update_ils_loc`) ends
    THAT rollout as `CRASHED` rather than killing the whole worker pool.

    `job = (star_name, seed, max_steps, warmup_wpts, runway, airport)`.
    """
    import random as rnd
    import traceback

    import numpy as np

    import math

    star_name, seed, max_steps, warmup_wpts, runway, airport = job
    rnd.seed(seed)
    np.random.seed(seed)

    a_traj: list[float] = []
    c_traj: list[float] = []
    callsign: str | None = None
    try:
        from environment import SimulationEnv

        runtime = _RUNTIME
        if runtime is None:
            return {'star': star_name, 'seed': seed, 'outcome': 'INIT_FAILED',
                    'a_traj': a_traj, 'c_traj': c_traj}

        runtime.reset()

        sim = SimulationEnv(airport_name=airport, spawn_single=True,
                            star_mode=True)
        if star_name not in sim.spawner.procedures:
            return {'star': star_name, 'seed': seed, 'outcome': 'STAR_MISSING',
                    'a_traj': a_traj, 'c_traj': c_traj}
        sim.spawner.procedures = {star_name: sim.spawner.procedures[star_name]}
        sim.spawner.last_spawned_star = None

        # Pre-compute the (x, y) → (a, c) runway-aligned rotation.
        geom = runtime.geom
        phi = math.radians((geom.course_deg + 180.0) % 360.0)
        sp, cp = math.sin(phi), math.cos(phi)

        def _push_pos(ac_obj, nmpp, ax, ay):
            x_nm = (ac_obj.x - ax) * nmpp
            y_nm = -(ac_obj.y - ay) * nmpp
            dx = x_nm - geom.thr_x_nm
            dy = y_nm - geom.thr_y_nm
            a_traj.append(dx * sp + dy * cp)
            c_traj.append(-dx * cp + dy * sp)

        sim.step(1.0)
        if not sim.aircraft_list:
            return {'star': star_name, 'seed': seed, 'outcome': 'NO_SPAWN',
                    'a_traj': a_traj, 'c_traj': c_traj}

        nmpp = sim.nm_per_pixel
        ax_air = sim.airport_x
        ay_air = sim.airport_y
        callsign = next(iter(sim.aircraft_list.keys()))
        _push_pos(sim.aircraft_list[callsign], nmpp, ax_air, ay_air)

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

            if callsign in sim.aircraft_list:
                ac = sim.aircraft_list[callsign]
                _push_pos(ac, nmpp, ax_air, ay_air)

                # Early-terminate at LOC capture. The localizer is the
                # gating event for ILS — once the aircraft is on it, the
                # only remaining question is "at or below the 3° glide?"
                # (proj_alt = 300 ft per nm to threshold). At/below ⇒
                # captureable from below ⇒ success. Above ⇒ overshot
                # the glide; no recovery ⇒ failure.
                #
                # Sim quirk fix: if the aircraft has overshot the runway
                # threshold along the approach direction, the simulator
                # can still flip loc_intercepted true (back-course
                # capture). The along-course coordinate `a` from the
                # runway-aligned frame is negative on the wrong side of
                # the threshold — fail those as LOC_BEHIND_THR.
                if ac.loc_intercepted and ac.ils_runway:
                    rwy = ac.coords.get(ac.ils_runway)
                    if rwy is not None:
                        dist_nm = ac.nm_per_pixel * math.hypot(
                            ac.x - rwy['x'], ac.y - rwy['y'])
                        proj_alt = dist_nm * 300.0
                        # Along-runway coordinate `a`: > 0 ⇒ approach
                        # side; < 0 ⇒ past the threshold (wrong side).
                        x_nm_ac = (ac.x - ax_air) * nmpp
                        y_nm_ac = -(ac.y - ay_air) * nmpp
                        dx_thr = x_nm_ac - geom.thr_x_nm
                        dy_thr = y_nm_ac - geom.thr_y_nm
                        a_along = dx_thr * sp + dy_thr * cp
                        if a_along < 0.0:
                            outcome = 'LOC_BEHIND_THR'
                        elif ac.altitude <= proj_alt:
                            outcome = 'LOC_BELOW_GS'
                        else:
                            outcome = 'LOC_ABOVE_GS'
                        return {'star': star_name, 'seed': seed,
                                'outcome': outcome,
                                'sim_time': float(sim.sim_time),
                                'callsign': callsign,
                                'steps_used': step + 1,
                                'alt_ft': float(ac.altitude),
                                'gs_proj_ft': float(proj_alt),
                                'dist_nm': float(dist_nm),
                                'a_along_nm': float(a_along),
                                'a_traj': a_traj, 'c_traj': c_traj}
            else:
                if sim.num_landed > landed_before:
                    # Sim autoland completed before our LOC check could
                    # fire — treat as success (LOC capture must have
                    # happened upstream of autoland).
                    outcome = 'LOC_BELOW_GS'
                elif sim.improper_exits > improper_before:
                    outcome = 'IMPROPER_EXIT'
                else:
                    outcome = 'REMOVED_UNKNOWN'
                return {'star': star_name, 'seed': seed, 'outcome': outcome,
                        'sim_time': float(sim.sim_time), 'callsign': callsign,
                        'steps_used': step + 1,
                        'a_traj': a_traj, 'c_traj': c_traj}

        return {'star': star_name, 'seed': seed, 'outcome': 'TIMEOUT',
                'sim_time': float(sim.sim_time), 'callsign': callsign,
                'steps_used': max_steps,
                'a_traj': a_traj, 'c_traj': c_traj}

    except Exception as e:
        return {
            'star': star_name,
            'seed': seed,
            'outcome': 'CRASHED',
            'callsign': callsign or '?',
            'error_type': type(e).__name__,
            'error': str(e),
            'traceback': traceback.format_exc(limit=4),
            'a_traj': a_traj, 'c_traj': c_traj,
        }


# --------------------------------------------------------------------------- #
# Aggregation + CLI
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
    ap.add_argument('--config', type=str, default='bc_fm_single',
                    help='Config name; used to locate the latest run when --ckpt omitted.')
    ap.add_argument('--ckpt', type=str, default=None,
                    help='Explicit checkpoint path. Overrides --run / --config.')
    ap.add_argument('--run', type=int, default=None,
                    help='Specific run number under rl_bc/runs/<config>/.')
    ap.add_argument('--cases', type=int, default=500,
                    help='Rollouts per STAR (default 500). Total = cases * 6.')
    ap.add_argument('--max-steps', type=int, default=0,
                    help='Global sim-second cap per rollout. 0 (default) '
                         'uses STAR_MAX_STEPS (1200 for NORTH1/2 + '
                         'SOUTH1/2, 500 for NORTH3 + SOUTH3). >0 '
                         'overrides every STAR with that single cap.')
    ap.add_argument('--warmup-wpts', type=int, default=2,
                    help='STAR waypoints to fly before BC takes over (default 2).')
    ap.add_argument('--runway', type=str, default='27')
    ap.add_argument('--airport', type=str, default='test')
    ap.add_argument('--workers', type=int,
                    default=max(1, mp.cpu_count() // 2),
                    help='Parallel worker processes (default = cpu_count // 2 '
                         'to approximate physical core count and avoid '
                         'hyperthread oversubscription on laptops; bump higher '
                         'on machines with many real cores).')
    ap.add_argument('--seed-base', type=int, default=0,
                    help='First seed; case i in star s gets seed = seed_base + i.')
    ap.add_argument('--out-dir', type=str, default=None,
                    help='Full destination folder for the 6 output files. '
                         'Default: eval/<config>/. Pass e.g. '
                         '--out-dir eval/bc_gmm_single_distill to write the '
                         'files directly into that folder (no extra <config> '
                         'subfolder appended). Overwrites whatever is there.')
    ap.add_argument('--lim-nm', type=float, default=30.0,
                    help='Plot half-extent in nm for trajectory PNGs (default 30).')
    args = ap.parse_args()

    ckpt_path = _resolve_ckpt(args.ckpt, args.config, args.run)
    out_dir = Path(args.out_dir) if args.out_dir else Path('eval') / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'rollouts.csv'
    summary_path = out_dir / 'summary.json'
    raw_path = out_dir / 'raw.json'
    traj_path = out_dir / 'trajectories.npz'

    summary, results = run_eval_parallel(
        ckpt_path=ckpt_path,
        cases=args.cases, max_steps=args.max_steps,
        warmup_wpts=args.warmup_wpts,
        runway=args.runway, airport=args.airport,
        workers=args.workers, seed_base=args.seed_base,
        config_name=args.config,
        csv_path=csv_path,
    )

    # raw.json holds outcome metadata only; positions go to the npz to keep
    # the JSON small and human-grokkable.
    raw_for_json = [{k: v for k, v in r.items()
                     if k not in ('a_traj', 'c_traj')}
                    for r in results]
    summary_path.write_text(json.dumps(summary, indent=2))
    raw_path.write_text(json.dumps(raw_for_json, indent=2))

    from rl_bc.eval.viz_trajectories import (save_trajectories,
                                              render_from_results)
    save_trajectories(traj_path, results)
    print("\n  rendering trajectory plots...")
    render_from_results(results, out_dir, lim_nm=args.lim_nm)

    print(f"\nSaved (overwritten in place):"
          f"\n  {csv_path}"
          f"\n  {summary_path}"
          f"\n  {raw_path}"
          f"\n  {traj_path}"
          f"\n  {out_dir / 'successful_trajectories.png'}"
          f"\n  {out_dir / 'unsuccessful_trajectories.png'}")


def run_eval_parallel(
    ckpt_path: Path,
    cases: int,
    max_steps: int,
    warmup_wpts: int,
    runway: str,
    airport: str,
    workers: int,
    seed_base: int = 0,
    config_name: str = '',
    chunksize: int = 4,
    csv_path: 'Path | None' = None,
) -> tuple[dict, list[dict]]:
    """Run `cases` rollouts on each of the 6 STARs in a `workers`-process pool.

    Returns (summary_dict, raw_results_list).

    If `csv_path` is given, each rollout result is appended to that CSV as
    soon as it completes (flushed after every row) so a mid-run crash still
    leaves you the partial log.
    """
    import csv as csv_module
    import sys

    ckpt_path = Path(ckpt_path)
    print(f"Evaluating {ckpt_path}", flush=True)
    if max_steps > 0:
        cap_for = lambda star: max_steps  # noqa: E731
        cap_desc = f"global cap={max_steps}s"
    else:
        cap_for = lambda star: STAR_MAX_STEPS[star]  # noqa: E731
        cap_desc = ('per-STAR: ' +
                    ', '.join(f"{s}={STAR_MAX_STEPS[s]}" for s in STARS))
    print(f"  6 STARs × {cases} cases = {6 * cases} rollouts  ({cap_desc})",
          flush=True)
    print(f"  workers={workers}  warmup_wpts={warmup_wpts}  "
          f"runway={runway}\n", flush=True)

    jobs = []
    for star in STARS:
        cap = cap_for(star)
        for i in range(cases):
            seed = seed_base + i
            jobs.append((star, seed, cap, warmup_wpts, runway, airport))

    total = len(jobs)
    # Per-rollout logging — one line per completed rollout with outcome.
    # Outcome tags get a single-char prefix so a quick eye-scan tells you
    # how the run is trending without parsing every field.
    _TAG = {
        'LOC_BELOW_GS': '✓',
        'LANDED':       '✓',   # legacy autoland fallback
        'LOC_ABOVE_GS': '↑',
        'LOC_BEHIND_THR': '←',  # back-course capture past the threshold
        'IMPROPER_EXIT': '✗',
        'TIMEOUT':      '·',
        'CRASHED':      '!',
    }

    print(f"  spawning pool: {workers} workers, each loads the ckpt once "
          f"(may take 30-90s)...", flush=True)
    sys.stdout.flush()
    pool_start = time.time()

    csv_fields = ('star', 'seed', 'outcome', 'callsign', 'sim_time',
                  'steps_used', 'error_type', 'error')
    csv_file = None
    csv_writer = None
    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv_module.DictWriter(csv_file, fieldnames=csv_fields,
                                            extrasaction='ignore')
        csv_writer.writeheader()
        csv_file.flush()
        print(f"  streaming per-rollout rows → {csv_path}", flush=True)

    results: list[dict] = []
    landed = 0
    with mp.Pool(workers, initializer=_init_worker,
                 initargs=(str(ckpt_path),)) as pool:
        print(f"  pool ready in {time.time() - pool_start:.1f}s — "
              f"logging every rollout", flush=True)
        sys.stdout.flush()
        t_start = time.time()
        for r in pool.imap_unordered(_run_case, jobs, chunksize=chunksize):
            results.append(r)
            n = len(results)
            outcome = r.get('outcome', '?')
            tag = _TAG.get(outcome, '?')
            if outcome in SUCCESS_OUTCOMES:
                landed += 1

            # Stream this row to CSV before we even print so a fatal crash
            # right after this point still leaves the file with this row.
            if csv_writer is not None:
                csv_writer.writerow(r)
                csv_file.flush()
            elapsed = time.time() - t_start
            rate = n / max(elapsed, 1e-6)
            eta = (total - n) / max(rate, 1e-3)
            cs = r.get('callsign', '-') or '-'
            t_sim = r.get('sim_time', 0.0)
            steps = r.get('steps_used', 0)
            extra = ''
            if outcome == 'CRASHED':
                extra = f"  ← {r.get('error_type', '?')}: {r.get('error', '?')[:60]}"
            print(
                f"  [{n:>4}/{total}] {tag} {outcome:<13} "
                f"{r['star']:<8} seed={r['seed']:<4} {cs:<8} "
                f"steps={steps:>4} sim_t={t_sim:>5.0f}s  |  "
                f"{landed}/{n} landed ({landed / n * 100:>5.1f}%)  "
                f"elapsed={elapsed:>4.0f}s  eta={eta:>4.0f}s{extra}",
                flush=True,
            )
            sys.stdout.flush()

    elapsed = time.time() - t_start
    if csv_file is not None:
        csv_file.close()
    print(f"\n  done in {elapsed:.1f}s "
          f"({len(results) / elapsed:.1f} rollouts/s)\n")

    buckets = ('LOC_BELOW_GS', 'LOC_ABOVE_GS', 'LOC_BEHIND_THR',
               'IMPROPER_EXIT', 'TIMEOUT', 'CRASHED', 'LANDED')
    per_star = {s: {b: 0 for b in buckets} | {'OTHER': 0} for s in STARS}
    for r in results:
        out = r.get('outcome')
        bucket = out if out in buckets else 'OTHER'
        per_star[r['star']][bucket] += 1

    def _succ(c):
        return c['LOC_BELOW_GS'] + c['LANDED']

    total_landed = sum(_succ(per_star[s]) for s in STARS)
    total = sum(sum(per_star[s].values()) for s in STARS)

    print(f"{'STAR':<10}{'OK<GS':>8}{'>GS':>6}{'BEHIND':>8}"
          f"{'IMPROPER':>10}{'TIMEOUT':>10}{'CRASHED':>10}"
          f"{'OTHER':>8}{'success%':>10}")
    print('-' * 80)
    for s in STARS:
        c = per_star[s]
        n = sum(c.values())
        succ = _succ(c)
        rate = succ / n if n else 0.0
        print(f"{s:<10}{succ:>8}{c['LOC_ABOVE_GS']:>6}"
              f"{c['LOC_BEHIND_THR']:>8}"
              f"{c['IMPROPER_EXIT']:>10}{c['TIMEOUT']:>10}"
              f"{c['CRASHED']:>10}{c['OTHER']:>8}{rate * 100:>9.1f}%")
    overall_rate = total_landed / total if total else 0.0
    print('-' * 80)
    print(f"{'OVERALL':<10}{total_landed:>8}{'':>6}{'':>8}"
          f"{'':>10}{'':>10}{'':>10}{'':>8}{overall_rate * 100:>9.1f}%")

    summary = {
        '_meta': {
            'ckpt': str(ckpt_path),
            'config': config_name,
            'cases_per_star': cases,
            'max_steps': max_steps,        # 0 ⇒ per-STAR caps in use
            'star_max_steps': STAR_MAX_STEPS if max_steps == 0
                              else {s: max_steps for s in STARS},
            'warmup_wpts': warmup_wpts,
            'runway': runway,
            'airport': airport,
            'workers': workers,
            'wall_seconds': elapsed,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'family': _FAMILY,
            'success_rule': 'LOC_BELOW_GS (or legacy LANDED)',
        },
        'per_star': {
            s: {**per_star[s],
                'n': sum(per_star[s].values()),
                'success_rate': _succ(per_star[s]) / max(1, sum(per_star[s].values()))}
            for s in STARS
        },
        'overall': {
            'n': total,
            'landed': total_landed,        # name kept for back-compat; semantics = SUCCESS
            'success_rate': overall_rate,
        },
    }
    return summary, results


if __name__ == '__main__':
    main()
