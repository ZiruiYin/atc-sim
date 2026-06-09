"""Detect circling / loop behavior in rollout trajectories.

Why this is needed: trajectory-length checks alone don't catch every
loop pattern. A policy can perform tight orbits in the green zone and
still come out below the per-STAR max length cap if the orbit count is
small. This module looks at the position trace directly.

Detection criterion (one trajectory):
    A timestep i is "looping" iff there exists a later timestep j with
        (a) j - i >= MIN_GAP_STEPS          # enough time elapsed
        (b) dist(p_i, p_j) < PROX_RADIUS    # back within R nm
        (c) max(dist(p_i, p_k)) > MIN_DETOUR for some k in (i, j)
            # we actually went somewhere in between (else it's slow drift)
    A trajectory is flagged if `n_looping / T >= MIN_LOOP_FRAC`.

Hyperparameters (all CLI flags, all tunable):

    --prox-radius-nm     R   (default 0.5 nm)
        How close counts as "same position." 0.5 nm catches tight orbits;
        0.25 nm = stricter; 1.0 nm = anything in the same general spot.

    --min-gap-steps      G   (default 60 sec)
        How much time must elapse between visits. 60 s catches short
        orbits (~180 kt × 30 s ≈ 1.5 nm radius). 120 s catches
        racetrack-pattern holds. <30 s is just adjacent points.

    --min-detour-nm      D   (default 1.5 nm)
        Must leave area by this distance between the two visits. Stops
        slow-drift-through-a-region from counting. 0 = ignore detour
        check (most permissive).

    --min-loop-frac      F   (default 0.05 = 5 %)
        Fraction of trajectory steps that must be marked looping for
        the whole trajectory to be flagged. Higher = stricter.

Output (under `eval/looping_analysis/`):
    looping_summary.csv             — one row per flagged traj across all 3 runs
    hyperparameters.json            — the snapshot used for this analysis
    run_5_looping_trajectories.png  — 6 STAR subplots, looping trajs only
    run_6_looping_trajectories.png
    run_7_looping_trajectories.png

CLI:
    python -m rl_ppo.loop_detector
    python -m rl_ppo.loop_detector --prox-radius-nm 0.3 --min-gap-steps 90
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SUCCESS_OUTCOMES = frozenset({'LOC_BELOW_GS', 'LANDED'})
STARS_ALL = ('NORTH1', 'NORTH2', 'NORTH3', 'SOUTH1', 'SOUTH2', 'SOUTH3')

# Default hyperparameters (override via CLI).
# Tuned to a fairly aggressive setting after the first pass under
# (0.5, 60, 1.5, 0.05) missed visible orbits in run_7. The looser
# settings below catch ~3-7% looping rates on the existing PPO runs;
# run_6 (the "okay" reference) still only flags 1 / 491 ≈ 0.2%,
# confirming the detector isn't false-positive-prone.
PROX_RADIUS_NM_DEFAULT = 0.75   # was 0.5 — wider revisit window
MIN_GAP_STEPS_DEFAULT = 45      # was 60  — catches tighter orbits
MIN_DETOUR_NM_DEFAULT = 1.0     # was 1.5 — smaller required detour
MIN_LOOP_FRAC_DEFAULT = 0.03    # was 0.05 — fewer loop steps to flag

# Default eval folders to scan. Override via --eval-dirs.
# These point at the archived 6-pack evals under legacy_runs/. The shipped
# model's eval is at rl_ppo/runs/continuous_runs/continuous_03/iter_0160_eval.
DEFAULT_EVAL_DIRS = {
    'run_5': 'rl_ppo/runs/legacy_runs/run_5/eval_6pack',
    'run_6': 'rl_ppo/runs/legacy_runs/run_6/eval_6pack',
    'run_7': 'rl_ppo/runs/legacy_runs/run_7/eval_6pack',
}


# --------------------------------------------------------------------- #
# Core detector
# --------------------------------------------------------------------- #
def detect_looping(a: np.ndarray, c: np.ndarray,
                   prox_radius_nm: float = PROX_RADIUS_NM_DEFAULT,
                   min_gap_steps: int = MIN_GAP_STEPS_DEFAULT,
                   min_detour_nm: float = MIN_DETOUR_NM_DEFAULT,
                   min_loop_frac: float = MIN_LOOP_FRAC_DEFAULT,
                   ) -> dict:
    """Score one (a, c) trajectory for looping. See module docstring."""
    a = np.asarray(a, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    T = int(a.size)
    if T < min_gap_steps + 2:
        return {
            'is_looping': False, 'loop_frac': 0.0,
            'n_looping_steps': 0, 'n_revisit_pairs': 0,
            'max_revisit_gap': 0, 'T': T,
            'looping_step_mask': np.zeros(T, dtype=bool),
        }

    R2 = float(prox_radius_nm) ** 2
    D2 = float(min_detour_nm) ** 2
    use_detour = min_detour_nm > 0.0
    looping = np.zeros(T, dtype=bool)
    n_pairs = 0
    max_gap = 0

    # For each i, find j-candidates with sufficient gap and close enough.
    # Inner work is fully vectorized over j; outer i loop is O(T).
    for i in range(T - min_gap_steps - 1):
        j_start = i + min_gap_steps
        dx = a[j_start:] - a[i]
        dy = c[j_start:] - c[i]
        d2 = dx * dx + dy * dy
        close_local = np.flatnonzero(d2 < R2)
        if close_local.size == 0:
            continue
        # Cheaper detour check: compute squared dist from p_i to every
        # k in (i, T-1) ONCE per i, then for each candidate j look up the
        # max over [i+1, j-1]. We use a running-max via cumulative max
        # on the squared distance series.
        if use_detour:
            seg_dx = a[i + 1:] - a[i]
            seg_dy = c[i + 1:] - c[i]
            seg_d2 = seg_dx * seg_dx + seg_dy * seg_dy
            cum_max_d2 = np.maximum.accumulate(seg_d2)
            for jl in close_local:
                j = jl + j_start
                # max d2 over k in [i+1, j-1] is cum_max_d2[j - i - 2]
                # (only if j >= i + 2; given j_start >= i + min_gap >= i + 2
                # this is always true when min_gap_steps >= 2).
                if cum_max_d2[j - i - 2] > D2:
                    looping[i] = True
                    looping[j] = True
                    n_pairs += 1
                    gap = j - i
                    if gap > max_gap:
                        max_gap = gap
                    break  # one qualifying j per i is enough
        else:
            j = int(close_local[0]) + j_start
            looping[i] = True
            looping[j] = True
            n_pairs += 1
            gap = j - i
            if gap > max_gap:
                max_gap = gap

    n_looping = int(looping.sum())
    loop_frac = n_looping / T
    return {
        'is_looping': loop_frac >= min_loop_frac,
        'loop_frac': float(loop_frac),
        'n_looping_steps': n_looping,
        'n_revisit_pairs': n_pairs,
        'max_revisit_gap': int(max_gap),
        'T': T,
        'looping_step_mask': looping,
    }


# --------------------------------------------------------------------- #
# Eval-dir analysis + plotting
# --------------------------------------------------------------------- #
def analyze_eval_dir(eval_dir: str | Path,
                     **hp) -> tuple[list[dict], dict]:
    """Score every SUCCESSFUL trajectory in `eval_dir`. Returns
    (results, traj_arrays) where `traj_arrays` is a dict with the raw
    concatenated (a, c) buffers + offsets so the plotter can slice them
    without re-loading."""
    eval_dir = Path(eval_dir)
    nzp = np.load(eval_dir / 'trajectories.npz', allow_pickle=False)
    stars = nzp['stars']
    outcomes = nzp['outcomes']
    lengths = np.asarray(nzp['lengths'], dtype=np.int64)
    a_concat = nzp['a_concat']
    c_concat = nzp['c_concat']
    offsets = np.concatenate([[0], np.cumsum(lengths)])

    results: list[dict] = []
    for i in range(stars.size):
        outcome = str(outcomes[i])
        if outcome not in SUCCESS_OUTCOMES:
            continue
        a = a_concat[offsets[i]:offsets[i + 1]]
        c = c_concat[offsets[i]:offsets[i + 1]]
        d = detect_looping(a, c, **hp)
        d['idx'] = int(i)
        d['star'] = str(stars[i])
        d['outcome'] = outcome
        results.append(d)
    return results, {
        'a_concat': a_concat, 'c_concat': c_concat, 'offsets': offsets,
        'stars': stars, 'outcomes': outcomes,
    }


def plot_run_looping(run_label: str, eval_dir: str | Path,
                     out_path: Path,
                     **hp) -> list[dict]:
    """Render one figure per run with 2x3 STAR subplots showing every
    LOOPING successful trajectory overlaid on the STAR's green zone.

    Looping steps are drawn in red; the rest of the trajectory in a
    fainter color. Returns the list of flagged trajectories (each dict
    enriched with 'run_label').
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from rl_ppo.reward_zones import STAR_GREEN_ZONES

    results, arr = analyze_eval_dir(eval_dir, **hp)
    looping = [r for r in results if r['is_looping']]
    for r in looping:
        r['run_label'] = run_label
    n_total = len(results)
    n_loop = len(looping)
    print(f"  [{run_label}] {n_loop} / {n_total} successful trajs flagged "
          f"as looping ({100.0 * n_loop / max(1, n_total):.1f}%)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    for ax, star in zip(axes.flat, STARS_ALL):
        # Green zone shading.
        for poly in STAR_GREEN_ZONES.get(star, ()):
            ax.add_patch(MplPolygon(poly, closed=True, alpha=0.18,
                                    facecolor='#7ec47e', edgecolor='#2c8a2c',
                                    linewidth=1.0))
        # Runway threshold marker.
        ax.plot(0.0, 0.0, marker='^', color='black', markersize=10,
                label='RWY 27 thr')

        # Plot every looping traj of this STAR.
        star_loopers = [r for r in looping if r['star'] == star]
        for r in star_loopers:
            idx = r['idx']
            a = arr['a_concat'][arr['offsets'][idx]:arr['offsets'][idx + 1]]
            c = arr['c_concat'][arr['offsets'][idx]:arr['offsets'][idx + 1]]
            mask = r['looping_step_mask']
            # Non-loop steps faint
            ax.plot(a, c, color='#4477aa', alpha=0.35, linewidth=0.7)
            # Loop steps highlighted
            if mask.any():
                # Plot only the looping points (scatter) so the segments
                # don't form spurious lines across non-loop gaps.
                ax.scatter(a[mask], c[mask], color='#cc2222',
                           s=4, alpha=0.55, marker='.')

        ax.set_title(f'{star}  ({len(star_loopers)} loopers)', fontsize=11)
        ax.set_xlabel('a (nm along runway)')
        ax.set_ylabel('c (nm cross-track)')
        ax.set_xlim(-35, 35)
        ax.set_ylim(-32, 32)
        ax.grid(alpha=0.25, linestyle=':')
        ax.set_aspect('equal', adjustable='box')

    hp_str = (f"R={hp.get('prox_radius_nm')}nm  "
              f"gap>={hp.get('min_gap_steps')}s  "
              f"detour>={hp.get('min_detour_nm')}nm  "
              f"loop_frac>={hp.get('min_loop_frac')*100:.1f}%")
    fig.suptitle(f"{run_label} — looping successful trajectories  "
                 f"({n_loop}/{n_total})\n{hp_str}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")
    return looping


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--prox-radius-nm', type=float,
                    default=PROX_RADIUS_NM_DEFAULT,
                    help=f'(default {PROX_RADIUS_NM_DEFAULT})')
    ap.add_argument('--min-gap-steps', type=int,
                    default=MIN_GAP_STEPS_DEFAULT,
                    help=f'(default {MIN_GAP_STEPS_DEFAULT})')
    ap.add_argument('--min-detour-nm', type=float,
                    default=MIN_DETOUR_NM_DEFAULT,
                    help=f'(default {MIN_DETOUR_NM_DEFAULT}; 0 = disable)')
    ap.add_argument('--min-loop-frac', type=float,
                    default=MIN_LOOP_FRAC_DEFAULT,
                    help=f'(default {MIN_LOOP_FRAC_DEFAULT})')
    ap.add_argument('--out-dir', type=str,
                    default='eval/looping_analysis',
                    help='Output directory for CSV + per-run PNGs.')
    ap.add_argument('--eval-dir', action='append', default=None,
                    metavar='LABEL=PATH',
                    help='Override default scan list. May be repeated. '
                         'e.g. --eval-dir run_X=eval/foo')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_dir:
        eval_dirs = {}
        for spec in args.eval_dir:
            label, path = spec.split('=', 1)
            eval_dirs[label] = path
    else:
        eval_dirs = DEFAULT_EVAL_DIRS

    hp = {
        'prox_radius_nm': float(args.prox_radius_nm),
        'min_gap_steps': int(args.min_gap_steps),
        'min_detour_nm': float(args.min_detour_nm),
        'min_loop_frac': float(args.min_loop_frac),
    }
    print(f"hyperparameters: {hp}")

    all_loopers: list[dict] = []
    for label, eval_dir in eval_dirs.items():
        eval_dir = Path(eval_dir)
        if not (eval_dir / 'trajectories.npz').exists():
            print(f"  MISSING: {eval_dir}/trajectories.npz — skipping")
            continue
        out_png = out_dir / f'{label}_looping_trajectories.png'
        loopers = plot_run_looping(label, eval_dir, out_png, **hp)
        all_loopers.extend(loopers)

    # CSV summary (one row per flagged traj).
    csv_path = out_dir / 'looping_summary.csv'
    fields = ['run_label', 'idx', 'star', 'outcome', 'T',
              'loop_frac', 'n_looping_steps', 'n_revisit_pairs',
              'max_revisit_gap']
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_loopers:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\nwrote {csv_path} ({len(all_loopers)} flagged trajs total)")

    # Hyperparameter snapshot.
    with (out_dir / 'hyperparameters.json').open('w') as f:
        json.dump(hp, f, indent=2)
    print(f"wrote {out_dir / 'hyperparameters.json'}")

    # Per-STAR / per-run counts to stdout.
    print('\nflagged-traj counts (run x STAR):')
    print(f"  {'run':<7} | " + ' '.join(f'{s:>7}' for s in STARS_ALL)
          + ' |  total')
    by_run: dict[str, dict[str, int]] = {}
    for r in all_loopers:
        by_run.setdefault(r['run_label'], {}).setdefault(r['star'], 0)
        by_run[r['run_label']][r['star']] += 1
    for run, d in sorted(by_run.items()):
        row = ' '.join(f'{d.get(s, 0):>7d}' for s in STARS_ALL)
        total = sum(d.values())
        print(f"  {run:<7} | {row} | {total:>5d}")


if __name__ == '__main__':
    main()
