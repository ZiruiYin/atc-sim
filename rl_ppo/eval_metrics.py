"""Score a rollout-eval directory on the four metrics we care about.

Input: a directory containing the standard pair produced by
`rl_bc.eval.runner` (and the existing PPO eval workflow):

    rollouts.csv          — one row per episode
        columns: star, seed, outcome, callsign, sim_time, steps_used, ...

    trajectories.npz      — concatenated per-step (a, c) for all episodes
        keys: stars, seeds, outcomes, lengths, a_concat, c_concat
        where `lengths[i]` is episode i's step count and the per-step
        slice is `a_concat[off[i]:off[i+1]]` with `off = cumsum(lengths)`.

Metrics returned (per-STAR + overall where it makes sense):
    success_rate              — fraction of episodes with outcome in
                                SUCCESS_OUTCOMES. Higher = better.
    pct_steps_in_green        — over SUCCESSFUL episodes only:
                                mean fraction of steps that fall inside
                                the STAR's green zone. Higher = better.
    pct_within_length_range   — over SUCCESSFUL episodes only:
                                fraction whose length falls in
                                LENGTH_RANGES[star]. Higher = better.
                                Too short → premature base turn.
                                Too long → green-zone circling exploit.
    length_var                — variance (std too) of successful-episode
                                lengths. Higher = better. Low values
                                indicate mixture mode collapse — the
                                policy is always producing the same
                                approach geometry.

Length ranges are calibrated from human multi-plane sessions
(see eval/_inspect_human_traj_lens.py; per-STAR p10/p90 rounded out).

CLI: `python -m rl_ppo.eval_metrics <eval_dir>`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from rl_ppo.reward_zones import point_in_zone


# Outcome → success classification. Matches reward_zones.SUCCESS_OUTCOMES
# but we re-declare here so this module can be read standalone.
SUCCESS_OUTCOMES: frozenset = frozenset({'LOC_BELOW_GS', 'LANDED'})


# Per-STAR length validity windows (steps = seconds, 1 Hz sim).
# Derived from `eval/_inspect_human_traj_lens.py`:
#   STAR_1 family (NORTH1, SOUTH1): humans land 761-948, p10-p90 ≈ 770-900.
#   STAR_2 family (NORTH2, SOUTH2): humans land 677-893, p10-p90 ≈ 710-840.
#   STAR_3 family (NORTH3, SOUTH3): humans land 253-261, very tight.
# Ranges below are rounded outward from those bands.
LENGTH_RANGES: dict[str, tuple[int, int]] = {
    'NORTH1': (720, 960),
    'NORTH2': (670, 880),
    'NORTH3': (250, 265),
    'SOUTH1': (720, 960),
    'SOUTH2': (670, 880),
    'SOUTH3': (250, 265),
}

STARS_ALL: tuple = ('NORTH1', 'NORTH2', 'NORTH3', 'SOUTH1', 'SOUTH2', 'SOUTH3')


# --------------------------------------------------------------------- #
def _load_rollouts_csv(path: Path) -> list[dict]:
    """Read rollouts.csv into a list of dicts (one per episode)."""
    with path.open() as f:
        return list(csv.DictReader(f))


def _load_trajectories_npz(path: Path) -> dict[str, np.ndarray]:
    """Load the trajectories.npz pair-arrays and per-step offsets."""
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    lens = np.asarray(out['lengths'], dtype=np.int64)
    # Per-episode start offsets into a_concat / c_concat.
    out['_offsets'] = np.concatenate([[0], np.cumsum(lens)])
    return out


def _frac_in_zone(star: str, a: np.ndarray, c: np.ndarray) -> float:
    """Fraction of (a, c) samples inside the STAR's green zone.

    Point-in-polygon is cheap per call; for typical 700-step trajectories
    the loop is fine and avoids the numpy vectorize overhead.
    """
    if a.size == 0:
        return 0.0
    n_in = 0
    for i in range(a.size):
        if point_in_zone(star, float(a[i]), float(c[i])):
            n_in += 1
    return n_in / a.size


# --------------------------------------------------------------------- #
def compute_eval_metrics(eval_dir: str | Path,
                         save: bool = True) -> dict:
    """Compute success / quality / length metrics for one eval directory.

    Returns a dict shaped like:
        {
          'overall': {'n': ..., 'n_success': ..., 'success_rate': ...},
          'per_star': {
              'NORTH1': {
                  'n': 100, 'n_success': 88, 'success_rate': 0.88,
                  'pct_steps_in_green': 0.91,
                  'pct_within_length_range': 0.74,
                  'length_mean': 812.3, 'length_std': 47.1,
                  'length_var': 2218.4, 'n_in_range': 65,
                  'length_range': [720, 960],
              }, ...
          },
          '_meta': {'eval_dir': ..., 'success_outcomes': [...]},
        }

    If `save=True`, the dict is also written as `eval_metrics.json`
    inside `eval_dir`.
    """
    eval_dir = Path(eval_dir)
    rcsv = eval_dir / 'rollouts.csv'
    nzp  = eval_dir / 'trajectories.npz'
    if not rcsv.exists():
        raise FileNotFoundError(rcsv)
    if not nzp.exists():
        raise FileNotFoundError(nzp)

    rollouts = _load_rollouts_csv(rcsv)
    traj = _load_trajectories_npz(nzp)
    stars = traj['stars']
    outcomes = traj['outcomes']
    lengths = np.asarray(traj['lengths'], dtype=np.int64)
    a_concat = traj['a_concat']
    c_concat = traj['c_concat']
    offsets = traj['_offsets']

    # Sanity: rollouts.csv row count should match trajectories.npz episode count.
    if len(rollouts) != stars.size:
        raise ValueError(
            f'rollouts.csv ({len(rollouts)}) and trajectories.npz '
            f'({stars.size}) episode counts disagree in {eval_dir}'
        )

    # --- Per-STAR aggregation ---
    per_star: dict[str, dict] = {}
    success_mask_all = np.array(
        [o in SUCCESS_OUTCOMES for o in outcomes], dtype=bool)

    for star in STARS_ALL:
        ep_mask = (stars == star)
        n_ep = int(ep_mask.sum())
        success = success_mask_all & ep_mask
        n_succ = int(success.sum())
        succ_lens = lengths[success]

        if n_ep == 0:
            per_star[star] = {
                'n': 0, 'n_success': 0, 'success_rate': 0.0,
                'pct_steps_in_green': 0.0,
                'pct_within_length_range': 0.0,
                'n_in_range': 0,
                'length_mean': 0.0, 'length_std': 0.0, 'length_var': 0.0,
                'length_range': list(LENGTH_RANGES[star]),
            }
            continue

        # Per-successful-episode fraction in green zone (averaged).
        if n_succ > 0:
            frac_per_ep = []
            idxs = np.flatnonzero(success)
            for i in idxs:
                a = a_concat[offsets[i]:offsets[i + 1]]
                c = c_concat[offsets[i]:offsets[i + 1]]
                frac_per_ep.append(_frac_in_zone(star, a, c))
            pct_green = float(np.mean(frac_per_ep))
        else:
            pct_green = 0.0

        # Length range check — three buckets so we know which direction
        # to tune in. `pct_below_min` rising is the "premature termination
        # / corner-cut exploit" signal (raise γ + drop everywhere_pen).
        # `pct_above_max` rising is the "circling / loop" signal
        # (drop γ + raise everywhere_pen).
        lo, hi = LENGTH_RANGES[star]
        below_mask = succ_lens < lo
        above_mask = succ_lens > hi
        in_range_mask = (succ_lens >= lo) & (succ_lens <= hi)
        n_in_range = int(in_range_mask.sum())
        n_below_min = int(below_mask.sum())
        n_above_max = int(above_mask.sum())
        pct_in_range = (n_in_range / n_succ) if n_succ else 0.0
        pct_below_min = (n_below_min / n_succ) if n_succ else 0.0
        pct_above_max = (n_above_max / n_succ) if n_succ else 0.0

        # Variance / std of successful lengths. ddof=1 (sample); needs n>=2.
        if succ_lens.size >= 2:
            length_mean = float(succ_lens.mean())
            length_std = float(succ_lens.std(ddof=1))
            length_var = float(succ_lens.var(ddof=1))
        elif succ_lens.size == 1:
            length_mean = float(succ_lens[0])
            length_std = 0.0
            length_var = 0.0
        else:
            length_mean = length_std = length_var = 0.0

        per_star[star] = {
            'n': n_ep,
            'n_success': n_succ,
            'success_rate': n_succ / n_ep,
            'pct_steps_in_green': pct_green,
            'pct_within_length_range': pct_in_range,
            'pct_below_min_length': pct_below_min,
            'pct_above_max_length': pct_above_max,
            'n_in_range': n_in_range,
            'n_below_min': n_below_min,
            'n_above_max': n_above_max,
            'length_mean': length_mean,
            'length_std': length_std,
            'length_var': length_var,
            'length_range': [int(lo), int(hi)],
        }

    # --- Overall aggregation ---
    n_total = int(stars.size)
    n_succ_total = int(success_mask_all.sum())
    overall = {
        'n': n_total,
        'n_success': n_succ_total,
        'success_rate': (n_succ_total / n_total) if n_total else 0.0,
        # Macro average across STARs (so it doesn't get dominated by
        # any one STAR's sample count).
        'macro_pct_steps_in_green': float(np.mean(
            [per_star[s]['pct_steps_in_green'] for s in STARS_ALL
             if per_star[s]['n'] > 0])) if any(
            per_star[s]['n'] > 0 for s in STARS_ALL) else 0.0,
        'macro_pct_within_length_range': float(np.mean(
            [per_star[s]['pct_within_length_range'] for s in STARS_ALL
             if per_star[s]['n'] > 0])) if any(
            per_star[s]['n'] > 0 for s in STARS_ALL) else 0.0,
    }

    out = {
        'overall': overall,
        'per_star': per_star,
        '_meta': {
            'eval_dir': str(eval_dir),
            'success_outcomes': sorted(SUCCESS_OUTCOMES),
            'length_ranges': {k: list(v) for k, v in LENGTH_RANGES.items()},
        },
    }

    if save:
        with (eval_dir / 'eval_metrics.json').open('w') as f:
            json.dump(out, f, indent=2)

    return out


# --------------------------------------------------------------------- #
def _print_summary(m: dict) -> None:
    """Compact, fixed-column print of the metrics dict."""
    o = m['overall']
    print(f"\n{m['_meta']['eval_dir']}")
    print(f"  overall: n={o['n']} success={o['n_success']} "
          f"SR={o['success_rate']*100:.1f}%  "
          f"macro_green={o['macro_pct_steps_in_green']*100:.1f}%  "
          f"macro_in_range={o['macro_pct_within_length_range']*100:.1f}%")
    print(f"  {'STAR':<7} {'n':>4} {'succ':>5} {'SR%':>5} "
          f"{'green%':>7} {'inRng%':>7} {'<min%':>6} {'>max%':>6} "
          f"{'len_mean':>9} {'len_std':>8}  range")
    for star in STARS_ALL:
        s = m['per_star'][star]
        if s['n'] == 0:
            continue
        lo, hi = s['length_range']
        print(f"  {star:<7} {s['n']:>4d} {s['n_success']:>5d} "
              f"{s['success_rate']*100:>5.1f} "
              f"{s['pct_steps_in_green']*100:>7.1f} "
              f"{s['pct_within_length_range']*100:>7.1f} "
              f"{s['pct_below_min_length']*100:>6.1f} "
              f"{s['pct_above_max_length']*100:>6.1f} "
              f"{s['length_mean']:>9.1f} {s['length_std']:>8.1f}  "
              f"[{lo},{hi}]")


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m rl_ppo.eval_metrics <eval_dir> '
              '[<eval_dir> ...]', file=sys.stderr)
        sys.exit(2)
    for d in sys.argv[1:]:
        m = compute_eval_metrics(d)
        _print_summary(m)


if __name__ == '__main__':
    main()
