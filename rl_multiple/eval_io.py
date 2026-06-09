"""Shared I/O for multi-plane PPO evaluation.

Given a list of `Trajectory` objects (rl_multiple.rollout.Trajectory),
write a standardised 6-pack of files to an output directory:

    rollouts.csv          — one row per episode (star, seed, outcome,
                             callsign, sim_time, steps_used, error_type,
                             error)
    trajectories.npz      — per-step (a, c) for all episodes, concat
                             format used by rl_ppo.eval_metrics + the
                             rl_bc trajectory visualizer
    summary.json          — per-STAR Counter of outcomes + means
    raw.json              — per-episode lightweight summary (no big
                             arrays); useful for debugging
    eval_metrics.json     — output of rl_ppo.eval_metrics.compute_eval_metrics
                             (SR, macro_green, per-STAR breakdown)

PNG renders (`successful_trajectories.png`, `unsuccessful_trajectories.png`)
are handled separately by `render_pngs()` because matplotlib is not in
the Modal training image — the PNGs get drawn locally after a Modal run
is pulled back.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------- #
# Individual file writers
# --------------------------------------------------------------------------- #


def write_rollouts_csv(trajectories: Iterable, csv_path: Path) -> None:
    """One row per episode. Schema matches rl_bc.eval.runner so existing
    rl_ppo.eval_metrics machinery reads it without changes."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['star', 'seed', 'outcome', 'callsign',
                    'sim_time', 'steps_used', 'error_type', 'error'])
        for t in trajectories:
            err = t.error
            err_type = ''
            err_msg = ''
            if err:
                # err is "Type: message" from worker. Split for the
                # error_type column when possible.
                if ':' in err:
                    err_type, _, err_msg = err.partition(':')
                    err_type = err_type.strip()
                    err_msg = err_msg.strip()
                else:
                    err_msg = err
            w.writerow([
                t.star, t.seed, t.outcome, t.callsign,
                t.steps, t.steps,    # sim_time == steps_used at 1 Hz
                err_type, err_msg,
            ])


def write_trajectories_npz(trajectories: Iterable, npz_path: Path) -> None:
    """Per-step (a, c) cache. Same key schema as rl_bc.eval.viz_trajectories.

    Keys: stars, seeds, outcomes, lengths, a_concat, c_concat.
    Use `offsets = np.concatenate([[0], np.cumsum(lengths)])` to slice
    per-episode arrays out of the concatenated buffers.
    """
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    trajs = list(trajectories)
    stars = [t.star for t in trajs]
    seeds = [int(t.seed) for t in trajs]
    outcomes = [t.outcome for t in trajs]
    lengths = np.array(
        [int(t.a_traj.size) if t.a_traj is not None else 0 for t in trajs],
        dtype=np.int64,
    )
    if trajs:
        a_concat = np.concatenate(
            [np.asarray(t.a_traj, dtype=np.float32)
             if t.a_traj is not None else np.empty(0, np.float32)
             for t in trajs]
        )
        c_concat = np.concatenate(
            [np.asarray(t.c_traj, dtype=np.float32)
             if t.c_traj is not None else np.empty(0, np.float32)
             for t in trajs]
        )
    else:
        a_concat = np.empty(0, np.float32)
        c_concat = np.empty(0, np.float32)

    np.savez_compressed(
        npz_path,
        stars=np.array(stars),
        seeds=np.array(seeds, dtype=np.int64),
        outcomes=np.array(outcomes),
        lengths=lengths,
        a_concat=a_concat,
        c_concat=c_concat,
    )


def write_summary_json(trajectories: Iterable, summary_path: Path) -> None:
    """Per-STAR outcome counter + means. Cheap summary parallel to the
    eval_metrics.json output (which is the heavyweight scoring file)."""
    summary_path = Path(summary_path)
    per_star: dict[str, Counter] = {}
    total = Counter()
    total_steps = 0
    total_n = 0
    total_R = 0.0
    for t in trajectories:
        c = per_star.setdefault(t.star, Counter())
        c[t.outcome] += 1
        c['_n'] += 1
        c['_steps'] += t.steps
        c['_R_sum'] += float(t.rewards.sum()) if t.rewards.size else 0.0
        total[t.outcome] += 1
        total_n += 1
        total_steps += t.steps
        total_R += float(t.rewards.sum()) if t.rewards.size else 0.0
    out = {
        'overall': {
            'n': total_n,
            'mean_steps': total_steps / max(1, total_n),
            'mean_reward': total_R / max(1, total_n),
            'outcomes': dict(total),
        },
        'per_star': {
            star: {
                'n': c['_n'],
                'mean_steps': c['_steps'] / max(1, c['_n']),
                'mean_reward': c['_R_sum'] / max(1, c['_n']),
                'outcomes': {k: v for k, v in c.items()
                             if not k.startswith('_')},
            }
            for star, c in per_star.items()
        },
    }
    summary_path.write_text(json.dumps(out, indent=2))


def write_raw_json(trajectories: Iterable, raw_path: Path) -> None:
    """Per-episode lightweight dump (no big arrays). Mirrors what
    rl_ppo.eval_runner produces."""
    raw_path = Path(raw_path)
    rows = []
    for t in trajectories:
        rows.append({
            'star': t.star,
            'seed': int(t.seed),
            'callsign': t.callsign,
            'outcome': t.outcome,
            'steps': t.steps,
            'd_thr_nm': t.d_thr_nm,
            'altitude_ft': t.altitude_ft,
            'gs_alt_ft': t.gs_alt_ft,
            'n_loop_steps': t.n_loop_steps,
            'loop_penalty_total': t.loop_penalty_total,
            'error': t.error,
        })
    raw_path.write_text(json.dumps(rows, indent=2))


# --------------------------------------------------------------------------- #
# 6-pack writer (no PNGs — that's a separate `render_pngs` call below)
# --------------------------------------------------------------------------- #


def write_eval_six_pack(trajectories: list,
                        out_dir: Path,
                        score: bool = True) -> dict | None:
    """Write rollouts.csv + trajectories.npz + summary.json + raw.json
    into `out_dir`. If `score=True`, also compute and write
    eval_metrics.json (the SR / macro_green scoring file).

    Returns the metrics dict when score=True, else None.

    Skips PNG rendering — those are produced separately by `render_pngs`
    (matplotlib not available in the Modal PPO image).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajs = list(trajectories)

    write_rollouts_csv(trajs, out_dir / 'rollouts.csv')
    write_trajectories_npz(trajs, out_dir / 'trajectories.npz')
    write_summary_json(trajs, out_dir / 'summary.json')
    write_raw_json(trajs, out_dir / 'raw.json')

    if not score:
        return None

    # eval_metrics.compute_eval_metrics reads rollouts.csv +
    # trajectories.npz from the same dir, writes eval_metrics.json,
    # returns the metrics dict.
    from rl_ppo.eval_metrics import compute_eval_metrics
    return compute_eval_metrics(out_dir, save=True)


# --------------------------------------------------------------------------- #
# PNG rendering (call AFTER training, locally, where matplotlib is present)
# --------------------------------------------------------------------------- #


def render_pngs(eval_dir: Path, lim_nm: float = 30.0) -> tuple[int, int]:
    """Render the two PNGs by reading the npz/csv just written.

    Returns (n_successful_drawn, n_unsuccessful_drawn).
    Raises ImportError if matplotlib isn't available (Modal image).
    """
    from rl_bc.eval.viz_trajectories import render_from_disk
    return render_from_disk(Path(eval_dir), lim_nm=lim_nm)
