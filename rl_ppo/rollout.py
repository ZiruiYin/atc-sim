"""Parallel rollout collection for PPO.

A worker pool of `n_workers` processes, each holding its own `PPOEnv` and
`PPOPolicy`. Per PPO iteration:

  1. The main process broadcasts the latest policy state_dict to all
     workers (via a multiprocessing.Manager dict).
  2. The pool runs `n_rollouts_per_iter` episodes in parallel, balanced
     across STARs.
  3. Each episode returns its full trajectory + outcome metadata. The
     main process flattens them into a batch for GAE + PPO updates.

Workers pin BLAS/OMP to 1 thread each (same pattern as eval/runner.py) —
our model is small and threaded BLAS just thrashes caches when the pool
oversubscribes cores.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Trajectory:
    """One episode's worth of (s, a, r, log_prob_old, value, done) tuples
    plus its outcome and per-STAR identity."""
    obs: np.ndarray             # (T, 3) standardized
    actions: np.ndarray         # (T, 4) physical units
    log_probs: np.ndarray       # (T,)
    rewards: np.ndarray         # (T,)
    values: np.ndarray          # (T,) V(s_t)
    dones: np.ndarray           # (T,) bool
    star: str
    outcome: str
    steps: int
    error: str | None = None
    # Terminal-step diagnostics — populated for SUCCESS / LOC_HIGH:
    #   d_thr_nm     distance to runway threshold at LOC interception
    #   altitude_ft  aircraft altitude at LOC interception
    #   gs_alt_ft    glideslope altitude at that d_thr (= d·300)
    # Lets us spot spurious "drive-by" LOC trips far from the threshold
    # — the sim flips loc_intercepted on any centerline crossing, with
    # no distance/altitude gate (environment/core/aircraft.py:289).
    d_thr_nm: float | None = None
    altitude_ft: float | None = None
    gs_alt_ft: float | None = None
    # Runway-aligned position trace for visualizing the rollout shape
    # (NaN-filled if the worker crashed before any step ran). Same length
    # as `rewards`. Used by train.py to render successful/unsuccessful
    # trajectory PNGs every N iters.
    a_traj: np.ndarray | None = None
    c_traj: np.ndarray | None = None
    # Post-hoc loop-penalty bookkeeping. Populated by the worker after
    # running `loop_detector.detect_looping` on the final trajectory.
    # `n_loop_steps` is the total looping-step count (k);
    # `n_penalized` is the second-half count (⌈k/2⌉ onwards, the ones
    # whose rewards got the per-step penalty subtracted);
    # `loop_penalty_total` is the absolute amount subtracted from
    # `rewards`. All zero when the penalty is disabled (cfg knob = 0).
    n_loop_steps: int = 0
    n_penalized: int = 0
    loop_penalty_total: float = 0.0


# --------------------------------------------------------------------------- #
# Worker globals (one per child process)
# --------------------------------------------------------------------------- #


_ENV = None
_POLICY = None
_CONFIG = None


def _init_worker(actor_ckpt: str, cfg_dict: dict) -> None:
    """Pool initializer: each worker loads its env + policy once."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)

    global _ENV, _POLICY, _CONFIG
    from .env import PPOEnv
    from .policy import PPOPolicy
    _CONFIG = cfg_dict

    # CRITICAL: apply reward-zone runtime overrides INSIDE the worker
    # process. mp.Pool with spawn creates fresh processes that re-import
    # reward_zones with module defaults. Without this call, every worker
    # uses the DEFAULT reward shaping regardless of what the main
    # process called set_runtime_overrides() with. (This was a silent
    # bug from the moment runtime overrides were introduced.)
    from . import reward_zones as _rz
    _rz.set_runtime_overrides(
        everywhere_step_penalty=cfg_dict.get('reward_everywhere_step_penalty'),
        step_penalty_cap=cfg_dict.get('reward_step_penalty_cap'),
        step_penalty_per_nm=cfg_dict.get('reward_step_penalty_per_nm'),
        early_zone_multiplier=cfg_dict.get('reward_early_zone_multiplier'),
        early_window_steps=cfg_dict.get('reward_early_window_steps'),
        early_drift_penalty=cfg_dict.get('reward_early_drift_penalty'),
        clean_terminal_threshold=cfg_dict.get('reward_clean_terminal_threshold'),
        drifty_success_value=cfg_dict.get('reward_drifty_success_value'),
        out_of_zone_terminate=cfg_dict.get('reward_out_of_zone_terminate'),
        out_of_zone_max_consecutive=cfg_dict.get('reward_out_of_zone_max_consecutive'),
        per_star_sr_scale=cfg_dict.get('reward_per_star_sr_scale'),
        per_star_recent_sr=cfg_dict.get('reward_per_star_recent_sr'),
        heading_intercept_enabled=cfg_dict.get('reward_heading_intercept_enabled'),
        turn_final_enabled=cfg_dict.get('reward_turn_final_enabled'),
    )

    _ENV = PPOEnv(
        actor_ckpt=actor_ckpt,
        airport_name=cfg_dict['airport_name'],
        runway=cfg_dict['runway'],
        warmup_wpts=cfg_dict['warmup_wpts'],
        max_timesteps_star_1_2=cfg_dict['max_timesteps_star_1_2'],
        max_timesteps_star_3=cfg_dict['max_timesteps_star_3'],
        success_reward=cfg_dict['success_reward'],
        failure_reward=cfg_dict['failure_reward'],
        gs_capture_buffer_ft=cfg_dict.get('gs_capture_buffer_ft', 50.0),
    )
    _POLICY = PPOPolicy.from_bc_checkpoint(
        actor_ckpt,
        value_hidden=cfg_dict['value_hidden'],
        value_dropout=cfg_dict['value_dropout'],
        device='cpu',
    )
    _POLICY.eval()


def _run_episode(job: tuple) -> dict:
    """Roll one episode. `job = (star, seed, policy_state)`.

    `policy_state` is the latest state_dict from the main process (sent
    every iteration). Worker reloads it before sampling so rollouts
    reflect the freshly-updated policy.

    Sampling mode matches BC Runtime: a per-aircraft `torch.Generator` is
    re-seeded with the SAME value every tick within the episode. This
    "frozen noise" pattern is what `bc_gmm.watch` and the BC eval use, so
    PPO optimizes the policy as it actually behaves at deployment — not
    the noise-averaged version PPO would see with fresh per-tick random
    sampling (which gave artificially high success rates).

    The seed is salted with the episode `seed` so different episodes of
    the same callsign explore different frozen-noise patterns. Within an
    episode it's stable, matching BC's per-aircraft determinism.
    """
    import traceback
    import torch

    star, seed, policy_state = job
    try:
        if policy_state is not None:
            _POLICY.load_state_split(policy_state, device='cpu')

        obs = _ENV.reset(star=star, seed=seed)

        # Per-episode seeded generator. Matches BC's rl_bc/bc_gmm/rollout.py
        # pattern: hash(callsign) & 0x7FFFFFFF re-seeded every tick. We salt
        # with the episode seed so PPO explores diverse frozen-noise
        # patterns across episodes.
        callsign = _ENV._callsign or '?'
        fm_seed = (hash((callsign, seed)) & 0x7FFFFFFF) or 1
        gen = torch.Generator(device='cpu')

        obs_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []
        lp_list: list[float] = []
        rw_list: list[float] = []
        val_list: list[float] = []
        done_list: list[bool] = []
        a_list: list[float] = []
        c_list: list[float] = []

        outcome = 'UNKNOWN'
        n_steps = 0
        err: str | None = None
        final_info: dict = {}
        while True:
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            # Re-seed every tick → same noise sequence at each tick, matches
            # BC Runtime. The action varies tick-to-tick because the state
            # (and thus the policy's distribution) changes, but the noise
            # offset applied to the distribution is identical.
            gen.manual_seed(fm_seed)
            out = _POLICY.act(obs_t, generator=gen)
            action = out['action'].squeeze(0).cpu().numpy()
            log_prob = float(out['log_prob'].squeeze(0).item())
            value = float(out['value'].squeeze(0).item())

            step = _ENV.step(action)
            obs_list.append(obs.copy())
            act_list.append(action.astype(np.float32))
            lp_list.append(log_prob)
            val_list.append(value)
            rw_list.append(float(step.reward))
            done_list.append(bool(step.done))
            a_list.append(float(step.info.get('a_nm', float('nan'))))
            c_list.append(float(step.info.get('c_nm', float('nan'))))
            n_steps += 1
            obs = step.obs
            if step.done:
                outcome = step.info.get('outcome', 'UNKNOWN')
                err = step.info.get('error')
                final_info = step.info
                break

        rewards = np.asarray(rw_list, dtype=np.float32)
        a_traj = np.asarray(a_list, dtype=np.float32)
        c_traj = np.asarray(c_list, dtype=np.float32)

        # ---- Post-hoc loop penalty ----
        # If cfg.loop_penalty_per_step > 0, run the loop detector and
        # subtract the per-step penalty from `rewards[n]` for every n
        # in the SECOND HALF of the looping-step list. First-half loop
        # steps get a "warning shot" (no penalty); the policy only pays
        # for persistent looping, not brief recoveries.
        loop_pps = float(_CONFIG.get('loop_penalty_per_step', 0.0) or 0.0)
        n_loop_steps = 0
        n_penalized = 0
        loop_penalty_total = 0.0
        if loop_pps > 0.0 and a_traj.size >= 2:
            from .loop_detector import detect_looping
            d = detect_looping(
                a_traj, c_traj,
                prox_radius_nm=float(_CONFIG.get('loop_prox_radius_nm', 0.75)),
                min_gap_steps=int(_CONFIG.get('loop_min_gap_steps', 45)),
                min_detour_nm=float(_CONFIG.get('loop_min_detour_nm', 1.0)),
                # min_loop_frac=0 — penalty applies to ANY loop steps,
                # regardless of fraction; the per-step scale handles
                # the "this trajectory is barely looping" case naturally.
                min_loop_frac=0.0,
            )
            mask = d['looping_step_mask']
            if mask.any():
                loop_indices = np.flatnonzero(mask)
                k = int(loop_indices.size)
                # ⌈k/2⌉ — slice 0..half is the warning, half..k pays.
                half = (k + 1) // 2
                penalize = loop_indices[half:]
                n_loop_steps = k
                n_penalized = int(penalize.size)
                if n_penalized > 0:
                    rewards[penalize] -= loop_pps
                    loop_penalty_total = float(loop_pps * n_penalized)

        return {
            'ok': True,
            'star': star, 'seed': seed,
            'outcome': outcome, 'steps': n_steps, 'error': err,
            'obs': np.stack(obs_list).astype(np.float32),
            'actions': np.stack(act_list).astype(np.float32),
            'log_probs': np.asarray(lp_list, dtype=np.float32),
            'rewards': rewards,
            'values': np.asarray(val_list, dtype=np.float32),
            'dones': np.asarray(done_list, dtype=bool),
            'a_traj': a_traj,
            'c_traj': c_traj,
            'd_thr_nm': final_info.get('d_thr_nm'),
            'altitude_ft': final_info.get('altitude_ft'),
            'gs_alt_ft': final_info.get('gs_alt_ft'),
            'n_loop_steps': n_loop_steps,
            'n_penalized': n_penalized,
            'loop_penalty_total': loop_penalty_total,
        }
    except Exception as exc:
        return {
            'ok': False,
            'star': star, 'seed': seed,
            'outcome': 'WORKER_CRASHED',
            'error': f"{type(exc).__name__}: {exc}",
            'traceback': traceback.format_exc(limit=4),
        }


# --------------------------------------------------------------------------- #
# Main-process orchestration
# --------------------------------------------------------------------------- #


def make_rollout_pool(actor_ckpt: str, cfg_dict: dict, n_workers: int):
    """Create the pool once at the start of training and reuse it across
    PPO iterations (constructing the pool is the expensive part — each
    worker loads the model and standardizers from disk)."""
    return mp.Pool(n_workers, initializer=_init_worker,
                   initargs=(str(actor_ckpt), cfg_dict))


def collect_rollouts(pool, n_rollouts: int, stars: tuple[str, ...],
                     seed_offset: int, policy_state: dict,
                     verbose: bool = True) -> list[Trajectory]:
    """Submit `n_rollouts` jobs balanced across `stars`, return Trajectories.

    Jobs that crash are returned with their error info (for debugging) and
    counted as failures — they don't enter the training batch.
    """
    n_stars = len(stars)
    base_per_star = n_rollouts // n_stars
    remainder = n_rollouts - base_per_star * n_stars
    jobs = []
    seed = seed_offset
    for i, star in enumerate(stars):
        k = base_per_star + (1 if i < remainder else 0)
        for _ in range(k):
            jobs.append((star, seed, policy_state))
            seed += 1

    t0 = time.time()
    trajectories: list[Trajectory] = []
    n_done = 0
    n_success = 0
    n_crashed = 0
    for r in pool.imap_unordered(_run_episode, jobs, chunksize=1):
        n_done += 1
        if not r.get('ok'):
            n_crashed += 1
            if verbose:
                print(f"  [{n_done}/{len(jobs)}] WORKER_CRASHED "
                      f"{r['star']} seed={r['seed']}: {r.get('error', '?')}",
                      flush=True)
            continue
        traj = Trajectory(
            obs=r['obs'], actions=r['actions'],
            log_probs=r['log_probs'], rewards=r['rewards'],
            values=r['values'], dones=r['dones'],
            star=r['star'], outcome=r['outcome'],
            steps=r['steps'], error=r.get('error'),
            d_thr_nm=r.get('d_thr_nm'),
            altitude_ft=r.get('altitude_ft'),
            gs_alt_ft=r.get('gs_alt_ft'),
            a_traj=r.get('a_traj'),
            c_traj=r.get('c_traj'),
            n_loop_steps=int(r.get('n_loop_steps', 0)),
            n_penalized=int(r.get('n_penalized', 0)),
            loop_penalty_total=float(r.get('loop_penalty_total', 0.0)),
        )
        if traj.outcome == 'LOC_BELOW_GS':
            n_success += 1
        trajectories.append(traj)

    if verbose:
        elapsed = time.time() - t0
        print(f"  rollouts: {len(trajectories)} ok ({n_success} success, "
              f"{n_crashed} crashed) in {elapsed:.1f}s "
              f"({sum(t.steps for t in trajectories)} env steps total)",
              flush=True)
    return trajectories
