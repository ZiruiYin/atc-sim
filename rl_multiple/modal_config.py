"""Modal infra for multi-plane PPO.

Shares the `atc-bc` Modal volume with `rl_bc` and `rl_ppo` (so BC seeds
and the shipped single-plane PPO ckpt are already on disk), but uses a
distinct app name `atc-ppo-multi` so runs show up separately in the
dashboard and a separate volume path prefix `runs_ppo_multi/` so outputs
don't collide with the rl_ppo single-plane track.

Public surface:
  train_block_and_eval_multi(args)   — remote function (cpu=64)
  pull_multi_run_back(run_name)      — local helper to fetch results
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "atc-ppo-multi"
VOLUME_NAME = "atc-bc"
WANDB_SECRET_NAME = "wandb"
REMOTE_REPO_ROOT = "/root/atc-sim"
REMOTE_VOLUME_MOUNT = "/root/atc-sim/_modal"

_LOCAL_MULTI_RUNS = Path("rl_multiple/runs")
_MODAL_CMD = [sys.executable, "-m", "modal"]


app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".",
        remote_path=REMOTE_REPO_ROOT,
        ignore=[
            "__pycache__", "*.pyc", ".git", ".vscode", ".pytest_cache",
            "rl_bc/cache", "rl_bc/runs", "rl_bc/_modal", "rl_bc/rollouts",
            "rl_ppo/runs", "rl_ppo/_modal",
            "rl_multiple/runs",
            "_internal", "*.exe", "ATC-Sim.exe",
            "human_data", "doc", "eval", "data_viz",
        ],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
_secrets = [modal.Secret.from_name(WANDB_SECRET_NAME)]


# --------------------------------------------------------------------------- #
# Local-side helpers
# --------------------------------------------------------------------------- #


def pull_multi_run_back(run_name: str) -> Path:
    """Pull /runs_ppo_multi/<run_name>/ back to rl_multiple/runs/<run_name>/.

    Same Windows-friendly tmp-dir-then-move dance as rl_ppo's helper.

    Handles nested run_name like `phase2/continuous_02`:
    - tmp_parent name is sanitized (slashes → underscores)
    - modal volume get places contents at <tmp_parent>/<leaf-of-run_name>
    """
    import shutil
    import tempfile
    import time

    # Sanitize run_name when building tmp_parent so a nested run_name
    # ("phase2/continuous_02") doesn't get interpreted as a multi-level
    # path. Use the leaf to locate the modal-pulled subdir.
    safe_name = run_name.replace('/', '_').replace('\\', '_')
    leaf_name = Path(run_name).name

    remote_dir = f"/runs_ppo_multi/{run_name}"
    local_dir = _LOCAL_MULTI_RUNS / run_name
    local_dir.mkdir(parents=True, exist_ok=True)

    tmp_parent = _LOCAL_MULTI_RUNS / f'_pull_{safe_name}_{int(time.time())}'
    tmp_parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [*_MODAL_CMD, "volume", "get", "--force", VOLUME_NAME,
             remote_dir, str(tmp_parent)],
            check=True,
        )
        # `modal volume get` places contents at <tmp_parent>/<leaf>.
        tmp_run = tmp_parent / leaf_name
        if not tmp_run.exists():
            raise RuntimeError(f"pulled tree missing expected dir {tmp_run}")
        n_moved, n_skipped = 0, 0
        for src in tmp_run.rglob('*'):
            rel = src.relative_to(tmp_run)
            dst = local_dir / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))
                n_moved += 1
            except OSError as e:
                print(f"  [warn] skip {rel}: {e}")
                n_skipped += 1
        print(f"  [pull] moved {n_moved}, skipped {n_skipped}")
    finally:
        try:
            shutil.rmtree(tmp_parent, ignore_errors=True)
        except Exception:
            pass

    return local_dir


# --------------------------------------------------------------------------- #
# Remote training function — one BLOCK at a time, identical contract to
# rl_ppo.modal_config.train_block_and_eval but for the multi-plane stack.
# --------------------------------------------------------------------------- #


@app.function(
    image=image,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    secrets=_secrets,
    cpu=64,
    memory=32 * 1024,
    timeout=6 * 60 * 60,
)
def train_block_and_eval_multi(args: dict):
    """Train iters [block_from_iter, block_to_iter) of multi-plane PPO.

    Required args:
        run_name, block_from_iter, block_to_iter, ppo_ckpt_relpath
    Optional args (forwarded into MultiPPOConfig if present):
        bc_seed_relpath, gamma, gae_lambda, lr_actor, lr_critic,
        entropy_coef, target_kl, clip_epsilon, batch_size, n_epochs,
        n_rollouts_per_iter, n_workers,
        everywhere_step_penalty, step_penalty_cap, step_penalty_per_nm,
        early_zone_multiplier, early_window_steps, early_drift_penalty,
        clean_terminal_threshold, drifty_success_value,
        out_of_zone_terminate, out_of_zone_max_consecutive,
        per_star_sr_scale, loop_penalty_per_step,
        delta_hidden, delta_hdg_clamp_deg, delta_alt_clamp_kft,
        delta_spd_clamp_kt, delta_log_sigma_init,
        seed, use_wandb, wandb_group.
    """
    import os
    import sys
    import multiprocessing as mp
    from pathlib import Path

    os.chdir(REMOTE_REPO_ROOT)
    if REMOTE_REPO_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_REPO_ROOT)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    from rl_multiple.config import MultiPPOConfig
    from rl_multiple.train import train

    vol_root = Path(REMOTE_VOLUME_MOUNT)
    run_name = args['run_name']
    run_dir = vol_root / 'runs_ppo_multi' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    block_from = int(args['block_from_iter'])
    block_to = int(args['block_to_iter'])
    n_iters_block = block_to - block_from
    if n_iters_block <= 0:
        raise ValueError(
            f'block_to_iter ({block_to}) must exceed block_from_iter ({block_from})'
        )

    # Resume from previous block if block_from > 0.
    resume_ckpt = None
    if block_from > 0:
        resume_ckpt = run_dir / f'iter_{block_from:04d}.pt'
        if not resume_ckpt.exists():
            raise FileNotFoundError(
                f'resume ckpt not on volume: {resume_ckpt}. '
                f'Did the previous block save at iter_{block_from:04d}?'
            )

    ppo_ckpt = vol_root / args['ppo_ckpt_relpath']
    if not ppo_ckpt.exists():
        raise FileNotFoundError(f'PPO seed not on volume: {ppo_ckpt}')

    cfg = MultiPPOConfig(ppo_ckpt=ppo_ckpt, run_dir=run_dir)
    cfg.n_iters = n_iters_block

    if 'bc_seed_relpath' in args and args['bc_seed_relpath']:
        cfg.bc_seed_ckpt = vol_root / args['bc_seed_relpath']

    # Translate user-friendly local init_radar_head_from path
    # (e.g. `rl_multiple/runs/phase1_v1/best.pt`) into the Modal
    # volume path (`/root/atc-sim/_modal/runs_ppo_multi/phase1_v1/best.pt`).
    # The local rl_multiple/runs tree is in the image-ignore list, so
    # the local path doesn't exist inside the container — but the
    # phase1 / phase2 ckpts were saved on the volume, so we redirect.
    irhf = args.get('init_radar_head_from')
    if irhf:
        irhf_norm = irhf.replace('\\', '/')
        if irhf_norm.startswith('rl_multiple/runs/'):
            rel = irhf_norm[len('rl_multiple/runs/'):]
            args['init_radar_head_from'] = str(vol_root / 'runs_ppo_multi' / rel)
            print(f"[multi-ppo] init_radar_head_from translated: "
                  f"{irhf!r} -> {args['init_radar_head_from']!r}", flush=True)

    # Generic config-passthrough: any MultiPPOConfig field present in
    # args overrides the default.
    for k in (
        'gamma', 'gae_lambda', 'lr_actor', 'lr_critic', 'entropy_coef',
        'target_kl', 'clip_epsilon', 'batch_size', 'n_epochs',
        'n_rollouts_per_iter', 'n_workers', 'value_coef',
        'value_hidden', 'value_dropout',
        'everywhere_step_penalty', 'step_penalty_cap',
        'step_penalty_per_nm', 'early_zone_multiplier',
        'early_window_steps', 'early_drift_penalty',
        'clean_terminal_threshold', 'drifty_success_value',
        'out_of_zone_terminate', 'out_of_zone_max_consecutive',
        'per_star_sr_scale', 'loop_penalty_per_step',
        'loop_prox_radius_nm', 'loop_min_gap_steps', 'loop_min_detour_nm',
        'delta_hidden', 'delta_hdg_clamp_deg', 'delta_alt_clamp_kft',
        'delta_spd_clamp_kt', 'delta_log_sigma_init',
        'delta_log_sigma_min', 'delta_log_sigma_max',
        'density_cutoff_nm', 'seed', 'eval_six_pack_every', 'save_every',
        # Phase-2 multi-plane
        'multi_plane', 'spawn_rate', 'collision_warning_penalty',
        'crash_extra_penalty', 'init_radar_head_from', 'drop_truncated',
    ):
        if k in args and args[k] is not None:
            setattr(cfg, k, args[k])

    print(f'[multi-ppo] block iters {block_from + 1}..{block_to} '
          f'(={n_iters_block} iters)', flush=True)
    print(f'[multi-ppo] resume_ckpt={resume_ckpt}', flush=True)
    print(f'[multi-ppo] frozen-GMM ckpt: {cfg.ppo_ckpt}', flush=True)
    print(f'[multi-ppo] lr_actor={cfg.lr_actor}  '
          f'lr_critic={cfg.lr_critic}  target_kl={cfg.target_kl}  '
          f'entropy={cfg.entropy_coef}', flush=True)

    metric_hook = None
    use_wandb = bool(args.get('use_wandb', False))
    if use_wandb:
        import wandb
        wandb.init(
            project='atc-ppo-multi',
            name=f'{run_name}_block_{block_from:04d}_{block_to:04d}',
            group=args.get('wandb_group', 'ppo_multi'),
            config={
                'block_from_iter': block_from,
                'block_to_iter': block_to,
                'gamma': cfg.gamma,
                'lr_actor': cfg.lr_actor,
                'lr_critic': cfg.lr_critic,
                'target_kl': cfg.target_kl,
                'entropy_coef': cfg.entropy_coef,
                'n_rollouts_per_iter': cfg.n_rollouts_per_iter,
            },
        )

        def metric_hook(metrics: dict):
            step = metrics.get('iter')
            payload = {k: v for k, v in metrics.items() if k != 'iter'}
            wandb.log(payload, step=step)

    summary = {}
    try:
        summary = train(
            cfg,
            metric_hook=metric_hook,
            resume_ckpt=resume_ckpt,
            start_iter=block_from,
        )
    finally:
        if use_wandb:
            import wandb
            wandb.finish()
        volume.commit()

    end_ckpt = run_dir / f'iter_{block_to:04d}.pt'
    return {
        'run_name': run_name,
        'block_from_iter': block_from,
        'block_to_iter': block_to,
        'end_ckpt_exists': end_ckpt.exists(),
        'train_summary': summary,
    }


# --------------------------------------------------------------------------- #
# Standalone eval remote — no training, no disk-blowing rollouts (rollouts
# are inside the function and written to a fresh subdir under the run's
# volume path). Returns SR + macro_green per STAR.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Paired A/B comparison remote — used by modal_compare_eval.
# Worker fn must live at module scope (mp.Pool with spawn can't pickle
# closures or local defs).
# --------------------------------------------------------------------------- #


def _compare_worker_chunk(args_tuple):
    import os
    import sys
    import time
    (chunk_idx, seeds, ckpt_a, multi_ckpt_a, ckpt_b, multi_ckpt_b,
     bc_seed, tag_a, tag_b, out_root, spawn_rate, max_steps,
     airport, runway, deterministic) = args_tuple

    os.chdir(REMOTE_REPO_ROOT)
    if REMOTE_REPO_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_REPO_ROOT)
    # Critical: cap torch's per-process thread pool BEFORE torch is
    # imported anywhere. Without this, 32 workers x 8 BLAS threads
    # oversubscribe the 64 CPUs by 4x and everything thrashes.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    import torch
    torch.set_num_threads(1)

    from rl_multiple.compare_eval import (
        run_one_scenario, fair_cap, build_runtime,
    )
    from pathlib import Path as _Path

    out_root = _Path(out_root)
    dir_a = out_root / tag_a
    dir_b = out_root / tag_b
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    rt_a = build_runtime(ckpt_a, multi_ckpt_a, bc_seed,
                          deterministic=deterministic)
    rt_b = build_runtime(ckpt_b, multi_ckpt_b, bc_seed,
                          deterministic=deterministic)

    out_kept_a, out_kept_b = [], []
    scenarios = []
    for s in seeds:
        t0 = time.time()
        csv_a = dir_a / f'seed_{s:06d}.csv'
        csv_b = dir_b / f'seed_{s:06d}.csv'
        res_a = run_one_scenario(rt_a, seed=s, spawn_rate=spawn_rate,
                                  max_steps=max_steps, out_csv=csv_a,
                                  airport=airport, runway=runway)
        res_b = run_one_scenario(rt_b, seed=s, spawn_rate=spawn_rate,
                                  max_steps=max_steps, out_csv=csv_b,
                                  airport=airport, runway=runway)
        kept_a, kept_b, t_cap = fair_cap(res_a, res_b, max_steps)
        kept_a = [t.to_dict() for t in kept_a if t.outcome != 'TRUNCATED']
        kept_b = [t.to_dict() for t in kept_b if t.outcome != 'TRUNCATED']
        out_kept_a.extend(kept_a)
        out_kept_b.extend(kept_b)
        scenarios.append({
            'seed': s,
            'crash_time_a': res_a.crash_time,
            'crash_time_b': res_b.crash_time,
            't_cap': t_cap,
            'n_kept_a': len(kept_a),
            'n_kept_b': len(kept_b),
            'csv_a': str(csv_a.relative_to(out_root)),
            'csv_b': str(csv_b.relative_to(out_root)),
        })
        print(f"  [chunk {chunk_idx} seed={s}] "
              f"crash_a={res_a.crash_time} crash_b={res_b.crash_time} "
              f"t_cap={t_cap:.0f} +A={len(kept_a)} +B={len(kept_b)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return chunk_idx, out_kept_a, out_kept_b, scenarios


@app.function(
    image=image,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    cpu=64,
    memory=64 * 1024,         # 64 GB — 32 GB was causing per-worker swap
    timeout=4 * 60 * 60,
)
def compare_remote(args: dict):
    """Paired multi-plane A/B eval. See modal_compare_eval.py for the
    local entrypoint that drives this."""
    import json
    import os
    import sys
    import multiprocessing as mp
    from pathlib import Path

    os.chdir(REMOTE_REPO_ROOT)
    if REMOTE_REPO_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_REPO_ROOT)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    from rl_multiple.compare_eval import metrics_from, Traj

    vol_root = Path(REMOTE_VOLUME_MOUNT)
    out_root = vol_root / args['out_relpath']
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_a = str(vol_root / args['ckpt_a_relpath'])
    multi_ckpt_a = (str(vol_root / args['multi_ckpt_a_relpath'])
                    if args.get('multi_ckpt_a_relpath') else None)
    ckpt_b = str(vol_root / args['ckpt_b_relpath'])
    multi_ckpt_b = (str(vol_root / args['multi_ckpt_b_relpath'])
                    if args.get('multi_ckpt_b_relpath') else None)
    bc_seed = (str(vol_root / args['bc_seed_relpath'])
               if args.get('bc_seed_relpath') else None)

    tag_a = args['tag_a']
    tag_b = args['tag_b']
    n_target = int(args['n_target'])
    seed_base = int(args.get('seed_base', 0))
    spawn_rate = int(args.get('spawn_rate', 90))
    max_steps = int(args.get('max_steps', 1500))
    max_scenarios = int(args.get('max_scenarios', 300))
    n_workers = int(args.get('n_workers', 32))
    airport = args.get('airport', 'test')
    runway = args.get('runway', '27')
    deterministic = bool(args.get('deterministic', False))

    seeds = list(range(seed_base, seed_base + max_scenarios))
    chunks = [seeds[i::n_workers] for i in range(n_workers)]
    worker_args = [
        (i, ch, ckpt_a, multi_ckpt_a, ckpt_b, multi_ckpt_b, bc_seed,
         tag_a, tag_b, str(out_root), spawn_rate, max_steps,
         airport, runway, deterministic)
        for i, ch in enumerate(chunks) if ch
    ]
    print(f"[compare] dispatching {len(worker_args)} workers, "
          f"{len(seeds)} seeds total", flush=True)

    all_kept_a, all_kept_b = [], []
    all_scenarios = []
    with mp.Pool(processes=len(worker_args)) as pool:
        for chunk_idx, ka, kb, scn in pool.imap_unordered(
                _compare_worker_chunk, worker_args):
            all_kept_a.extend(ka)
            all_kept_b.extend(kb)
            all_scenarios.extend(scn)
            print(f"[compare] chunk {chunk_idx} done: "
                  f"+A={len(ka)} +B={len(kb)} "
                  f"cum A={len(all_kept_a)} B={len(all_kept_b)}", flush=True)

    all_scenarios.sort(key=lambda r: r['seed'])
    # Workers process their assigned chunks in seed order, but chunks are
    # interleaved across workers — accept that the final n_target may not be
    # the strictly-lowest seeds.
    final_a = all_kept_a[:n_target]
    final_b = all_kept_b[:n_target]

    summary = {
        'config': {
            'tag_a': tag_a, 'tag_b': tag_b,
            'ckpt_a': args['ckpt_a_relpath'],
            'multi_ckpt_a': args.get('multi_ckpt_a_relpath'),
            'ckpt_b': args['ckpt_b_relpath'],
            'multi_ckpt_b': args.get('multi_ckpt_b_relpath'),
            'n_target': n_target,
            'seed_base': seed_base,
            'spawn_rate': spawn_rate,
            'max_steps': max_steps,
            'max_scenarios': max_scenarios,
            'n_workers_used': len(worker_args),
            'n_scenarios_run': len(all_scenarios),
            'reached_target_a': len(all_kept_a) >= n_target,
            'reached_target_b': len(all_kept_b) >= n_target,
        },
        'metrics_a': metrics_from([Traj(**t) for t in final_a]),
        'metrics_b': metrics_from([Traj(**t) for t in final_b]),
        'scenarios': all_scenarios,
        'trajs_a': final_a,
        'trajs_b': final_b,
    }
    (out_root / 'summary.json').write_text(json.dumps(summary, indent=2))
    volume.commit()

    return {
        'out_relpath': args['out_relpath'],
        'summary': {k: v for k, v in summary.items() if k != 'scenarios'},
    }


@app.function(
    image=image,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    cpu=64,
    memory=32 * 1024,
    timeout=30 * 60,
)
def eval_multi_remote(args: dict):
    """Standalone eval on a multi-PPO ckpt. Mirrors the contract of
    rl_ppo.modal_config.eval_sanity_remote but for the multi stack.

    Required args:
        ckpt_relpath          — path on the volume to the multi-PPO ckpt
                                  (e.g. runs_ppo_multi/<run>/iter_NNNN.pt)
        out_dir_relpath       — where to write the 6-pack (e.g.
                                  runs_ppo_multi/<run>/iter_NNNN_eval_full)
    Optional:
        ppo_seed_relpath      — frozen-GMM seed used during training
                                  (default: continuous_03 best.pt)
        bc_seed_relpath       — BC seed (arch + standardizer)
        n_per_star            — episodes per STAR (default 50)
        seed_base             — RNG seed base (default 999_000)
        n_workers             — pool size (default 32)
        out_of_zone_terminate — (default True)
        out_of_zone_max_consecutive — (default 10)
        everywhere_step_penalty     — (default 0.002)
    """
    import os
    import sys
    import multiprocessing as mp
    from pathlib import Path

    os.chdir(REMOTE_REPO_ROOT)
    if REMOTE_REPO_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_REPO_ROOT)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    from rl_multiple.eval_runner import run_eval

    vol_root = Path(REMOTE_VOLUME_MOUNT)
    ckpt_path = vol_root / args['ckpt_relpath']
    out_dir = vol_root / args['out_dir_relpath']

    ppo_seed_relpath = args.get(
        'ppo_seed_relpath', 'runs_ppo/continuous_03/iter_0160.pt')
    ppo_seed = vol_root / ppo_seed_relpath
    bc_seed_relpath = args.get(
        'bc_seed_relpath', 'runs/bc_gmm_single_full/run_11/best.pt')
    bc_seed = vol_root / bc_seed_relpath

    metrics = run_eval(
        ckpt_path=ckpt_path,
        ppo_seed_ckpt=ppo_seed,
        out_dir=out_dir,
        n_per_star=int(args.get('n_per_star', 50)),
        bc_seed=bc_seed,
        seed_base=int(args.get('seed_base', 999_000)),
        n_workers=int(args.get('n_workers', 32)),
        out_of_zone_terminate=bool(
            args.get('out_of_zone_terminate', True)),
        out_of_zone_max_consecutive=int(
            args.get('out_of_zone_max_consecutive', 10)),
        everywhere_step_penalty=float(
            args.get('everywhere_step_penalty', 0.002)),
        render=False,            # no matplotlib in this image
    )
    volume.commit()

    return {
        'ckpt_relpath': args['ckpt_relpath'],
        'out_dir_relpath': args['out_dir_relpath'],
        'overall': metrics['overall'],
        'per_star': metrics['per_star'],
    }
