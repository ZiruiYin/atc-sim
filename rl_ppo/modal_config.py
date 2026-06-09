"""Modal infrastructure for the PPO trainer.

Self-contained — does NOT import `rl_bc.modal_config` at module level.
Modal mounts each PythonPackage at `/root/<pkg>/` in the container but
only the entry script's package gets `/root` on `sys.path` automatically,
so a top-level `from rl_bc...` import would fail in the container even
though both packages are mounted. Internal rl_bc / rl_ppo imports happen
LAZILY inside `train_ppo_remote` after we explicitly fix `sys.path`.

Shares the same Modal-backend resources as `rl_bc.modal_config` by using
the same names: `modal.App("atc-bc")` and `modal.Volume.from_name("atc-bc")`
resolve to the same backend app / volume regardless of which Python file
constructs them. So the volume already has `human_data/`, `cache/`, and
`rl_bc/runs/` from the BC pipeline.

The PPO trainer reads the actor checkpoint from
`/runs/<bc_config>/run_N/best.pt` on the volume and writes PPO checkpoints
into `/runs_ppo/<run_name>/`. Separate prefix keeps BC and PPO outputs
isolated on the same volume.

CPU allocation
--------------
PPO rollouts are CPU-bound (sim + small policy forward pass per step) and
embarrassingly parallel. On AMD EPYC vCPUs (what Modal hands out),
hyperthread siblings share L2 cache — running one rollout worker per vCPU
oversubscribes physical cores and degrades throughput by ~30–50%.

The remote function below allocates `cpu=64` vCPUs (= ~32 physical cores
on EPYC). The PPO worker pool autodetects `cpu_count // 2 = 32` workers
so each one lands on its own physical core. Edit `cpu=` to scale; the
worker count autoscales with it.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

# Cross-platform "modal" invocation. On Windows the `modal` script lives
# in the env's Scripts/ folder and isn't on PATH for subprocess; using
# `python -m modal` works on every OS.
_MODAL_CMD = [sys.executable, "-m", "modal"]
from pathlib import Path

import modal


# --------------------------------------------------------------------------- #
# Names.
#
#   APP_NAME    = "atc-ppo-single"  — separate Modal app, so PPO runs don't
#                                     mix with the BC train/eval functions
#                                     in the Modal dashboard.
#   VOLUME_NAME = "atc-bc"          — SHARED with rl_bc/modal_config.py.
#                                     The volume already has `human_data/`
#                                     and `rl_bc/runs/.../best.pt` (the
#                                     actor seed). PPO outputs go to
#                                     `/runs_ppo/<run_name>/` on the same
#                                     volume — separate path prefix keeps
#                                     them isolated.
#   WANDB_PROJECT = "atc-ppo"       — see PPOConfig.wandb_project.
# --------------------------------------------------------------------------- #


APP_NAME = "atc-ppo-single"
VOLUME_NAME = "atc-bc"
WANDB_SECRET_NAME = "wandb"
REMOTE_REPO_ROOT = "/root/atc-sim"
REMOTE_VOLUME_MOUNT = "/root/atc-sim/_modal"

# Per-block result pulls land here (default: the in-repo runs dir).
# IMPORTANT for CONCURRENT runs: a pull (or its PNG render) writing into
# the repo while ANOTHER run's `modal run` is hashing the local mount
# triggers Modal's "modified during build" error — which HANGS the build.
# So when running multiple continuous jobs at once, set $ATC_PPO_PULLS to
# a path OUTSIDE the repo (the ablation driver `run_ablation.ps1` does
# this) so concurrent pulls never touch the mounted tree.
_LOCAL_PPO_RUNS = Path(os.environ.get("ATC_PPO_PULLS") or "rl_ppo/runs")


# --------------------------------------------------------------------------- #
# Modal app / image / volume / secret — independent construction, same names
# as rl_bc so they resolve to the same backend resources.
# --------------------------------------------------------------------------- #


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


def next_ppo_run_number(prefix: str = 'run') -> int:
    """Next free run number under `/runs_ppo/<prefix>_N` on the Modal volume."""
    proc = subprocess.run(
        [*_MODAL_CMD, "volume", "ls", VOLUME_NAME, "/runs_ppo"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return 1
    nums = []
    for line in proc.stdout.splitlines():
        m = re.search(rf"\b{re.escape(prefix)}_(\d+)\b", line)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def pull_ppo_run_back(run_name: str) -> Path:
    """Download a finished PPO run from the volume into rl_ppo/runs/<run_name>/.

    Robust to Windows file locks: an earlier Read on a rendered PNG can
    hold an OS-level handle on its parent directory, preventing both
    `shutil.rmtree` AND `modal volume get` from writing inside it. We
    work around this by pulling into a SIBLING temp dir, then moving
    files into place per-file (which only fails on the locked subtree,
    not the rest of the run).
    """
    import shutil
    import tempfile
    import time

    remote_dir = f"/runs_ppo/{run_name}"
    local_dir = _LOCAL_PPO_RUNS / run_name
    local_dir.mkdir(parents=True, exist_ok=True)

    # Pull to a tmp sibling first to dodge any locked subdir in local_dir.
    tmp_parent = _LOCAL_PPO_RUNS / f'_pull_{run_name}_{int(time.time())}'
    tmp_parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [*_MODAL_CMD, "volume", "get", "--force", VOLUME_NAME,
             remote_dir, str(tmp_parent)],
            check=True,
        )
        # `modal volume get <remote_dir> <tmp_parent>` drops the tree at
        # `tmp_parent/<basename(remote_dir)>` — the LAST path component only,
        # even when run_name is nested (e.g. 'continuous_runs/foo' lands at
        # tmp_parent/foo, not tmp_parent/continuous_runs/foo). Use basename
        # to locate it; files still copy into the nested local_dir below.
        tmp_run = tmp_parent / Path(run_name).name
        if not tmp_run.exists():
            raise RuntimeError(f"pulled tree missing expected dir {tmp_run}")
        # Move files individually into local_dir; skip ones that can't
        # land because of file locks (locked dirs are stale anyway).
        n_moved, n_skipped = 0, 0
        for src in tmp_run.rglob('*'):
            rel = src.relative_to(tmp_run)
            dst = local_dir / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n_moved += 1
            except (PermissionError, OSError) as exc:
                n_skipped += 1
        if n_skipped:
            print(f"  [pull] {n_moved} files copied, {n_skipped} skipped "
                  f"(likely held by Windows; data is on volume anyway)",
                  flush=True)
    finally:
        try:
            shutil.rmtree(tmp_parent, ignore_errors=True)
        except Exception:
            pass
    return local_dir


# --------------------------------------------------------------------------- #
# Remote PPO training
# --------------------------------------------------------------------------- #


@app.function(
    image=image,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    secrets=_secrets,
    cpu=64,                 # = ~32 physical cores on AMD EPYC after SMT.
                            # PPO worker pool autodetects cpu_count // 2 = 32,
                            # giving one worker per physical core (avoids the
                            # SMT-oversubscription slowdown we saw on bc_eval).
                            # Drop to 16 / 32 for tighter cost / smaller jobs.
    memory=32 * 1024,       # 32 GiB; each worker holds a sim + policy
                            # (~200 MB), 32 workers + main = ~7 GB peak.
    timeout=6 * 60 * 60,
)
def train_ppo_remote(args: dict):
    """Run PPO inside a Modal container.

    `args` keys:
      run_name, actor_ckpt_relpath, n_iters, n_rollouts_per_iter, n_workers,
      n_epochs, lr_actor, lr_critic, clip_epsilon, gamma, gae_lambda,
      target_kl, entropy_coef, batch_size, max_timesteps_star_1_2,
      max_timesteps_star_3, seed, use_wandb.
    """
    import os
    import sys
    import multiprocessing as mp
    from pathlib import Path

    os.chdir(REMOTE_REPO_ROOT)
    if REMOTE_REPO_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_REPO_ROOT)

    # Required for "spawn" start method when fork would inherit torch state.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    from rl_ppo.config import PPOConfig
    from rl_ppo.train import train

    vol_root = Path(REMOTE_VOLUME_MOUNT)
    run_name = args['run_name']
    run_dir = vol_root / "runs_ppo" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    actor_ckpt_rel = args['actor_ckpt_relpath']
    actor_ckpt = vol_root / actor_ckpt_rel
    if not actor_ckpt.exists():
        raise FileNotFoundError(f"actor ckpt not on volume: {actor_ckpt}")

    cfg = PPOConfig(actor_ckpt=actor_ckpt, run_dir=run_dir)
    # Apply every override the caller passed (only the ones present).
    for k in ('n_iters', 'n_rollouts_per_iter', 'n_workers', 'n_epochs',
              'lr_actor', 'lr_critic', 'clip_epsilon', 'gamma', 'gae_lambda',
              'target_kl', 'entropy_coef', 'batch_size',
              'max_timesteps_star_1_2', 'max_timesteps_star_3',
              'seed', 'value_hidden', 'value_dropout', 'normalize_advantages',
              'max_grad_norm', 'value_coef', 'gs_capture_buffer_ft',
              'log_every', 'save_every', 'eval_every', 'eval_cases',
              # Reward-shaping knobs — so a single straight `--n-iters` run
              # can reproduce a continuous-driver reward recipe (train()
              # already forwards these from cfg to the rollout workers).
              'everywhere_step_penalty', 'step_penalty_cap',
              'step_penalty_per_nm', 'out_of_zone_terminate',
              'out_of_zone_max_consecutive', 'per_star_sr_scale',
              'loop_penalty_per_step',
              'heading_intercept_enabled', 'turn_final_enabled'):
        if k in args and args[k] is not None:
            setattr(cfg, k, args[k])

    # Report the resolved worker count + machine vCPU count so the log
    # makes it obvious whether we're SMT-oversubscribed or not.
    print(f"[ppo-remote] os.cpu_count()={os.cpu_count()}  "
          f"cfg.n_workers={cfg.n_workers}  "
          f"resolved_n_workers={cfg.resolve_n_workers()}", flush=True)

    metric_hook = None
    use_wandb = bool(args.get('use_wandb', False))
    if use_wandb:
        import wandb
        wandb.init(
            project='atc-ppo',
            name=run_name,
            group='ppo',
            config={k: (str(v) if isinstance(v, Path) else v)
                    for k, v in cfg.__dict__.items()},
        )

        def metric_hook(metrics: dict):
            # train() now hands us a fully-flat metrics dict (per-STAR
            # already broken out into `per_star/<STAR>/<key>` keys), so
            # we just forward everything except the step counter.
            step = metrics.get('iter')
            payload = {k: v for k, v in metrics.items() if k != 'iter'}
            wandb.log(payload, step=step)

    try:
        summary = train(cfg, metric_hook=metric_hook)
    finally:
        if use_wandb:
            import wandb
            wandb.finish()
        volume.commit()

    summary['run_name'] = run_name
    return summary


# --------------------------------------------------------------------------- #
# Continuous-run: train ONE block + eval, save into run_dir on the volume.
# Mirrors `train_ppo_remote` but supports per-block resume + tunable rewards
# + a post-train eval. All `rl_ppo.*` imports are LAZY (inside the function)
# so this module loads cleanly even if `/root/atc-sim` isn't yet on sys.path
# at import time (Modal mounts the entry script as /root/<file>).
# --------------------------------------------------------------------------- #


@app.function(
    image=image,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    secrets=_secrets,
    cpu=64,
    memory=32 * 1024,
    timeout=6 * 60 * 60,
)
def train_block_and_eval(args: dict):
    """Train iters [block_from_iter, block_to_iter) on the resumed
    policy, then eval the resulting checkpoint.

    Required `args` keys:
        run_name, block_from_iter, block_to_iter, actor_ckpt_relpath,
        gamma, everywhere_step_penalty, step_penalty_cap
    Optional:
        n_rollouts_per_iter, entropy_coef, target_kl, lr_actor,
        lr_critic, batch_size, eval_cases, eval_seed_base, seed,
        use_wandb, wandb_group.
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

    from rl_ppo.config import PPOConfig
    from rl_ppo.train import train
    from rl_ppo.eval_runner import run_eval

    vol_root = Path(REMOTE_VOLUME_MOUNT)
    run_name = args['run_name']
    run_dir = vol_root / 'runs_ppo' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    block_from = int(args['block_from_iter'])
    block_to   = int(args['block_to_iter'])
    n_iters_block = block_to - block_from
    if n_iters_block <= 0:
        raise ValueError(
            f'block_to_iter ({block_to}) must exceed block_from_iter ({block_from})'
        )

    # block_from == 0 → seed from BC; else resume from previous block's ckpt.
    resume_ckpt = None
    if block_from > 0:
        resume_ckpt = run_dir / f'iter_{block_from:04d}.pt'
        if not resume_ckpt.exists():
            raise FileNotFoundError(
                f'resume ckpt not on volume: {resume_ckpt}. '
                f'Did the previous block save at iter_{block_from:04d}?'
            )

    actor_ckpt = vol_root / args['actor_ckpt_relpath']
    if not actor_ckpt.exists():
        raise FileNotFoundError(f'actor (BC seed) not on volume: {actor_ckpt}')

    cfg = PPOConfig(actor_ckpt=actor_ckpt, run_dir=run_dir)
    cfg.n_iters = n_iters_block
    cfg.gamma = float(args['gamma'])
    cfg.everywhere_step_penalty = float(args['everywhere_step_penalty'])
    cfg.step_penalty_cap = float(args['step_penalty_cap'])
    if 'step_penalty_per_nm' in args and args['step_penalty_per_nm'] is not None:
        cfg.step_penalty_per_nm = float(args['step_penalty_per_nm'])
    for k in ('early_zone_multiplier', 'early_window_steps',
              'early_drift_penalty', 'clean_terminal_threshold',
              'drifty_success_value', 'out_of_zone_terminate',
              'out_of_zone_max_consecutive', 'per_star_sr_scale',
              'heading_intercept_enabled', 'turn_final_enabled'):
        if k in args and args[k] is not None:
            setattr(cfg, k, args[k])
    # Loop penalty + its detector hyperparams (defaults if absent).
    if 'loop_penalty_per_step' in args and args['loop_penalty_per_step'] is not None:
        cfg.loop_penalty_per_step = float(args['loop_penalty_per_step'])
    for k in ('loop_prox_radius_nm', 'loop_min_gap_steps', 'loop_min_detour_nm'):
        if k in args and args[k] is not None:
            setattr(cfg, k, args[k])

    for k in ('n_rollouts_per_iter', 'entropy_coef', 'target_kl',
              'lr_actor', 'lr_critic', 'batch_size', 'n_workers',
              'n_epochs', 'seed', 'eval_every', 'eval_cases'):
        if k in args and args[k] is not None:
            setattr(cfg, k, args[k])

    print(f'[continuous] block iters {block_from + 1}..{block_to} '
          f'(={n_iters_block} iters)', flush=True)
    print(f'[continuous] resume_ckpt={resume_ckpt}', flush=True)
    print(f'[continuous] gamma={cfg.gamma} everywhere={cfg.everywhere_step_penalty} '
          f'slope={cfg.step_penalty_per_nm} cap={cfg.step_penalty_cap} '
          f'loop_pen={cfg.loop_penalty_per_step}',
          flush=True)
    print(f'[continuous] n_rollouts={cfg.n_rollouts_per_iter}  '
          f'entropy_coef={cfg.entropy_coef}', flush=True)

    metric_hook = None
    use_wandb = bool(args.get('use_wandb', False))
    if use_wandb:
        import wandb
        wandb.init(
            project='atc-ppo',
            name=f'{run_name}_block_{block_from:04d}_{block_to:04d}',
            group=args.get('wandb_group', 'ppo_continuous'),
            config={
                'block_from_iter': block_from,
                'block_to_iter': block_to,
                'gamma': cfg.gamma,
                'everywhere_step_penalty': cfg.everywhere_step_penalty,
                'step_penalty_cap': cfg.step_penalty_cap,
                'n_rollouts_per_iter': cfg.n_rollouts_per_iter,
                'entropy_coef': cfg.entropy_coef,
                'lr_actor': cfg.lr_actor,
                'lr_critic': cfg.lr_critic,
                'target_kl': cfg.target_kl,
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

    # ---- Eval the end-of-block ckpt ----
    end_ckpt = run_dir / f'iter_{block_to:04d}.pt'
    if not end_ckpt.exists():
        raise RuntimeError(
            f"expected end-of-block ckpt at {end_ckpt} but train() didn't save it"
        )
    eval_out = run_dir / f'iter_{block_to:04d}_eval'
    eval_cases = int(args.get('eval_cases', 200))
    eval_seed_base = int(args.get('eval_seed_base',
                                  10_000 + block_to * 1000))
    print(f'[continuous] eval {eval_cases} cases x 6 STARs -> {eval_out}',
          flush=True)
    metrics = run_eval(end_ckpt, eval_out,
                       cases=eval_cases, seed_base=eval_seed_base)
    volume.commit()

    o = metrics['overall']
    print(f'\n[continuous] iter_{block_to:04d} eval:')
    print(f'  overall  SR={o["success_rate"]*100:.1f}%  '
          f'macro_green={o["macro_pct_steps_in_green"]*100:.1f}%  '
          f'macro_in_range={o["macro_pct_within_length_range"]*100:.1f}%',
          flush=True)
    for star, s in metrics['per_star'].items():
        if s['n'] == 0:
            continue
        print(f'  {star}: SR={s["success_rate"]*100:5.1f}%  '
              f'green={s["pct_steps_in_green"]*100:5.1f}%  '
              f'in_range={s["pct_within_length_range"]*100:5.1f}%  '
              f'<min={s["pct_below_min_length"]*100:4.1f}%  '
              f'>max={s["pct_above_max_length"]*100:4.1f}%  '
              f'len_mean={s["length_mean"]:6.0f}  '
              f'len_std={s["length_std"]:5.0f}', flush=True)

    return {
        'run_name': run_name,
        'block_from_iter': block_from,
        'block_to_iter': block_to,
        'train_summary': summary,
        'eval_metrics_overall': metrics['overall'],
        'eval_metrics_per_star': metrics['per_star'],
    }


# --------------------------------------------------------------------------- #
# Sanity eval — no train, no disk writes. Runs N rollouts/STAR through the
# real PPOEnv (same termination logic as training) and returns SR. Used to
# verify a shipped checkpoint behaves as expected without touching any
# stored trajectories.
# --------------------------------------------------------------------------- #


@app.function(
    image=image,
    volumes={REMOTE_VOLUME_MOUNT: volume},
    cpu=64,
    memory=32 * 1024,
    timeout=30 * 60,
)
def eval_sanity_remote(args: dict):
    """Roll N episodes per STAR through PPOEnv, count LOC_BELOW_GS as
    success. Returns per-STAR + overall SR. No disk output.

    Required `args` keys:
        ckpt_relpath           — path to PPO ckpt under the volume mount
    Optional:
        bc_seed_relpath        — BC seed for arch + standardizer
                                 (default runs/bc_gmm_single_full/run_11/best.pt)
        n_per_star             — episodes per STAR (default 50)
        seed_base              — seed offset (default 999000, well outside
                                 the training seed space)
        n_workers              — worker pool size (default 32)
        stars                  — list of STARs (default all 6)
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

    import torch
    from rl_ppo.rollout import make_rollout_pool, collect_rollouts
    from rl_ppo.policy import PPOPolicy

    vol_root = Path(REMOTE_VOLUME_MOUNT)
    ckpt_path = vol_root / args['ckpt_relpath']
    bc_seed = vol_root / args.get(
        'bc_seed_relpath', 'runs/bc_gmm_single_full/run_11/best.pt')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'PPO ckpt not on volume: {ckpt_path}')
    if not bc_seed.exists():
        raise FileNotFoundError(f'BC seed not on volume: {bc_seed}')

    stars = tuple(args.get('stars',
                           ['NORTH1', 'NORTH2', 'NORTH3',
                            'SOUTH1', 'SOUTH2', 'SOUTH3']))
    n_per_star = int(args.get('n_per_star', 50))
    seed_base = int(args.get('seed_base', 999_000))
    n_workers = int(args.get('n_workers', 32))

    # PPOEnv + PPOPolicy defaults matching the trained policy. Reward
    # overrides intentionally LEFT UNSET (defaults) — eval only needs
    # the termination outcome (LOC_BELOW_GS vs everything else).
    cfg_dict = {
        'airport_name': 'test',
        'runway': '27',
        'warmup_wpts': 2,
        'max_timesteps_star_1_2': 1200,
        'max_timesteps_star_3': 500,
        'success_reward': 10.0,
        'failure_reward': -10.0,
        'gs_capture_buffer_ft': 50.0,
        'value_hidden': 64,
        'value_dropout': 0.0,
    }

    print(f'[sanity_eval] ckpt={ckpt_path}', flush=True)
    print(f'[sanity_eval] bc_seed={bc_seed}', flush=True)
    print(f'[sanity_eval] {n_per_star} cases x {len(stars)} STARs = '
          f'{n_per_star * len(stars)} total  '
          f'workers={n_workers}  seed_base={seed_base}', flush=True)

    pool = make_rollout_pool(str(bc_seed), cfg_dict, n_workers)

    # Build a main-process policy with BC arch, then load the PPO weights
    # into it. `collect_rollouts` broadcasts this state to every worker.
    policy = PPOPolicy.from_bc_checkpoint(str(bc_seed))
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'actor_state' not in blob:
        raise ValueError(
            f"ckpt at {ckpt_path} has no `actor_state` "
            f"(keys: {list(blob.keys())[:6]}...)")
    policy.actor.load_state_dict(blob['actor_state'])
    if 'critic_state' in blob:
        policy.critic.load_state_dict(blob['critic_state'])

    n_total = n_per_star * len(stars)
    trajs = collect_rollouts(
        pool, n_total, stars,
        seed_offset=seed_base,
        policy_state=policy.state_dict_split(),
        verbose=False,
    )
    pool.close()
    pool.join()

    by_star: dict = {}
    crash_details: list = []
    for t in trajs:
        bs = by_star.setdefault(t.star, {'n': 0, 'succ': 0, 'outcomes': {}})
        bs['n'] += 1
        if t.outcome == 'LOC_BELOW_GS':
            bs['succ'] += 1
        bs['outcomes'][t.outcome] = bs['outcomes'].get(t.outcome, 0) + 1
        if t.outcome == 'CRASHED' and t.error:
            crash_details.append({
                'star': t.star,
                'steps': t.steps,
                'error': t.error,
            })

    n_succ = sum(bs['succ'] for bs in by_star.values())
    return {
        'ckpt_iter': int(blob.get('iter', -1)),
        'ckpt_relpath': args['ckpt_relpath'],
        'total': len(trajs),
        'n_success': n_succ,
        'sr_overall': n_succ / max(1, len(trajs)),
        'per_star': by_star,
        'crash_details': crash_details,
    }
