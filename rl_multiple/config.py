"""Multi-plane PPO hyperparameters.

Mirrors `rl_ppo.config.PPOConfig` and inherits its defaults for everything
that's structurally identical (gamma, GAE, batch, lr scale). Adds knobs
specific to the delta-head architecture and the 79-D state-with-density
observation.

Phase 1 (single-plane stabilization) uses the same single-aircraft sim
and the same reward shaping as `rl_ppo`; the only differences vs vanilla
PPO are:

  - Observation is 79-D (ego_7 + density_36 + density_delta_36), with
    density_* identically zero when no other planes are present.
  - Trainable action is the 3-D delta (Δhdg_deg, Δalt_kft, Δspd_kt);
    final 4-D sim action = frozen_gmm_mode + delta (heading-wrapped,
    altitude/speed clamped).
  - PPO log_prob / entropy come from the delta head's 3-D diagonal
    Gaussian. The GMM is frozen and contributes no gradient.

Phase 2 will spawn multiple planes simultaneously; the density inputs
become non-zero and the delta head learns separation. No config change
needed here — the multi-plane sim is selected at the env level.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STARS_ALL = ('NORTH1', 'NORTH2', 'NORTH3', 'SOUTH1', 'SOUTH2', 'SOUTH3')


@dataclass
class MultiPPOConfig:
    # ------------------------------------------------------------------ #
    # Frozen-GMM seed (the shipped single-plane PPO ckpt)
    # ------------------------------------------------------------------ #
    # The "actor seed" is the PPO checkpoint whose actor_state we load
    # into a BCActor and FREEZE. Default = continuous_03 iter 160 ship.
    ppo_ckpt: Path = Path(
        'rl_ppo/runs/continuous_runs/continuous_03/best.pt')
    # The BC seed referenced by the PPO ckpt — needed for the
    # standardizer + arch config. If empty, resolved from the PPO
    # ckpt's `config.actor_ckpt` field.
    bc_seed_ckpt: Path = Path('')

    # ------------------------------------------------------------------ #
    # Environment / episode setup (Phase 1 = single plane)
    # ------------------------------------------------------------------ #
    airport_name: str = 'test'
    runway: str = '27'
    stars: tuple = STARS_ALL
    warmup_wpts: int = 2
    max_timesteps_star_1_2: int = 1200
    max_timesteps_star_3: int = 500

    # ------------------------------------------------------------------ #
    # Density observation (rl_multiple.density)
    # ------------------------------------------------------------------ #
    density_cutoff_nm: float = 10.0     # must match density.CUTOFF_NM
    # 36 hard-binned angular wedges (read at runtime from density.N_BINS;
    # exposed here only so config.json captures it for posterity)
    density_n_bins: int = 36

    # ------------------------------------------------------------------ #
    # Reward (same shape as rl_ppo Phase 1; loosened for delta-head
    # stabilization). Phase 1 disables OUT_OF_ZONE termination and
    # CLEAN_TERMINAL — the delta head needs room to wiggle without
    # immediate death.
    # ------------------------------------------------------------------ #
    success_reward: float = 10.0
    failure_reward: float = -10.0
    gs_capture_buffer_ft: float = 50.0

    everywhere_step_penalty: float = 0.001
    step_penalty_cap: float = 0.005
    step_penalty_per_nm: float = 0.0005
    early_zone_multiplier: float = 1.0
    early_window_steps: int = 300
    early_drift_penalty: float = 0.0
    clean_terminal_threshold: float = 0.0
    drifty_success_value: float = 0.0
    out_of_zone_terminate: bool = False        # Phase 1: off
    out_of_zone_max_consecutive: int = 5
    per_star_sr_scale: float = 0.0             # Phase 1: off

    loop_penalty_per_step: float = 0.0
    loop_prox_radius_nm: float = 0.75
    loop_min_gap_steps: int = 45
    loop_min_detour_nm: float = 1.0

    # ------------------------------------------------------------------ #
    # GAE / discount
    # ------------------------------------------------------------------ #
    gamma: float = 0.999
    gae_lambda: float = 0.95

    # ------------------------------------------------------------------ #
    # PPO update
    # ------------------------------------------------------------------ #
    clip_epsilon: float = 0.2
    n_epochs: int = 4
    batch_size: int = 256
    target_kl: float | None = 0.02
    entropy_coef: float = 0.01           # delta head Gaussian, NOT mixture
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True

    # ------------------------------------------------------------------ #
    # Optimizer. Only delta_head + critic params are updated; the frozen
    # GMM is excluded from the optimizer entirely (see policy.py).
    # ------------------------------------------------------------------ #
    lr_actor: float = 1e-5         # delta_head LR
    lr_critic: float = 3e-5
    weight_decay: float = 0.0

    # ------------------------------------------------------------------ #
    # Rollout collection
    # ------------------------------------------------------------------ #
    n_rollouts_per_iter: int = 128
    n_workers: int = 0
    n_iters: int = 200

    # ------------------------------------------------------------------ #
    # Phase-2 multi-plane controls
    # ------------------------------------------------------------------ #
    # When True, rollout uses MultiRolloutSim (multi-plane continuous
    # sim, success = `landed`, traffic density carries real signal,
    # collision-warning penalty applies). When False (Phase-1 legacy),
    # uses single-plane PPOEnv (success = LOC_BELOW_GS).
    multi_plane: bool = False
    spawn_rate: int = 120                # seconds between spawns
    # Drop TRUNCATED trajectories (planes flushed by other-plane crashes)
    # from the training batch. These bootstrap on V(s_T) which biases
    # the critic toward "near-crash" state distributions that aren't
    # the policy's fault. With this on, the collector keeps rolling
    # until n_rollouts_per_iter NON-TRUNCATED trajectories are closed.
    drop_truncated: bool = True
    # Per-step penalty per plane while sim's collision_warning flag is
    # True for that plane. Both planes in a pair get the penalty each
    # tick they're in warning. Calibrated below failure_reward so a
    # 30-tick brush doesn't dominate a -10 terminal:
    #   30 ticks × 0.10 = -3, well under |failure_reward|=10.
    # 100-tick sustained warning (-10) does — that's the point.
    collision_warning_penalty: float = 0.10
    # Sim-level crash: both colliding planes get terminal FAILURE_REWARD
    # (-10), all other in-flight planes get TRUNCATED with V(s_T)
    # bootstrap. Sim restarts fresh, collection continues.
    crash_extra_penalty: float = 0.0     # set >0 if you want crashed > regular fail

    # Cross-run warm start. If set, load delta_head + critic weights
    # from this ckpt at training start (fresh optimizer state). Used
    # to seed Phase 2 from a Phase 1 best.pt.
    init_radar_head_from: str = ''

    # ------------------------------------------------------------------ #
    # Delta-head architecture
    # ------------------------------------------------------------------ #
    delta_hidden: int = 64
    delta_hdg_clamp_deg: float = 30.0
    delta_alt_clamp_kft: float = 1.0
    delta_spd_clamp_kt: float = 30.0
    delta_log_sigma_init: float = -1.5    # σ ≈ 0.22 at init
    delta_log_sigma_min: float = -3.5
    delta_log_sigma_max: float = 0.0

    # ------------------------------------------------------------------ #
    # Critic architecture
    # ------------------------------------------------------------------ #
    value_hidden: int = 64
    value_dropout: float = 0.0

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    seed: int = 0
    device: str = 'cpu'
    run_dir: Path = Path('rl_multiple/runs/default')
    log_every: int = 1
    save_every: int = 10
    # Every N iters, dump a 6-pack (rollouts.csv, trajectories.npz,
    # eval_metrics.json, summary.json, raw.json) into
    # run_dir/iter_NNNN_eval/ alongside the iter ckpt. PNGs are
    # rendered locally after the run is pulled back (matplotlib not in
    # the Modal training image).
    eval_six_pack_every: int = 20

    # W&B
    wandb: bool = False
    wandb_project: str = 'atc-ppo-multi'
    wandb_run_name: str = ''
    wandb_group: str = 'ppo_multi'

    def max_steps_for(self, star: str) -> int:
        if star.endswith('3'):
            return self.max_timesteps_star_3
        return self.max_timesteps_star_1_2

    def resolve_n_workers(self) -> int:
        if self.n_workers > 0:
            return int(self.n_workers)
        import os
        cpu = os.cpu_count() or 2
        return max(1, cpu // 2)
