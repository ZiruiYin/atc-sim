"""PPO hyperparameters — single source of truth for the trainer.

Every knob lives here so a run is fully described by one `PPOConfig`.
Defaults are tuned for the run_11 BC seed (bc_gmm_single_full, the 7-D
soft-DAgger model trained with σ_hdg=20° / σ_alt=500ft / σ_spd=20kt
noise + per-channel dropout + 200× dataset repeat + clamps) and the
ATC sim's 1-Hz sparse-terminal reward setup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


STARS_ALL = ('NORTH1', 'NORTH2', 'NORTH3', 'SOUTH1', 'SOUTH2', 'SOUTH3')


@dataclass
class PPOConfig:
    # ------------------------------------------------------------------ #
    # BC actor seed (the GMM checkpoint PPO starts from)
    # ------------------------------------------------------------------ #
    actor_ckpt: Path = Path('rl_bc/runs/bc_gmm_single_full/run_11/best.pt')

    # ------------------------------------------------------------------ #
    # Environment / episode setup
    # ------------------------------------------------------------------ #
    airport_name: str = 'test'
    runway: str = '27'
    stars: tuple = STARS_ALL
    warmup_wpts: int = 2

    # Per-STAR episode timestep cap. PPO control begins AFTER the STAR
    # warm-up (i.e. after `warmup_wpts` waypoints are popped); the cap
    # counts only the PPO-controlled steps. Sized to allow LOC + GS both
    # captured (the sim's full ILS-capture condition).
    #   NORTH1 / NORTH2 / SOUTH1 / SOUTH2  →  max_timesteps_star_1_2
    #   NORTH3 / SOUTH3                    →  max_timesteps_star_3
    max_timesteps_star_1_2: int = 1200
    max_timesteps_star_3: int = 500

    # ------------------------------------------------------------------ #
    # Reward (purely terminal — no shaping)
    # ------------------------------------------------------------------ #
    #   success: the sim's BOTH `loc_intercepted` AND `gs_intercepted`
    #            flags are True (i.e. the aircraft has fully captured ILS:
    #            on the localizer AND on the glideslope).
    #   failure: timeout / improper exit / sim error  OR
    #            LOC captured ABOVE the GS — sim blocks re-vectoring once
    #            on LOC and only captures GS from below, so the approach
    #            is doomed; we fail-fast on this one-shot check at the
    #            moment LOC fires.
    # Terminal magnitudes — frozen at ±10 (env reads from `reward_zones`).
    # Kept here as PPOConfig fields purely so they land in `config.json`
    # alongside the tunable shaping knobs below. Do not edit per run.
    success_reward: float = 10.0
    failure_reward: float = -10.0
    # Buffer (ft) added to the LOC-above-GS fail-fast check. Matches the
    # sim's own GS-capture window (`abs(altitude - projected_alt) <= 50`
    # in environment/core/aircraft.py::_update_ils_gs). Don't change unless
    # the sim's window changes.
    gs_capture_buffer_ft: float = 50.0

    # ------------------------------------------------------------------ #
    # Tunable shaping (these are the ones I edit per training block).
    # The env reads these into `reward_zones` at train start via
    # `reward_zones.set_runtime_overrides()`, so every run's `config.json`
    # is self-describing — no hidden module constants to chase.
    # ------------------------------------------------------------------ #

    # Per-step penalty applied EVERY step regardless of zone. On/off-like
    # ladder per the continuous-run plan: {0.0, 0.001, 0.002}.
    #   0.0    → no time pressure; policy free to linger / loop in green
    #   0.001  → gentle; default for block 1
    #   0.002  → strong; turn on when trajectories are too long
    everywhere_step_penalty: float = 0.001

    # Cap on the out-of-zone step penalty (per step, in nm-weighted units).
    # Higher = stronger push to stay in the L/triangle green zone.
    # Ladder: {0.005, 0.010, 0.020, 0.040, 0.050}.
    #   0.005  → permissive (BC seed level); macro green% tends to 40-50%
    #   0.010  → moderate; expect macro green% ~60-70%
    #   0.020  → aggressive; for when corner-cutting is dominant
    #   0.040+ → very aggressive; teach in-zone behavior early in c02
    step_penalty_cap: float = 0.005

    # Slope of the out-of-zone step penalty (per nm out of zone, per step).
    # Linear regime up to step_penalty_cap; the cap fires at distance
    # = cap / slope. In the typical 1-3 nm out regime, the SLOPE is the
    # binding lever — the cap rarely matters. Ladder: {0.0005, 0.001,
    # 0.002, 0.005}. Default 0.0005 matches the historical constant.
    step_penalty_per_nm: float = 0.0005

    # Early-window zone-penalty multiplier. For the first
    # `early_window_steps` POLICY-controlled steps of each episode, the
    # out-of-zone zone penalty is multiplied by `early_zone_multiplier`.
    # Targets the BC failure mode where policy drifts off-zone on the
    # triangle→downwind merge (NORTH2/SOUTH2 first-300-policy-step in-zone
    # fraction is only 53-62% in BC, vs 98% for NORTH1/SOUTH1).
    # Ladder: multiplier {1.0, 3.0, 5.0, 10.0}; window {100, 200, 300, 500}.
    early_zone_multiplier: float = 1.0     # 1.0 = disabled
    early_window_steps: int = 300

    # FLAT per-out-of-zone-step penalty during early window. Independent
    # of distance. Designed to make BC's drift-off modes (NORTH2/SOUTH2)
    # net to "half failure" reward instead of full +10 — strong PPO
    # gradient toward in-zone GMM modes.
    # Ladder: {0, 0.02, 0.05, 0.10}. STAR-3 family is exempt.
    # 0.05 default rationale: BC NORTH2 ~37% out in 300 steps →
    #   0.37*300*0.05 = 5.55 penalty → net reward +4.5 ("half failure").
    # BC NORTH1 ~1.7% out → 0.26 penalty → net +9.7 (barely touched).
    early_drift_penalty: float = 0.0   # 0 = disabled (backward compat)

    # CLEAN_TERMINAL_THRESHOLD: For STAR-1/2 family, a LOC_BELOW_GS
    # success only gets the full +SUCCESS_REWARD terminal if the
    # fraction-in-zone during policy control >= this threshold.
    # Otherwise terminal becomes drifty_success_value (default 0).
    # Step penalties always accumulate, so drifty "successes" net
    # strictly negative. STAR-3 family is exempt.
    # Gate condition: `frac_in_zone < clean_terminal_threshold`.
    #   threshold = 0.0  → never fires, mechanism DISABLED (default).
    #   threshold = 0.90 → require 90%+ in zone for full +10 terminal.
    #   threshold = 1.0  → require 100% (very strict).
    # Ladder: {0.0 (disabled), 0.7, 0.8, 0.9, 0.95, 1.0}.
    clean_terminal_threshold: float = 0.0    # 0.0 = disabled
    drifty_success_value: float = 0.0        # what to use instead of +10

    # OUT_OF_ZONE termination (c03 mechanism). When enabled, episode
    # terminates as FAILURE (RZ_FAILURE_REWARD = -10) once a STAR-1/2
    # aircraft has been out of zone for `out_of_zone_max_consecutive`
    # policy-controlled steps in a row. STAR-3 family is exempt.
    # Typically used WITH `step_pen_per_nm=0, step_pen_cap=0` — the
    # termination IS the penalty.
    # Ladder: out_of_zone_max_consecutive {1 (zero tol), 3, 5, 10, 20}.
    out_of_zone_terminate: bool = False
    out_of_zone_max_consecutive: int = 5

    # Per-STAR multiplicative success-reward scaling. Amplifies
    # terminal reward on STARs with low recent SR — fights "STAR
    # trading" mode collapse where the policy over-optimizes one
    # STAR at the expense of another.
    # 0 = disabled. 0.5 = mild (max +5 bonus for 0%-SR STAR).
    per_star_sr_scale: float = 0.0

    # Always-on base shaping toggles — default True (design baseline);
    # flip to False ONLY for reward-ablation runs. Forwarded to rollout
    # workers like the other reward knobs.
    heading_intercept_enabled: bool = True
    turn_final_enabled: bool = True

    # ------------------------------------------------------------------ #
    # Post-hoc loop penalty (applied at episode end, retroactively to
    # the *second half* of looping timesteps as identified by
    # `rl_ppo.loop_detector.detect_looping`).
    #
    # The first half of any detected loop pays nothing — gives the policy
    # a "warning shot" and avoids over-penalizing brief recoveries. The
    # second half pays `loop_penalty_per_step` per timestep, so a long
    # persistent loop is much more expensive than a short transient one.
    #
    # The three detector hyperparams below default to the same values
    # `loop_detector.py` uses for analysis, so the reward signal matches
    # what we see in the `eval/looping_analysis/` plots.
    # ------------------------------------------------------------------ #

    # Per-step penalty applied to second-half loop timesteps. 0 disables.
    # Ladder: {0, 0.02, 0.05, 0.10}.
    #   0.02 → a 100-loop-step trajectory pays ~−1 (mild nudge)
    #   0.05 → same trajectory pays ~−2.5 (significant vs terminal +10)
    #   0.10 → same trajectory pays ~−5 (terminal-magnitude penalty)
    # We don't know the right scale yet — pick from the ladder per block.
    loop_penalty_per_step: float = 0.0

    # Loop-detector hyperparameters (mirror rl_ppo.loop_detector defaults).
    loop_prox_radius_nm: float = 0.75
    loop_min_gap_steps: int = 45
    loop_min_detour_nm: float = 1.0

    # ------------------------------------------------------------------ #
    # GAE / discount
    # ------------------------------------------------------------------ #
    # γ=0.999 (restored from 0.995). Run_6 (γ=0.995 + −0.005/step
    # penalty) over-pressured the policy into terminating early,
    # collapsing NORTH1 from 88% → 52%. With the gentler 0.002/step
    # penalty for attempt 3, restoring γ=0.999 preserves longer-horizon
    # credit propagation so the policy can value a full proper downwind
    # setup over an early premature turn.
    gamma: float = 0.999
    gae_lambda: float = 0.95

    # ------------------------------------------------------------------ #
    # PPO update
    # ------------------------------------------------------------------ #
    clip_epsilon: float = 0.2
    n_epochs: int = 4                 # passes over each rollout batch
    batch_size: int = 256
    target_kl: float | None = 0.02    # restored to standard — the v2-v4
                                      # tightening was chasing a dropout
                                      # bug, not real policy drift.
    entropy_coef: float = 0.02        # bumped from 0.005 — run_5 collapsed
                                      # mixture entropy 0.79 → 0.22; need
                                      # more entropy bonus to preserve
                                      # mixture diversity during training.
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #
    # 3× lower than the textbook PPO defaults (3e-4 / 1e-3). The BC seed
    # is already a competent policy, so we're fine-tuning rather than
    # learning from scratch — smaller steps preserve what BC learned while
    # still leaving room for PPO to improve. Also gives KL a wider safety
    # margin against the sparse-reward advantage spikes that derailed our
    # first (easier-task) PPO run.
    # 1e-7 (was 1e-5) — the run_11 GMM has high κ (~45) and σ at the
    # structural floor, so log_prob is steep around the mean and even a
    # single SGD step at 1e-5 produces KL ~ 2.0 (100× the old 3-D model
    # at the same LR). 1e-7 brings per-batch KL into the 0.005 target
    # range. lr_critic dropped proportionally for stable value-function
    # learning at the lower reward scale.
    lr_actor: float = 1e-5           # standard BC-finetune LR (the v3-v4
                                      # 100× reductions were chasing the
                                      # dropout-mode bug, not policy drift)
    lr_critic: float = 3e-5
    weight_decay: float = 0.0

    # ------------------------------------------------------------------ #
    # Rollout collection
    # ------------------------------------------------------------------ #
    # More rollouts → lower per-STAR variance in the success-rate / advantage
    # estimates → more stable updates. 128 = 21–22 per STAR × 6 STARs gives
    # per-STAR success-rate stderr of ~0.11, a reasonable balance.
    n_rollouts_per_iter: int = 128
    # 0 = autodetect from `os.cpu_count() // 2`. The // 2 skips SMT siblings:
    # on hyperthreaded CPUs (AMD EPYC on Modal, Intel laptops) running one
    # worker per vCPU oversubscribes physical cores, and the sim+policy
    # forward is L2-bound enough that two SMT threads on the same physical
    # core actively contend rather than overlap. cpu_count // 2 ≈ physical
    # core count. Explicit override is honored if non-zero.
    n_workers: int = 0
    n_iters: int = 200

    # ------------------------------------------------------------------ #
    # Value network architecture
    # ------------------------------------------------------------------ #
    value_hidden: int = 64
    value_dropout: float = 0.0

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    seed: int = 0
    device: str = 'cpu'
    run_dir: Path = Path('rl_ppo/runs/default')
    log_every: int = 1
    save_every: int = 10
    # In-run periodic eval (for single straight `--n-iters` runs, so you
    # get the per-K-iter eval curve WITHOUT the per-block modal overhead
    # of the continuous driver). 0 = no in-run eval. When > 0, every
    # `eval_every` iters the just-saved ckpt is evaluated on
    # `eval_cases`×6 STARs into `iter_NNNN_eval/` (same artifacts the
    # continuous driver produces). NOTE: this does NOT refresh
    # `per_star_recent_sr` mid-run — `per_star_sr_scale` stays frozen at
    # its start-of-run value. For a faithful adaptive c03-style run use
    # `modal_continuous.py`; for clean ablations set `per_star_sr_scale=0`.
    eval_every: int = 0
    eval_cases: int = 200

    # ------------------------------------------------------------------ #
    # W&B
    # ------------------------------------------------------------------ #
    wandb: bool = False                # disabled by default for local runs
    wandb_project: str = 'atc-ppo'
    wandb_run_name: str = ''           # '' → train.py derives from run_dir
    wandb_group: str = 'ppo'

    def max_steps_for(self, star: str) -> int:
        """Return the PPO-control timestep cap for `star`."""
        if star.endswith('3'):
            return self.max_timesteps_star_3
        return self.max_timesteps_star_1_2

    def resolve_n_workers(self) -> int:
        """Return the actual rollout-worker count, autodetecting if 0.

        Defaults to `max(1, cpu_count // 2)` to skip SMT siblings — running
        one worker per vCPU oversubscribes physical cores, since the sim
        + policy forward are L2-bound and don't overlap well across SMT.
        """
        if self.n_workers > 0:
            return int(self.n_workers)
        import os
        cpu = os.cpu_count() or 2
        return max(1, cpu // 2)
