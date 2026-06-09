"""bc_gmm_single — single-plane joint Gaussian-mixture BC actor.

Same encoder shape as bc_fm_single, but the head outputs the parameters of
a K-component Gaussian mixture over the 4-D action space
`(sin θ, cos θ, alt_std, spd_std)` with diagonal covariance. Training is
NLL (maximize log-likelihood of the demonstrator action under the GMM).
At rollout time we sample a component, then a Gaussian conditional on it;
both steps reuse a per-aircraft seeded torch.Generator so each plane
commits to a stable mode tick-to-tick.

Why GMM over flow matching: GMM gives an explicit density function, so
later PPO rollouts can compute `log_prob(action | state)` cheaply for the
importance-sampling ratio. Flow matching only gives samples.
"""

NAME = "bc_gmm_single"
FAMILY = "bc_gmm"

# Optimization
EPOCHS = 30
BATCH_SIZE = 256
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 10
SEED = 0

# Val split
FINAL_VAL_FRACTION = 0.15

# Network
DROPOUT = 0.1
INPUT_INDICES = (0, 1, 2)       # position-only (a_nm, c_nm, d_thr_nm)
HIDDEN = 64

# Gaussian mixture
N_COMPONENTS = 4                # K mixture components, diagonal covariance

# Per-dim loss weights — applied as a per-dim NLL multiplier (targets are
# standardized so equal weights are reasonable defaults).
HDG_LOSS_WEIGHT = 1.0
ALT_LOSS_WEIGHT = 1.0
SPD_LOSS_WEIGHT = 1.0

# Drop loc==1 rows' hdg dims from the NLL (marginal log-likelihood over
# alt + spd dims only on those rows).
FILTER_LOC_FOR_HDG = True

# Batch-level load-balancing weight — only constrains the batch-mean
# mixture distribution, NOT per-row sharpness. Kept low; the per-row floor
# below does the real work of preventing collapse.
MIXTURE_BALANCE_WEIGHT = 0.05

# Per-row mixture entropy floor: penalize states where the mixture
# distribution is too sharp. H_target ≈ 0.7 (about half of log 4 = 1.386)
# means we want ≥2 components carrying non-trivial mass at each state.
#
# Calibration note (May 2026): run_8 with weight=0.5 produced mean_max_pi
# = 0.986, mean entropy = 0.034 — the penalty was overwhelmed by NLL's
# gradient on the dominant logit. Bumped 10× to 5.0.
ROW_ENTROPY_WEIGHT = 5.0
ROW_ENTROPY_TARGET = 0.7

# σ floor and κ ceiling were here as soft-penalty knobs through run_9.
# Run_9 showed soft penalties are gameable — the optimizer complied on
# unused components and parked the dominant one at σ→0, κ→clamp. The
# bounds are now STRUCTURAL via softplus reparameterization inside
# `rl_bc/bc_gmm/model.py` (`LOG_STD_FLOOR`, `LOG_KAPPA_CEILING`) and
# cannot be violated. No knobs at the training level.

WANDB_PROJECT = "atc-bc-gmm"
