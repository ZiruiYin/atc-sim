"""bc_fm_single — single-plane joint conditional flow-matching BC actor.

One model family (`bc_fm`), one variant for now (`single`). Future siblings
will be e.g. `bc_gmm_single` (Gaussian mixture), kept under their own
package directories with their own configs.

Architecture (`rl_bc/bc_fm/model.py:BCActor`):
  - One position-only encoder over `(a_nm, c_nm, d_thr_nm)`
  - One velocity field over ℝ⁴ → ℝ⁴ for joint (sin θ, cos θ, alt_std, spd_std)
  - Targets standardized per-dim on the training set; un-standardized at sample

LOC filter: per-dim. When loc==1, the hdg dims of the velocity-field loss
are masked out; alt and spd dims still contribute (LOC capture doesn't lock
altitude or speed).

Runtime (`rl_bc/bc_fm/rollout.py`): same command-emission contract as before
— `C XXX` for heading, `S NNN` for speed, direct-assignment for altitude.
"""

NAME = "bc_fm_single"
FAMILY = "bc_fm"

# Optimization
EPOCHS = 30
BATCH_SIZE = 256
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 10
SEED = 0

# Final-actor val split
FINAL_VAL_FRACTION = 0.15

# Network
DROPOUT = 0.1
INPUT_INDICES = (0, 1, 2)       # position-only (a_nm, c_nm, d_thr_nm)
HIDDEN = 64

# Flow matching
FM_T_EMBED_DIM = 16
FM_N_STEPS = 10
FM_NOISE_SCALE = 1.0

# Per-dim loss weights at the FM velocity-field MSE. Targets are standardized
# upstream so all dims are roughly unit-variance — keep equal unless you want
# to deliberately under-weight one channel.
HDG_LOSS_WEIGHT = 1.0
ALT_LOSS_WEIGHT = 1.0
SPD_LOSS_WEIGHT = 1.0

# Drop loc==1 rows from the hdg dims of the joint FM loss.
FILTER_LOC_FOR_HDG = True

WANDB_PROJECT = "atc-bc-fm"
