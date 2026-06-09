"""bc_fm_single_full — bc_fm_single but with a full state input.

Identical to `bc_fm_single` except the encoder reads a 7-D state instead of
a 3-D position-only state:

    INPUT_INDICES = (0, 1, 2, 4, 5, 6, 7)

  cache idx → meaning
    0  a_nm                    (standardized)
    1  c_nm                    (standardized)
    2  d_thr_nm                (standardized)
    4  current_alt / 1000      (standardized)
    5  (ias - 200) / 100       (standardized)
    6  sin(current_heading)    (raw, already on the circle)
    7  cos(current_heading)    (raw, already on the circle)

Index 3 (relative-heading delta) is intentionally skipped — it carries the
same information as the (sin, cos) pair under a different parameterization
and has a discontinuity at ±180° that we don't want in the encoder input.

The (sin, cos) circular encoding matches the action-side convention so the
network can learn relations like "turn-to-target" as a consistent vector
operation everywhere on the circle.

Same family (`bc_fm`), same loss, same FM head — the only thing that
changes is the encoder's input width (3→7). All other code paths (model,
train, rollout, eval) auto-size off `INPUT_INDICES` and need no edits.
The `bc_fm_single` checkpoint and its inference path are unaffected.
"""

NAME = "bc_fm_single_full"
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
INPUT_INDICES = (0, 1, 2, 4, 5, 6, 7)  # pos + alt + spd + sin/cos(hdg)
HIDDEN = 64

# Flow matching
FM_T_EMBED_DIM = 16
FM_N_STEPS = 10
FM_NOISE_SCALE = 1.0

# Per-dim loss weights at the FM velocity-field MSE.
HDG_LOSS_WEIGHT = 1.0
ALT_LOSS_WEIGHT = 1.0
SPD_LOSS_WEIGHT = 1.0

# Drop loc==1 rows from the hdg dims of the joint FM loss.
FILTER_LOC_FOR_HDG = True

WANDB_PROJECT = "atc-bc-fm"
