"""bc_gmm_single_full — bc_gmm_single but with a full state input.

Same K=4 vMF×N×N mixture head as `bc_gmm_single`. The only change is the
encoder's input width: a 7-D state instead of 3-D position-only.

    INPUT_INDICES = (0, 1, 2, 4, 5, 6, 7)

  cache idx → meaning
    0  a_nm                    (standardized)
    1  c_nm                    (standardized)
    2  d_thr_nm                (standardized)
    4  current_alt / 1000      (standardized)
    5  (ias - 200) / 100       (standardized)
    6  sin(current_heading)    (raw)
    7  cos(current_heading)    (raw)

Index 3 (relative-heading delta) is intentionally skipped — same info as
the (sin, cos) pair with a discontinuity at ±180°.

Why the imbalance-handling knobs below are critical for `_full`:
==============================================================
The prior `_full` model achieved excellent val NLL (-4.94) but eval landed
~1/300 aircraft — model learned to parrot current heading because ~85% of
training rows have `target ≈ current` (steady-state autopilot). The
shortcut "target = current_heading" minimizes NLL on the bulk of data, so
NLL alone trains a silent policy. At deployment, silence → no heading
command → straight-line drift → out of airspace.

Two fixes layered together:

  INPUT_NOISE_* (heading 10°, alt 200 ft, spd 10 kt)
    On every step, perturb the current-state inputs with Gaussian noise
    in physical units while keeping the target unchanged. The model sees
    (slightly off-trajectory state → human's original target) pairs and
    learns a stable recovery action, instead of the `target = current`
    shortcut. Soft-DAgger style: the policy gets a wider attractor basin
    around the human trajectory so it can rejoin when it drifts.

  FOCAL_GAMMA = 2.0
    Focal-style reweighting of NLL: weight each row by (hardness)^γ where
    hardness = sigmoid((NLL - batch_median_NLL) / temp). Steady-state rows
    (model fits them well → low NLL) get weight ~0; active-control rows
    (high NLL) keep full weight. Concentrates the gradient on the
    decisions that matter.

Together: noise breaks the shortcut by making `current ≠ next target` for
most rows, and focal concentrates the gradient on the active-control
decisions. The model is forced to learn an actual control law.
"""

NAME = "bc_gmm_single_full"
FAMILY = "bc_gmm"

# Optimization
# DATASET_REPEAT=100 → each epoch iterates 100× the cached rows with
# fresh noise/dropout per access. EPOCHS=25 with cosine LR over the
# full 25×100 = 2500 effective epoch-equivalents; ~5× the prior recipe
# (10 × 50). BATCH_SIZE=1024 saturates the GPU for h=64.
EPOCHS = 25
BATCH_SIZE = 1024
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
INPUT_INDICES = (0, 1, 2, 4, 5, 6, 7)  # pos + alt + spd + sin/cos(hdg)
HIDDEN = 64

# Gaussian mixture — K=4 ("4 means" per the request).
N_COMPONENTS = 4

# Per-dim loss weights
HDG_LOSS_WEIGHT = 1.0
ALT_LOSS_WEIGHT = 1.0
SPD_LOSS_WEIGHT = 1.0

# Drop loc==1 rows' hdg dims from the NLL.
FILTER_LOC_FOR_HDG = True

# Mixture regularizers (same shape as bc_gmm_single defaults).
MIXTURE_BALANCE_WEIGHT = 0.05
ROW_ENTROPY_WEIGHT = 5.0
ROW_ENTROPY_TARGET = 0.7   # ≈ 0.5 · log(K=4)

# Imbalance fixes (the actual shortcut-killers for _full).
FOCAL_GAMMA = 2.0
FOCAL_TEMPERATURE = 1.0    # sigmoid sharpness on (NLL - thr) / temp

# Soft-DAgger via input noise. Targets stay unchanged; only the current
# heading/alt/spd channels get perturbed. Physical units → trainer
# converts alt/spd into standardizer space internally. All Gaussian
# (torch.randn) — symmetric around zero, unbounded tails are clipped
# by the NOISE_CLAMP_* bounds below to keep states in the ATC envelope.
#
# Sweet-spot recipe from the run_8 sweep: σ=15°/200ft/10kt hit 48.2% SR.
# Bumping to 20°/500/20 (run_11) regressed to 45.2% — over-aggressive,
# SOUTH STARs got worse and BEHIND_THR cases appeared. Keeping the run_8
# noise envelope but with the larger 200× repeat + 25 ep + clamps stack.
INPUT_NOISE_HEADING_DEG = 15.0
INPUT_NOISE_ALT_FT      = 200.0
INPUT_NOISE_SPD_KT      = 10.0

# Physical envelope clamps applied after noise. Keeps the Gaussian
# tails inside flyable bounds: σ_alt=500 over a 5000ft state can push
# alt to negative or above ceiling without these.
NOISE_CLAMP_ALT_MIN_FT = 1000.0
NOISE_CLAMP_ALT_MAX_FT = 18000.0
NOISE_CLAMP_SPD_MIN_KT = 140.0
NOISE_CLAMP_SPD_MAX_KT = 280.0

# Per-channel input dropout. Independently zeros each channel (heading
# both sin/cos together) with 20% probability per row. Combined with
# noise this forces the encoder to predict from any subset of the
# state — kills any single-channel `current ≈ target` shortcut.
HEADING_DROPOUT_PROB = 0.20
ALT_DROPOUT_PROB     = 0.20
SPD_DROPOUT_PROB     = 0.20

# Multiplicative repeat of train rows per epoch. 200× = 200 independent
# noisy samples per row per epoch. With EPOCHS=25 that's 5000 effective
# epoch-equivalents — more noise mass needed because both σ_hdg and
# σ_alt are now larger (more recovery space to cover).
DATASET_REPEAT = 200

WANDB_PROJECT = "atc-bc-gmm"
