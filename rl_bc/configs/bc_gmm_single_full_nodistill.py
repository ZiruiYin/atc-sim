"""bc_gmm_single_full_nodistill — data ablation of the shipped 7-D GMM.

IDENTICAL recipe to `bc_gmm_single_full` (same 7-D state, same K=4 vMF×N×N
head, same soft-DAgger input noise / per-channel dropout / focal NLL /
200× repeat / 25 epochs). The ONLY difference is the training data pool:

    TRAIN_SOURCES = (single, multi, dagger)   ← distillation DROPPED

i.e. keep the human single-plane + multi-plane sessions AND the DAgger
corrections, but exclude the FM-distillation rollouts. Isolates "what did
distillation buy us?" against the shipped model (which trains on all four
sources). Source filtering happens at train time over the shared cache —
see `rl_bc/config.py::train_sources` and `rl_bc/bc_gmm/train.py`.
"""

NAME = "bc_gmm_single_full_nodistill"
FAMILY = "bc_gmm"

# Data-source ablation. Codes (rl_bc/data.py): single=0, multi=1, distill=2,
# dagger=3. Keep single + multi + dagger; drop distill.
TRAIN_SOURCES = (0, 1, 3)

# Optimization
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

# Gaussian mixture — K=4.
N_COMPONENTS = 4

# Per-dim loss weights
HDG_LOSS_WEIGHT = 1.0
ALT_LOSS_WEIGHT = 1.0
SPD_LOSS_WEIGHT = 1.0

# Drop loc==1 rows' hdg dims from the NLL.
FILTER_LOC_FOR_HDG = True

# Mixture regularizers (same as shipped).
MIXTURE_BALANCE_WEIGHT = 0.05
ROW_ENTROPY_WEIGHT = 5.0
ROW_ENTROPY_TARGET = 0.7   # ≈ 0.5 · log(K=4)

# Imbalance fixes (the actual shortcut-killers).
FOCAL_GAMMA = 2.0
FOCAL_TEMPERATURE = 1.0

# Soft-DAgger input noise (physical units).
INPUT_NOISE_HEADING_DEG = 15.0
INPUT_NOISE_ALT_FT      = 200.0
INPUT_NOISE_SPD_KT      = 10.0

# Physical envelope clamps applied after noise.
NOISE_CLAMP_ALT_MIN_FT = 1000.0
NOISE_CLAMP_ALT_MAX_FT = 18000.0
NOISE_CLAMP_SPD_MIN_KT = 140.0
NOISE_CLAMP_SPD_MAX_KT = 280.0

# Per-channel input dropout.
HEADING_DROPOUT_PROB = 0.20
ALT_DROPOUT_PROB     = 0.20
SPD_DROPOUT_PROB     = 0.20

# Multiplicative repeat of train rows per epoch.
DATASET_REPEAT = 200

WANDB_PROJECT = "atc-bc-gmm"
