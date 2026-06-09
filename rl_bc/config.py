"""Central knobs for the BC training pipeline."""

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
HUMAN_DATA_DIR = REPO_ROOT / 'human_data'
CACHE_DIR = REPO_ROOT / 'rl_bc' / 'cache'

# The first 6 of the 10-dim feature vector are standardized.
N_CONT = 6
N_FEATURES = 10


@dataclass
class Config:
    # Data / runway
    airport_name: str = 'test'
    runway: str = '27'
    radar_side: int = 800
    nm_range: int = 60
    cache_path: Path = CACHE_DIR / 'bc_dataset.npz'

    # Optimization
    batch_size: int = 256
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    dropout: float = 0.1

    # Validation split
    final: bool = False
    final_val_fraction: float = 0.15
    n_folds: int = 5
    fold: int = 0
    seed: int = 0
    early_stop_patience: int = 10

    # IO
    run_dir: Path = REPO_ROOT / 'rl_bc' / 'runs'
    log_every_n_steps: int = 50
    num_workers: int = 0
    pin_memory: bool = True

    # Per-dim loss weights for the joint FM velocity-field MSE.
    hdg_loss_weight: float = 1.0
    alt_loss_weight: float = 1.0
    spd_loss_weight: float = 1.0

    # Joint encoder slice + width.
    input_indices: tuple = (0, 1, 2)
    hidden: int = 64

    # Flow matching (bc_fm only).
    fm_t_embed_dim: int = 16
    fm_n_steps: int = 10
    fm_noise_scale: float = 1.0

    # Gaussian mixture (bc_gmm only).
    n_components: int = 4
    # Batch-level load-balancing weight: penalizes the batch-mean mixture
    # for deviating from uniform via (log K - H(mean_pi)). This alone cannot
    # prevent *per-state* collapse — different rows can each be winner-take-all
    # while the batch-mean stays uniform.
    mixture_balance_weight: float = 0.0
    # Per-row mixture entropy floor: pushes the per-state entropy H(pi(x))
    # above a target so the mixture stays meaningfully multi-component at
    # each individual state, not just on average. Penalty is
    #   λ_row · ReLU(H_target - H_row).pow(2).mean()
    # H_target ≈ 0.5·log(K) keeps ≥2 components active per state without
    # overpowering NLL fit. 0 disables.
    row_entropy_weight: float = 0.0
    row_entropy_target: float = 0.0    # nats; e.g. 0.7 for K=4 (~half of log 4)

    # NOTE: σ floor and κ ceiling were once soft penalty weights here;
    # they're now structural bounds inside `rl_bc/bc_gmm/model.py` via
    # softplus reparameterization. No knobs to tune at the training level —
    # to change the bounds, edit `LOG_STD_FLOOR` / `LOG_KAPPA_CEILING` in
    # the model module.

    # Drop loc==1 rows from the hdg dims of the action-space loss.
    filter_loc_for_hdg: bool = True

    # ------------------------------------------------------------------ #
    # Imbalance-handling (focal NLL + heading dropout) — added to fight
    # the "do nothing" shortcut on the _full variant where ~85% of rows
    # have target ≈ current and the model learned to be silent.
    # ------------------------------------------------------------------ #
    # Focal loss: scales each row's NLL by `(hardness)^γ` where
    # `hardness = sigmoid((NLL - threshold) / temperature)`. Easy rows
    # (low NLL → model already fits them well) get crushed to ~0 weight;
    # hard rows (high NLL) keep ~full weight. γ=0 disables.
    focal_gamma: float = 0.0
    # Threshold for "easy" — when set to 0, uses the per-batch median
    # NLL (adaptive). Set a fixed value if you want a stable boundary.
    focal_threshold: float = 0.0
    # Sharpness of the easy/hard boundary in the sigmoid.
    focal_temperature: float = 1.0

    # Per-channel state dropout. With probability p (independent per
    # channel, per row), zero the listed feature column(s) BEFORE the
    # encoder sees them. 0 in standardized space = the channel's mean →
    # standard "missing data" trick. Used to break `target ≈ current`
    # shortcuts in the _full variant. Only fires in train mode.
    #   heading_dropout_prob → drops cols 6 AND 7 together (sin/cos pair)
    #   alt_dropout_prob     → drops col 4
    #   spd_dropout_prob     → drops col 5
    heading_dropout_prob: float = 0.0
    alt_dropout_prob: float = 0.0
    spd_dropout_prob: float = 0.0

    # Dataset repetition factor. When > 1, each training epoch iterates
    # the cached training rows N times via modular indexing — combined
    # with stochastic noise/dropout this yields N effectively-independent
    # noisy samples per row per epoch (cheap soft-DAgger). Validation
    # never repeats. 1 disables. Use this instead of bumping `epochs` so
    # the LR cosine schedule still completes one cycle over the run.
    dataset_repeat: int = 1

    # Input state noise (covariate-shift fix). At each training step, add
    # Gaussian noise to the current-heading / current-alt / current-spd
    # input features while keeping the targets unchanged. The model learns
    # that "small drift from the trained trajectory" still maps to the
    # original human target — a soft DAgger-style recovery behavior.
    # Noise sigmas are in PHYSICAL units (degrees, feet, knots); the
    # trainer converts them to the standardizer's units internally.
    # 0 disables. Suggested: 10° / 200 ft / 10 kt.
    input_noise_heading_deg: float = 0.0
    input_noise_alt_ft: float = 0.0
    input_noise_spd_kt: float = 0.0

    # Physical bounds applied AFTER the noise is added (and before
    # dropout). Stops the Gaussian tails from sending alt negative or
    # speeds out of the flyable envelope. Defaults match the sim's
    # realistic operational envelope. Set min >= max to disable.
    noise_clamp_alt_min_ft: float = 1000.0
    noise_clamp_alt_max_ft: float = 18000.0
    noise_clamp_spd_min_kt: float = 140.0
    noise_clamp_spd_max_kt: float = 280.0

    # Data-source ablation filter. When non-empty, training restricts to
    # cache rows whose `source` code is in this tuple (codes:
    # single=0, multi=1, distill=2, dagger=3 — see rl_bc/data.py). Applied
    # to BOTH the train and val index sets after the episode split, so the
    # model never sees the excluded sources. Empty () = use everything
    # (default; identical to the shipped recipe). Lets us reuse the single
    # shared cache for source ablations instead of building variant caches.
    train_sources: tuple = ()

    # Identifier + family — populated from configs/<name>.py NAME / FAMILY.
    name: str = 'bc_fm_single'
    family: str = 'bc_fm'
