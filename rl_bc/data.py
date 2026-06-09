"""CSV → cache → Dataset pipeline shared by v9 and v10.

- Reads CSVs from `human_data/{single_plane,multiple_planes,dagger,*_distillation}/`.
- Groups into episodes by `(source_file, callsign)`, splitting at terminal
  rows so a re-used callsign in one file becomes two episodes.
- Drops `IMPROPER_EXIT` and incomplete episodes (we train on landings only).
- Drops rows with `on_ground != ""`.
- Computes a 10-dim feature vector and the four absolute-target labels per
  surviving row. Caches everything to a single `.npz`.
- Fingerprints CSVs so the cache auto-rebuilds when `human_data/` changes.

Feature layout (10-dim):
    0 a_nm                          5 (ias - 200) / 100
    1 c_nm                          6 sin(current_heading)
    2 d_thr_nm                      7 cos(current_heading)
    3 (current_heading - course) / 180
    4 current_alt / 1000            8 loc flag (0/1)
                                    9 gs  flag (0/1)

v9 and v10 only feed cols (0, 1, 2) to their heads. Cols 3..9 are kept in
the cache for diagnostics (viz, probing) and so the masking flags loc/gs
can be read off each row at training time.

Labels:
    target_hdg_sin, target_hdg_cos     (sin, cos of target_heading)
    target_alt_kft                     (target_altitude / 1000)
    target_spd_norm                    ((target_airspeed - 200) / 100)
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from rl_bc.config import CACHE_DIR, HUMAN_DATA_DIR, N_CONT, N_FEATURES, Config


SOURCE_SINGLE = 0
SOURCE_MULTI = 1
SOURCE_DISTILL = 2
SOURCE_DAGGER = 3

SOURCE_LABEL = {SOURCE_SINGLE: 'single', SOURCE_MULTI: 'multi',
                SOURCE_DISTILL: 'distill', SOURCE_DAGGER: 'dagger'}
TAG_TO_SOURCE = {'single': SOURCE_SINGLE, 'multi': SOURCE_MULTI,
                 'distill': SOURCE_DISTILL, 'dagger': SOURCE_DAGGER}

# When this file is present we treat it as the SINGLE source of truth and
# skip the per-folder walk. The `_source` column carries the per-row source
# tag (written by `rl_bc.eval.prepare_training_data`).
PREPARED_COMBINED_REL = Path('prepared') / 'combined.csv'
SOURCE_COL = '_source'


# --------------------------------------------------------------------------- #
# CSV fingerprint (drives cache auto-rebuild + Modal volume re-sync)
# --------------------------------------------------------------------------- #


def data_fingerprint(human_dir: Path) -> str:
    """SHA-256 over every CSV's (relpath, size, mtime). When
    `prepared/combined.csv` exists, the fingerprint is computed over JUST
    that file so the local sync and remote cache-freshness check agree on
    what counts as 'the data' — the prepared CSV is the sole input."""
    human_dir = Path(human_dir)
    h = hashlib.sha256()
    if not human_dir.exists():
        return h.hexdigest()
    prepared = human_dir / PREPARED_COMBINED_REL
    if prepared.exists():
        stat = prepared.stat()
        h.update(str(prepared.relative_to(human_dir)).encode())
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
        return h.hexdigest()
    for path in sorted(human_dir.rglob('*.csv')):
        stat = path.stat()
        h.update(str(path.relative_to(human_dir)).encode())
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()


def _fingerprint_file(cache_path: Path) -> Path:
    return cache_path.with_suffix('.fingerprint')


def cache_is_fresh(cache_path: Path, human_dir: Path) -> bool:
    if not cache_path.exists():
        return False
    fp_file = _fingerprint_file(cache_path)
    if not fp_file.exists():
        return False
    return fp_file.read_text().strip() == data_fingerprint(human_dir)


# --------------------------------------------------------------------------- #
# Runway geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunwayGeometry:
    thr_x_nm: float        # threshold, airport-centered nm (east-positive)
    thr_y_nm: float        # north-positive
    course_deg: float      # landing course (0 = N, clockwise)


def load_runway_geometry(airport_name: str, runway: str,
                         radar_side: int = 800, nm_range: int = 60) -> RunwayGeometry:
    from environment.display.generate_game_coordinates import generate_game_coordinates

    data = generate_game_coordinates(radar_side, radar_side, nm_range, airport_name)
    nmpp = data['screen_info']['nm_per_pixel']
    ax = data['airport']['coordinates']['x']
    ay = data['airport']['coordinates']['y']

    thr_px = None
    other_px = None
    for pair_data in data['runways'].values():
        thresholds = pair_data['thresholds']
        if runway in thresholds:
            thr_px = thresholds[runway]
            for end_id, end_data in thresholds.items():
                if end_id != runway:
                    other_px = end_data
                    break
            break
    if thr_px is None:
        raise ValueError(f"runway {runway!r} not found in {airport_name!r} data")

    def px_to_nm(p):
        return (p['x'] - ax) * nmpp, -(p['y'] - ay) * nmpp

    thr_x_nm, thr_y_nm = px_to_nm(thr_px)
    other_x_nm, other_y_nm = px_to_nm(other_px) if other_px else (thr_x_nm, thr_y_nm)

    dx = other_x_nm - thr_x_nm
    dy = other_y_nm - thr_y_nm
    course = math.degrees(math.atan2(dx, dy)) % 360.0
    return RunwayGeometry(thr_x_nm=thr_x_nm, thr_y_nm=thr_y_nm, course_deg=course)


# --------------------------------------------------------------------------- #
# CSV → episodes
# --------------------------------------------------------------------------- #


def _parse_bool(s: str) -> int:
    s = s.strip().lower()
    return 1 if s in ('true', '1', 't', 'yes', 'y') else 0


def _iter_csv_files(human_dir: Path) -> Iterable[tuple[Path, int | None]]:
    """Walk CSVs feeding the cache. Two modes:

    - If `human_dir/prepared/combined.csv` exists, that single file is the
      sole input. Per-row source tagging comes from its `_source` column
      (yielded source_id is `None` to signal this to the caller).
    - Otherwise fall back to the per-folder walk of `single_plane/`,
      `multiple_planes/`, `dagger/`, `*_distillation/`.
    """
    prepared = human_dir / PREPARED_COMBINED_REL
    if prepared.exists():
        yield prepared, None
        return

    single = human_dir / 'single_plane'
    multi = human_dir / 'multiple_planes'
    dagger = human_dir / 'dagger'
    if single.exists():
        for p in sorted(single.glob('*.csv')):
            yield p, SOURCE_SINGLE
    if multi.exists():
        for p in sorted(multi.glob('*.csv')):
            yield p, SOURCE_MULTI
    if dagger.exists():
        for p in sorted(dagger.glob('*.csv')):
            yield p, SOURCE_DAGGER   # broken out from MULTI so the train banner
                                     # shows dagger row count separately
    # Distillation folders: any `human_data/*_distillation/` directory holds
    # CSVs replayed from a learned policy rolling out single-aircraft
    # scenarios (one STAR at a time). Tagged as their own source so the
    # train banner can report the human/dagger/distill split, and the
    # stratifier keeps distill episodes balanced across train/val folds.
    for sub in sorted(human_dir.glob('*_distillation')):
        if sub.is_dir():
            for p in sorted(sub.glob('*.csv')):
                yield p, SOURCE_DISTILL


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _split_into_episodes(rows: Sequence[dict]) -> list[list[dict]]:
    """Split a (file, callsign) row sequence into episodes — ends at any row
    with non-empty `terminal`.
    """
    episodes: list[list[dict]] = []
    cur: list[dict] = []
    for r in rows:
        cur.append(r)
        if r.get('terminal', '').strip():
            episodes.append(cur)
            cur = []
    if cur:
        episodes.append(cur)
    return episodes


def _episode_features_labels(episode: Sequence[dict], geom: RunwayGeometry) -> tuple:
    """Compute the 10-dim feature vector and four absolute-target labels per row.

    Returns:
        features         (T, 10) float32 (unstandardized; cols 0..5 standardized at __getitem__)
        target_hdg_sin   (T,)   float32
        target_hdg_cos   (T,)   float32
        target_alt_kft   (T,)   float32
        target_spd_norm  (T,)   float32
    """
    T = len(episode)
    features = np.zeros((T, N_FEATURES), dtype=np.float32)

    # Vectorized extraction.
    x_nm = np.array([float(r['x_nm']) for r in episode], dtype=np.float64)
    y_nm = np.array([float(r['y_nm']) for r in episode], dtype=np.float64)
    altitude = np.array([float(r['altitude']) for r in episode], dtype=np.float64)
    heading = np.array([float(r['heading']) for r in episode], dtype=np.float64)
    airspeed = np.array([float(r['airspeed']) for r in episode], dtype=np.float64)
    target_heading = np.array([float(r['target_heading']) for r in episode], dtype=np.float64)
    target_alt_ft = np.array([float(r['target_altitude']) for r in episode], dtype=np.float64)
    target_spd_kt = np.array([float(r['target_airspeed']) for r in episode], dtype=np.float64)
    loc = np.array([_parse_bool(r['loc']) for r in episode], dtype=np.int64)
    gs = np.array([_parse_bool(r['gs']) for r in episode], dtype=np.int64)

    # Position in runway-aligned frame.
    phi = math.radians((geom.course_deg + 180.0) % 360.0)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    dx = x_nm - geom.thr_x_nm
    dy = y_nm - geom.thr_y_nm
    a_nm = dx * sin_phi + dy * cos_phi
    c_nm = -dx * cos_phi + dy * sin_phi
    d_thr = np.sqrt(a_nm * a_nm + c_nm * c_nm)

    dtheta = ((heading - geom.course_deg + 540.0) % 360.0) - 180.0
    hdg_rad = np.radians(heading)

    features[:, 0] = a_nm.astype(np.float32)
    features[:, 1] = c_nm.astype(np.float32)
    features[:, 2] = d_thr.astype(np.float32)
    features[:, 3] = (dtheta / 180.0).astype(np.float32)
    features[:, 4] = (altitude / 1000.0).astype(np.float32)
    features[:, 5] = ((airspeed - 200.0) / 100.0).astype(np.float32)
    features[:, 6] = np.sin(hdg_rad).astype(np.float32)
    features[:, 7] = np.cos(hdg_rad).astype(np.float32)
    features[:, 8] = loc.astype(np.float32)
    features[:, 9] = gs.astype(np.float32)

    target_hdg_rad = np.radians(target_heading)
    return (
        features,
        np.sin(target_hdg_rad).astype(np.float32),
        np.cos(target_hdg_rad).astype(np.float32),
        (target_alt_ft / 1000.0).astype(np.float32),
        ((target_spd_kt - 200.0) / 100.0).astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# Cache builder
# --------------------------------------------------------------------------- #


def build_cache(cfg: Config, human_dir: Path | None = None,
                cache_path: Path | None = None, verbose: bool = True) -> Path:
    """Read every CSV under `human_dir`, build per-row features+labels for
    completed-landing episodes, and save to a single `.npz`.

    Stored arrays:
      - features         (N, 10) float32
      - target_hdg_sin   (N,)    float32
      - target_hdg_cos   (N,)    float32
      - target_alt_kft   (N,)    float32
      - target_spd_norm  (N,)    float32
      - episode_id       (N,)    int64
      - source           (N,)    int64
      - episode_lens     (n_eps,) int64
      - episode_sources  (n_eps,) int64
      - runway_thr_x, runway_thr_y, runway_course (scalars)
    """
    human_dir = human_dir or HUMAN_DATA_DIR
    cache_path = cache_path or cfg.cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    geom = load_runway_geometry(cfg.airport_name, cfg.runway,
                                cfg.radar_side, cfg.nm_range)
    if verbose:
        print(f"runway {cfg.runway} threshold @ ({geom.thr_x_nm:.3f}, {geom.thr_y_nm:.3f}) nm, "
              f"course {geom.course_deg:.1f}°")

    all_feats, all_t_hsin, all_t_hcos, all_t_akft, all_t_snorm = [], [], [], [], []
    all_epid, all_src = [], []
    episode_lens: list[int] = []
    episode_sources: list[int] = []

    n_files = 0
    dropped_improper = 0
    dropped_incomplete = 0
    dropped_onground = 0
    dropped_goaround = 0
    kept_episodes = 0
    kept_rows = 0
    next_episode_id = 0

    for csv_path, source_id in _iter_csv_files(human_dir):
        n_files += 1
        rows = _read_csv_rows(csv_path)
        # When source_id is None (single combined CSV), partition by
        # (callsign, _source) so two sources sharing a callsign stay separate.
        if source_id is None:
            by_key: dict[tuple, list[dict]] = {}
            for r in rows:
                tag = r.get(SOURCE_COL, '').strip()
                src = TAG_TO_SOURCE.get(tag, SOURCE_SINGLE)
                by_key.setdefault((r['callsign'], src), []).append(r)
            grouped = [(cs, src, ep) for (cs, src), ep in by_key.items()]
        else:
            by_cs: dict[str, list[dict]] = {}
            for r in rows:
                by_cs.setdefault(r['callsign'], []).append(r)
            grouped = [(cs, source_id, ep) for cs, ep in by_cs.items()]

        for cs, src, cs_rows in grouped:
            # DO NOT sort by sim_time: the distillation collector resets
            # sim_time to ~2.0 at the start of every rollout, so a sort
            # would interleave rows from different episodes for the same
            # callsign. The CSV is already in chronological per-rollout
            # order; rely on it.
            for ep_rows in _split_into_episodes(cs_rows):
                terminal = ep_rows[-1].get('terminal', '').strip()
                if terminal == 'IMPROPER_EXIT':
                    dropped_improper += 1; continue
                if terminal != 'LANDED':
                    dropped_incomplete += 1; continue
                # Drop the entire episode if the human ever issued an `A` (abort)
                # command — go-around trajectories take a different shape than
                # the steady downwind→base→final pattern and would pollute the
                # supervised signal.
                if any(r.get('cmd_abort', '').strip().upper() == 'Y' for r in ep_rows):
                    dropped_goaround += 1; continue
                filtered = [r for r in ep_rows if not r.get('on_ground', '').strip()]
                dropped_onground += len(ep_rows) - len(filtered)
                if len(filtered) < 2:
                    dropped_incomplete += 1; continue

                feats, t_hsin, t_hcos, t_akft, t_snorm = _episode_features_labels(
                    filtered, geom)
                T = feats.shape[0]
                all_feats.append(feats)
                all_t_hsin.append(t_hsin); all_t_hcos.append(t_hcos)
                all_t_akft.append(t_akft); all_t_snorm.append(t_snorm)
                all_epid.append(np.full(T, next_episode_id, dtype=np.int64))
                all_src.append(np.full(T, src, dtype=np.int64))
                episode_lens.append(T)
                episode_sources.append(src)
                next_episode_id += 1
                kept_episodes += 1
                kept_rows += T

    if not all_feats:
        raise RuntimeError(
            f"no usable episodes found under {human_dir}. "
            "Run the simulator with recording enabled to produce CSVs.")

    np.savez_compressed(
        cache_path,
        features=np.concatenate(all_feats, axis=0),
        target_hdg_sin=np.concatenate(all_t_hsin, axis=0),
        target_hdg_cos=np.concatenate(all_t_hcos, axis=0),
        target_alt_kft=np.concatenate(all_t_akft, axis=0),
        target_spd_norm=np.concatenate(all_t_snorm, axis=0),
        episode_id=np.concatenate(all_epid, axis=0),
        source=np.concatenate(all_src, axis=0),
        episode_lens=np.array(episode_lens, dtype=np.int64),
        episode_sources=np.array(episode_sources, dtype=np.int64),
        runway_thr_x=geom.thr_x_nm,
        runway_thr_y=geom.thr_y_nm,
        runway_course=geom.course_deg,
    )
    _fingerprint_file(cache_path).write_text(data_fingerprint(human_dir))

    if verbose:
        print(f"files scanned    : {n_files}")
        print(f"episodes kept    : {kept_episodes}")
        print(f"episodes dropped : {dropped_improper} improper-exit, "
              f"{dropped_incomplete} incomplete, "
              f"{dropped_goaround} go-around (cmd_abort)")
        print(f"rows kept        : {kept_rows}")
        print(f"on_ground rows dropped: {dropped_onground}")
        print(f"cache → {cache_path}")
    return cache_path


# --------------------------------------------------------------------------- #
# Cache loading + splits
# --------------------------------------------------------------------------- #


@dataclass
class CachedData:
    features: np.ndarray
    target_hdg_sin: np.ndarray
    target_hdg_cos: np.ndarray
    target_alt_kft: np.ndarray
    target_spd_norm: np.ndarray
    episode_id: np.ndarray
    source: np.ndarray
    episode_lens: np.ndarray
    episode_sources: np.ndarray


def load_cache(cache_path: Path) -> CachedData:
    z = np.load(cache_path)
    return CachedData(
        features=z['features'],
        target_hdg_sin=z['target_hdg_sin'],
        target_hdg_cos=z['target_hdg_cos'],
        target_alt_kft=z['target_alt_kft'],
        target_spd_norm=z['target_spd_norm'],
        episode_id=z['episode_id'],
        source=z['source'],
        episode_lens=z['episode_lens'],
        episode_sources=z['episode_sources'],
    )


def describe_data(cached: CachedData,
                  train_idx: 'np.ndarray | None' = None,
                  val_idx: 'np.ndarray | None' = None,
                  tag: str = '') -> str:
    """Build a multi-line banner of what's in the cache (and, if provided,
    in the train/val splits) broken down by source folder. Used by the
    family trainers so Modal/local logs make it obvious at run start how
    much human / dagger / distillation data the model is being trained on.
    """
    src_row = cached.source
    src_ep = cached.episode_sources
    src_codes = sorted(set(src_ep.tolist()) | set(src_row.tolist()))
    prefix = f"[{tag}] " if tag else ''
    lines = [f"{prefix}data composition (cache @ row + episode level):"]
    total_rows = int(src_row.size)
    total_eps = int(src_ep.size)
    for code in src_codes:
        label = SOURCE_LABEL.get(int(code), f'src={code}')
        n_rows = int((src_row == code).sum())
        n_eps = int((src_ep == code).sum())
        pct_rows = 100.0 * n_rows / max(1, total_rows)
        pct_eps = 100.0 * n_eps / max(1, total_eps)
        lines.append(
            f"{prefix}  {label:<8} rows={n_rows:>7} ({pct_rows:>4.1f}%)  "
            f"episodes={n_eps:>4} ({pct_eps:>4.1f}%)"
        )
    lines.append(f"{prefix}  {'TOTAL':<8} rows={total_rows:>7}          "
                 f"episodes={total_eps:>4}")
    if train_idx is not None and val_idx is not None:
        lines.append(f"{prefix}split: train rows={len(train_idx)}  "
                     f"val rows={len(val_idx)}  "
                     f"(val_fraction by row ≈ "
                     f"{100.0 * len(val_idx) / max(1, len(train_idx) + len(val_idx)):.1f}%)")
        lines.append(f"{prefix}  train by source:")
        for code in src_codes:
            label = SOURCE_LABEL.get(int(code), f'src={code}')
            n = int((src_row[train_idx] == code).sum())
            pct = 100.0 * n / max(1, len(train_idx))
            lines.append(f"{prefix}    {label:<8} rows={n:>7} ({pct:>4.1f}%)")
    return '\n'.join(lines)


def final_split(cached: CachedData, val_fraction: float = 0.05, seed: int = 0
                ) -> tuple[np.ndarray, np.ndarray]:
    """Episode-level split for `--final` training."""
    n_eps = cached.episode_lens.shape[0]
    if n_eps < 2:
        raise ValueError(f"need ≥2 episodes for a final split (got {n_eps})")
    rng = np.random.default_rng(seed)
    eps = np.arange(n_eps); rng.shuffle(eps)
    n_val = max(1, int(round(n_eps * val_fraction)))
    val_eps = eps[:n_val]
    val_mask = np.isin(cached.episode_id, val_eps)
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def kfold_episode_splits(cached: CachedData, n_folds: int = 5, seed: int = 0
                         ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Episode-level stratified k-fold (length quartile × source)."""
    n_eps = cached.episode_lens.shape[0]
    if n_eps < n_folds:
        raise ValueError(f"need ≥{n_folds} episodes for {n_folds}-fold (got {n_eps})")
    lens = cached.episode_lens
    srcs = cached.episode_sources
    order = np.argsort(lens)
    quartile = np.empty(n_eps, dtype=np.int64)
    for rank, eid in enumerate(order):
        quartile[eid] = min(3, rank * 4 // n_eps)
    strata = quartile * 2 + srcs.astype(np.int64)
    rng = np.random.default_rng(seed)
    fold_of_episode = np.full(n_eps, -1, dtype=np.int64)
    for s in np.unique(strata):
        members = np.where(strata == s)[0]
        rng.shuffle(members)
        for i, eid in enumerate(members):
            fold_of_episode[eid] = i % n_folds
    splits = []
    for k in range(n_folds):
        val_eps = np.where(fold_of_episode == k)[0]
        val_mask = np.isin(cached.episode_id, val_eps)
        splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    return splits


# --------------------------------------------------------------------------- #
# Standardizer + Dataset
# --------------------------------------------------------------------------- #


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, train_idx: np.ndarray) -> 'Standardizer':
        cont = features[train_idx, :N_CONT]
        mean = cont.mean(axis=0)
        std = cont.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        out[..., :N_CONT] = (out[..., :N_CONT] - self.mean) / self.std
        return out


class BCDataset(Dataset):
    """Returns the standardized 10-dim feature row + the four target labels +
    the loc/gs flags (used for the v9/v10 hdg-loss mask).
    """

    def __init__(self, cached: CachedData, row_idx: np.ndarray,
                 standardizer: Standardizer):
        self.cached = cached
        self.row_idx = row_idx.astype(np.int64)
        self.standardizer = standardizer

    def __len__(self) -> int:
        return self.row_idx.shape[0]

    def __getitem__(self, i):
        r = self.row_idx[i]
        x = self.cached.features[r].copy()
        x[:N_CONT] = (x[:N_CONT] - self.standardizer.mean) / self.standardizer.std
        return {
            'x': torch.from_numpy(x),
            'target_hdg_sincos': torch.tensor(
                [self.cached.target_hdg_sin[r], self.cached.target_hdg_cos[r]],
                dtype=torch.float32),
            'target_alt_kft':  torch.tensor(self.cached.target_alt_kft[r], dtype=torch.float32),
            'target_spd_norm': torch.tensor(self.cached.target_spd_norm[r], dtype=torch.float32),
            'loc': torch.tensor(self.cached.features[r, 8], dtype=torch.float32),
            'gs':  torch.tensor(self.cached.features[r, 9], dtype=torch.float32),
        }
