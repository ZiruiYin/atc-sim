"""Local preprocessing for BC training: produces ONE consolidated CSV that
Modal trains on, so the remote side only has to shuffle/split/train.

Pipeline (all local):

  1. Walk every raw human CSV under `human_data/{single_plane, multiple_planes,
     dagger}/`. Each row is tagged with its source folder.
  2. Walk every distillation CSV under `human_data/*_distillation/`. Per-STAR,
     pick a random subset of `--distill-fraction` of trajectories so the
     near-deterministic distill rollouts don't dominate the dataset. Sampling
     is independent per STAR, so each STAR contributes its own share.
  3. Concatenate all surviving rows into `human_data/prepared/combined.csv`,
     adding a `_source` column ('single' | 'multi' | 'dagger' | 'distill') so
     the Modal-side cache builder can preserve per-source statistics.
  4. Save a per-STAR trajectory plot of the sampled distillation to
     `data_viz/distill_sampled_<ts>.png` so it's easy to eyeball coverage.

Run before `modal run rl_bc/modal_train.py ...`. The modal launcher will
pick up `human_data/prepared/combined.csv` instead of the raw CSV tree, so
the volume sync is one file rather than thousands.

Usage:
    python -m rl_bc.eval.prepare_training_data
    python -m rl_bc.eval.prepare_training_data --distill-fraction 0.5 --seed 1
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


STARS = ['NORTH1', 'NORTH2', 'NORTH3', 'SOUTH1', 'SOUTH2', 'SOUTH3']

# Each source directory pattern → tag written into the `_source` column.
HUMAN_DIRS = [('single_plane', 'single'),
              ('multiple_planes', 'multi'),
              ('dagger', 'dagger')]
DISTILL_DIR_SUFFIX = '_distillation'
DISTILL_TAG = 'distill'

SOURCE_COL = '_source'


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return header, rows


def _group_by_callsign(rows: list[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r['callsign']].append(r)
    return by


def _split_episodes_by_terminal(rows: list[dict]) -> list[list[dict]]:
    """A single callsign can carry multiple landings in one CSV (the
    distillation collector reuses random callsigns across seeds, and
    multi-plane sessions retire/respawn callsigns). Episodes end at any
    row whose `terminal` field is non-empty.

    IMPORTANT: rows are iterated in FILE order — DO NOT sort by sim_time.
    The distillation collector resets sim_time to ~2.0 at the start of every
    rollout, so sorting would interleave rows from different episodes."""
    out: list[list[dict]] = []
    cur: list[dict] = []
    for r in rows:
        cur.append(r)
        if r.get('terminal', '').strip():
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _star_of_episode(ep: list[dict]) -> str | None:
    for r in ep:
        s = r.get('star', '')
        if s:
            return s
    return None


# --------------------------------------------------------------------------- #
# Distillation: per-STAR random subset
# --------------------------------------------------------------------------- #


def sample_distill_csv(path: Path, fraction: float,
                       rng: np.random.Generator):
    """Per-STAR random episode sampler.

    Returns
        kept_episodes        list of (episode_id, rows) for the sampled subset
        all_episodes_by_star {star: [episode_id, ...]} for every episode in the file
        chosen_by_star       {star: [episode_id, ...]} for sampled episodes only

    Episode IDs are tuples `(file_path_str, callsign, ep_idx)` where `ep_idx`
    counts terminal-bounded chunks within the callsign. The distillation
    collector reuses callsigns across STARs; splitting by terminal restores
    the true per-episode unit.
    """
    _, rows = _read_csv_rows(path)
    by_cs = _group_by_callsign(rows)
    sp = str(path)

    # Enumerate every episode.
    episodes: dict[tuple, list[dict]] = {}
    by_star_all: dict[str, list[tuple]] = defaultdict(list)
    for cs, cs_rows in by_cs.items():
        for ep_idx, ep in enumerate(_split_episodes_by_terminal(cs_rows)):
            star = _star_of_episode(ep)
            if star is None:
                continue
            ident = (sp, cs, ep_idx)
            episodes[ident] = ep
            by_star_all[star].append(ident)

    chosen_by_star: dict[str, list[tuple]] = {}
    kept: list[tuple] = []
    for star, group in by_star_all.items():
        if not group:
            continue
        k = max(1, int(round(len(group) * fraction)))
        k = min(k, len(group))
        # Deterministic ordering before sampling.
        group_sorted = sorted(group, key=lambda t: (t[1], t[2]))
        idx = rng.choice(len(group_sorted), size=k, replace=False)
        chosen = [group_sorted[i] for i in idx]
        chosen_by_star[star] = chosen
        kept.extend(chosen)
    # Return episodes themselves so the writer can iterate without re-grouping.
    kept_pairs = [(ident, episodes[ident]) for ident in kept]
    return kept_pairs, episodes, by_star_all, chosen_by_star


# --------------------------------------------------------------------------- #
# Driver: build one combined CSV from human + sampled distill
# --------------------------------------------------------------------------- #


def build_combined_csv(human_root: Path, out_csv: Path,
                       distill_fraction: float, rng: np.random.Generator,
                       verbose: bool = True) -> dict:
    """Writes `out_csv` and returns a summary dict (also used by the viz)."""
    all_headers: list[str] = []

    def merge_header(h):
        for col in h:
            if col not in all_headers:
                all_headers.append(col)

    summary: dict = {
        'sources': {tag: {'files': 0, 'episodes': 0, 'rows': 0}
                    for _, tag in HUMAN_DIRS},
        'distill': {'files': 0, 'episodes_total': 0, 'episodes_kept': 0,
                    'rows': 0, 'per_star_total': {}, 'per_star_kept': {}},
    }
    out_rows: list[dict] = []

    # --- HUMAN sources ---
    for sub, tag in HUMAN_DIRS:
        d = human_root / sub
        if not d.exists():
            continue
        for path in sorted(d.glob('*.csv')):
            header, rows = _read_csv_rows(path)
            merge_header(header)
            if not rows:
                continue
            summary['sources'][tag]['files'] += 1
            seen = set()
            for r in rows:
                r[SOURCE_COL] = tag
                out_rows.append(r)
                seen.add(r.get('callsign', ''))
            summary['sources'][tag]['episodes'] += len(seen)
            summary['sources'][tag]['rows'] += len(rows)
            if verbose:
                print(f"  human/{tag}: {path.name} "
                      f"({len(rows):,} rows, {len(seen)} callsigns)")

    # --- DISTILL sources (per-STAR random EPISODE sample) ---
    plot_payload: list[tuple] = []
    distill_dirs = sorted(p for p in human_root.glob(f'*{DISTILL_DIR_SUFFIX}')
                          if p.is_dir())
    for d in distill_dirs:
        for path in sorted(d.glob('*.csv')):
            kept_pairs, all_eps, by_star_all, chosen_by_star = sample_distill_csv(
                path, fraction=distill_fraction, rng=rng)
            header, _ = _read_csv_rows(path)
            merge_header(header)
            summary['distill']['files'] += 1
            for star, idents in by_star_all.items():
                summary['distill']['per_star_total'][star] = (
                    summary['distill']['per_star_total'].get(star, 0)
                    + len(idents))
            kept_n_rows = 0
            for ident, ep in kept_pairs:
                star = _star_of_episode(ep)
                summary['distill']['per_star_kept'][star] = (
                    summary['distill']['per_star_kept'].get(star, 0) + 1)
                for r in ep:
                    r[SOURCE_COL] = DISTILL_TAG
                    out_rows.append(r)
                    kept_n_rows += 1
            n_total_eps = sum(len(v) for v in by_star_all.values())
            summary['distill']['episodes_total'] += n_total_eps
            summary['distill']['episodes_kept'] += len(kept_pairs)
            summary['distill']['rows'] += kept_n_rows
            plot_payload.append((path, all_eps, by_star_all, chosen_by_star))
            if verbose:
                print(f"  distill: {path.name} "
                      f"({n_total_eps} → {len(kept_pairs)} episodes, "
                      f"{kept_n_rows:,} rows kept)")

    if SOURCE_COL in all_headers:
        all_headers.remove(SOURCE_COL)
    all_headers.append(SOURCE_COL)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=all_headers, extrasaction='ignore')
        w.writeheader()
        for r in out_rows:
            for col in all_headers:
                r.setdefault(col, '')
            w.writerow(r)
    summary['out_csv'] = str(out_csv)
    summary['out_rows'] = len(out_rows)
    summary['plot_payload'] = plot_payload
    return summary


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #


def plot_distill_sampling(plot_payload, fraction, out_png, title_suffix=""):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Merge per-STAR keep/drop across all distill files. Identifiers are
    # episode-level (callsign + ep_idx within file).
    chosen_per_star: dict[str, set[tuple]] = defaultdict(set)
    all_per_star: dict[str, list[tuple]] = defaultdict(list)
    eps_by_id: dict[tuple, list[dict]] = {}
    for _path, all_eps, by_star_all, chosen in plot_payload:
        eps_by_id.update(all_eps)
        for star, idents in by_star_all.items():
            all_per_star[star].extend(idents)
        for star, idents in chosen.items():
            chosen_per_star[star].update(idents)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, star in zip(axes.flat, STARS):
        all_ids = all_per_star.get(star, [])
        chosen_set = chosen_per_star.get(star, set())
        for ident in all_ids:
            if ident in chosen_set: continue
            ep = eps_by_id[ident]
            xs = [float(r['x_nm']) for r in ep]
            ys = [float(r['y_nm']) for r in ep]
            ax.plot(xs, ys, color='lightgray', lw=0.6, alpha=0.65, zorder=1)
        for ident in all_ids:
            if ident not in chosen_set: continue
            ep = eps_by_id[ident]
            xs = [float(r['x_nm']) for r in ep]
            ys = [float(r['y_nm']) for r in ep]
            ax.plot(xs, ys, color='tab:blue', lw=0.9, alpha=0.85, zorder=2)
        ax.set_title(f"{star}: kept {len(chosen_set)}/{len(all_ids)} episodes",
                     fontsize=11)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')
    fig.suptitle(
        f"Distillation random per-STAR episode sampling — fraction = {fraction:.2f}"
        f"{(' · ' + title_suffix) if title_suffix else ''}\n"
        f"gray = dropped · blue = kept",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--human-root', type=str, default='human_data',
                    help='Root containing single_plane/, multiple_planes/, '
                         'dagger/, and *_distillation/ subdirs.')
    ap.add_argument('--out-csv', type=str,
                    default='human_data/prepared/combined.csv',
                    help='Path for the combined CSV (one row per tick, '
                         'with a trailing `_source` column).')
    ap.add_argument('--distill-fraction', type=float, default=0.25,
                    help='Per-STAR fraction of distillation trajectories to keep.')
    ap.add_argument('--seed', type=int, default=0,
                    help='RNG seed for distillation sampling.')
    ap.add_argument('--viz-dir', type=str, default='data_viz',
                    help='Where to write the per-STAR sampled-trajectory plot.')
    args = ap.parse_args()

    human_root = Path(args.human_root)
    if not human_root.exists():
        sys.exit(f"--human-root not found: {human_root}")

    out_csv = Path(args.out_csv)
    viz_dir = Path(args.viz_dir)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_png = viz_dir / f'distill_sampled_{ts}.png'

    print(f"Preparing combined training CSV")
    print(f"  human_root      : {human_root}")
    print(f"  out_csv         : {out_csv}")
    print(f"  distill-fraction: {args.distill_fraction:.2f}  seed={args.seed}")
    print(f"  viz             : {out_png}\n", flush=True)

    rng = np.random.default_rng(args.seed)
    summary = build_combined_csv(human_root, out_csv,
                                 distill_fraction=args.distill_fraction,
                                 rng=rng)

    print()
    print(f"Combined CSV  : {summary['out_csv']}  ({summary['out_rows']:,} rows)")
    for tag, s in summary['sources'].items():
        if s['files']:
            print(f"  human/{tag:<7s}: {s['files']} files, "
                  f"{s['episodes']} episodes, {s['rows']:,} rows")
    d = summary['distill']
    print(f"  distill      : {d['files']} files, "
          f"{d['episodes_kept']}/{d['episodes_total']} episodes "
          f"({100.0 * d['episodes_kept'] / max(1, d['episodes_total']):.1f}%), "
          f"{d['rows']:,} rows")
    print(f"  distill per-STAR (kept / total):")
    for star in STARS:
        t = d['per_star_total'].get(star, 0)
        k = d['per_star_kept'].get(star, 0)
        print(f"    {star}: {k:>3d}/{t:<3d}")

    if d['files'] and summary['plot_payload']:
        plot_distill_sampling(summary['plot_payload'],
                              fraction=args.distill_fraction,
                              out_png=out_png,
                              title_suffix=f"seed={args.seed}")
        print(f"\nWrote {out_png}")


if __name__ == '__main__':
    main()
