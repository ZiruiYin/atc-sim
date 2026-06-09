"""Local entrypoint for Modal-hosted BC evaluation.

The actual `eval_remote` @app.function lives in `rl_bc.modal_config` (same
pattern as `train_remote`) so Modal's import path resolves cleanly. This
file is the local CLI that picks a checkpoint on the volume, kicks off the
remote eval, and writes results back into `eval/<config>/` locally with
the SAME fixed-filename layout as the local runner — overwritten on re-run:

    eval/<config>/rollouts.csv
    eval/<config>/summary.json
    eval/<config>/raw.json
    eval/<config>/trajectories.npz
    eval/<config>/successful_trajectories.png
    eval/<config>/unsuccessful_trajectories.png

Usage:
    modal run rl_bc/modal_eval.py --config bc_fm_single
    modal run rl_bc/modal_eval.py --config bc_gmm_single --cases 200
    modal run rl_bc/modal_eval.py --config bc_fm_single --run 2
"""

from __future__ import annotations

import csv as csv_module
import json
from pathlib import Path

from rl_bc.modal_config import app, eval_remote


CSV_FIELDS = ('star', 'seed', 'outcome', 'callsign', 'sim_time',
              'steps_used', 'error_type', 'error')


@app.local_entrypoint()
def main(
    config: str,                # required, e.g. bc_fm_single
    run: int = 0,               # 0 → latest on volume
    cases: int = 500,
    max_steps: int = 0,             # 0 ⇒ per-STAR caps (NORTH1/2,SOUTH1/2=1200; NORTH3,SOUTH3=500)
    warmup_wpts: int = 2,
    runway: str = '27',
    airport: str = 'test',
    cpu: int = 16,              # mp.Pool worker count INSIDE the container.
                                # Container CPU allocation is set on the
                                # @app.function decorator in modal_config.py;
                                # keep this ≤ that or you'll over-subscribe.
    seed_base: int = 0,
    lim_nm: float = 30.0,
    out_dir: str = '',           # full destination folder; '' → eval/<config>/
):
    args = {
        "config_name": config,
        "run_n": (None if run <= 0 else int(run)),
        "cases": int(cases),
        "max_steps": int(max_steps),
        "warmup_wpts": int(warmup_wpts),
        "runway": str(runway),
        "airport": str(airport),
        "workers": int(cpu),
        "seed_base": int(seed_base),
    }
    print(f"  → dispatching eval to Modal (workers={cpu}, config={config}, "
          f"run={'latest' if run <= 0 else run}, cases={cases})")
    out = eval_remote.remote(args)
    summary = out["summary"]
    results = out["results"]
    ckpt_path_remote = out["ckpt_path"]

    print(f"\nremote ckpt: {ckpt_path_remote}")
    print(f"wall time  : {summary['_meta']['wall_seconds']:.1f}s "
          f"on {summary['_meta']['workers']} workers")
    print(f"overall    : {summary['overall']['landed']}/{summary['overall']['n']}"
          f" = {summary['overall']['success_rate'] * 100:.1f}% landed")

    out_dir = Path(out_dir) if out_dir else Path("eval") / config
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'rollouts.csv'
    summary_path = out_dir / 'summary.json'
    raw_path = out_dir / 'raw.json'
    traj_path = out_dir / 'trajectories.npz'

    # Stream the CSV from the returned results (Modal-side eval doesn't
    # stream because the workers can't write to the local FS).
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv_module.DictWriter(f, fieldnames=CSV_FIELDS,
                                  extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow(r)

    raw_for_json = [{k: v for k, v in r.items()
                     if k not in ('a_traj', 'c_traj')}
                    for r in results]
    summary_path.write_text(json.dumps(summary, indent=2))
    raw_path.write_text(json.dumps(raw_for_json, indent=2))

    from rl_bc.eval.viz_trajectories import (save_trajectories,
                                              render_from_results)
    save_trajectories(traj_path, results)
    print("\n  rendering trajectory plots...")
    render_from_results(results, out_dir, lim_nm=lim_nm)

    print(f"\nsaved (overwritten in place):"
          f"\n  {csv_path}"
          f"\n  {summary_path}"
          f"\n  {raw_path}"
          f"\n  {traj_path}"
          f"\n  {out_dir / 'successful_trajectories.png'}"
          f"\n  {out_dir / 'unsuccessful_trajectories.png'}")
