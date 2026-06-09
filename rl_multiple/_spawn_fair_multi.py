"""Fire-and-forget launcher for the multi_ppo fair eval.

Uses Function.spawn() instead of .remote() so the call is dispatched to
Modal and we exit immediately — no streaming connection that can be
cancelled when this process dies.
"""
import os
import sys

# Make sure MODAL_NONPREEMPTIBLE is set BEFORE importing modal_eval_fair
# (the decorator reads it at module load).
os.environ['MODAL_NONPREEMPTIBLE'] = '1'

# Local repo on sys.path.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl_multiple.modal_eval_fair import app, eval_fair_remote

ARGS = {
    'out_relpath': 'rollout_comparisons/baseline_vs_multi_02_v2/multi_ppo',
    'tag': 'multi_ppo',
    'ckpt_relpath': 'runs_ppo/continuous_03/iter_0160.pt',
    'multi_ckpt_relpath': 'runs_ppo_multi/phase2/continuous_02/best.pt',
    'bc_seed_relpath': 'runs/bc_gmm_single_full/run_11/best.pt',
    'n_target': 512,
    'seed_base': 0,
    'max_scenarios': 10000,
    'spawn_rate': 90,
    'max_steps': 1_000_000,
    'max_steps_1_2': 1200,
    'max_steps_3': 500,
    'warmup_wpts': 2,
}

if __name__ == '__main__':
    # detach=True keeps the app alive even after this process exits, AND
    # spawn() dispatches without holding a streaming connection. Together
    # they give true fire-and-forget.
    with app.run(detach=True):
        fc = eval_fair_remote.spawn(ARGS)
        print(f"Spawned function call: {fc.object_id}", flush=True)
        print(f"Args: {ARGS}", flush=True)
        print(f"Tail $TEMP/fair_full_multi_v3.log? No — there is no local "
              f"log file. Check Modal dashboard or the atc-bc volume at "
              f"/{ARGS['out_relpath']}/summary.json when complete.",
              flush=True)
