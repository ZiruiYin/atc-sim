"""Multi-plane PPO playground.

Scaffold for the multi-plane phase of the project. The single-plane PPO
ship (rl_ppo/runs/continuous_runs/continuous_03/best.pt) is reused
unchanged — each spawned aircraft gets its own per-callsign frozen-noise
seed and the policy is queried independently for each plane per tick.

Entry points:

    python -m rl_multiple.watch       # Flask server; model flies the game
"""
