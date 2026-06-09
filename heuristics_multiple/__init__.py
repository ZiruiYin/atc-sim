"""heuristics_multiple — rollout-based multi-plane conflict resolution.

Drives the base single-GMM PPO (the same ckpt that seeds rl_multiple)
across multiple spawned planes, and is the place where heuristic
rollout-and-pick logic will live.

For now: just `watch.py`, the live radar driver. Trajectory rollout +
intersection scoring lands in follow-ups.
"""
