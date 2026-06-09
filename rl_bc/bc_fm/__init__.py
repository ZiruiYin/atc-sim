"""bc_fm — joint conditional flow-matching BC actor.

Self-contained package:

  - model.py   : BCActor (one joint 4-D FM head over hdg/alt/spd)
  - rollout.py : Runtime — load checkpoint, encode_state, predict, translate, tick
  - watch.py   : Flask app (`python -m rl_bc.bc_fm.watch`); supports --dagger
  - probe.py   : sample many x_0 per canonical state; report per-dim spread
"""
