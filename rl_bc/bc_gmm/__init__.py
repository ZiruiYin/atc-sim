"""bc_gmm — joint Gaussian-mixture BC actor.

Self-contained package:

  - model.py   : BCActor (one encoder + K-component diagonal-covariance
                 mixture head over the 4-D action vector)
  - train.py   : NLL training loop; LOC mask drops hdg dims on loc==1 rows
  - rollout.py : Runtime — per-aircraft seeded sampling for deterministic
                 mode commitment, exposes log_prob for future PPO use
  - watch.py   : Flask app, supports --dagger
  - probe.py   : per-state mixture inspection
"""
