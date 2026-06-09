"""Behavior-cloning training package.

Current family: `bc_fm` (joint conditional flow matching). Future siblings
(Gaussian mixture, etc.) will live alongside in their own packages.

Layout:

  - rl_bc.config           Config dataclass + paths
  - rl_bc.data             CSV → cache → Dataset
  - rl_bc.train            single-process training loop
  - rl_bc.viz_quiver       static model-quiver renderer (per-cell arrows)
  - rl_bc.viz_interactive  click-anywhere heading-distribution probe
  - rl_bc.eval.viz_trajectories  rollout trajectory plots (npz or CSV input)
  - rl_bc.modal_config     Modal app/image/volume + remote train function
  - rl_bc.modal_train      one-command Modal launcher (auto-sync + auto-pull)
  - rl_bc.bc_fm.*          BCActor, Runtime, watcher, probe

Configs live in `rl_bc.configs.<name>` and are discovered automatically.
"""
