# auto_plan — AUTO multi-aircraft planner

Self-contained port of the rl-branch `heuristics_multiple` planner + GMM policy
into the deployable simulator. Drives every aircraft on the **SIMULATED**
airport autonomously: vectoring each plane to the runway with the trained
policy while a conflict-resolution search keeps any two planes from violating
separation.

**Backend-only.** This package needs PyTorch + multiprocessing and therefore
**cannot run in the GitHub Pages (Pyodide / WebAssembly) build** — the AUTO
button only appears when a Flask backend is present. Nothing here is shipped to
the static site (`build_pages.py` lists only `environment/`).

## Files

| File | Role |
|------|------|
| `best.pt` | continuous_03 PPO checkpoint, **iter 160** (byte-identical to the rl `rl_ppo/best.pt`). |
| `policy_config.json` | Baked architecture + input standardizer + `input_indices` + geometry, so no BC-seed checkpoint is needed at runtime. |
| `model.py` | `BCActor` — mixture of (von Mises heading, Gaussian alt, Gaussian speed). Verbatim port. |
| `runtime.py` | Loads `best.pt`, encodes the 7-feature state, samples the GMM, translates to sim commands. |
| `rollout.py` | Single-plane rollout in a frozen-spawning sub-sim + the spawn worker pool + handoff helpers. |
| `flight_plan.py` | Conflict-resolution planner (backtracking search + greedy max-separation). |
| `planner.py` | `AutoPlanner` — per-tick arming, replanning, teleport-replay. |

## Model

7-D input `(0,1,2,4,5,6,7)` = along-track nm, cross-track nm, distance-to-
threshold nm, altitude/1000, (airspeed−200)/100, sin(heading), cos(heading).
Encoder 7→64→64, 4 mixture components. Output is a 4-D action
`(sin θ, cos θ, alt_kft, spd_norm)` translated to `C`/`S` commands plus a direct
target-altitude write. Architecture is recovered from the checkpoint's
`actor_state` tensor shapes; the 6-feature input standardizer + `input_indices`
are baked into `policy_config.json`.

## Planning

Matches the rl `heuristics_multiple` **watch** (live) loop:

- **Horizon:** `plan_steps = 400`.
- **Async planning:** a replan runs in a background thread; while it's in flight
  the sim **holds** (does not advance) and the UI shows a `PLANNING…` overlay.
  `/step`, `/auto off`, and `/restart` never block on it.
- **Arming:** a plane is taken over once it has flown 2 STAR waypoints, measured
  from the **full STAR length** — so a plane already past warmup when AUTO is
  switched on mid-scenario is armed immediately (the watch always runs from
  spawn, so it never needed this). On arming the STAR is cleared and `L <runway>`
  issued.
- **Replan trigger:** a newly-armed active plane with no plan, or every active
  plan depleted. Existing plans are extended to the horizon; conflicts re-rolled
  up to `max_conflict_iters`.
- **Conflict rule (planning + live):** two planes conflict iff **same medium**
  (both airborne or both on the ground) AND **lateral < 2.0 NM** AND
  **vertical < 1000 ft**. LOC/ILS status is ignored. Matches the live
  `CollisionMonitor` SIMULATED rule exactly.
- **Live application:** each tick the planner forces the planned state onto the
  aircraft (teleport replay) and no-ops its physics `update`; the GMM drives any
  armed plane that has no plan. The remaining plan tail is sent to the UI and
  drawn as a light-blue line (red if in an unresolved conflict).
- **Termination is the simulator's:** crash / landed / improper-exit. There are
  **no timeouts** and no LOC-above-glideslope dropping (unlike the rl eval
  harness).
- **AUTO off:** physics is restored and every armed plane not already on the
  localizer is left **cleared to land** (`L <runway>`), so planes keep flying
  the approach instead of freezing on their last vector.

## CPU / parallelism

Rollouts run on a persistent **spawn** worker pool sized adaptively to
`max(1, cpu_count − 1)` (favoring wall-clock over footprint, per request). On a
1–2 core host it falls back to serial. Each worker loads the policy once at init
and caps BLAS/torch threads to 1 to avoid oversubscription. The pool is warmed
when AUTO is enabled — expect a few seconds of one-time startup (Windows spawn +
per-worker torch load). `GET /auto` reports the chosen worker count and mode.

Per-callsign RNG is seeded with a **stable CRC32** of the callsign (not Python's
`hash()`, which is randomized per process and would desync rollouts across
workers).

## Use

The Flask app (`app.py`) drives this via `auto_plan.get_planner()`:
`POST /auto {on: true|false}` toggles it (`enable`/`disable`), `GET /auto`
returns status, and when AUTO is on `/step` calls `planner.step(sim)` (which runs
its own fast_forward loop and holds the sim during background replans). The
`/state` and `/step` payloads include `planning` (bool) and `flight_plans`
(per-plane remaining path) for the UI overlay.
