"""Single-aircraft PPO environment.

Each episode:
  1. Spawn one aircraft on a chosen STAR.
  2. Step the sim until `warmup_wpts` STAR waypoints are popped (so the BC
     policy isn't training on the trivial "follow the lateral nav" phase).
  3. PPO control begins. Each `step(action)` applies the 4-D action via the
     standard runtime translator (heading + speed → sim.command, altitude →
     direct ac.target_altitude write) and advances the sim 1 second.
  4. Termination — using the sim's own ILS-capture flags directly:
       SUCCESS  — `ac.loc_intercepted AND ac.gs_intercepted` both True.
                  The sim says we're on full ILS (localizer + glideslope).
       LOC_HIGH — On the very first step LOC fires, if the aircraft is
                  above the glideslope by > 50 ft, abort early. Sim blocks
                  re-vectoring once on LOC AND GS only captures from below
                  → the approach is provably doomed; don't burn rollout time
                  waiting for a guaranteed failure.
       TIMEOUT       — per-STAR step cap reached without SUCCESS.
       IMPROPER_EXIT — aircraft removed from sim (left the radar, etc.).
       CRASHED       — sim raised an exception (vMF sampler edge cases, etc.)

Observation: standardized 3-D `(a_nm, c_nm, d_thr_nm)` — same as the BC
encoder input. The 10-dim raw feature vector is computed under the hood
(needed for sim-aligned standardization buffers).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# NOTE: import the module (not the constants) for EVERYWHERE_STEP_PENALTY
# so the runtime override installed by train.py via
# `reward_zones.set_runtime_overrides()` propagates to step().
# `from X import CONST` would freeze the value at import time, making
# the setter useless. `step_reward` reads STEP_PENALTY_CAP through its
# own module globals so it picks up the override automatically.
from rl_ppo import reward_zones as _rz
from rl_ppo.reward_zones import (
    SUCCESS_REWARD as RZ_SUCCESS_REWARD,
    FAILURE_REWARD as RZ_FAILURE_REWARD,
    step_reward as zone_step_reward,
    point_in_zone,
    heading_intercept_reward,
    final_turn_action_reward,
)


# Import lazily inside methods so multiprocessing workers can spawn before
# the heavy imports (torch, environment) get fork-loaded.


@dataclass
class StepResult:
    obs: np.ndarray         # (3,) standardized
    reward: float
    done: bool
    info: dict              # outcome tag and any sim metadata


class PPOEnv:
    """One STAR, one aircraft, one PPO episode.

    Re-used across episodes by calling `reset(star)` again. NOT thread-safe
    — keep one instance per worker.
    """

    def __init__(self,
                 actor_ckpt: str | Path,
                 airport_name: str = 'test',
                 runway: str = '27',
                 warmup_wpts: int = 2,
                 max_timesteps_star_1_2: int = 1200,
                 max_timesteps_star_3: int = 500,
                 success_reward: float = 1.0,
                 failure_reward: float = -1.0,
                 gs_capture_buffer_ft: float = 50.0):
        from rl_bc.data import Standardizer, load_runway_geometry

        self.airport_name = airport_name
        self.runway = runway
        self.warmup_wpts = int(warmup_wpts)
        self.max_steps_1_2 = int(max_timesteps_star_1_2)
        self.max_steps_3 = int(max_timesteps_star_3)
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.gs_capture_buffer_ft = float(gs_capture_buffer_ft)

        # Pull standardizer + runway geometry from the BC checkpoint so the
        # observation matches what the actor was trained on.
        import torch
        blob = torch.load(actor_ckpt, map_location='cpu', weights_only=False)
        saved = blob['config']
        radar_side = int(saved.get('radar_side', 800))
        nm_range = int(saved.get('nm_range', 60))
        self.geom = load_runway_geometry(self.airport_name, self.runway,
                                         radar_side, nm_range)
        self.standardizer = Standardizer(
            mean=np.asarray(blob['standardizer_mean'], dtype=np.float32),
            std=np.asarray(blob['standardizer_std'], dtype=np.float32),
        )
        # Mirror the BC actor's input slice. _full uses (0,1,2,4,5,6,7);
        # the original 3-D variant uses (0,1,2). Default to (0,1,2) for
        # legacy checkpoints that don't carry an input_indices field.
        self._input_indices = tuple(int(i) for i in
                                    saved.get('input_indices', (0, 1, 2)))
        self.obs_dim = len(self._input_indices)

        # Per-episode state.
        self._sim = None
        self._callsign: str | None = None
        self._star: str | None = None
        self._steps: int = 0
        self._max_steps: int = 0
        self._closed_episode = True
        # Tracks whether we've already done the one-shot "LOC above GS?"
        # check at the moment LOC first fires. Set True once consumed so
        # subsequent steps don't re-fail on the same flag transition.
        self._loc_high_checked = False
        # One-shot flag for the turn-final (s, a) shaping bonus.
        self._turn_final_fired = False
        # Aircraft direction (NORTH/SOUTH) cached for reward calls.
        self._direction: str = 'NORTH'
        # Counters for the CLEAN_TERMINAL_THRESHOLD mechanism — track
        # how many policy-controlled steps were in-zone vs total.
        # At LOC_BELOW_GS termination for STAR-1/2 family, if
        # n_in_zone / n_total < threshold, replace SUCCESS_REWARD
        # with DRIFTY_SUCCESS_VALUE (default 0) so drift-but-successful
        # trajectories end up strictly negative once step penalties
        # accumulate.
        self._n_in_zone: int = 0
        self._n_policy_steps_total: int = 0
        # Out-of-zone termination counter (c03 mechanism).
        self._consecutive_out_of_zone: int = 0

    # -------------------------------------------------------------- #
    # Reset / step
    # -------------------------------------------------------------- #

    def reset(self, star: str, seed: int) -> np.ndarray:
        """Spawn a fresh aircraft on `star`, run STAR warm-up internally,
        return the first PPO observation."""
        import random as rnd
        from environment import SimulationEnv

        rnd.seed(seed)
        np.random.seed(seed)

        self._sim = SimulationEnv(airport_name=self.airport_name,
                                  spawn_single=True, star_mode=True)
        if star not in self._sim.spawner.procedures:
            raise RuntimeError(f"STAR {star!r} not in sim.spawner.procedures")
        self._sim.spawner.procedures = {star: self._sim.spawner.procedures[star]}
        self._sim.spawner.last_spawned_star = None

        self._sim.step(1.0)
        if not self._sim.aircraft_list:
            raise RuntimeError(f"sim did not spawn an aircraft on {star!r}")
        self._callsign = next(iter(self._sim.aircraft_list.keys()))
        self._star = star
        self._max_steps = (self.max_steps_3 if star.endswith('3')
                           else self.max_steps_1_2)

        # Run the STAR warm-up loop until the aircraft has popped
        # `warmup_wpts` waypoints (matches BC eval/runner's warm-up).
        initial_star_len = None
        for _ in range(2000):  # safety bound; STARs are short
            ac = self._sim.aircraft_list.get(self._callsign)
            if ac is None:
                raise RuntimeError(
                    f"aircraft {self._callsign} disappeared during warm-up")
            if initial_star_len is None:
                initial_star_len = len(ac.star) if ac.star else 0
            if initial_star_len == 0:
                break
            current = len(ac.star) if ac.star else 0
            popped = initial_star_len - current
            threshold = min(self.warmup_wpts, initial_star_len)
            if popped >= threshold:
                # End of STAR phase — clear for landing, hand to PPO.
                ac.star = None
                ac.star_name = None
                ac.target_wpt = None
                res = self._sim.command(self._callsign, f"L {self.runway}")
                if not res.get('ok'):
                    raise RuntimeError(
                        f"failed to issue landing clearance: {res}")
                break
            self._sim.step(1.0)

        self._steps = 0
        self._closed_episode = False
        self._loc_high_checked = False
        self._turn_final_fired = False
        self._direction = 'NORTH' if star.startswith('NORTH') else 'SOUTH'
        # Reset clean-terminal counters.
        self._n_in_zone = 0
        self._n_policy_steps_total = 0
        self._consecutive_out_of_zone = 0
        return self._observe()

    def step(self, action_4d: np.ndarray) -> StepResult:
        """Apply a 4-D action `(sin θ, cos θ, alt_kft, spd_norm)`, advance
        the sim 1 second, check termination, return shaped reward.

        Termination semantics MIRROR the BC eval runner exactly
        (rl_bc/eval/runner.py) so PPO and BC are scored on the same
        success criterion:
          LOC_BELOW_GS  — loc_intercepted fires AND aircraft is at/below
                          the 3° glide projection AND on the approach side
                          (a_along ≥ 0). SUCCESS.
          LOC_ABOVE_GS  — loc_intercepted fires AND aircraft is above
                          the projected glide → can't capture, FAILURE.
          LOC_BEHIND_THR — loc_intercepted fires past the threshold
                          (a_along < 0, back-course capture). FAILURE.
          TIMEOUT       — step count ≥ per-STAR cap without LOC fire.
          IMPROPER_EXIT — aircraft removed from sim.
          CRASHED       — sim raised an exception.

        Reward composition (see rl_ppo.reward_zones for the schedule):
            per-step  : zone_step_reward(s) + final_turn_action_reward(s, a)
                        (turn-final is one-shot per episode)
            terminal  : ±SUCCESS_REWARD/FAILURE_REWARD + heading_intercept_reward(s)
                        (heading bonus uses actual aircraft heading at term)
        """
        if self._closed_episode:
            raise RuntimeError("episode finished — call reset() first")
        try:
            cmd, target_alt_ft = self._action_to_cmd(action_4d)
            if cmd:
                self._sim.command(self._callsign, cmd)
            ac_obj = self._sim.aircraft_list.get(self._callsign)
            if (ac_obj is not None
                    and not getattr(ac_obj, 'gs_intercepted', False)
                    and not getattr(ac_obj, 'on_ground', False)
                    and not getattr(ac_obj, 'landed', False)):
                if abs(target_alt_ft
                       - float(getattr(ac_obj, 'target_altitude',
                                        target_alt_ft))) > 25.0:
                    ac_obj.target_altitude = target_alt_ft
                    ac_obj.star_apply_alt = False

            self._sim.step(1.0)
            self._steps += 1

            ac = self._sim.aircraft_list.get(self._callsign)

            # ---- Per-step shaping (zone penalty + turn-final R(s,a)
            #      + everywhere-step penalty) ----
            # Read EVERYWHERE_STEP_PENALTY through the module so runtime
            # overrides land here (vs frozen `from ... import CONST`).
            shape_r = -_rz.EVERYWHERE_STEP_PENALTY    # everywhere penalty
            a_nm = float('nan'); c_nm = float('nan')
            if ac is not None:
                a_nm, c_nm = self._runway_aligned(ac)
                # Out-of-zone step penalty. For the first
                # EARLY_WINDOW_STEPS policy-controlled steps, multiply by
                # EARLY_ZONE_MULTIPLIER — targets the "drift on
                # triangle-to-downwind merge" failure mode without
                # crushing the policy through the whole episode.
                # STAR-3 family is EXEMPT from any early-window pressure
                # (we only care about SR there).
                zone_pen = zone_step_reward(self._star, a_nm, c_nm)
                in_early_window = self._steps < _rz.EARLY_WINDOW_STEPS
                early_eligible = (in_early_window
                                  and self._star not in _rz.EARLY_DRIFT_EXEMPT)
                if early_eligible:
                    zone_pen *= _rz.EARLY_ZONE_MULTIPLIER
                    # FLAT early-drift penalty: every out-of-zone step in
                    # the window pays this regardless of distance. This
                    # is the "half-failure" lever — calibrated so a
                    # BC-like NORTH2 drift (~37% out in first 300) nets
                    # roughly +5 reward instead of +10.
                    if zone_pen < 0.0:
                        zone_pen -= _rz.EARLY_DRIFT_PENALTY
                shape_r += zone_pen
                # Track in-zone vs total policy steps for the
                # CLEAN_TERMINAL_THRESHOLD mechanism. Use point_in_zone
                # DIRECTLY — NOT zone_step_reward, because when
                # step_pen_per_nm=0 and step_pen_cap=0 (c03 v2 config),
                # zone_step_reward returns 0 even when out of zone.
                # Only count when STAR-1/2 family — STAR-3 is exempt
                # from clean-terminal too.
                self._n_policy_steps_total += 1
                if (self._star not in _rz.EARLY_DRIFT_EXEMPT
                        and point_in_zone(self._star, a_nm, c_nm)):
                    self._n_in_zone += 1
                if _rz.TURN_FINAL_ENABLED and not self._turn_final_fired:
                    sin_h = float(action_4d[0]); cos_h = float(action_4d[1])
                    target_h_deg = (math.degrees(math.atan2(sin_h, cos_h))
                                    + 360.0) % 360.0
                    bonus = final_turn_action_reward(
                        self._star, a_nm, c_nm, target_h_deg)
                    if bonus > 0.0:
                        shape_r += bonus
                        self._turn_final_fired = True

            # Out-of-zone termination (c03 mechanism). Fires when
            # STAR-1/2 aircraft has been out of zone for too many
            # consecutive steps. STAR-3 is exempt. Note this check
            # happens BEFORE the LOC capture check, so we don't allow
            # captures from out-of-zone positions if termination would
            # have fired anyway. zone_pen is the (possibly-mutated)
            # zone penalty — its un-mutated value is 0 iff in-zone.
            if (_rz.OUT_OF_ZONE_TERMINATE
                    and ac is not None
                    and self._star not in _rz.EARLY_DRIFT_EXEMPT):
                # Use point_in_zone directly — NOT zone_step_reward,
                # since zone_step_reward returns 0 when slope=cap=0.
                in_zone = point_in_zone(self._star, a_nm, c_nm)
                if in_zone:
                    self._consecutive_out_of_zone = 0
                else:
                    self._consecutive_out_of_zone += 1
                if self._consecutive_out_of_zone >= _rz.OUT_OF_ZONE_MAX_CONSECUTIVE:
                    self._closed_episode = True
                    return StepResult(
                        obs=self._observe(),
                        reward=RZ_FAILURE_REWARD + shape_r,
                        done=True,
                        info={'outcome': 'OUT_OF_ZONE',
                              'star': self._star,
                              'steps': self._steps,
                              'a_nm': a_nm, 'c_nm': c_nm,
                              'consecutive_out_of_zone': self._consecutive_out_of_zone},
                    )

            # Termination: aircraft removed → improper exit.
            if ac is None:
                self._closed_episode = True
                return StepResult(
                    obs=np.zeros(self.obs_dim, dtype=np.float32),
                    reward=RZ_FAILURE_REWARD + shape_r,
                    done=True,
                    info={'outcome': 'IMPROPER_EXIT',
                          'star': self._star,
                          'steps': self._steps},
                )

            loc_on = bool(getattr(ac, 'loc_intercepted', False))

            # Termination on first LOC-capture — same 3-way classification
            # as the BC eval runner. The sim has a hard heading gate on
            # LOC capture (±30° within 20nm), so once loc_intercepted
            # fires the aircraft IS geometrically committed; we just
            # classify by altitude (vs projected glide) and side (vs the
            # threshold along the runway centerline).
            if loc_on:
                d_thr_nm, gs_alt_ft, capturable = self._gs_capturable(ac)
                a_along, _c_off = self._runway_aligned(ac)
                hb = (heading_intercept_reward(
                          self._direction, float(ac.heading))
                      if _rz.HEADING_INTERCEPT_ENABLED else 0.0)
                if a_along < 0.0:
                    outcome = 'LOC_BEHIND_THR'   # past threshold, wrong-side capture
                    terminal_r = RZ_FAILURE_REWARD
                elif capturable:
                    outcome = 'LOC_BELOW_GS'     # SUCCESS — same as BC eval
                    terminal_r = RZ_SUCCESS_REWARD
                    # Per-STAR multiplicative reward scaling.
                    # Amplifies terminal on STARs with low recent SR
                    # to combat STAR-trading mode collapse.
                    if _rz.PER_STAR_SR_SCALE > 0.0:
                        recent_sr = _rz.PER_STAR_RECENT_SR.get(self._star, 1.0)
                        scale = 1.0 + _rz.PER_STAR_SR_SCALE * (1.0 - recent_sr)
                        terminal_r = terminal_r * scale
                    # CLEAN_TERMINAL gate: for STAR-1/2 family, demand
                    # the trajectory was mostly in zone. Otherwise we
                    # don't reward the success — replace the +10
                    # terminal with the configurable drifty value
                    # (default 0). Step penalties have already accumulated
                    # over the trajectory so the net reward will be
                    # strictly negative for drifty "successes".
                    if (self._star not in _rz.EARLY_DRIFT_EXEMPT
                            and self._n_policy_steps_total > 0):
                        frac_in = (self._n_in_zone
                                   / self._n_policy_steps_total)
                        if frac_in < _rz.CLEAN_TERMINAL_THRESHOLD:
                            terminal_r = _rz.DRIFTY_SUCCESS_VALUE
                else:
                    outcome = 'LOC_ABOVE_GS'     # too high to capture, FAILURE
                    terminal_r = RZ_FAILURE_REWARD
                self._closed_episode = True
                return StepResult(
                    obs=self._observe(),
                    reward=terminal_r + shape_r + hb,
                    done=True,
                    info={'outcome': outcome,
                          'star': self._star,
                          'steps': self._steps,
                          'd_thr_nm': d_thr_nm,
                          'altitude_ft': float(ac.altitude),
                          'gs_alt_ft': gs_alt_ft,
                          'a_along_nm': a_along,
                          'heading_bonus': hb},
                )

            # Termination: timeout.
            if self._steps >= self._max_steps:
                self._closed_episode = True
                hb = (heading_intercept_reward(self._direction, float(ac.heading))
                      if _rz.HEADING_INTERCEPT_ENABLED else 0.0)
                return StepResult(
                    obs=self._observe(),
                    reward=RZ_FAILURE_REWARD + shape_r + hb,
                    done=True,
                    info={'outcome': 'TIMEOUT',
                          'star': self._star,
                          'steps': self._steps,
                          'loc_intercepted': loc_on,
                          'gs_intercepted': bool(
                              getattr(ac, 'gs_intercepted', False)),
                          'heading_bonus': hb},
                )
            # Step continues with just the per-step shaping signal.
            return StepResult(
                obs=self._observe(),
                reward=shape_r,
                done=False,
                info={'star': self._star, 'steps': self._steps,
                      'a_nm': a_nm, 'c_nm': c_nm},
            )
        except Exception as exc:
            self._closed_episode = True
            return StepResult(
                obs=np.zeros(self.obs_dim, dtype=np.float32),
                reward=RZ_FAILURE_REWARD,    # no heading bonus on crash
                done=True,
                info={'outcome': 'CRASHED',
                      'star': self._star,
                      'steps': self._steps,
                      'error_type': type(exc).__name__,
                      'error': str(exc)[:200]},
            )

    def _runway_aligned(self, ac) -> tuple[float, float]:
        """Return (a_nm, c_nm) for the aircraft in the runway-aligned
        frame — same math as `_observe` uses for cols 0/1."""
        nm_per_pixel = self._sim.nm_per_pixel
        x_nm = (ac.x - self._sim.airport_x) * nm_per_pixel
        y_nm = -(ac.y - self._sim.airport_y) * nm_per_pixel
        phi = math.radians((self.geom.course_deg + 180.0) % 360.0)
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        dx = x_nm - self.geom.thr_x_nm
        dy = y_nm - self.geom.thr_y_nm
        a_nm = dx * sin_phi + dy * cos_phi
        c_nm = -dx * cos_phi + dy * sin_phi
        return float(a_nm), float(c_nm)

    def _gs_capturable(self, ac) -> tuple[float, float, bool]:
        """Compute (distance_to_threshold_nm, gs_altitude_ft, capturable).

        GS-capturable iff `altitude ≤ d_thr_nm·300 + gs_capture_buffer_ft`,
        matching the sim's own GS-capture window in
        environment/core/aircraft.py::_update_ils_gs. The 300 ft/nm slope
        is the sim's hard-coded 3°/nm glideslope.
        """
        thr_coords = ac.coords.get(self.runway)
        if thr_coords is None:
            # Shouldn't happen — we cleared for the same runway.
            return 0.0, 0.0, False
        nm_per_pixel = self._sim.nm_per_pixel
        dx_px = ac.x - thr_coords['x']
        dy_px = ac.y - thr_coords['y']
        d_thr_nm = nm_per_pixel * math.sqrt(dx_px * dx_px + dy_px * dy_px)
        gs_alt_ft = d_thr_nm * 300.0
        capturable = float(ac.altitude) <= gs_alt_ft + self.gs_capture_buffer_ft
        return d_thr_nm, gs_alt_ft, capturable

    def close(self):
        self._sim = None
        self._closed_episode = True

    # -------------------------------------------------------------- #
    # Helpers
    # -------------------------------------------------------------- #

    def _observe(self) -> np.ndarray:
        """Standardized observation vector matching the BC actor's
        `input_indices`. Build the full 10-D feature row exactly as
        `rl_bc/bc_gmm/rollout.py::Runtime._build_feat` does (sim
        observable → standardized features), then SLICE to whatever
        columns the BC actor was trained on (3-D for `bc_gmm_single`,
        7-D for `bc_gmm_single_full`)."""
        from rl_bc.config import N_CONT, N_FEATURES
        ac = self._sim.aircraft_list.get(self._callsign)
        if ac is None:
            return np.zeros(len(self._input_indices), dtype=np.float32)
        nm_per_pixel = self._sim.nm_per_pixel
        x_nm = (ac.x - self._sim.airport_x) * nm_per_pixel
        y_nm = -(ac.y - self._sim.airport_y) * nm_per_pixel
        phi = math.radians((self.geom.course_deg + 180.0) % 360.0)
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        dx = x_nm - self.geom.thr_x_nm
        dy = y_nm - self.geom.thr_y_nm
        a_nm = dx * sin_phi + dy * cos_phi
        c_nm = -dx * cos_phi + dy * sin_phi
        d_thr = math.sqrt(a_nm * a_nm + c_nm * c_nm)
        heading = float(getattr(ac, 'heading', 0.0))
        altitude = float(getattr(ac, 'altitude', 0.0))
        airspeed = float(getattr(ac, 'airspeed', 0.0))
        loc_flag = 1.0 if getattr(ac, 'loc_intercepted', False) else 0.0
        gs_flag = 1.0 if getattr(ac, 'gs_intercepted', False) else 0.0
        dtheta = ((heading - self.geom.course_deg + 540.0) % 360.0) - 180.0
        x = np.zeros(N_FEATURES, dtype=np.float32)
        x[0] = a_nm; x[1] = c_nm; x[2] = d_thr
        x[3] = dtheta / 180.0
        x[4] = altitude / 1000.0
        x[5] = (airspeed - 200.0) / 100.0
        x[6] = math.sin(math.radians(heading))
        x[7] = math.cos(math.radians(heading))
        x[8] = loc_flag; x[9] = gs_flag
        # Standardize the 6 continuous columns.
        x[:N_CONT] = (x[:N_CONT] - self.standardizer.mean) / self.standardizer.std
        return x[list(self._input_indices)].astype(np.float32)

    def _action_to_cmd(self, action_4d: np.ndarray) -> tuple[str | None, float]:
        """Convert (sin θ, cos θ, alt_kft, spd_norm) → (cmd_string, target_alt_ft).

        Heading: only issue a `C` command if it differs from current target
        by >=1°. Speed: only issue an `S` command if it differs by >5 kt.
        Altitude is set directly on the aircraft object (the sim's command
        syntax for altitude requires extra plumbing we don't need here).
        """
        sin_h, cos_h, alt_kft, spd_norm = (float(v) for v in action_4d)
        target_hdg = (math.degrees(math.atan2(sin_h, cos_h)) + 360.0) % 360.0
        target_spd_kt = max(140.0, min(280.0, float(spd_norm) * 100.0 + 200.0))
        target_alt_ft = max(1000.0, min(18000.0, float(alt_kft) * 1000.0))

        ac = self._sim.aircraft_list.get(self._callsign)
        if ac is None:
            return None, target_alt_ft
        loc_on = bool(getattr(ac, 'loc_intercepted', False))
        cur_hdg = float(getattr(ac, 'target_heading', ac.heading))
        cur_spd = float(getattr(ac, 'target_airspeed', ac.airspeed))

        parts: list[str] = []
        if not loc_on:
            tgt_int = int(round(target_hdg)) % 360
            diff = abs(((tgt_int - cur_hdg + 540.0) % 360.0) - 180.0)
            if diff >= 1.0:
                h = tgt_int if tgt_int != 0 else 360
                parts.append(f"C {h:03d}")
        tgt_kt = int(round(target_spd_kt / 10.0)) * 10
        tgt_kt = max(140, min(280, tgt_kt))
        if abs(tgt_kt - cur_spd) > 5.0:
            parts.append(f"S {tgt_kt}")
        cmd = " ".join(parts) if parts else None
        return cmd, target_alt_ft
