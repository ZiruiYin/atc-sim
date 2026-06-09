"""Multi-plane continuous rollout sim for Phase-2 PPO.

One sim per worker. Runs indefinitely, spawning planes per `spawn_rate`,
collecting one per-plane Trajectory per landed/failed plane. Terminates
when `n_target` trajectories have been closed (with safety-bound max
ticks).

Per-plane lifecycle (mirrors rl_ppo.env.PPOEnv's structure but lives
inside a multi-plane sim):

    1. Plane spawns at sim's spawn cadence with a random STAR.
    2. Plane follows the STAR for `warmup_wpts` waypoints.
    3. At handover, `L 27` is issued, the trajectory's recording begins.
    4. The radar head (delta-augmented PPO policy) drives the plane.
    5. Trajectory closes on first of:
         LANDED         (success +10)        ← NEW vs Phase 1
         LOC_ABOVE_GS   (fail    -10)
         LOC_BEHIND_THR (fail    -10)
         TIMEOUT        (fail    -10)        ← pre-LOC step cap
         IMPROPER_EXIT  (fail    -10)
       Sim wipes the plane for any non-LANDED outcome (LANDED self-wipes).
    6. On full sim crash (any plane has `crash`), all in-flight
       trajectories are closed as TRUNCATED with `done=False` and
       `bootstrap_value = last V(s_T)`. The 2 crashing planes get
       terminal failure_reward. Sim restarts fresh, collection continues.

Reward composition per tick (radar head policy):
    everywhere_pen     applies always                          (default -0.002)
    collision_warning  applies when plane.collision_warning      (default -0.10/step)
    heading_intercept  one-shot at LOC capture (terminal-time)   (up to +1)
    final_turn         one-shot at first turn-in                 (up to +0.5)
    loop_penalty       post-hoc, second-half of detected loops   (default -0.10/step)
    terminal           success +10 / failure -10

Logs (per worker, one of each):
    Training:  list[Trajectory]   — (s, a, r, log_p, V, done) tuples
    Replay:    HumanDataRecorder  — per-tick CSV in human_data format
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from environment import SimulationEnv
from environment.core.human_data_logger import HumanDataRecorder

from rl_ppo import reward_zones as _rz
from rl_ppo.reward_zones import (
    SUCCESS_REWARD as RZ_SUCCESS_REWARD,
    FAILURE_REWARD as RZ_FAILURE_REWARD,
    heading_intercept_reward,
    final_turn_action_reward,
)

from rl_multiple.density import N_BINS, build_density
from rl_multiple.policy import CombinedPolicy
from rl_multiple.runtime import resolve_bc_seed_path


# --------------------------------------------------------------------------- #
# Per-plane trajectory accumulator (in-flight; finalized to a Trajectory).
# --------------------------------------------------------------------------- #


@dataclass
class _PlaneBuffer:
    star: str
    callsign: str
    seed: int                       # rollout seed unique-ish per plane
    spawn_sim_time: float           # for replay alignment
    full_obs:    list = field(default_factory=list)   # (T, 79)
    ego_obs:     list = field(default_factory=list)   # (T, 7)
    actions:     list = field(default_factory=list)   # (T, 3) delta
    log_probs:   list = field(default_factory=list)
    rewards:     list = field(default_factory=list)
    values:      list = field(default_factory=list)
    a_traj:      list = field(default_factory=list)   # along-runway nm, for viz
    c_traj:      list = field(default_factory=list)   # cross-track nm, for viz
    # Bookkeeping
    pre_loc_steps: int = 0          # ticks before LOC capture (for cap)
    loc_captured: bool = False
    turn_final_fired: bool = False
    warning_seconds: int = 0
    # Set on close:
    outcome: str = 'UNKNOWN'
    error:   Optional[str] = None
    d_thr_nm: Optional[float] = None
    altitude_ft: Optional[float] = None
    gs_alt_ft: Optional[float] = None
    bootstrap_value: float = 0.0    # used by GAE when truncated (done=False)
    truncated: bool = False         # True for crash-survivors


# --------------------------------------------------------------------------- #
# Finalized trajectory — matches the schema rl_multiple.rollout.Trajectory
# uses so train.py's flatten_trajectories can consume both single- and
# multi-plane rollouts without code changes.
# --------------------------------------------------------------------------- #


@dataclass
class MultiTrajectory:
    obs: np.ndarray
    ego_obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    dones: np.ndarray
    star: str
    outcome: str
    steps: int
    seed: int = 0
    callsign: str = ''
    error: Optional[str] = None
    d_thr_nm: Optional[float] = None
    altitude_ft: Optional[float] = None
    gs_alt_ft: Optional[float] = None
    a_traj: Optional[np.ndarray] = None
    c_traj: Optional[np.ndarray] = None
    n_loop_steps: int = 0
    n_penalized: int = 0
    loop_penalty_total: float = 0.0
    # Phase-2 specific
    warning_seconds: int = 0
    truncated: bool = False
    bootstrap_value: float = 0.0


# --------------------------------------------------------------------------- #
# MultiRolloutSim — one continuous sim, owns a policy + density cache +
# replay recorder.
# --------------------------------------------------------------------------- #


class MultiRolloutSim:
    """One sim per worker. Run `collect()` to get a list of trajectories."""

    def __init__(self,
                 ppo_seed_ckpt: str | Path,
                 bc_seed_ckpt: str | Path,
                 cfg_dict: dict,
                 worker_id: int = 0,
                 replay_dir: Optional[Path] = None):
        from rl_bc.data import Standardizer, load_runway_geometry

        self.cfg_dict = dict(cfg_dict)
        self.worker_id = worker_id
        self.replay_dir = Path(replay_dir) if replay_dir else None
        self.airport_name = cfg_dict['airport_name']
        self.runway = cfg_dict['runway']
        self.warmup_wpts = int(cfg_dict['warmup_wpts'])
        self.max_steps_1_2 = int(cfg_dict['max_timesteps_star_1_2'])
        self.max_steps_3 = int(cfg_dict['max_timesteps_star_3'])
        self.spawn_rate = int(cfg_dict['spawn_rate'])
        self.gs_capture_buffer_ft = float(cfg_dict.get('gs_capture_buffer_ft', 50.0))
        self.density_cutoff_nm = float(cfg_dict.get('density_cutoff_nm', 10.0))
        self.collision_warning_penalty = float(
            cfg_dict.get('collision_warning_penalty', 0.10))
        self.crash_extra_penalty = float(cfg_dict.get('crash_extra_penalty', 0.0))
        # Loop detector
        self.loop_pps = float(cfg_dict.get('loop_penalty_per_step', 0.0) or 0.0)
        self.loop_prox_radius_nm = float(cfg_dict.get('loop_prox_radius_nm', 0.75))
        self.loop_min_gap_steps = int(cfg_dict.get('loop_min_gap_steps', 45))
        self.loop_min_detour_nm = float(cfg_dict.get('loop_min_detour_nm', 1.0))

        # Pull standardizer + geom from BC seed once.
        blob = torch.load(bc_seed_ckpt, map_location='cpu', weights_only=False)
        saved = blob['config']
        radar_side = int(saved.get('radar_side', 800))
        nm_range = int(saved.get('nm_range', 60))
        self.geom = load_runway_geometry(self.airport_name, self.runway,
                                          radar_side, nm_range)
        self.standardizer = Standardizer(
            mean=np.asarray(blob['standardizer_mean'], dtype=np.float32),
            std=np.asarray(blob['standardizer_std'], dtype=np.float32),
        )
        self._input_indices = tuple(int(i) for i in
                                    saved.get('input_indices', (0, 1, 2)))
        self._ego_dim = len(self._input_indices)
        self._full_dim = self._ego_dim + 2 * N_BINS

        # Build the policy ONCE; load_state_split before each collect().
        self.policy = CombinedPolicy.from_ppo_ckpt(
            ppo_seed_ckpt,
            bc_seed_path=bc_seed_ckpt,
            density_n_bins=N_BINS,
            delta_hidden=int(cfg_dict.get('delta_hidden', 64)),
            value_hidden=int(cfg_dict.get('value_hidden', 64)),
            value_dropout=float(cfg_dict.get('value_dropout', 0.0)),
            device='cpu',
        )
        self.policy.eval()

        self._sim: Optional[SimulationEnv] = None
        self._planes: dict[str, _PlaneBuffer] = {}
        self._initial_star_len: dict[str, int] = {}
        self._armed: set[str] = set()
        # Per-callsign prev density (for delta in env obs). Lives across
        # ticks until that callsign departs.
        self._prev_density: dict[str, np.ndarray] = {}
        # Crash bookkeeping
        self.n_crashes = 0
        self._next_seed = 1_000_000 * worker_id    # disjoint per worker

    # ------------------------------------------------------------------ #
    # Sim lifecycle
    # ------------------------------------------------------------------ #

    def _start_sim(self):
        self._sim = SimulationEnv(
            radar_side=800, airport_name=self.airport_name,
            spawn_single=False, star_mode=True,
        )
        self._sim.set_spawn_rate(self.spawn_rate)
        if self.replay_dir:
            self.replay_dir.mkdir(parents=True, exist_ok=True)
            # in_memory=False writes to disk in HUMAN_DATA_DIR by default;
            # in_memory=True keeps it buffered.
            recorder = HumanDataRecorder(spawn_single=False, in_memory=True)
            recorder.start()
            self._sim.recorder = recorder
        self._planes.clear()
        self._initial_star_len.clear()
        self._armed.clear()
        self._prev_density.clear()

    def _restart_sim(self) -> str:
        """Dump any active replay buffer, start a fresh sim. Returns
        the saved replay path or '' if not recording."""
        replay_path = ''
        if self._sim is not None and self._sim.recorder is not None:
            rec = self._sim.recorder
            if self.replay_dir and getattr(rec, 'in_memory', False):
                csv_text = rec.to_csv()
                fname = (f'worker_{self.worker_id:02d}_'
                         f'session_{int(getattr(self, "_session_idx", 0)):03d}.csv')
                p = self.replay_dir / fname
                p.write_text(csv_text, encoding='utf-8')
                replay_path = str(p)
                self._session_idx = getattr(self, '_session_idx', 0) + 1
            try:
                rec.close()
            except Exception:
                pass
        self._start_sim()
        return replay_path

    # ------------------------------------------------------------------ #
    # Observation building
    # ------------------------------------------------------------------ #

    def _runway_aligned(self, ac) -> tuple[float, float]:
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
        thr_coords = ac.coords.get(self.runway)
        if thr_coords is None:
            return 0.0, 0.0, False
        nm_per_pixel = self._sim.nm_per_pixel
        dx_px = ac.x - thr_coords['x']
        dy_px = ac.y - thr_coords['y']
        d_thr_nm = nm_per_pixel * math.sqrt(dx_px * dx_px + dy_px * dy_px)
        gs_alt_ft = d_thr_nm * 300.0
        capturable = float(ac.altitude) <= gs_alt_ft + self.gs_capture_buffer_ft
        return d_thr_nm, gs_alt_ft, capturable

    def _build_ego_obs(self, ac) -> np.ndarray:
        from rl_bc.config import N_CONT, N_FEATURES
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
        heading = float(ac.heading)
        altitude = float(ac.altitude)
        airspeed = float(ac.airspeed)
        loc_flag = 1.0 if ac.loc_intercepted else 0.0
        gs_flag = 1.0 if ac.gs_intercepted else 0.0
        dtheta = ((heading - self.geom.course_deg + 540.0) % 360.0) - 180.0
        x = np.zeros(N_FEATURES, dtype=np.float32)
        x[0] = a_nm; x[1] = c_nm; x[2] = d_thr
        x[3] = dtheta / 180.0
        x[4] = altitude / 1000.0
        x[5] = (airspeed - 200.0) / 100.0
        x[6] = math.sin(math.radians(heading))
        x[7] = math.cos(math.radians(heading))
        x[8] = loc_flag; x[9] = gs_flag
        x[:N_CONT] = (x[:N_CONT] - self.standardizer.mean) / self.standardizer.std
        return x[list(self._input_indices)].astype(np.float32)

    def _build_density_now(self, ego, others) -> np.ndarray:
        if not others:
            return np.zeros(N_BINS, dtype=np.float32)
        return build_density(ego, others,
                              nm_per_pixel=self._sim.nm_per_pixel,
                              cutoff_nm=self.density_cutoff_nm)

    # ------------------------------------------------------------------ #
    # Arming new planes (post-STAR warmup, before policy takes over)
    # ------------------------------------------------------------------ #

    def _check_arm(self):
        for cs, ac in list(self._sim.aircraft_list.items()):
            if cs in self._armed:
                continue
            if cs not in self._initial_star_len:
                self._initial_star_len[cs] = len(ac.star) if ac.star else 0
                continue
            initial = self._initial_star_len[cs]
            current = len(ac.star) if ac.star else 0
            popped = initial - current
            threshold = min(self.warmup_wpts, initial) if initial > 0 else 0
            if initial == 0 or popped >= threshold:
                self._arm_plane(cs, ac)

    def _arm_plane(self, cs: str, ac) -> None:
        self._armed.add(cs)
        # Capture STAR name BEFORE wiping ac.star_name.
        star_name = getattr(ac, 'star_name', None) or 'UNKNOWN'
        if ac.star is not None or ac.target_wpt is not None:
            ac.star = None
            ac.star_name = None
            ac.target_wpt = None
        self._sim.command(cs, f"L {self.runway}")
        self._planes[cs] = _PlaneBuffer(
            star=star_name,
            callsign=cs,
            seed=self._next_seed,
            spawn_sim_time=float(self._sim.sim_time),
        )
        self._next_seed += 1

    # ------------------------------------------------------------------ #
    # Action → sim command
    # ------------------------------------------------------------------ #

    def _action_to_cmd(self, cs: str, action_4d: np.ndarray):
        sin_h, cos_h, alt_kft, spd_norm = (float(v) for v in action_4d)
        target_hdg = (math.degrees(math.atan2(sin_h, cos_h)) + 360.0) % 360.0
        target_spd_kt = max(140.0, min(280.0, float(spd_norm) * 100.0 + 200.0))
        target_alt_ft = max(1000.0, min(18000.0, float(alt_kft) * 1000.0))

        ac = self._sim.aircraft_list.get(cs)
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

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def collect(self,
                n_target: int,
                policy_state: dict,
                max_ticks: int = 30_000,
                drop_truncated: bool = True) -> tuple[list[MultiTrajectory], dict]:
        """Run the sim until `n_target` trajectories are closed.

        When `drop_truncated=True` (default), the loop only counts
        NON-TRUNCATED trajectories toward `n_target` and removes
        TRUNCATED ones from the returned list. This means rollout
        wall time goes up (need to collect more raw to hit target),
        but the training signal stays clean — TRUNCATED trajectories
        bootstrap on V(s_T) which biases the critic toward
        near-crash-site state distributions that weren't caused by
        this plane's own actions.

        Returns (trajectories, stats). `stats` includes n_crashes,
        n_sim_restarts, n_truncated_dropped, and total raw counts.
        """
        self.policy.load_state_split(policy_state, device='cpu')
        closed: list[MultiTrajectory] = []
        n_sim_restarts = 0
        self._start_sim()
        self._session_idx = 0
        # Need one initial step so the spawner places the first plane.
        self._sim.step(1.0)

        def _count_for_target():
            if drop_truncated:
                return sum(1 for t in closed if t.outcome != 'TRUNCATED')
            return len(closed)

        tick = 0
        while _count_for_target() < n_target and tick < max_ticks:
            tick += 1
            # Arm any planes ready.
            self._check_arm()
            # Build batched obs for ALL armed, currently-alive planes.
            armed_alive = [cs for cs in self._armed
                           if cs in self._sim.aircraft_list]
            full_obs_list = []
            ego_obs_list = []
            a_nm_list = []
            c_nm_list = []
            for cs in armed_alive:
                ac = self._sim.aircraft_list[cs]
                ego_obs = self._build_ego_obs(ac)
                others = [o for c2, o in self._sim.aircraft_list.items()
                          if c2 != cs]
                now_d = self._build_density_now(ac, others)
                prev_d = self._prev_density.get(cs, np.zeros_like(now_d))
                delta_d = now_d - prev_d
                self._prev_density[cs] = now_d
                full_obs = np.concatenate(
                    [ego_obs, now_d, delta_d]).astype(np.float32)
                ego_obs_list.append(ego_obs)
                full_obs_list.append(full_obs)
                a, c = self._runway_aligned(ac)
                a_nm_list.append(a); c_nm_list.append(c)

            # Sample one tick of actions for all armed planes.
            if armed_alive:
                ego_t = torch.from_numpy(np.stack(ego_obs_list))
                full_t = torch.from_numpy(np.stack(full_obs_list))
                act = self.policy.act(ego_t, full_t)
                deltas = act['action_delta'].cpu().numpy()
                finals = act['action_final'].cpu().numpy()
                log_probs = act['log_prob'].cpu().numpy()
                values = act['value'].cpu().numpy()

                # Apply commands + write target_alt direct (matches PPOEnv).
                for i, cs in enumerate(armed_alive):
                    cmd, target_alt_ft = self._action_to_cmd(cs, finals[i])
                    if cmd:
                        self._sim.command(cs, cmd)
                    ac_obj = self._sim.aircraft_list.get(cs)
                    if (ac_obj is not None
                            and not getattr(ac_obj, 'gs_intercepted', False)
                            and not getattr(ac_obj, 'on_ground', False)
                            and not getattr(ac_obj, 'landed', False)):
                        cur_ta = float(getattr(ac_obj, 'target_altitude',
                                                target_alt_ft))
                        if abs(target_alt_ft - cur_ta) > 25.0:
                            ac_obj.target_altitude = target_alt_ft
                            ac_obj.star_apply_alt = False

                    # Record into the plane's trajectory buffer.
                    buf = self._planes[cs]
                    buf.full_obs.append(full_obs_list[i])
                    buf.ego_obs.append(ego_obs_list[i])
                    buf.actions.append(deltas[i].astype(np.float32))
                    buf.log_probs.append(float(log_probs[i]))
                    buf.values.append(float(values[i]))
                    buf.a_traj.append(a_nm_list[i])
                    buf.c_traj.append(c_nm_list[i])

            # Step the sim 1 sec.
            self._sim.step(1.0)

            # Per-plane reward + termination check for armed planes.
            for i, cs in enumerate(armed_alive):
                self._post_step(cs, closed)

            # Sim-level crash → truncate everything in flight, restart.
            if self._sim.crash_occurred:
                self.n_crashes += 1
                # The 2 colliding planes (anyone with .crash set) get
                # FAILURE terminal. Anyone else in flight is TRUNCATED.
                crashed_callsigns = set()
                for cs in list(self._planes.keys()):
                    ac = self._sim.aircraft_list.get(cs)
                    if ac is not None and getattr(ac, 'crash', None):
                        crashed_callsigns.add(cs)
                for cs in list(self._planes.keys()):
                    buf = self._planes[cs]
                    if not buf.values:
                        continue   # plane that never got armed enough to act
                    if cs in crashed_callsigns:
                        # Add failure terminal reward to last reward step.
                        if buf.rewards:
                            buf.rewards[-1] += float(RZ_FAILURE_REWARD)
                            buf.rewards[-1] -= float(self.crash_extra_penalty)
                        buf.outcome = 'CRASH'
                    else:
                        buf.outcome = 'TRUNCATED'
                        buf.truncated = True
                        buf.bootstrap_value = float(buf.values[-1])
                    closed.append(self._finalize(cs))
                self._planes.clear()
                self._armed.clear()
                self._initial_star_len.clear()
                self._prev_density.clear()
                n_sim_restarts += 1
                self._restart_sim()
                self._sim.step(1.0)
                continue

        n_raw = len(closed)
        n_truncated = sum(1 for t in closed if t.outcome == 'TRUNCATED')
        if drop_truncated:
            closed = [t for t in closed if t.outcome != 'TRUNCATED']
        stats = {
            'n_trajectories': len(closed),
            'n_raw_trajectories': n_raw,
            'n_truncated_dropped': n_truncated if drop_truncated else 0,
            'n_crashes': self.n_crashes,
            'n_sim_restarts': n_sim_restarts,
            'n_ticks': tick,
        }
        # Dump the last replay too.
        self._restart_sim()
        return closed, stats

    # ------------------------------------------------------------------ #
    # Per-plane post-step bookkeeping (called after sim.step)
    # ------------------------------------------------------------------ #

    def _post_step(self, cs: str, closed: list):
        buf = self._planes.get(cs)
        if buf is None:
            return
        ac = self._sim.aircraft_list.get(cs)

        # Reward this step: everywhere_pen + collision_warning_pen.
        r = -float(_rz.EVERYWHERE_STEP_PENALTY)
        if ac is not None and getattr(ac, 'collision_warning', False):
            r -= self.collision_warning_penalty
            buf.warning_seconds += 1

        # Heading-intercept + final-turn bonuses (one-shot each per plane).
        if ac is not None:
            a_nm, c_nm = self._runway_aligned(ac)
            direction = 'NORTH' if buf.star.startswith('NORTH') else 'SOUTH'
            if not buf.turn_final_fired and buf.actions:
                last_action = buf.actions[-1]
                sin_h = float(last_action[0]); cos_h = float(last_action[1])
                target_h_deg = (math.degrees(math.atan2(sin_h, cos_h))
                                 + 360.0) % 360.0
                bonus = final_turn_action_reward(
                    buf.star, a_nm, c_nm, target_h_deg)
                if bonus > 0.0:
                    r += float(bonus)
                    buf.turn_final_fired = True

        # ---- Per-plane termination ----
        # 1. Plane removed by sim (landed OR improper_exit).
        if ac is None:
            # The recorder's removal_terminal dict would tell us why,
            # but it's not directly exposed here. Heuristic: if the
            # sim's num_landed bumped this tick we treat as LANDED;
            # else IMPROPER_EXIT. Simpler: stash the prior state and
            # check `landed`/`on_ground` from the LAST tick we saw.
            #
            # Both outcomes use a terminal reward already added below.
            buf.rewards.append(r)   # last step's r (before terminal)
            # We need to know if this was a successful landing or an
            # exit. The plane is gone now; check the sim's counters
            # change vs what we last saw. Easier: stash the plane
            # object in the buffer at each tick and check `landed`.
            # Since we already lost the ref, infer via outcome heuristic:
            outcome = self._infer_disappearance_outcome(cs)
            terminal = (RZ_SUCCESS_REWARD if outcome == 'LANDED'
                        else RZ_FAILURE_REWARD)
            # Heading intercept bonus on terminal (if landed cleanly).
            if outcome == 'LANDED':
                buf.rewards[-1] += float(terminal)
            else:
                buf.rewards[-1] += float(RZ_FAILURE_REWARD)
            buf.outcome = outcome
            self._prev_density.pop(cs, None)
            closed.append(self._finalize(cs))
            del self._planes[cs]
            self._armed.discard(cs)
            return

        # 2. LOC firing — categorize and possibly terminate.
        loc_on = bool(getattr(ac, 'loc_intercepted', False))
        if loc_on and not buf.loc_captured:
            buf.loc_captured = True
            # Categorize the capture: BELOW_GS (good), ABOVE_GS (fail),
            # BEHIND_THR (fail).
            d_thr_nm, gs_alt_ft, capturable = self._gs_capturable(ac)
            a_along, _c = self._runway_aligned(ac)
            hb = heading_intercept_reward(
                'NORTH' if buf.star.startswith('NORTH') else 'SOUTH',
                float(ac.heading))
            if a_along < 0.0:
                outcome = 'LOC_BEHIND_THR'
                terminal_r = float(RZ_FAILURE_REWARD)
                buf.rewards.append(r + terminal_r + float(hb))
                buf.outcome = outcome
                buf.d_thr_nm = d_thr_nm
                buf.altitude_ft = float(ac.altitude)
                buf.gs_alt_ft = gs_alt_ft
                self._wipe_plane(cs)
                closed.append(self._finalize(cs))
                return
            if not capturable:
                outcome = 'LOC_ABOVE_GS'
                terminal_r = float(RZ_FAILURE_REWARD)
                buf.rewards.append(r + terminal_r + float(hb))
                buf.outcome = outcome
                buf.d_thr_nm = d_thr_nm
                buf.altitude_ft = float(ac.altitude)
                buf.gs_alt_ft = gs_alt_ft
                self._wipe_plane(cs)
                closed.append(self._finalize(cs))
                return
            # LOC_BELOW_GS: keep going — episode continues until LANDED.
            # Apply the heading bonus now (one-shot at capture), no
            # terminal reward yet.
            r += float(hb)
            buf.d_thr_nm = d_thr_nm
            buf.altitude_ft = float(ac.altitude)
            buf.gs_alt_ft = gs_alt_ft

        # 3. Pre-LOC step-cap (TIMEOUT). Post-LOC ticks don't count.
        if not buf.loc_captured:
            buf.pre_loc_steps += 1
            cap = (self.max_steps_3 if buf.star.endswith('3')
                   else self.max_steps_1_2)
            if buf.pre_loc_steps >= cap:
                buf.rewards.append(r + float(RZ_FAILURE_REWARD))
                buf.outcome = 'TIMEOUT'
                self._wipe_plane(cs)
                closed.append(self._finalize(cs))
                return

        # Normal continuing step.
        buf.rewards.append(r)

    def _wipe_plane(self, cs: str):
        """Remove a failed-but-still-flying plane from the sim."""
        if self._sim is not None and cs in self._sim.aircraft_list:
            del self._sim.aircraft_list[cs]
        self._armed.discard(cs)
        self._initial_star_len.pop(cs, None)
        self._prev_density.pop(cs, None)
        # Don't delete from self._planes — _finalize needs it. Caller
        # deletes after finalize.

    def _infer_disappearance_outcome(self, cs: str) -> str:
        """Plane just vanished from sim.aircraft_list. The sim only
        removes for landed or improper_exit. We don't have direct
        access, so we use a tracked flag set on prior ticks."""
        # The buffer can stash the last seen state in a `_last_landed`
        # field. We don't currently — quick & dirty: assume IMPROPER_EXIT
        # unless we infer from sim counters. Best-effort heuristic:
        # since we increment `num_landed` only on LANDED removal, we
        # can compare counters before/after sim.step in the main loop
        # (more invasive). For now: check the sim's num_landed delta.
        # Since we don't track that here either, fall back to the
        # buffer's last "loc_captured" + "low altitude" hint.
        buf = self._planes.get(cs)
        if buf is None:
            return 'IMPROPER_EXIT'
        # If we'd just captured LOC_BELOW_GS and the plane disappeared,
        # it most likely landed cleanly. Otherwise improper exit.
        if buf.loc_captured:
            return 'LANDED'
        return 'IMPROPER_EXIT'

    def _finalize(self, cs: str) -> MultiTrajectory:
        buf = self._planes[cs]
        # Build done mask: True at terminal, False if truncated.
        T = len(buf.actions)
        if T == 0:
            # Empty traj (rare edge case from immediate crash).
            return MultiTrajectory(
                obs=np.zeros((0, self._full_dim), dtype=np.float32),
                ego_obs=np.zeros((0, self._ego_dim), dtype=np.float32),
                actions=np.zeros((0, 3), dtype=np.float32),
                log_probs=np.zeros(0, dtype=np.float32),
                rewards=np.zeros(0, dtype=np.float32),
                values=np.zeros(0, dtype=np.float32),
                dones=np.zeros(0, dtype=bool),
                star=buf.star, outcome=buf.outcome, steps=0,
                seed=buf.seed, callsign=cs,
                warning_seconds=buf.warning_seconds,
                truncated=buf.truncated,
                bootstrap_value=buf.bootstrap_value,
            )
        # Some rewards may be missing if termination fired before we
        # appended one this tick. Pad with 0 to align length.
        while len(buf.rewards) < T:
            buf.rewards.append(0.0)
        dones = np.zeros(T, dtype=bool)
        if not buf.truncated:
            dones[-1] = True
        a_traj = np.asarray(buf.a_traj[:T], dtype=np.float32)
        c_traj = np.asarray(buf.c_traj[:T], dtype=np.float32)

        # Loop penalty (post-hoc, only on landed/failed — skip if truncated).
        rewards_arr = np.asarray(buf.rewards[:T], dtype=np.float32)
        n_loop_steps = n_penalized = 0
        loop_penalty_total = 0.0
        if self.loop_pps > 0.0 and not buf.truncated and a_traj.size >= 2:
            from rl_ppo.loop_detector import detect_looping
            d = detect_looping(
                a_traj, c_traj,
                prox_radius_nm=self.loop_prox_radius_nm,
                min_gap_steps=self.loop_min_gap_steps,
                min_detour_nm=self.loop_min_detour_nm,
                min_loop_frac=0.0,
            )
            mask = d['looping_step_mask']
            if mask.any():
                loop_indices = np.flatnonzero(mask)
                k = int(loop_indices.size)
                half = (k + 1) // 2
                penalize = loop_indices[half:]
                n_loop_steps = k
                n_penalized = int(penalize.size)
                if n_penalized > 0:
                    rewards_arr[penalize] -= self.loop_pps
                    loop_penalty_total = float(self.loop_pps * n_penalized)

        return MultiTrajectory(
            obs=np.stack(buf.full_obs[:T]).astype(np.float32),
            ego_obs=np.stack(buf.ego_obs[:T]).astype(np.float32),
            actions=np.stack(buf.actions[:T]).astype(np.float32),
            log_probs=np.asarray(buf.log_probs[:T], dtype=np.float32),
            rewards=rewards_arr,
            values=np.asarray(buf.values[:T], dtype=np.float32),
            dones=dones,
            star=buf.star, outcome=buf.outcome, steps=T,
            seed=buf.seed, callsign=cs,
            d_thr_nm=buf.d_thr_nm,
            altitude_ft=buf.altitude_ft,
            gs_alt_ft=buf.gs_alt_ft,
            a_traj=a_traj, c_traj=c_traj,
            n_loop_steps=n_loop_steps,
            n_penalized=n_penalized,
            loop_penalty_total=loop_penalty_total,
            warning_seconds=buf.warning_seconds,
            truncated=buf.truncated,
            bootstrap_value=buf.bootstrap_value,
        )
