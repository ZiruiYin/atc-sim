"""bc_fm rollout — load a bc_fm checkpoint and drive the simulator per tick.

The only place this package talks to the simulator. Per-tick flow:

  1. Build the 10-dim feature vector for each live aircraft.
  2. Standardize the continuous cols with the checkpoint's mean/std.
  3. Encode position → c (FM condition).
  4. Sample a 4-D *standardized* action via Euler ODE with a per-aircraft
     seeded x_0 (same x_0 every tick → stable mode commitment).
  5. Un-standardize with the model's stored target mean/std.
  6. `atan2` the first two dims for target_heading_deg; 3rd = target_alt_kft;
     4th = target_spd_norm.
  7. Emit `C XXX` (heading, 1° res) if diff ≥ 1° AND loc==0.
  8. Emit `S NNN` (speed, snapped to 10 kt, clamped [140, 280]) if diff > 5 kt.
  9. Direct-write `aircraft.target_altitude = pred_ft` (clamped, 25-ft dedup).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rl_bc.config import N_CONT, N_FEATURES
from rl_bc.data import Standardizer, load_runway_geometry
from rl_bc.bc_fm.model import build_from_saved


# --------------------------------------------------------------------------- #
# Per-aircraft state
# --------------------------------------------------------------------------- #


@dataclass
class AircraftState:
    cleared: bool = False
    fm_seed: int | None = None


# --------------------------------------------------------------------------- #
# Command-string formatters
# --------------------------------------------------------------------------- #


def _format_heading_cmd(deg_int: int) -> str:
    h = deg_int % 360
    if h == 0:
        h = 360
    return f"C {h:03d}"


def _snap_speed_kt(kt: float) -> int:
    return int(round(kt / 10.0)) * 10


def _format_speed_cmd(kt: float) -> str:
    return f"S {max(140, min(280, _snap_speed_kt(kt)))}"


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #


class Runtime:
    """Load a bc_fm checkpoint and drive the sim per tick."""

    def __init__(self, ckpt_path: str | Path,
                 device: torch.device | str = 'cpu',
                 alt_floor_ft: float = 1000.0,
                 runway: str = '27',
                 issue_speed: bool = True):
        self.device = torch.device(device)
        self.alt_floor_ft = float(alt_floor_ft)
        self.runway = runway
        self.issue_speed = issue_speed

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        saved = ckpt['config']

        airport = saved.get('airport_name', 'test')
        radar_side = int(saved.get('radar_side', 800))
        nm_range = int(saved.get('nm_range', 60))
        self.geom = load_runway_geometry(airport, runway, radar_side, nm_range)

        self.standardizer = Standardizer(
            mean=np.asarray(ckpt['standardizer_mean'], dtype=np.float32),
            std=np.asarray(ckpt['standardizer_std'], dtype=np.float32),
        )

        self.model = build_from_saved(saved).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        self.fm_n_steps = int(saved.get('fm_n_steps', 10))

        self._states: dict[str, AircraftState] = {}

    def state_for(self, callsign: str) -> AircraftState:
        st = self._states.get(callsign)
        if st is None:
            st = AircraftState()
            self._states[callsign] = st
        return st

    def forget(self, callsign: str) -> None:
        self._states.pop(callsign, None)

    def reset(self) -> None:
        self._states.clear()

    # ---------------- primitives ---------------- #

    def encode_state(self, ac: dict, nm_per_pixel: float,
                     airport_x: float, airport_y: float) -> np.ndarray:
        """Build the standardized 10-dim feature vector for one aircraft."""
        x_nm = (ac['x'] - airport_x) * nm_per_pixel
        y_nm = -(ac['y'] - airport_y) * nm_per_pixel

        phi = math.radians((self.geom.course_deg + 180.0) % 360.0)
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        dx = x_nm - self.geom.thr_x_nm
        dy = y_nm - self.geom.thr_y_nm
        a_nm = dx * sin_phi + dy * cos_phi
        c_nm = -dx * cos_phi + dy * sin_phi
        d_thr = math.sqrt(a_nm * a_nm + c_nm * c_nm)

        heading = float(ac['heading'])
        altitude = float(ac['altitude'])
        airspeed = float(ac['airspeed'])
        loc = 1.0 if ac.get('loc') else 0.0
        gs = 1.0 if ac.get('gs') else 0.0
        dtheta = ((heading - self.geom.course_deg + 540.0) % 360.0) - 180.0

        x = np.zeros(N_FEATURES, dtype=np.float32)
        x[0] = a_nm; x[1] = c_nm; x[2] = d_thr
        x[3] = dtheta / 180.0
        x[4] = altitude / 1000.0
        x[5] = (airspeed - 200.0) / 100.0
        x[6] = math.sin(math.radians(heading))
        x[7] = math.cos(math.radians(heading))
        x[8] = loc; x[9] = gs

        x[:N_CONT] = (x[:N_CONT] - self.standardizer.mean) / self.standardizer.std
        return x

    def predict(self, x: np.ndarray,
                fm_generator: torch.Generator | None = None) -> dict:
        """Joint FM sample → un-standardize → 4-D action vector → physical units."""
        if x.ndim == 1:
            x = x[None, :]
        with torch.no_grad():
            t = torch.from_numpy(x).to(self.device)
            c = self.model.encode(t)
            sampled_std = self.model.sample(
                c, n_steps=self.fm_n_steps, generator=fm_generator)
            sampled = self.model.unstandardize_sample(sampled_std)
            sin_h, cos_h, alt_kft, spd_norm = sampled[0].cpu().numpy().tolist()
            target_hdg_deg = (math.degrees(math.atan2(sin_h, cos_h)) + 360.0) % 360.0
            return {
                'target_hdg_deg': float(target_hdg_deg),
                'target_alt_kft': float(alt_kft),
                'target_spd_kt':  float(spd_norm) * 100.0 + 200.0,
            }

    def translate(self, ac: dict, actions: dict,
                  st: AircraftState) -> Optional[str]:
        if ac.get('landed') or ac.get('on_ground'):
            return None

        loc_on = bool(ac.get('loc'))
        target_heading = float(ac.get('target_heading', ac['heading']))
        target_airspeed = float(ac.get('target_airspeed', ac['airspeed']))

        parts: list[str] = []

        if not loc_on:
            tgt = int(round(actions['target_hdg_deg'])) % 360
            diff = abs(((tgt - target_heading + 540.0) % 360.0) - 180.0)
            if diff >= 1.0:
                parts.append(_format_heading_cmd(tgt))

        if self.issue_speed:
            tgt_kt = max(140, min(280, _snap_speed_kt(actions['target_spd_kt'])))
            if abs(tgt_kt - target_airspeed) > 5.0:
                parts.append(_format_speed_cmd(actions['target_spd_kt']))

        return " ".join(parts) if parts else None

    def tick(self, sim, armed: Optional[set] = None) -> list[dict]:
        env_state = sim.get_state()
        static = env_state['static']
        nm_per_pixel = static['nm_per_pixel']
        airport_x = sim.airport_x
        airport_y = sim.airport_y

        report = []
        for ac in env_state['aircraft']:
            cs = ac['callsign']
            if armed is not None and cs not in armed:
                continue
            st = self.state_for(cs)

            x = self.encode_state(ac, nm_per_pixel, airport_x, airport_y)

            if st.fm_seed is None:
                st.fm_seed = hash(cs) & 0x7FFFFFFF
            fm_gen = torch.Generator(device=self.device)
            fm_gen.manual_seed(st.fm_seed)

            actions = self.predict(x, fm_generator=fm_gen)
            cmd = self.translate(ac, actions, st)

            sim_result = None
            if cmd:
                sim_result = sim.command(cs, cmd)

            ac_obj = sim.aircraft_list.get(cs)
            if (ac_obj is not None
                    and not getattr(ac_obj, 'gs_intercepted', False)
                    and not getattr(ac_obj, 'on_ground', False)
                    and not getattr(ac_obj, 'landed', False)):
                pred_ft = float(actions['target_alt_kft']) * 1000.0
                pred_ft = max(self.alt_floor_ft, min(18000.0, pred_ft))
                if abs(pred_ft - float(getattr(ac_obj, 'target_altitude', pred_ft))) > 25.0:
                    ac_obj.target_altitude = pred_ft
                    ac_obj.star_apply_alt = False

            report.append({
                'callsign': cs, 'actions': actions,
                'cmd': cmd, 'sim_result': sim_result,
            })

        live = {ac['callsign'] for ac in env_state['aircraft']}
        for dead in [c for c in self._states if c not in live]:
            del self._states[dead]
        return report
