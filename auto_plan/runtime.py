"""Self-contained inference runtime for the AUTO planner.

Loads the continuous_03 PPO checkpoint (`best.pt`, actor weights) and drives
the simulator one tick at a time: encode the 7-feature state -> sample the GMM
-> translate to sim commands. Ported from the rl branch's
`rl_bc/bc_gmm/rollout.Runtime` + `rl_multiple/runtime.Runtime` and trimmed to
the inference path only.

The architecture (encoder 7->64->64, 4 components) is read from the
checkpoint's `actor_state` shapes via `policy_config.json`; the input
standardizer (6 continuous features) and `input_indices` are baked into that
config (the BC seed they came from is not needed at runtime).
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from auto_plan.model import build_actor

# Feature layout matches the trained model (see rl_bc/data.py):
#   0 a_nm  1 c_nm  2 d_thr_nm  3 (hdg-course)/180  4 alt/1000
#   5 (ias-200)/100  6 sin(hdg)  7 cos(hdg)  8 loc  9 gs
N_CONT = 6        # first 6 features are standardized
N_FEATURES = 10

_CFG_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunwayGeometry:
    thr_x_nm: float
    thr_y_nm: float
    course_deg: float


def load_runway_geometry(airport_name: str, runway: str,
                         radar_side: int = 800,
                         nm_range: int = 60) -> RunwayGeometry:
    """Threshold position (airport-centered nm) + landing course. Computed
    from the live airport data so it stays correct if the data changes."""
    from environment.display.generate_game_coordinates import generate_game_coordinates

    data = generate_game_coordinates(radar_side, radar_side, nm_range, airport_name)
    nmpp = data['screen_info']['nm_per_pixel']
    ax = data['airport']['coordinates']['x']
    ay = data['airport']['coordinates']['y']

    thr_px = None
    other_px = None
    for pair_data in data['runways'].values():
        thresholds = pair_data['thresholds']
        if runway in thresholds:
            thr_px = thresholds[runway]
            for end_id, end_data in thresholds.items():
                if end_id != runway:
                    other_px = end_data
                    break
            break
    if thr_px is None:
        raise ValueError(f"runway {runway!r} not found in {airport_name!r} data")

    def px_to_nm(p):
        return (p['x'] - ax) * nmpp, -(p['y'] - ay) * nmpp

    thr_x_nm, thr_y_nm = px_to_nm(thr_px)
    other_x_nm, other_y_nm = px_to_nm(other_px) if other_px else (thr_x_nm, thr_y_nm)
    dx = other_x_nm - thr_x_nm
    dy = other_y_nm - thr_y_nm
    course = math.degrees(math.atan2(dx, dy)) % 360.0
    return RunwayGeometry(thr_x_nm=thr_x_nm, thr_y_nm=thr_y_nm, course_deg=course)


def _stable_seed(callsign: str) -> int:
    """Process-stable per-callsign seed. `hash()` is randomized across worker
    processes (PYTHONHASHSEED), which would make rollouts non-reproducible; a
    crc32 of the name is identical in every process."""
    return zlib.crc32(callsign.encode('utf-8')) & 0x7FFFFFFF


@dataclass
class _CallsignState:
    cleared: bool = False
    fm_seed: int | None = None


class Runtime:
    """Load the GMM actor and drive armed aircraft per tick."""

    def __init__(self, ckpt_path: str | Path | None = None,
                 config_path: str | Path | None = None,
                 device: str = 'cpu',
                 runway: str | None = None,
                 alt_floor_ft: float | None = None,
                 issue_speed: bool = True,
                 deterministic: bool = False):
        self.device = torch.device(device)
        cfg_path = Path(config_path) if config_path else (_CFG_DIR / 'policy_config.json')
        self.config = json.loads(Path(cfg_path).read_text())

        self.runway = runway or self.config.get('runway', '27')
        self.alt_floor_ft = float(alt_floor_ft if alt_floor_ft is not None
                                  else self.config.get('alt_floor_ft', 1000.0))
        self.issue_speed = bool(issue_speed)
        self.deterministic = bool(deterministic)

        ck = Path(ckpt_path) if ckpt_path else (_CFG_DIR / self.config['ckpt'])
        blob = torch.load(ck, map_location=self.device, weights_only=False)
        state = blob[self.config.get('state_key', 'actor_state')]

        self.model = build_actor(
            input_indices=self.config['input_indices'],
            hidden=self.config['hidden'],
            n_components=self.config['n_components'],
            dropout=self.config.get('dropout', 0.1),
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self._mean = np.asarray(self.config['standardizer_mean'], dtype=np.float32)
        self._std = np.asarray(self.config['standardizer_std'], dtype=np.float32)

        self.geom = load_runway_geometry(
            self.config.get('airport_name', 'test'), self.runway,
            int(self.config.get('radar_side', 800)),
            int(self.config.get('nm_range', 60)))

        self._iter = blob.get('iter')
        self._ckpt_path = str(ck)
        self._states: dict[str, _CallsignState] = {}

    # ---------------- per-callsign state ---------------- #

    def state_for(self, callsign: str) -> _CallsignState:
        st = self._states.get(callsign)
        if st is None:
            st = _CallsignState()
            self._states[callsign] = st
        return st

    def forget(self, callsign: str) -> None:
        self._states.pop(callsign, None)

    def reset(self) -> None:
        self._states.clear()

    # ---------------- primitives ---------------- #

    def encode_state(self, ac: dict, nm_per_pixel: float,
                     airport_x: float, airport_y: float) -> np.ndarray:
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

        x[:N_CONT] = (x[:N_CONT] - self._mean) / self._std
        return x

    def predict(self, x: np.ndarray,
                generator: torch.Generator | None = None) -> dict:
        if x.ndim == 1:
            x = x[None, :]
        with torch.no_grad():
            t = torch.from_numpy(x).to(self.device)
            c = self.model.encode(t)
            sampled = self.model.sample(
                c, generator=generator, deterministic=self.deterministic)
            sin_h, cos_h, alt_kft, spd_norm = sampled[0].cpu().numpy().tolist()
            target_hdg_deg = (math.degrees(math.atan2(sin_h, cos_h)) + 360.0) % 360.0
            return {
                'target_hdg_deg': float(target_hdg_deg),
                'target_alt_kft': float(alt_kft),
                'target_spd_kt': float(spd_norm) * 100.0 + 200.0,
            }

    def translate(self, ac: dict, actions: dict) -> str | None:
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
                h = tgt if tgt != 0 else 360
                parts.append(f"C {h:03d}")
        if self.issue_speed:
            tgt_kt = max(140, min(280, int(round(actions['target_spd_kt'] / 10.0)) * 10))
            if abs(tgt_kt - target_airspeed) > 5.0:
                parts.append(f"S {tgt_kt}")
        return " ".join(parts) if parts else None

    def tick(self, sim, armed: set | None = None) -> list[dict]:
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
                st.fm_seed = _stable_seed(cs)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(st.fm_seed)

            actions = self.predict(x, generator=gen)
            cmd = self.translate(ac, actions)

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

            report.append({'callsign': cs, 'actions': actions,
                           'cmd': cmd, 'sim_result': sim_result})

        live = {ac['callsign'] for ac in env_state['aircraft']}
        for dead in [c for c in self._states if c not in live]:
            del self._states[dead]
        return report
