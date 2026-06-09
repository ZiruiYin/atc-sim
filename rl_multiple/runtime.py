"""Multi-plane PPO runtime — load a PPO actor and drive every spawned
aircraft per tick.

Subclasses `rl_bc.bc_gmm.rollout.Runtime` so the model logic (encode →
sample → translate to sim commands) is shared verbatim. Only the
constructor differs: PPO checkpoints store the actor weights under
`actor_state` (not `model_state`), and the standardizer + arch config
live in the BC seed the run started from. We resolve that seed via the
PPO ckpt's `config.actor_ckpt` field (same approach as
`rl_ppo.eval_runner._wrap_ppo_to_bc`) so callers only need to point at
the PPO ckpt.

Sampling matches single-plane: per-aircraft `torch.Generator` re-seeded
every tick → frozen noise per callsign. With multiple planes live the
fm_seed is `hash(callsign)`, so different aircraft commit to different
mixture-noise patterns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rl_bc.bc_gmm.model import build_from_saved
from rl_bc.bc_gmm.rollout import Runtime as _BCRuntime
from rl_bc.data import Standardizer, load_runway_geometry


def resolve_bc_seed_path(ppo_blob: dict) -> Path:
    """Find the BC seed ckpt referenced by a PPO ckpt's config.

    The PPO ckpt's `config.actor_ckpt` carries the seed path. Modal jobs
    write that as an absolute container path (`/root/...`); local runs
    write a repo-relative path. Try the path verbatim first, then fall
    back to `rl_bc/runs/...` based on the tail past the last `/runs/`.
    """
    seed_path_str = ppo_blob.get('config', {}).get('actor_ckpt', '')
    if not seed_path_str:
        raise RuntimeError(
            "PPO ckpt has no `config.actor_ckpt` field — can't recover "
            "the BC seed needed for standardizer + arch config."
        )
    p = Path(seed_path_str)
    if p.exists():
        return p
    s_norm = seed_path_str.replace('\\', '/')
    for marker in ('/_modal/runs/', '/runs/'):
        if marker in s_norm:
            tail = 'rl_bc/runs/' + s_norm.split(marker, 1)[1]
            cand = Path(tail)
            if cand.exists():
                return cand
    raise FileNotFoundError(
        f"could not resolve BC seed referenced by PPO ckpt: {seed_path_str}"
    )


class Runtime(_BCRuntime):
    """PPO actor on top of the BC GMM architecture.

    Same interface as `rl_bc.bc_gmm.rollout.Runtime` — `tick(sim)` drives
    every armed aircraft each call, and `predict / log_prob / translate`
    behave identically. The differences vs the BC runtime are isolated
    to `__init__`:

      - reads `actor_state` from the PPO ckpt
      - pulls standardizer + arch config + runway geometry from the BC
        seed referenced by `config.actor_ckpt`
    """

    def __init__(self, ckpt_path: str | Path,
                 bc_seed_path: str | Path | None = None,
                 device: torch.device | str = 'cpu',
                 alt_floor_ft: float = 1000.0,
                 runway: str = '27',
                 issue_speed: bool = True,
                 deterministic: bool = False):
        # NOTE: don't call super().__init__ — its loading logic assumes
        # the BC ckpt format. Replicate the field set it would have set.
        self.device = torch.device(device)
        self.alt_floor_ft = float(alt_floor_ft)
        self.runway = runway
        self.issue_speed = issue_speed
        self.deterministic = bool(deterministic)

        ppo_blob = torch.load(ckpt_path, map_location=self.device,
                              weights_only=False)
        if 'actor_state' not in ppo_blob:
            raise ValueError(
                f"{ckpt_path}: not a PPO checkpoint "
                f"(no `actor_state`; keys={list(ppo_blob.keys())[:6]})"
            )

        seed_path = (Path(bc_seed_path) if bc_seed_path
                     else resolve_bc_seed_path(ppo_blob))
        seed_blob = torch.load(seed_path, map_location=self.device,
                               weights_only=False)
        saved = seed_blob['config']

        airport = saved.get('airport_name', 'test')
        radar_side = int(saved.get('radar_side', 800))
        nm_range = int(saved.get('nm_range', 60))
        self.geom = load_runway_geometry(airport, runway, radar_side,
                                         nm_range)

        self.standardizer = Standardizer(
            mean=np.asarray(seed_blob['standardizer_mean'],
                            dtype=np.float32),
            std=np.asarray(seed_blob['standardizer_std'],
                           dtype=np.float32),
        )

        self.model = build_from_saved(saved).to(self.device)
        self.model.load_state_dict(ppo_blob['actor_state'])
        self.model.eval()

        # Match _BCRuntime's per-callsign state dict (used by tick()).
        self._states = {}

        self._ppo_iter = ppo_blob.get('iter')
        self._ppo_ckpt_path = str(ckpt_path)
        self._bc_seed_path = str(seed_path)


# --------------------------------------------------------------------------- #
# MultiRuntime — frozen GMM + trained radar head (delta head + critic) for
# the watch. Replaces the single-GMM `Runtime` when serving a multi-PPO
# checkpoint that has `delta_head_state` + `critic_state`.
# --------------------------------------------------------------------------- #


class MultiRuntime:
    """Drives every armed aircraft per tick using the combined policy:
    frozen GMM (sampled stochastically) + radar head (Gaussian Δ).

    Per tick, for each armed plane:
      1. Build ego_obs (7-D, same encoding as `Runtime`)
      2. Compute density_now (36-D) + density_delta (36-D) for this plane
      3. CombinedPolicy.act → 4-D sim action (sin θ, cos θ, alt_kft,
         spd_norm) = frozen GMM mode + sampled Δ
      4. Translate (sin θ, cos θ, spd_norm) → C/S sim commands;
         write alt_kft directly to ac.target_altitude

    Matches `Runtime.tick`'s output shape so `watch.py` can swap them
    interchangeably.
    """

    def __init__(self,
                 multi_ckpt_path,
                 ppo_seed_ckpt_path,
                 bc_seed_path=None,
                 *,
                 device='cpu',
                 alt_floor_ft=1000.0,
                 runway='27',
                 issue_speed=True,
                 deterministic_base=False,
                 density_cutoff_nm=10.0):
        import math
        from rl_multiple.density import DensityCache, N_BINS
        from rl_multiple.policy import CombinedPolicy

        self.device = torch.device(device)
        self.alt_floor_ft = float(alt_floor_ft)
        self.runway = runway
        self.issue_speed = issue_speed
        self.deterministic_base = bool(deterministic_base)
        self._N_BINS = N_BINS

        # Resolve BC seed (for standardizer + geometry) from the PPO
        # seed ckpt — same logic as `Runtime` uses.
        ppo_seed_blob = torch.load(ppo_seed_ckpt_path,
                                    map_location=self.device,
                                    weights_only=False)
        seed_path = (Path(bc_seed_path) if bc_seed_path
                     else resolve_bc_seed_path(ppo_seed_blob))
        seed_blob = torch.load(seed_path, map_location=self.device,
                               weights_only=False)
        saved = seed_blob['config']

        airport = saved.get('airport_name', 'test')
        radar_side = int(saved.get('radar_side', 800))
        nm_range = int(saved.get('nm_range', 60))
        self.geom = load_runway_geometry(airport, runway, radar_side,
                                          nm_range)
        self.standardizer = Standardizer(
            mean=np.asarray(seed_blob['standardizer_mean'], dtype=np.float32),
            std=np.asarray(seed_blob['standardizer_std'], dtype=np.float32),
        )
        self._input_indices = tuple(int(i) for i in
                                    saved.get('input_indices', (0, 1, 2)))
        self._ego_dim = len(self._input_indices)

        # Build CombinedPolicy: frozen GMM weights from ppo_seed_ckpt,
        # delta head + critic weights from the multi-PPO ckpt.
        self.policy = CombinedPolicy.from_ppo_ckpt(
            ppo_seed_ckpt_path,
            bc_seed_path=seed_path,
            density_n_bins=N_BINS,
            device=self.device,
        )
        multi_blob = torch.load(multi_ckpt_path, map_location=self.device,
                                weights_only=False)
        if 'delta_head_state' not in multi_blob:
            raise ValueError(
                f"{multi_ckpt_path}: not a multi-PPO ckpt "
                f"(no delta_head_state; keys={list(multi_blob.keys())[:6]})"
            )
        self.policy.delta_head.load_state_dict(multi_blob['delta_head_state'])
        if 'critic_state' in multi_blob:
            self.policy.critic.load_state_dict(multi_blob['critic_state'])
        self.policy.eval()

        self._density_cache = DensityCache(cutoff_nm=density_cutoff_nm)
        self._cs_state: dict = {}

        # For status reporting in watch banner.
        self._multi_iter = multi_blob.get('iter')
        self._multi_ckpt_path = str(multi_ckpt_path)
        self._ppo_seed_ckpt_path = str(ppo_seed_ckpt_path)
        self._bc_seed_path = str(seed_path)

    # ---------------- per-callsign state for parity with bc_gmm.Runtime
    # The watch's _arm_ppo does `runtime.state_for(cs).cleared = True`.
    # Returning a fresh stub each call would discard the flag — keep one
    # stable object per callsign.
    class _CallsignState:
        __slots__ = ('cleared',)
        def __init__(self):
            self.cleared = False

    def state_for(self, callsign):
        st = self._cs_state.get(callsign)
        if st is None:
            st = MultiRuntime._CallsignState()
            self._cs_state[callsign] = st
        return st

    def forget(self, callsign):
        self._cs_state.pop(callsign, None)
        self._density_cache.forget(callsign)

    def reset(self):
        self._cs_state.clear()
        self._density_cache.reset()

    # ---------------- encoding helpers (mirror Runtime / PPOEnv) ----------
    def _build_ego_obs(self, ac_dict, nm_per_pixel, airport_x, airport_y):
        """Same as Runtime/PPOEnv ego encoding — returns standardized
        7-D feature vector (the input_indices slice)."""
        import math
        from rl_bc.config import N_CONT, N_FEATURES
        x_nm = (ac_dict['x'] - airport_x) * nm_per_pixel
        y_nm = -(ac_dict['y'] - airport_y) * nm_per_pixel
        phi = math.radians((self.geom.course_deg + 180.0) % 360.0)
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        dx = x_nm - self.geom.thr_x_nm
        dy = y_nm - self.geom.thr_y_nm
        a_nm = dx * sin_phi + dy * cos_phi
        c_nm = -dx * cos_phi + dy * sin_phi
        d_thr = math.sqrt(a_nm * a_nm + c_nm * c_nm)
        heading = float(ac_dict['heading'])
        altitude = float(ac_dict['altitude'])
        airspeed = float(ac_dict['airspeed'])
        loc = 1.0 if ac_dict.get('loc') else 0.0
        gs = 1.0 if ac_dict.get('gs') else 0.0
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
        return x[list(self._input_indices)].astype(np.float32)

    def _translate(self, ac_dict, action_4d):
        """4-D sim action → (cmd_string or None, target_alt_ft)."""
        import math
        sin_h, cos_h, alt_kft, spd_norm = (float(v) for v in action_4d)
        target_hdg = (math.degrees(math.atan2(sin_h, cos_h)) + 360.0) % 360.0
        target_spd_kt = max(140.0, min(280.0, float(spd_norm) * 100.0 + 200.0))
        target_alt_ft = max(self.alt_floor_ft, min(18000.0, float(alt_kft) * 1000.0))

        loc_on = bool(ac_dict.get('loc'))
        target_heading = float(ac_dict.get('target_heading', ac_dict['heading']))
        target_airspeed = float(ac_dict.get('target_airspeed', ac_dict['airspeed']))

        parts = []
        if not loc_on:
            tgt = int(round(target_hdg)) % 360
            diff = abs(((tgt - target_heading + 540.0) % 360.0) - 180.0)
            if diff >= 1.0:
                h = tgt if tgt != 0 else 360
                parts.append(f"C {h:03d}")
        if self.issue_speed:
            tgt_kt = max(140, min(280, int(round(target_spd_kt / 10.0)) * 10))
            if abs(tgt_kt - target_airspeed) > 5.0:
                parts.append(f"S {tgt_kt}")
        return (" ".join(parts) if parts else None), target_alt_ft

    # ---------------- tick: drive all armed planes ----------------
    @torch.no_grad()
    def tick(self, sim, armed=None):
        env_state = sim.get_state()
        static = env_state['static']
        nm_per_pixel = static['nm_per_pixel']
        airport_x = sim.airport_x
        airport_y = sim.airport_y

        # Build density for all live planes (using cache for delta tracking).
        density_map = self._density_cache.update(sim)

        report = []
        for ac in env_state['aircraft']:
            cs = ac['callsign']
            if armed is not None and cs not in armed:
                continue

            ego = self._build_ego_obs(ac, nm_per_pixel, airport_x, airport_y)
            d = density_map.get(cs, None)
            if d is None:
                # plane vanished from cache (shouldn't happen mid-tick); fall
                # back to zeros
                now = np.zeros(self._N_BINS, dtype=np.float32)
                delta = np.zeros(self._N_BINS, dtype=np.float32)
            else:
                now = d['now']
                delta = d['delta']
            full = np.concatenate([ego, now, delta]).astype(np.float32)

            ego_t = torch.from_numpy(ego).unsqueeze(0)
            full_t = torch.from_numpy(full).unsqueeze(0)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(hash(cs) & 0x7FFFFFFF)
            out = self.policy.act(ego_t, full_t, generator=gen,
                                   deterministic_base=self.deterministic_base)
            final = out['action_final'].squeeze(0).cpu().numpy()
            delta_3d = out['action_delta'].squeeze(0).cpu().numpy()

            # Translate to sim command (heading + speed) and apply.
            import math
            sin_h, cos_h, alt_kft, spd_norm = final
            target_hdg_deg = (math.degrees(math.atan2(float(sin_h),
                                                       float(cos_h)))
                              + 360.0) % 360.0
            cmd, target_alt_ft = self._translate(ac, final)
            sim_result = None
            if cmd:
                sim_result = sim.command(cs, cmd)

            ac_obj = sim.aircraft_list.get(cs)
            if (ac_obj is not None
                    and not getattr(ac_obj, 'gs_intercepted', False)
                    and not getattr(ac_obj, 'on_ground', False)
                    and not getattr(ac_obj, 'landed', False)):
                if abs(target_alt_ft - float(getattr(
                        ac_obj, 'target_altitude', target_alt_ft))) > 25.0:
                    ac_obj.target_altitude = target_alt_ft
                    ac_obj.star_apply_alt = False

            actions = {
                'target_hdg_deg': target_hdg_deg,
                'target_alt_kft': float(alt_kft),
                'target_spd_kt': float(spd_norm) * 100.0 + 200.0,
                # Phase-2 radar head's delta in physical units (for HUD).
                'delta_hdg_deg': float(delta_3d[0]),
                'delta_alt_kft': float(delta_3d[1]),
                'delta_spd_kt': float(delta_3d[2]),
            }
            report.append({
                'callsign': cs,
                'actions': actions,
                'cmd': cmd,
                'sim_result': sim_result,
            })

        return report
