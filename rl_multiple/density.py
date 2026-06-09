"""Traffic-density encoding for multi-plane PPO.

Per ego plane each tick: build a 36-bin angular density profile of all
*other* live aircraft within a 10 nm lateral cutoff, in the ego's frame
(forward = 0°, +90° = right of nose). A `DensityCache` tracks the
previous tick's density per callsign so the temporal delta — which
encodes closing vs. opening motion — can be computed as `now - prev`.

Schema:
    N_BINS           = 36
    BIN_WIDTH_DEG    = 10
    BIN_CENTERS      = [-175, -165, ..., -5, 5, 15, ..., 175]  (deg, ego-rel)
    CUTOFF_NM        = 10.0
    SIGMA_BETA_DEG   = 12.0      # Gaussian smear; ~1 bin to each side

Per neighbor — Gaussian smear, max-aggregation per bin:

    For each plane at bearing β, distance d_nm:
        amp(plane)      = max(0, 1 − d_nm / 10)              ← linear proximity
        ang_kernel(c)   = exp(−(c − β)² / (2·SIGMA_BETA²))   ← bell, wrap-aware
                                                              peak=1.0 at c=β
        contribution(c) = amp × ang_kernel(c)                ← shape (36,)
    Per bin:
        out[c] = max over planes of contribution(c)          ← NEAREST, not summed

Why a Gaussian smear AND a max aggregation:

    - **Smear** keeps the profile continuous as a plane moves smoothly
      across bearings. A plane at β=47° lights up bins {35°, 45°, 55°,
      65°} with falloff — when it drifts to β=48° next tick those values
      shift smoothly. This makes the temporal delta (now − prev) a clean
      signal that reflects ACTUAL motion, not bin-boundary artifacts.

    - **Max aggregation** (vs sum) keeps the bin's interpretation honest:
      "what's the strongest threat in this direction?" Two planes
      stacked at 5 nm in the same wedge produce a bin value of ~0.5
      (their shared per-plane max), not ~1.0 — because there's only one
      threat-amount worth of danger there.

    - **Linear proximity (1 − d/10)** keeps meaningful signal across the
      whole 0-10 nm range, not just in the close-in tail.

Per-bin amplitude is bounded in [0, 1] regardless of how many neighbors
are in range: the per-plane contribution is ≤ 1.0 and max-aggregation
never grows the value above that. The watch radar's FIXED_MAX_AMP=1.0
lines up exactly with this bound.

Phase-1 property (no traffic): all neighbors filtered out → bin vector
is identically zero, delta is identically zero.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


N_BINS = 36
BIN_WIDTH_DEG = 360.0 / N_BINS                                          # 10.0
BIN_CENTERS = np.arange(-180.0 + BIN_WIDTH_DEG / 2, 180.0,
                        BIN_WIDTH_DEG, dtype=np.float32)
assert BIN_CENTERS.shape == (N_BINS,)

CUTOFF_NM = 10.0

# Angular Gaussian smear width. At σ=12°, the kernel falls to e^(-0.5)
# ≈ 0.61 at ±12° from a plane's bearing, e^(-2) ≈ 0.14 at ±24°, and
# is effectively zero beyond ±36° (3σ). So one plane meaningfully
# lights up ~3 bins on each side of its peak bin — enough to make
# bin transitions smooth without smearing across half the radar.
SIGMA_BETA_DEG = 12.0


def _ego_relative_bearing_deg(ego_x: float, ego_y: float,
                              ego_heading_deg: float,
                              nb_x: float, nb_y: float) -> float:
    """Bearing from ego's forward axis to the neighbor, in degrees,
    wrapped to (-180, +180].

    Sim's coords have y-down (screen convention) and headings are
    compass-degrees with 0=N, 90=E, 180=S, 270=W. Matches
    `environment.utils.get_bearing_from_coords`.
    """
    dx = nb_x - ego_x
    dy = nb_y - ego_y
    abs_bearing = (math.degrees(math.atan2(dx, -dy))) % 360.0
    rel = ((abs_bearing - ego_heading_deg + 540.0) % 360.0) - 180.0
    return rel


def build_density(ego, neighbors: Iterable, *,
                  nm_per_pixel: float,
                  cutoff_nm: float = CUTOFF_NM,
                  sigma_beta_deg: float = SIGMA_BETA_DEG) -> np.ndarray:
    """36-bin angular density of `neighbors` relative to `ego`.

    Gaussian angular smear (σ ≈ 12°) so the profile is continuous as
    planes move across bearings; max-aggregation per bin so the value
    represents "strongest threat in this wedge" rather than a sum.

    `ego` and each entry in `neighbors` are simulator Aircraft objects
    (or anything with `.x`, `.y`, `.heading`). The ego itself is NOT
    expected in `neighbors` — caller filters it out.
    """
    out = np.zeros(N_BINS, dtype=np.float32)
    ego_x, ego_y, ego_h = float(ego.x), float(ego.y), float(ego.heading)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma_beta_deg * sigma_beta_deg)
    for nb in neighbors:
        nb_x, nb_y = float(nb.x), float(nb.y)
        dx = nb_x - ego_x
        dy = nb_y - ego_y
        d_px = math.hypot(dx, dy)
        if d_px <= 0.0:
            continue
        d_nm = d_px * nm_per_pixel
        if d_nm >= cutoff_nm:
            continue
        rel = _ego_relative_bearing_deg(ego_x, ego_y, ego_h, nb_x, nb_y)
        amp = 1.0 - d_nm / cutoff_nm                          # (0, 1]
        # Wrap-aware angular distance from each bin center to the
        # plane's bearing. (BIN_CENTERS − rel + 540) % 360 − 180 keeps
        # the result in (-180, +180], so a plane at +178° is correctly
        # close to the bin at -175° (Δ = 7°, not 353°).
        diff = ((BIN_CENTERS - rel + 540.0) % 360.0) - 180.0
        ang_kernel = np.exp(-(diff * diff) * inv_two_sigma_sq)
        contributions = (amp * ang_kernel).astype(np.float32)
        # NEAREST per bin: a stronger contributor wins, no stacking.
        np.maximum(out, contributions, out=out)
    return out


class DensityCache:
    """Per-callsign rolling density buffer.

    On each tick call `update(sim)` to get
        {callsign: {'now': arr(36), 'delta': arr(36)}}
    for every live plane.

    `delta = now − prev`, the per-bin numerical derivative of the
    smoothed density profile. Because the profile is continuous (smear
    in build_density), a plane drifting smoothly across bearings
    produces a smooth wave-shaped delta — positive at the bins the
    plane is moving toward, negative at the ones it's leaving — rather
    than the binary spikes a hard-bin formulation would give. A plane
    closing dead-ahead within its bin produces a positive bump centered
    at its bearing.

    Dead callsigns are GC'd.
    """

    def __init__(self, cutoff_nm: float = CUTOFF_NM):
        self.cutoff_nm = float(cutoff_nm)
        self._prev: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._prev.clear()

    def forget(self, callsign: str) -> None:
        self._prev.pop(callsign, None)

    def update(self, sim) -> dict[str, dict]:
        live = list(sim.aircraft_list.items())
        live_set = {cs for cs, _ in live}
        results: dict[str, dict] = {}
        for cs, ego in live:
            others = [ac for k, ac in live if k != cs]
            now = build_density(
                ego, others,
                nm_per_pixel=sim.nm_per_pixel,
                cutoff_nm=self.cutoff_nm,
            )
            prev = self._prev.get(cs)
            delta = (now - prev) if prev is not None \
                else np.zeros_like(now)
            results[cs] = {'now': now, 'delta': delta}
            self._prev[cs] = now
        for dead in [k for k in self._prev if k not in live_set]:
            del self._prev[dead]
        return results
