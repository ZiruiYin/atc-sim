"""Per-STAR "green zones" — the ideal-position regions where an aircraft
*should* be during the approach phase, BEFORE the localizer is captured.

Used by PPO as a dense shaping reward (positive while inside the STAR's
green zone, neutral / negative outside) so the agent gets feedback long
before the sparse LOC+GS terminal signal. Also imported by the BC eval
viz to overlay the zones on the trajectory plots.

Frame: runway-aligned `(a, c)` nm
    a = along-track from threshold 27, east-positive (incoming side > 0).
    c = cross-track from threshold,   north-positive.

Shape recipe (from the user's spec, runway 27, test airport):

    NORTH1 / SOUTH1  →  L-shaped polygon
        downwind strip ±1 nm around c = ±8 (mirrored north/south)
        base/turn rectangle: a in [earliest_turn_a, latest_turn_a],
                             c in [0, ±downwind_side]
        earliest_turn_a = where the 3° glide hits 2200 ft (7.333 nm
                          from threshold).
        latest_turn_a   = 5 nm past WP6 to the east (WP6 a=14 → 19).
        L stops at the centerline (c = 0).

    NORTH2 / SOUTH2  →  L (same as N1/S1) + a triangle entering from
        the north/south. Triangle vertices for NORTH2: WP17, WP19, WP3.
        For SOUTH2:                                    WP21, WP23, WP9.

    NORTH3 / SOUTH3  →  short triangle. Vertices: a dot 2 nm WEST of
        WP6, a dot 2 nm EAST of WP6, and WP13 (NORTH3) / WP15 (SOUTH3).

The zones are computed in this module as constants, then exported as
`STAR_GREEN_ZONES: dict[str, list[Polygon]]` where each Polygon is a list
of `(a_nm, c_nm)` vertices.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# --------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------- #

Vertex = tuple[float, float]
Polygon = list[Vertex]


# Waypoints in runway-aligned `(a, c)` nm. Derived from
# `environment/data/test_navigation.json` (lat/lon, scaled 1 deg ≈ 60 nm)
# minus the runway-27 threshold at (a=0, c=0). The threshold sits at
# +1.0 nm east of the airport center; that's already subtracted here.
WP: dict[str, Vertex] = {
    'WP1':  (-31.0,   8.0),
    'WP2':  (-21.0,   8.0),
    'WP3':  ( -9.0,   8.0),
    'WP4':  (  3.0,   8.0),
    'WP5':  ( 14.0,   8.0),
    'WP6':  ( 14.0,   0.0),
    'WP7':  (-31.0,  -8.0),
    'WP8':  (-21.0,  -8.0),
    'WP9':  ( -9.0,  -8.0),
    'WP10': (  3.0,  -8.0),
    'WP11': ( 14.0,  -8.0),
    'WP12': ( 29.0,   4.0),
    'WP13': ( 21.0,   4.0),
    'WP14': ( 29.0,  -4.0),
    'WP15': ( 21.0,  -4.0),
    'WP16': (-11.0,  30.0),
    'WP17': (-11.0,  22.0),
    'WP18': ( -6.0,  14.0),
    'WP19': ( -2.0,   9.0),
    'WP20': (-11.0, -30.0),
    'WP21': (-11.0, -22.0),
    'WP22': ( -6.0, -14.0),
    'WP23': ( -2.0,  -9.0),
}


# --------------------------------------------------------------------- #
# Tunables (with provenance for each)
# --------------------------------------------------------------------- #

# Standard 3° glide projects 300 ft per nm to the threshold. The
# "earliest turn" boundary is where that glide is at 2200 ft → 2200/300
# ≈ 7.333 nm from threshold. Inside this radius the aircraft is too
# close to the threshold to make a clean base→final turn.
GS_FT_PER_NM = 300.0
EARLIEST_TURN_ALT_FT = 2200.0
EARLIEST_TURN_A_NM = EARLIEST_TURN_ALT_FT / GS_FT_PER_NM     # 7.333

# Latest acceptable turn point: 5 nm east of WP6. Past this the
# aircraft has overshot the comfortable base-turn window.
LATEST_TURN_A_NM = WP['WP6'][0] + 5.0                        # 19.0

# Downwind half-widths, ASYMMETRIC. The downwind strip is wider on the
# outer side (away from the centerline) than the inner side. The intent:
# an aircraft slightly off the downwind on the outer side (drifting north
# of the NORTH downwind, etc.) is still on a workable approach geometry
# and shouldn't be penalized as hard as one that's drifted toward the
# centerline (which means it's overshooting toward the base turn). The
# inner edge stays tight at 1 nm; the outer edge gets +1 nm of slack.
DOWNWIND_HALF_WIDTH_INNER_NM = 1.0
DOWNWIND_HALF_WIDTH_OUTER_NM = 2.0
DOWNWIND_C_NORTH = WP['WP3'][1]                              # +8
DOWNWIND_C_SOUTH = WP['WP9'][1]                              # -8

# Downwind westward extent: NORTH1/SOUTH1 STARs start as far west as
# WP1/WP7 (a=-31). We cover from there east to the latest-turn boundary.
DOWNWIND_A_WEST = WP['WP1'][0]                               # -31

# NORTH3/SOUTH3 — short triangle. Base sits on the centerline straddling
# WP6 with an ASYMMETRIC span: 1 nm to the WEST (closer to threshold,
# less room because the aircraft has less time to capture LOC there) and
# 3 nm to the EAST (more permissive — captures the typical fan-out
# point for late LOC intercept). Peak at WP13/WP15 (the STAR's last
# off-centerline waypoint before WP6).
N3_S3_BASE_WEST_NM = 1.0    # nm WEST of WP6 (smaller `a`)
N3_S3_BASE_EAST_NM = 3.0    # nm EAST of WP6 (larger `a`)


# --------------------------------------------------------------------- #
# L-shape builder for NORTH1 / SOUTH1
# --------------------------------------------------------------------- #

def _l_zone(side: str, a_west: float | None = None) -> Polygon:
    """Build the L-shaped green zone for an N or S STAR family.

    `side ∈ {'N','S'}` flips the polygon about the centerline.
    `a_west` sets the western end of the downwind strip. Defaults to
    `DOWNWIND_A_WEST` (WP1/WP7 ≈ -31 nm), matching NORTH1 / SOUTH1 which
    fly the full downwind. For NORTH2 / SOUTH2 the inbound triangle
    merges into the downwind at WP3 / WP9 (a = -9), so passing
    `a_west=WP3 a` keeps the L compact and visually distinguishable from
    NORTH1 / SOUTH1's full-length L.

    Vertices traced counter-clockwise (in the c-north-positive frame),
    starting from the far-west top corner of the downwind strip.
    """
    if side == 'N':
        c_far  = DOWNWIND_C_NORTH + DOWNWIND_HALF_WIDTH_OUTER_NM   # +10
        c_near = DOWNWIND_C_NORTH - DOWNWIND_HALF_WIDTH_INNER_NM   # +7
    else:
        c_far  = DOWNWIND_C_SOUTH - DOWNWIND_HALF_WIDTH_OUTER_NM   # -10
        c_near = DOWNWIND_C_SOUTH + DOWNWIND_HALF_WIDTH_INNER_NM   # -7
    a_w = DOWNWIND_A_WEST if a_west is None else float(a_west)
    a_e = LATEST_TURN_A_NM
    a_turn = EARLIEST_TURN_A_NM
    # 6 vertices: far-west-top → far-west-near → turn-corner → centerline
    # corner → centerline-east → far-east-top → close.
    return [
        (a_w,    c_far),
        (a_w,    c_near),
        (a_turn, c_near),
        (a_turn, 0.0),
        (a_e,    0.0),
        (a_e,    c_far),
    ]


# --------------------------------------------------------------------- #
# Per-STAR green zones
# --------------------------------------------------------------------- #

STAR_GREEN_ZONES: dict[str, list[Polygon]] = {
    'NORTH1': [_l_zone('N')],
    'SOUTH1': [_l_zone('S')],
    # NORTH2: inbound triangle from the far north MERGING into the
    # downwind. The WEST edge of the triangle is now VERTICAL: top at
    # WP17 (a=-11, c=22), bottom at (a=-11, c=8) — directly south.
    # This lets the aircraft fly straight south (heading 180°) before
    # turning onto downwind, no slanted west wall. The downwind L is
    # extended west to a=-11 accordingly so the two pieces connect
    # without a gap. WP19 stays as the eastern triangle vertex (where
    # the inbound merges onto the downwind centerline near WP3's old
    # position).
    'NORTH2': [_l_zone('N', a_west=WP['WP17'][0]),
               [WP['WP17'],
                WP['WP19'],
                (WP['WP17'][0], WP['WP3'][1])]],
    # SOUTH2: mirror — vertical west edge at WP21's a (-11), apex going
    # straight up to WP21, eastern vertex at WP23, bottom at (-11, -8).
    'SOUTH2': [_l_zone('S', a_west=WP['WP21'][0]),
               [WP['WP21'],
                WP['WP23'],
                (WP['WP21'][0], WP['WP9'][1])]],
    # NORTH3 / SOUTH3: a short asymmetric triangle. Base on the
    # centerline straddling WP6 — 1 nm WEST + 3 nm EAST — peak at
    # WP13 / WP15.
    'NORTH3': [[(WP['WP6'][0] - N3_S3_BASE_WEST_NM, 0.0),
                (WP['WP6'][0] + N3_S3_BASE_EAST_NM, 0.0),
                WP['WP13']]],
    'SOUTH3': [[(WP['WP6'][0] - N3_S3_BASE_WEST_NM, 0.0),
                (WP['WP6'][0] + N3_S3_BASE_EAST_NM, 0.0),
                WP['WP15']]],
}


# --------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------- #

def _point_in_polygon(a: float, c: float, poly: Polygon) -> bool:
    """Crossing-number test. `poly` is a list of (a, c) vertices, last
    NOT repeated (segment from last → first is added implicitly)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        ai, ci = poly[i]
        aj, cj = poly[j]
        # Half-open edge crossing rule: avoids double-count at shared vertices.
        if ((ci > c) != (cj > c)):
            slope = (a - ai) * (cj - ci) - (aj - ai) * (c - ci)
            if (slope < 0) != (cj < ci):
                inside = not inside
        j = i
    return inside


def point_in_zone(star: str, a: float, c: float) -> bool:
    """True if (a, c) falls inside ANY polygon of the STAR's green zone."""
    for poly in STAR_GREEN_ZONES.get(star, ()):
        if _point_in_polygon(a, c, poly):
            return True
    return False


def vectorized_in_zone(star: str,
                       a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vectorized version of `point_in_zone` for shaping arrays of
    trajectory points. Returns a bool array of the same shape as `a`."""
    out = np.zeros_like(a, dtype=bool)
    for poly in STAR_GREEN_ZONES.get(star, ()):
        # Cheap implementation: per-polygon ray-cast loop. Fine for our
        # scale (one rollout = a few hundred ticks; one minibatch = a
        # few thousand). If this ever becomes a bottleneck, swap to
        # `matplotlib.path.Path.contains_points` (vectorized C call).
        poly_arr = np.asarray(poly, dtype=np.float64)
        ai, ci = poly_arr[:, 0], poly_arr[:, 1]
        aj, cj = np.roll(ai, 1), np.roll(ci, 1)
        for k in range(a.size):
            x, y = float(a.flat[k]), float(c.flat[k])
            cross = ((ci > y) != (cj > y))
            slope = (x - ai) * (cj - ci) - (aj - ai) * (y - ci)
            hit = np.logical_xor((slope < 0), (cj < ci)) & cross
            if hit.sum() % 2 == 1:
                out.flat[k] = True
    return out


def iter_zone_polygons() -> Iterable[tuple[str, Polygon]]:
    """Yield (star, polygon) for every polygon across all STARs.
    Convenience for plotting."""
    for star, polys in STAR_GREEN_ZONES.items():
        for poly in polys:
            yield star, poly


# --------------------------------------------------------------------- #
# Distance from a point to the STAR's green zone
# --------------------------------------------------------------------- #

import math


def _dist_point_to_segment(px: float, py: float,
                           ax: float, ay: float,
                           bx: float, by: float) -> float:
    """Euclidean distance from point (px, py) to segment (a)→(b)."""
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _dist_point_to_polygon_edge(px: float, py: float,
                                poly: Polygon) -> float:
    """Min distance from (px, py) to any edge of `poly`. Does NOT
    short-circuit on inside-ness; callers should check that first."""
    n = len(poly)
    best = math.inf
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        d = _dist_point_to_segment(px, py, ax, ay, bx, by)
        if d < best:
            best = d
    return best


def dist_to_zone(star: str, a: float, c: float) -> float:
    """Shortest nm distance from `(a, c)` to the STAR's green zone.

    Returns 0.0 if the point is inside ANY of the STAR's polygons
    (multiple polygons → union semantics), else the min Euclidean
    distance to the nearest polygon edge across all polygons.
    """
    polys = STAR_GREEN_ZONES.get(star, ())
    if not polys:
        return 0.0
    if point_in_zone(star, a, c):
        return 0.0
    best = math.inf
    for poly in polys:
        d = _dist_point_to_polygon_edge(a, c, poly)
        if d < best:
            best = d
    return best


# --------------------------------------------------------------------- #
# Reward constants and per-step / terminal reward
# --------------------------------------------------------------------- #
# Design rationale (matches the user's spec):
#
#   - In-zone: zero per-step reward. NO positive credit for staying in
#     the zone — would otherwise reward an aircraft for sitting in the
#     downwind forever without ever turning final.
#   - Out-of-zone: small NEGATIVE per-step penalty that grows linearly
#     with distance to the nearest polygon edge, then caps. The cap
#     keeps wildly-off trajectories from blowing past the terminal
#     reward magnitude (we want terminal to dominate the credit signal).
#   - Terminal SUCCESS (LOC_BELOW_GS / LANDED): big positive lump.
#   - Terminal FAILURE (TIMEOUT / IMPROPER_EXIT / LOC_ABOVE_GS /
#     LOC_BEHIND_THR / CRASHED): big negative lump, MORE negative in
#     magnitude than the realistic accumulated step penalty so that
#     failure is unambiguously worse than just drifting around.
#
# Sizing example for a 1500-step rollout that sits ~5 nm off the zone:
#     step accumulation = 5 * 0.005 * 1500 = -37.5
#     well under |FAILURE_REWARD| = 100  ✓
# At the cap (≥10 nm out for 1500 steps): step accumulation = -75,
# still under |FAILURE_REWARD|.

# Terminal rewards — scaled down 10× from the original ±100 design to
# match the PPO hyperparams (lr_actor=1e-5, value_coef=0.5, target_kl=0.02
# all tuned for ±1-to-±10 reward magnitudes). The relative structure of
# every shaping term below is preserved — only the absolute scale shrinks.
# A first attempt with ±100 + ±20 + ±5 shaping produced KL=1.37 (68× the
# 0.02 target) and the GMM head NaN'd out by iter 4.
SUCCESS_REWARD: float = 10.0
FAILURE_REWARD: float = -10.0

# Per-step shaping (scaled 10× down from ±0.05 cap to ±0.005 cap).
# `STEP_PENALTY_CAP` is RUNTIME-TUNABLE — set per training block via
# `set_runtime_overrides()` from PPOConfig. The default here is the
# baseline value (run_5-ish); the cap ladder we tune across is
# {0.005, 0.010, 0.020}.
STEP_PENALTY_PER_NM: float = 0.0005   # per step, per nm outside the zone
STEP_PENALTY_CAP: float = 0.005       # max -0.005 per step (≈10 nm out)

# Everywhere-step penalty (applied EVERY step, regardless of zone).
# RUNTIME-TUNABLE per block via `set_runtime_overrides()`. Ladder:
# {0.0, 0.001, 0.002}.
#   0.0    — no time pressure; policy free to linger / loop in green zone
#   0.001  — gentle; block-1 default
#   0.002  — strong; flip on when trajectories drift above the per-STAR
#            max length (loop / circling exploit forming)
EVERYWHERE_STEP_PENALTY: float = 0.001

# Early-window zone-penalty multiplier. For the first
# EARLY_WINDOW_STEPS POLICY-controlled steps in an episode, the
# OUT-OF-ZONE per-step penalty is multiplied by EARLY_ZONE_MULTIPLIER.
EARLY_ZONE_MULTIPLIER: float = 1.0   # 1.0 = disabled (backward compat)
EARLY_WINDOW_STEPS: int = 300

# Early-drift FLAT penalty. Independent of distance: every out-of-zone
# step during the early window adds this flat penalty (negative).
# Designed to make BC's "drift-off on triangle→downwind merge" mode
# pay roughly half the terminal reward — turning successful-but-sloppy
# trajectories into "half failures" so PPO's gradient picks the
# in-zone modes from the BC mixture.
#
# Magnitude calibration (0.05): BC NORTH2 has 37% out-of-zone over
# its first 300 policy steps → 0.37 × 300 × 0.05 = 5.55 total penalty.
# Net reward: +10 (terminal) − 5.55 = +4.5 (half failure). Meanwhile
# BC NORTH1 with 1.7% out pays only 0.26 — barely touched.
#
# EXEMPT for NORTH3/SOUTH3 (direct-vector STARs) — per user directive,
# we only care about SR there, not in-zone behavior.
EARLY_DRIFT_PENALTY: float = 0.0    # 0 = disabled (backward compat)
EARLY_DRIFT_EXEMPT: frozenset = frozenset({'NORTH3', 'SOUTH3'})

# Clean-terminal threshold. For STAR-1/2 family, a SUCCESSFUL trajectory
# (LOC_BELOW_GS) only gets the full +SUCCESS_REWARD terminal IF its
# fraction-in-zone over the policy-controlled phase is at least
# CLEAN_TERMINAL_THRESHOLD. Otherwise, the terminal is replaced with
# DRIFTY_SUCCESS_VALUE (default 0). Step penalties still accumulate
# throughout the trajectory regardless of outcome — so a drifty
# "successful" trajectory ends up STRICTLY NEGATIVE in total reward.
# This forces PPO to select clean-in-zone modes from the BC mixture.
# The gate condition is `frac_in_zone < CLEAN_TERMINAL_THRESHOLD`:
#   - threshold = 0.0  → never fires (no frac < 0), mechanism DISABLED.
#   - threshold = 0.90 → demand 90% in zone for full +10 terminal.
#   - threshold = 1.0  → demand 100% in zone (very strict).
CLEAN_TERMINAL_THRESHOLD: float = 0.0       # 0.0 = disabled
DRIFTY_SUCCESS_VALUE: float = 0.0           # what to replace terminal with

# OUT_OF_ZONE TERMINATION mechanism (c03 strategy).
# When enabled, terminate episode as FAILURE (RZ_FAILURE_REWARD = -10)
# the moment a STAR-1/2 aircraft is OUT of the green zone for
# OUT_OF_ZONE_MAX_CONSECUTIVE policy-controlled steps in a row.
# STAR-3 family (EARLY_DRIFT_EXEMPT) is untouched — no zone termination.
#
# When this mechanism is on, we typically disable the slope/cap zone
# penalty (set to 0) — termination IS the penalty.
#
# OUT_OF_ZONE_MAX_CONSECUTIVE: small tolerance allows brief excursions
# (e.g., 5 steps of "natural" slight off-zone during a turn) before
# termination fires. Set to 1 for ZERO tolerance.
OUT_OF_ZONE_TERMINATE: bool = False
OUT_OF_ZONE_MAX_CONSECUTIVE: int = 5

# Per-STAR multiplicative success-reward scaling. Amplifies the
# terminal reward on STARs where the policy's recent SR is low —
# combats the "STAR trading" failure mode where the policy
# over-optimizes one STAR at the cost of another.
#
#     terminal_r = SUCCESS_REWARD * (1 + scale * (1 - recent_SR[star]))
#
# scale=0 disables. scale=0.5 (mild) gives at most +5 bonus for a
# 0%-SR STAR; at 90% SR → +0.5 bonus.
# Recent SR is updated per block from the prior eval's
# eval_metrics.json (see train.py). Default 1.0 = no bonus.
PER_STAR_SR_SCALE: float = 0.0
PER_STAR_RECENT_SR: dict = {
    'NORTH1': 1.0, 'NORTH2': 1.0, 'NORTH3': 1.0,
    'SOUTH1': 1.0, 'SOUTH2': 1.0, 'SOUTH3': 1.0,
}

# Always-on base shaping components, exposed as runtime toggles ONLY for
# ablation studies (default True = the design baseline). When False, that
# bonus is simply not added at its call site in env.py.
#   HEADING_INTERCEPT_ENABLED — the capture-heading terminal bonus.
#   TURN_FINAL_ENABLED        — the one-shot base-turn R(s,a) bonus.
HEADING_INTERCEPT_ENABLED: bool = True
TURN_FINAL_ENABLED: bool = True


def set_runtime_overrides(*,
                          everywhere_step_penalty: float | None = None,
                          step_penalty_cap: float | None = None,
                          step_penalty_per_nm: float | None = None,
                          early_zone_multiplier: float | None = None,
                          early_window_steps: int | None = None,
                          early_drift_penalty: float | None = None,
                          clean_terminal_threshold: float | None = None,
                          drifty_success_value: float | None = None,
                          out_of_zone_terminate: bool | None = None,
                          out_of_zone_max_consecutive: int | None = None,
                          per_star_sr_scale: float | None = None,
                          per_star_recent_sr: dict | None = None,
                          heading_intercept_enabled: bool | None = None,
                          turn_final_enabled: bool | None = None) -> None:
    """Override the RUNTIME-TUNABLE shaping constants in-place.

    Called by the trainer at start-of-block from PPOConfig so the env
    sees the values declared in `config.json`. We deliberately do NOT
    expose terminal magnitudes (`SUCCESS_REWARD` / `FAILURE_REWARD`),
    the heading bonus weights, or the turn-final R(s,a) bonus — those
    are always-on per the design (and the user's directive).

    Each call snapshots the new values into module globals so other
    references in this file pick them up too. Pass `None` to leave a
    value at its current setting.
    """
    global EVERYWHERE_STEP_PENALTY, STEP_PENALTY_CAP, STEP_PENALTY_PER_NM
    global EARLY_ZONE_MULTIPLIER, EARLY_WINDOW_STEPS, EARLY_DRIFT_PENALTY
    global CLEAN_TERMINAL_THRESHOLD, DRIFTY_SUCCESS_VALUE
    global OUT_OF_ZONE_TERMINATE, OUT_OF_ZONE_MAX_CONSECUTIVE
    global PER_STAR_SR_SCALE, PER_STAR_RECENT_SR
    global HEADING_INTERCEPT_ENABLED, TURN_FINAL_ENABLED
    if everywhere_step_penalty is not None:
        EVERYWHERE_STEP_PENALTY = float(everywhere_step_penalty)
    if step_penalty_cap is not None:
        STEP_PENALTY_CAP = float(step_penalty_cap)
    if step_penalty_per_nm is not None:
        STEP_PENALTY_PER_NM = float(step_penalty_per_nm)
    if early_zone_multiplier is not None:
        EARLY_ZONE_MULTIPLIER = float(early_zone_multiplier)
    if early_window_steps is not None:
        EARLY_WINDOW_STEPS = int(early_window_steps)
    if early_drift_penalty is not None:
        EARLY_DRIFT_PENALTY = float(early_drift_penalty)
    if clean_terminal_threshold is not None:
        CLEAN_TERMINAL_THRESHOLD = float(clean_terminal_threshold)
    if drifty_success_value is not None:
        DRIFTY_SUCCESS_VALUE = float(drifty_success_value)
    if out_of_zone_terminate is not None:
        OUT_OF_ZONE_TERMINATE = bool(out_of_zone_terminate)
    if out_of_zone_max_consecutive is not None:
        OUT_OF_ZONE_MAX_CONSECUTIVE = int(out_of_zone_max_consecutive)
    if per_star_sr_scale is not None:
        PER_STAR_SR_SCALE = float(per_star_sr_scale)
    if per_star_recent_sr is not None:
        # Merge — keep entries for STARs not in the override at 1.0.
        for k, v in per_star_recent_sr.items():
            PER_STAR_RECENT_SR[k] = float(v)
    if heading_intercept_enabled is not None:
        HEADING_INTERCEPT_ENABLED = bool(heading_intercept_enabled)
    if turn_final_enabled is not None:
        TURN_FINAL_ENABLED = bool(turn_final_enabled)


def get_runtime_overrides() -> dict:
    """Current values of the tunable shaping knobs. For logging."""
    return {
        'everywhere_step_penalty': float(EVERYWHERE_STEP_PENALTY),
        'step_penalty_cap': float(STEP_PENALTY_CAP),
        'step_penalty_per_nm': float(STEP_PENALTY_PER_NM),
        'early_zone_multiplier': float(EARLY_ZONE_MULTIPLIER),
        'early_window_steps': int(EARLY_WINDOW_STEPS),
        'early_drift_penalty': float(EARLY_DRIFT_PENALTY),
        'early_drift_exempt': sorted(EARLY_DRIFT_EXEMPT),
        'clean_terminal_threshold': float(CLEAN_TERMINAL_THRESHOLD),
        'drifty_success_value': float(DRIFTY_SUCCESS_VALUE),
        'out_of_zone_terminate': bool(OUT_OF_ZONE_TERMINATE),
        'out_of_zone_max_consecutive': int(OUT_OF_ZONE_MAX_CONSECUTIVE),
        'per_star_sr_scale': float(PER_STAR_SR_SCALE),
        'per_star_recent_sr': {k: float(v) for k, v in PER_STAR_RECENT_SR.items()},
        # Always-on constants below:
        'success_reward': float(SUCCESS_REWARD),
        'failure_reward': float(FAILURE_REWARD),
        'heading_peak_bonus': float(HEADING_PEAK_BONUS),
        'final_turn_peak_bonus': float(FINAL_TURN_PEAK_BONUS),
        'heading_intercept_enabled': bool(HEADING_INTERCEPT_ENABLED),
        'turn_final_enabled': bool(TURN_FINAL_ENABLED),
    }

# Heading-at-intercept shaping. Applied AT TERMINATION based on the
# aircraft's heading averaged from the final few position segments —
# proxies the heading the model commanded just before LOC capture.
#
# Shape (NORTH; SOUTH is the mirror about 270°):
#   h = 240°  →  +10  (PEAK — ideal LOC-intercept angle, 30° closing)
#   h = 245°  →   +5  (still in the good window)
#   h = 250°  →    0  (no reward; angle is fine but not preferred)
#   h = 260°  →  -10  (penalized; converging too parallel to LOC)
#   h = 270°  →  -20  (at centerline; aircraft has crossed through)
#   h > 270°  →  past-centerline overshoot, continue slope, floor -20
#   h < 240°  →  -20  cliff (too shallow — sim wouldn't even let it capture
#                      here per the ±30° gate; this is a safety floor)
#
# Single linear slope (-1/°) from 240° to 270° gives the +10 → -20
# spread. Below the peak it's a hard cliff because the geometry is
# unrecoverable: an aircraft at 220° has 50° to close on the LOC and
# can't physically turn fast enough mid-intercept.
#
# Stacks on top of ±100 terminal as a SMALL final-turn nudge.
HEADING_PEAK_BONUS:    float =  1.0    # at the per-direction peak heading
HEADING_LATE_PENALTY:  float = -2.0    # at the centerline (270°)
HEADING_CLIFF_PENALTY: float = -2.0    # below peak (NORTH < 240, SOUTH > 300)
                                       #   AND floor far past centerline.
                                       # Matches LATE_PENALTY so the worst
                                       # heading cost is -20 regardless of
                                       # whether you fell short or overshot.

# Per-direction peak headings (the EARLIEST acceptable intercept angle).
# NORTH: turning RIGHT from south (180°) to west (270°), peak at 240°
# (30° short of full alignment). SOUTH: turning LEFT from north (000°)
# down through 350° → 300° → 270°, peak at 300° (30° short going the
# other way). Both peaks are 30° before the centerline.
NORTH_PEAK_HDG: float = 240.0
SOUTH_PEAK_HDG: float = 300.0
CENTERLINE_HDG: float = 270.0
HEADING_SLOPE_PER_DEG: float = 1.0     # -1 reward per degree, from peak toward centerline


def heading_intercept_reward(direction: str, heading_deg: float) -> float:
    """Bonus / penalty for the aircraft's heading at LOC-intercept
    time (or at episode end as a proxy when no intercept happened).

    See module-level docstring for the full schedule. Returns are
    bounded: at most +PEAK_BONUS (=+10), at least CLIFF_PENALTY
    (=-50). Sized as a small final-turn nudge on top of ±100 terminal.

        direction ∈ {'NORTH', 'SOUTH'}
        heading_deg : compass degrees (any range; reduced mod 360)
    """
    h = float(heading_deg) % 360.0
    if direction == 'NORTH':
        if h < NORTH_PEAK_HDG:
            # Cliff below the peak — geometrically unrecoverable.
            return HEADING_CLIFF_PENALTY
        # h ≥ 240. Linear -1/° down: +10 at 240, -20 at 270.
        # Past 270 continues the slope, floored at the cliff value.
        return max(HEADING_CLIFF_PENALTY,
                   HEADING_PEAK_BONUS - HEADING_SLOPE_PER_DEG * (h - NORTH_PEAK_HDG))
    # --- SOUTH (mirror about 270°) ---
    if h > SOUTH_PEAK_HDG:
        return HEADING_CLIFF_PENALTY
    # h ≤ 300. Linear +1/° up: +10 at 300, -20 at 270.
    # Past 270 (toward 0°) continues the slope, floored at the cliff.
    return max(HEADING_CLIFF_PENALTY,
               HEADING_PEAK_BONUS - HEADING_SLOPE_PER_DEG * (SOUTH_PEAK_HDG - h))


# ----------------------------------------------------------------------- #
# State-ACTION reward: turn-final commanded heading inside the lateral
# window where the human typically issues the turn-final command.
# Derived from analysis of recorded human commands — across 4 STARs
# (NORTH1/2, SOUTH1/2) the turn-final cmd lands at |c| ≈ 1.0-1.8 nm
# with heading 240° (NORTH) or 300° (SOUTH).
#
# This reward is R(s, a): it requires BOTH the lateral position (state)
# AND the commanded target heading (action). PPO env evaluates it
# per step using the model's emitted action.
# ----------------------------------------------------------------------- #

FINAL_TURN_C_MIN_NM: float = 1.0     # inner trigger (closer to centerline)
FINAL_TURN_C_MAX_NM: float = 2.0     # outer trigger (further from centerline)
FINAL_TURN_PEAK_BONUS: float = 0.5   # at exact peak heading
FINAL_TURN_EDGE_BONUS: float = 0.1   # at edge of acceptable range
NORTH_FINAL_PEAK_HDG: float = 240.0  # peak; +5
NORTH_FINAL_EDGE_HDG: float = 245.0  # edge; +1
SOUTH_FINAL_PEAK_HDG: float = 300.0  # peak; +5  (linearly DECREASING with h)
SOUTH_FINAL_EDGE_HDG: float = 295.0  # edge; +1


def final_turn_action_reward(star: str, a_nm: float, c_nm: float,
                             target_heading_deg: float) -> float:
    """Per-step (s, a) bonus when the aircraft is INSIDE the STAR's
    green zone AND in the turn-final lateral window AND its commanded
    target_heading sits in the ideal per-direction range.

        NORTH1 / NORTH2:  target_h ∈ [240°, 245°] → +5 → +1 linear
        SOUTH1 / SOUTH2:  target_h ∈ [295°, 300°] → +1 → +5 linear

    Returns 0 for:
      - NORTH3 / SOUTH3 (direct vector, no base turn)
      - Aircraft outside the green zone (e.g. too close to threshold,
        past the runway, behind the downwind strip)
      - |c| outside the [FINAL_TURN_C_MIN_NM, FINAL_TURN_C_MAX_NM] window
      - target_heading outside the per-direction ideal range
    """
    if star in ('NORTH3', 'SOUTH3'):
        return 0.0
    # Gate 1: aircraft must be inside the STAR's green zone. Stops the
    # bonus from firing when the aircraft is already past the runway,
    # in the back-course area, or otherwise off the ideal-approach
    # geometry — even if its target_heading happens to be right.
    if not point_in_zone(star, float(a_nm), float(c_nm)):
        return 0.0
    abs_c = abs(float(c_nm))
    if abs_c < FINAL_TURN_C_MIN_NM or abs_c > FINAL_TURN_C_MAX_NM:
        return 0.0
    h = float(target_heading_deg) % 360.0
    direction = 'NORTH' if star.startswith('NORTH') else 'SOUTH'
    if direction == 'NORTH':
        if h < NORTH_FINAL_PEAK_HDG or h > NORTH_FINAL_EDGE_HDG:
            return 0.0
        # 240 → +5, 245 → +1 linear
        t = (h - NORTH_FINAL_PEAK_HDG) / (NORTH_FINAL_EDGE_HDG - NORTH_FINAL_PEAK_HDG)
        return FINAL_TURN_PEAK_BONUS + t * (FINAL_TURN_EDGE_BONUS - FINAL_TURN_PEAK_BONUS)
    # SOUTH (mirror): 300 → +5, 295 → +1
    if h > SOUTH_FINAL_PEAK_HDG or h < SOUTH_FINAL_EDGE_HDG:
        return 0.0
    t = (SOUTH_FINAL_PEAK_HDG - h) / (SOUTH_FINAL_PEAK_HDG - SOUTH_FINAL_EDGE_HDG)
    return FINAL_TURN_PEAK_BONUS + t * (FINAL_TURN_EDGE_BONUS - FINAL_TURN_PEAK_BONUS)


def _final_heading_deg(a: np.ndarray, c: np.ndarray,
                       window: int = 1) -> float | None:
    """Compass heading from the LAST position delta of a trajectory.

    `window=1` measures the velocity vector over the single step that
    terminated the episode — i.e. the aircraft's heading at the
    moment of LOC capture (or whatever terminated it). A wider window
    averages over pre-termination motion: useful for smoothing but
    biased low for NORTH (which is mid-turn earlier in the window),
    and biased high for SOUTH (mirror). We default to 1 step for the
    capture-moment-faithful reading.
    """
    if len(a) < window + 1:
        return None
    da = float(a[-1] - a[-1 - window])
    dc = float(c[-1] - c[-1 - window])
    if (da * da + dc * dc) ** 0.5 < 1e-3:
        return None
    return math.degrees(math.atan2(da, dc)) % 360.0


# Outcome labels: which terminal classes count as success vs failure.
SUCCESS_OUTCOMES: frozenset = frozenset({'LOC_BELOW_GS', 'LANDED'})
FAILURE_OUTCOMES: frozenset = frozenset({
    'LOC_ABOVE_GS', 'LOC_BEHIND_THR',
    'IMPROPER_EXIT', 'TIMEOUT', 'CRASHED', 'REMOVED_UNKNOWN',
    'OUT_OF_ZONE',  # new c03 mechanism
})


def step_reward(star: str, a: float, c: float) -> float:
    """Per-step shaping reward for one (a, c) sample under `star`.

    Returns 0 inside the zone, else a small negative ∝ distance, capped.
    """
    d = dist_to_zone(star, a, c)
    if d <= 0.0:
        return 0.0
    return -min(d * STEP_PENALTY_PER_NM, STEP_PENALTY_CAP)


def terminal_reward(outcome: str) -> float:
    """Terminal lump payoff at episode end."""
    if outcome in SUCCESS_OUTCOMES:
        return SUCCESS_REWARD
    return FAILURE_REWARD


def trajectory_reward(star: str,
                      a_traj: np.ndarray,
                      c_traj: np.ndarray,
                      outcome: str) -> dict:
    """Compute step + terminal + total reward for a full trajectory.

    Returns a dict with `step_reward_sum`, `terminal_reward`,
    `total_reward`, plus diagnostic counts. Cheap point-by-point loop
    — fine for the ~hundreds of trajectories we score offline.
    """
    a = np.asarray(a_traj, dtype=np.float32)
    c = np.asarray(c_traj, dtype=np.float32)
    n = a.size
    # Everywhere-step penalty (one term per step regardless of zone).
    everywhere_pen_sum = -EVERYWHERE_STEP_PENALTY * n
    step_sum = float(everywhere_pen_sum)
    n_inside = 0
    n_outside = 0
    sum_dist_outside = 0.0
    max_dist_outside = 0.0
    sum_dist_all = 0.0
    # State-action turn-final bonus — ONCE PER EPISODE. Fires the first
    # time the aircraft satisfies both (a) in lateral trigger window
    # AND (b) commanded heading in the ideal range. Per-step firing
    # over-rewards lingering failures; one-shot keeps the signal clean.
    final_turn_bonus = 0.0
    final_turn_fire_step = -1
    final_turn_fire_c = float('nan')
    final_turn_fire_hdg = float('nan')
    excluded_from_turn = star in ('NORTH3', 'SOUTH3')
    for i in range(n):
        d = dist_to_zone(star, float(a[i]), float(c[i]))
        sum_dist_all += d
        if d <= 0.0:
            n_inside += 1
        else:
            n_outside += 1
            sum_dist_outside += d
            if d > max_dist_outside:
                max_dist_outside = d
            step_sum -= min(d * STEP_PENALTY_PER_NM, STEP_PENALTY_CAP)
        # Turn-final (s, a) bonus: check until we've already fired.
        if (not excluded_from_turn and final_turn_fire_step < 0
                and i >= 1):
            abs_c_i = abs(float(c[i]))
            if FINAL_TURN_C_MIN_NM <= abs_c_i <= FINAL_TURN_C_MAX_NM:
                da = float(a[i] - a[i - 1])
                dc = float(c[i] - c[i - 1])
                if (da * da + dc * dc) ** 0.5 >= 1e-3:
                    h_proxy = math.degrees(math.atan2(da, dc)) % 360.0
                    bonus = final_turn_action_reward(
                        star, float(a[i]), float(c[i]), h_proxy)
                    if bonus > 0.0:
                        final_turn_bonus = bonus
                        final_turn_fire_step = i
                        final_turn_fire_c = abs_c_i
                        final_turn_fire_hdg = h_proxy
    term = terminal_reward(outcome)
    # Length-invariant per-step penalty: divides the step accumulation
    # by the number of steps so two trajectories with the same OUT-OF-
    # ZONE *proportion* get the same score regardless of their lengths.
    # The raw `step_reward_sum` measures total off-zone time and biases
    # against long trajectories; `mean_step_penalty` strips that out and
    # is the fair-comparison metric for ranking.
    mean_step = (step_sum / n) if n else 0.0
    # Heading-at-intercept terminal modifier. Computed from the final
    # few segments of the trajectory (proxies the heading the model
    # commanded right before LOC capture / episode end).
    direction = 'NORTH' if star.startswith('NORTH') else 'SOUTH'
    # window=1 → heading from the SINGLE last step, i.e. the velocity
    # over the step that triggered termination. This is the actual
    # capture-moment heading, not an average over pre-capture turning.
    final_hdg = _final_heading_deg(a, c, window=1)
    heading_bonus = (heading_intercept_reward(direction, final_hdg)
                     if final_hdg is not None else 0.0)
    return {
        'step_reward_sum': step_sum,
        'mean_step_penalty': mean_step,
        'terminal_reward': term,
        'heading_bonus': heading_bonus,
        'final_heading_deg': (final_hdg if final_hdg is not None
                              else float('nan')),
        # Renamed to reflect one-shot semantics; counters track WHERE
        # along the episode the bonus fired (or -1 if it never did).
        'final_turn_bonus': final_turn_bonus,
        'final_turn_fire_step': final_turn_fire_step,
        'final_turn_fire_c_nm': final_turn_fire_c,
        'final_turn_fire_hdg_deg': final_turn_fire_hdg,
        'total_reward': (step_sum + term + heading_bonus
                         + final_turn_bonus),
        'n_steps': n,
        'n_inside': n_inside,
        'n_outside': n_outside,
        # Distance summaries. `mean_dist_outside_nm` is conditional on
        # being outside (NaN-style 0 if never outside). `mean_dist_nm`
        # averages over EVERY step (in-zone counts as 0) so it's a
        # smooth scalar that grows with how far/often you stray.
        # `max_dist_outside_nm` exposes the worst single-step excursion,
        # which the step-penalty cap (0.05/step ≈ 10nm) intentionally
        # masks in the reward signal.
        'mean_dist_outside_nm': (sum_dist_outside / n_outside) if n_outside else 0.0,
        'mean_dist_nm': (sum_dist_all / n) if n else 0.0,
        'max_dist_outside_nm': max_dist_outside,
        'frac_inside': (n_inside / n) if n else 0.0,
    }
