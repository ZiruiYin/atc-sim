"""Randomized valid-command generator for the dsl_parser training data.

Produces random but VALID sim commands (DSL) spanning the command space:
vectors (heading), altitude/speed changes, direct-to and holds over waypoints,
and ILS approach clearances -- with optional turn-direction (;L/;R) and expedite
(;X) modifiers. Structural validity is built in here; it is then *confirmed* by
running each command through the real aircraft validator in build_dataset.py
(which doubles as the "valid token filter").

Spec ranges:
  altitude : 1000-8000 ft -> DSL "C 1".."C 8"
  heading  : 010-360 deg  -> DSL "C 010".."C 360"  (3-digit, zero-padded)
  speed    : 140-240 kt    -> DSL "S 140".."S 240" (tens)

Target canonicalization
-----------------------
The training TARGET must be a deterministic function of the SPOKEN phraseology,
or the parser sees label noise. Two cases are otherwise ambiguous:
  * heading turn direction -- "turn left heading 240" can come from an explicit
    ;L OR from the shortest-turn geometry, so "C 240" and "C 240;L" can sound
    identical.
  * expedite on a no-op change -- "C 5;X" while already at 5000 is voiced as
    plain "maintain 5000" (the expedite is dropped).
canonicalize_target() rewrites the modifiers (;L/;R/;X) to match exactly what the
ATC line actually says, so target == f(phraseology).
"""
import re

from environment.params import CALLSIGN_PREFIXES


def random_callsign(rng):
    return f"{rng.choice(CALLSIGN_PREFIXES)}{rng.randint(1, 999)}"


def _hdg(rng):
    return f"{rng.randrange(1, 37) * 10:03d}"        # 010..360 (steps of 10)


def _alt(rng):
    return str(rng.randint(1, 8))                    # 1000..8000


def _spd(rng):
    return str(rng.randrange(140, 241, 10))          # 140..240 (tens)


def _dir(rng):
    return rng.choice(("L", "R"))


def random_init_state(rng):
    """Random *current* aircraft state -> drives phraseology variety
    (climb vs descend, left vs right turn, reduce vs increase speed)."""
    return {
        "heading": rng.randint(0, 359),
        "altitude": rng.randrange(2000, 12001, 1000),
        "airspeed": rng.randrange(160, 281, 10),
    }


def random_command(rng, waypoints, runways):
    """Return a random valid command string (DSL, no callsign).

    Heading turn-direction is left implicit (the shortest-turn geometry voices
    it, and canonicalize_target recovers it); direct/hold carry explicit ;L/;R
    because those phrasings only exist when stated.
    """
    wpt = lambda: rng.choice(waypoints)
    rwy = lambda: rng.choice(runways)
    hdg = lambda: _hdg(rng)
    alt = lambda: _alt(rng)
    spd = lambda: _spd(rng)
    expedite = lambda tok: tok + (";X" if rng.random() < 0.4 else "")
    turn = lambda tok: tok + (f";{_dir(rng)}" if rng.random() < 0.6 else "")
    ga_alt = lambda: str(rng.randint(3, 8))   # go-around climbs above the approach

    templates = [
        lambda: f"C {hdg()}",
        lambda: f"C {alt()}",
        lambda: f"C {expedite(alt())}",
        lambda: f"S {spd()}",
        lambda: f"S {expedite(spd())}",
        lambda: f"C {hdg()} C {alt()}",
        lambda: f"C {hdg()} C {expedite(alt())}",
        lambda: f"C {hdg()} C {alt()} S {spd()}",
        lambda: f"C {hdg()} S {spd()}",
        lambda: f"C {alt()} S {spd()}",
        lambda: f"C {turn(wpt())}",
        lambda: f"C {turn(wpt())} C {alt()}",
        lambda: f"C {turn(wpt())} C {alt()} S {spd()}",
        lambda: f"H {turn(wpt())}",
        lambda: f"C {hdg()} C {alt()} L {rwy()}",
        lambda: f"C {hdg()} C {alt()} S {spd()} L {rwy()}",
        lambda: f"C {hdg()} L {rwy()}",
        # Go-around (A): always carry an explicit heading and/or altitude so the
        # phraseology has no auto-added "fly current heading"/"maintain X" that
        # would be ambiguous to recover. The aircraft must be cleared for ILS for
        # A to validate -- build_dataset sets that up.
        lambda: f"A C {hdg()} C {ga_alt()}",
        lambda: f"A C {hdg()} C {ga_alt()} S {spd()}",
        lambda: f"A C {ga_alt()}",
    ]
    return rng.choice(templates)()


def canonicalize_target(command, atc):
    """Rewrite a command's ;L/;R/;X modifiers to match the spoken ATC line, so
    the training target is recoverable from the phraseology. `atc` is the raw
    (digit-form) controller string from aircraft.process_command()['atc'].
    """
    low = atc.lower()
    toks = command.split()
    has_l = "L" in toks                       # ILS clearance present?
    out = []
    i = 0
    while i < len(toks):
        ctype = toks[i]
        if ctype == "A":                      # go-around: no parameter
            out.append("A")
            i += 1
            continue
        param = toks[i + 1]
        i += 2
        base = param.split(";")[0]
        b = base.lower()
        if ctype == "C":
            if base.isdigit() and len(base) == 3:                       # heading
                if re.search(rf"turn left heading {base}", low):
                    param = f"{base};L"
                elif re.search(rf"turn right heading {base}", low):
                    param = f"{base};R"
                else:
                    param = base
            elif base.isdigit() and len(base) <= 2:                     # altitude
                # Folded into an ILS clearance and voiced as a plain "maintain X"
                # (no climb/descend)? Then the altitude was NOT a commanded change
                # -- it's just the aircraft's current level carried into the
                # clearance -- so drop it. Otherwise "C 5 L 27" while already at
                # 5000 is indistinguishable from "L 27" (both say "maintain 5000
                # until established"), which is label noise for the parser.
                if has_l and not re.search(r"(climb|descend) and maintain", low):
                    continue
                param = base + (";X" if re.search(r"expedite (climb|descent)", low) else "")
            else:                                                       # direct-to
                if re.search(rf"left turn direct to {b}", low):
                    param = f"{base};L"
                elif re.search(rf"right turn direct to {b}", low):
                    param = f"{base};R"
                else:
                    param = base
        elif ctype == "S":
            param = base + (";X" if re.search(r"knots, expedite", low) else "")
        elif ctype == "H":
            if re.search(rf"hold over {b} left turn", low):
                param = f"{base};L"
            else:
                param = f"{base};R"
        out += [ctype, param]
    return " ".join(out)
