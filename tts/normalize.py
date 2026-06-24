"""Rules-only normalizer: ATC/pilot command string -> spoken (TTS-ready) text.

No models, no ML -- pure deterministic rules tuned to this sim's phraseology
(environment/core/aircraft.py:_build_radio_messages).

Callsign decoupling is deterministic, so it needs no pattern matching:
  * controller message -> "<callsign>, <instruction...>"   (callsign FIRST)
  * pilot readback     -> "<instruction...>, <callsign>"   (callsign LAST)
Pass is_controller to pick which end the callsign sits on.

Number reading follows aviation convention:
  * callsign number  -> grouped airline style: 329 -> "three twenty nine"
  * heading/runway   -> digit-by-digit, 9 -> "niner": 030 -> "zero three zero"
  * altitude (feet)  -> "four thousand", "one one thousand", "three thousand five hundred"
  * speed            -> digit-by-digit + "knots": 180 -> "one eight zero knots"

CLI:
    python tts/normalize.py "DL329, turn right heading 030, descend and maintain 4000"
    python tts/normalize.py --pilot "Turn right heading 030, ..., DL329"
"""
from __future__ import annotations

import re

# Callsign numbers are spoken naturally ("nine", not "niner").
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEENS = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
         15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"}
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
        6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}

# Headings/runways/speeds are spelled digit-by-digit, with aviation "niner".
DIGIT_RADIO = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
               "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "niner"}

# ICAO prefix -> spoken telephony callsign. Prefixes from environment/params.py.
AIRLINES = {
    "DL": "Delta", "BA": "Speedbird", "AF": "Air France",
    "LH": "Lufthansa", "UA": "United", "AA": "American",
}

SIDE = {"L": "left", "R": "right", "C": "center"}


def _two_digit(n: int) -> str:
    """0..99 -> natural English ('forty two', 'nineteen', 'seven')."""
    if n < 10:
        return ONES[n]
    if n < 20:
        return TEENS[n]
    tens, ones = divmod(n, 10)
    return TENS[tens] + (f" {ONES[ones]}" if ones else "")


def _callsign_number(n: int) -> str:
    """1..999 grouped airline style: 7->'seven', 42->'forty two',
    329->'three twenty nine', 305->'three oh five', 500->'five hundred'.
    (Callsign numbers are 1-3 digits here -- randint(1, 999), never 4.)"""
    if n < 100:
        return _two_digit(n)
    hundreds, last_two = divmod(n, 100)
    if last_two == 0:
        return f"{ONES[hundreds]} hundred"
    if last_two < 10:
        return f"{ONES[hundreds]} oh {ONES[last_two]}"
    return f"{ONES[hundreds]} {_two_digit(last_two)}"


def _spell_digits(s: str) -> str:
    """'030' -> 'zero three zero', '27' -> 'two seven' (niner for 9)."""
    return " ".join(DIGIT_RADIO[c] for c in s if c in DIGIT_RADIO)


def _speak_altitude(feet: int) -> str:
    """4000->'four thousand', 11000->'one one thousand',
    3500->'three thousand five hundred', 800->'eight hundred'."""
    thousands, rem = divmod(feet, 1000)
    parts = []
    if thousands:
        if thousands < 10:
            parts.append(f"{ONES[thousands]} thousand")
        else:                                   # 10..18 -> 'one zero'..'one eight'
            parts.append(f"{_spell_digits(str(thousands))} thousand")
    hundreds = rem // 100
    if hundreds:
        parts.append(f"{ONES[hundreds]} hundred")
    return " ".join(parts) if parts else ONES[0]


def _speak_runway(rw: str) -> str:
    """'27R' -> 'two seven right', '09L' -> 'zero niner left', '36' -> 'three six'."""
    m = re.match(r"(\d{1,2})([LRC]?)$", rw)
    if not m:
        return _spell_digits(re.sub(r"\D", "", rw))
    digits, side = m.group(1), m.group(2)
    out = _spell_digits(digits)
    return out + (f" {SIDE[side]}" if side else "")


def _speak_callsign(cs: str) -> str:
    """'DL329' -> 'Delta three twenty nine'."""
    m = re.match(r"([A-Za-z]+)\s*(\d+)$", cs.strip())
    if not m:
        return cs
    prefix, num = m.group(1).upper(), int(m.group(2))
    airline = AIRLINES.get(prefix, prefix)
    return f"{airline} {_callsign_number(num)}"


def _normalize_body(s: str) -> str:
    """Spell out the numeric tokens inside an instruction phrase. Speed is handled
    before altitude so 'maintain 200 knots' isn't read as an altitude."""
    s = re.sub(r"heading (\d{3})",
               lambda m: "heading " + _spell_digits(m.group(1)), s)
    s = re.sub(r"(\d+) knots",
               lambda m: _spell_digits(m.group(1)) + " knots", s)
    s = re.sub(r"(climb and maintain|descend and maintain|maintain) (\d+)",
               lambda m: f"{m.group(1)} {_speak_altitude(int(m.group(2)))}", s)
    s = re.sub(r"runway (\d{1,2}[LRC]?)",
               lambda m: "runway " + _speak_runway(m.group(1)), s)
    return s


def to_spoken(text: str, is_controller: bool) -> str:
    """Convert a raw sim command/readback string into spoken TTS text.

    is_controller=True  -> callsign is the first comma-field (ATC instruction)
    is_controller=False -> callsign is the last  comma-field (pilot readback)
    """
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return text
    if is_controller:
        cs_raw, body = parts[0], parts[1:]
    else:
        cs_raw, body = parts[-1], parts[:-1]

    callsign = _speak_callsign(cs_raw)
    body_spoken = [_normalize_body(b) for b in body]
    fields = [callsign] + body_spoken if is_controller else body_spoken + [callsign]
    out = ", ".join(fields)
    return out[:1].upper() + out[1:]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Spell an ATC/pilot line for TTS.")
    ap.add_argument("text", help="raw command/readback string")
    ap.add_argument("--pilot", action="store_true",
                    help="treat as pilot readback (callsign at the END)")
    a = ap.parse_args()
    print("INPUT :", a.text)
    print("SPOKEN:", to_spoken(a.text, is_controller=not a.pilot))
