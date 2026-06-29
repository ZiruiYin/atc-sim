"""Cheap TEXT-level augmentation of cached transcripts (no TTS/STT rerun).

This is "option 2": the expensive audio->STT step is done once; here we expand the
training inputs by applying MEANING-PRESERVING surface variations to the cached
transcript. Every variant still maps to the SAME DSL target -- we never inject
unrecoverable garbage (that is the validator/error path's job, not training).

The transforms mirror variation actually seen in the TTS/STT runs, and reuse
tts/normalize.py's spelling conventions so word-forms match what the voices say:

  digit<->word numbers   "240"        <-> "two four zero", "3000" <-> "three thousand"
  grouped speed/callsign "180 knots"  ->  "one eighty knots", "Delta 377" -> "Delta three seventy seven"
  runway digit-spell     "runway 27"  ->  "runway two seven"
  comma toggle           "3000"       <-> "3,000"
  connective dropout     "descend and maintain" -> "descend maintain"
  ILS phrasing           "cleared ILS runway 27 approach" -> "cleared ILS 27"
  punctuation/case jitter   drop commas / lowercase / drop trailing period
  "correction" self-edit "heading 240" -> "heading 300, correction, 240"  (learnable)

CLI (augment the sampled rows -> sample_commands_augmented.txt):
    python -m dsl_parser_training.augment
"""
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts.normalize import _spell_digits, _speak_altitude, _callsign_number  # noqa: E402

_AIRLINES = r"(Delta|Speedbird|Air France|Lufthansa|United|American)"


# --- individual transforms (each returns possibly-modified text) --------------

def spell_heading(text):
    """'heading 240' -> 'heading two four zero' (digit-by-digit, niner for 9)."""
    return re.sub(r"\bheading (\d{2,3})\b",
                  lambda m: "heading " + _spell_digits(m.group(1).zfill(3)), text)


def word_altitude(text):
    """'maintain 3000' -> 'maintain three thousand' (altitudes only, not speeds)."""
    def repl(m):
        n = int(m.group(1).replace(",", ""))
        return f"maintain {_speak_altitude(n)}" if n >= 1000 else m.group(0)
    return re.sub(r"\bmaintain ([\d,]+)\b(?! knots)", repl, text)


def word_speed(text):
    """'180 knots' -> 'one eighty knots' (grouped, airline-number style)."""
    return re.sub(r"\b(\d{2,3}) knots\b",
                  lambda m: f"{_callsign_number(int(m.group(1)))} knots", text)


def spell_runway(text):
    """'runway 27' / 'ILS 27' -> 'runway two seven' / 'ILS two seven'."""
    text = re.sub(r"\brunway (\d{1,2})\b", lambda m: "runway " + _spell_digits(m.group(1)), text)
    text = re.sub(r"\bILS (\d{1,2})\b", lambda m: "ILS " + _spell_digits(m.group(1)), text)
    return text


def word_callsign(text):
    """'Delta 377' -> 'Delta three seventy seven'."""
    return re.sub(rf"\b{_AIRLINES} (\d{{1,3}})\b",
                  lambda m: f"{m.group(1)} {_callsign_number(int(m.group(2)))}", text)


def hyphen_heading(text):
    """'heading 250' -> 'heading 2-5-0' (Whisper often hyphenates digit strings)."""
    return re.sub(r"\bheading (\d{2,3})\b", lambda m: "heading " + "-".join(m.group(1)), text)


def hyphen_speed(text):
    """'210 knots' -> '2-1-0 knots'."""
    return re.sub(r"\b(\d{2,3}) knots\b", lambda m: "-".join(m.group(1)) + " knots", text)


def hyphen_runway(text):
    """'runway 27' / 'ILS 27' -> 'runway 2-7' / 'ILS 2-7'."""
    text = re.sub(r"\brunway (\d{1,2})\b", lambda m: "runway " + "-".join(m.group(1)), text)
    text = re.sub(r"\bILS (\d{1,2})\b", lambda m: "ILS " + "-".join(m.group(1)), text)
    return text


def split_callsign(text, rng):
    """Split a callsign number the way Whisper mangles it:
    '611' -> '6-11' / '6.11' / '6-1-1', '27' -> '2-7'. Maps to the same ICAO."""
    def repl(m):
        name, num = m.group(1), m.group(2)
        if len(num) >= 3:
            s = rng.choice(["-".join(num), num[0] + "-" + num[1:], num[0] + "." + num[1:]])
        elif len(num) == 2:
            s = rng.choice([num[0] + "-" + num[1], num])
        else:
            s = num
        return f"{name} {s}"
    return re.sub(rf"\b{_AIRLINES} (\d{{1,3}})\b", repl, text)


def drop_connectives(text):
    return (text.replace("descend and maintain", "descend maintain")
                .replace("climb and maintain", "climb maintain"))


def ils_phrasing(text, rng):
    return re.sub(r"cleared ILS runway (\d{1,2}[LRC]?) approach",
                  lambda m: rng.choice([f"cleared ILS {m.group(1)}",
                                        f"cleared for the ILS runway {m.group(1)}",
                                        m.group(0)]), text)


def toggle_commas(text, rng):
    def repl(m):
        n = m.group(0).replace(",", "")
        return n if rng.random() < 0.5 else f"{int(n):,}"
    return re.sub(r"\b\d{1,2},?\d{3}\b", repl, text)


def punct_case(text):
    return text.replace(",", "").rstrip(".").lower()


def insert_correction(text, rng):
    """Insert a distractor value + 'correction' before a heading (learnable: the
    parser must take the value AFTER the last correction)."""
    m = re.search(r"\bheading (\d{2,3})\b", text)
    if not m:
        return text
    real = m.group(1)
    wrong = real
    while wrong == real:
        wrong = f"{rng.randrange(1, 37) * 10:03d}"
    return text[:m.start()] + f"heading {wrong}, correction, {real}" + text[m.end():]


# --- composition --------------------------------------------------------------

def augment_one(transcript, rng, k=4):
    """Return up to k distinct meaning-preserving variants of `transcript`.

    Each number slot (heading / speed / runway / callsign) is rendered randomly as
    digits, spelled words, or hyphenated digits -- the three forms Whisper emits --
    so the parser learns to map all of them back to the canonical DSL.
    """
    variants = []
    for _ in range(k * 3):                       # oversample, then dedup
        t = transcript
        if rng.random() < 0.25: t = insert_correction(t, rng)   # before renderings

        r = rng.random()                          # heading: digits | words | hyphen
        if r < 0.30: t = spell_heading(t)
        elif r < 0.55: t = hyphen_heading(t)

        r = rng.random()                          # speed
        if r < 0.30: t = word_speed(t)
        elif r < 0.55: t = hyphen_speed(t)

        r = rng.random()                          # runway
        if r < 0.30: t = spell_runway(t)
        elif r < 0.50: t = hyphen_runway(t)

        r = rng.random()                          # callsign
        if r < 0.25: t = word_callsign(t)
        elif r < 0.50: t = split_callsign(t, rng)

        if rng.random() < 0.50: t = word_altitude(t)
        if rng.random() < 0.50: t = drop_connectives(t)
        if rng.random() < 0.30: t = ils_phrasing(t, rng)
        if rng.random() < 0.40: t = toggle_commas(t, rng)
        if rng.random() < 0.30: t = punct_case(t)
        if t != transcript and t not in variants:
            variants.append(t)
        if len(variants) >= k:
            break
    return variants


def _parse_pairs(path):
    """Yield (dsl, transcript) from a sample_pairs.txt file. The leading
    '[voice]' tag is optional (older files had it)."""
    rx = re.compile(r"^(?:\[[^\]]+\]\s+)?(.+?)\s+->\s+(.+)$")
    for line in open(path, encoding="utf-8"):
        m = rx.match(line.strip())
        if m and m.group(2) != "<EMPTY>":
            yield m.group(1), m.group(2)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "sample_pairs.txt")
    dst = os.path.join(here, "sample_commands_augmented.txt")
    rng = random.Random(7)

    pairs = list(_parse_pairs(src))
    with open(dst, "w", encoding="utf-8") as f:
        f.write("# Text-level augmentation of the sampled rows (no TTS/STT rerun).\n")
        f.write("# Each block: TARGET DSL, the cached transcript, then augmented input variants.\n\n")
        total = 0
        for dsl, transcript in pairs:
            f.write(f"{dsl}\n")
            f.write(f"  cached: {transcript}\n")
            for v in augment_one(transcript, rng, k=4):
                f.write(f"  aug   : {v}\n")
                total += 1
            f.write("\n")
    print(f"{len(pairs)} commands -> {total} augmented variants")
    print("wrote", dst)


if __name__ == "__main__":
    main()
