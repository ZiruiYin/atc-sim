"""Runtime text -> DSL parser (inference).

Loads the trained t5-small checkpoint and maps an ASR transcript to a sim command
DSL string, e.g. "Delta 377, turn left heading 240, cleared ILS 27" ->
"DL377 C 240;L L 27". This is the inference counterpart to dsl_parser_training/
(which produced the checkpoint); the runtime /stt service uses it after STT and
before the validator.

The checkpoint ships with the package at dsl_parser/model/ (the ~242MB
model.safetensors is committed via Git LFS); override with ATC_PARSER_CKPT.
available() lets callers degrade gracefully (transcript-only) when no model is
present -- e.g. the static/Pyodide build.
"""
import os
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("ATC_PARSER_CKPT", os.path.join(_HERE, "model"))
PREFIX = "parse atc: "
# Greedy (beam=1) by default: a beam sweep on the eval set showed identical
# accuracy at every beam (the DSL space is tiny -> greedy already finds the best
# path), so wider beams only add latency. Override with ATC_PARSER_BEAMS.
BEAMS = int(os.environ.get("ATC_PARSER_BEAMS", "1"))
# Cap torch threads so a parse doesn't grab every core (which on a 2-vCPU host
# would starve the /step tick); paired with the process-wide no-spin set in app.py.
_PARSER_THREADS = max(2, (os.cpu_count() or 4) // 2)

_MODEL = None
_TOK = None
_LOCK = threading.Lock()


def available():
    """True if a trained checkpoint is on disk (else the service stays transcript-only)."""
    return os.path.isdir(CKPT) and os.path.exists(os.path.join(CKPT, "config.json"))


def _load():
    global _MODEL, _TOK
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                import torch
                try:
                    torch.set_num_threads(_PARSER_THREADS)
                except Exception:
                    pass
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                _TOK = AutoTokenizer.from_pretrained(CKPT)
                m = AutoModelForSeq2SeqLM.from_pretrained(CKPT)
                m.eval()
                _MODEL = m
    return _MODEL, _TOK


def parse(transcript, beams=None):
    """Transcript -> DSL command string ("<CALLSIGN> <command...>"), or "" if empty."""
    if not transcript or not transcript.strip():
        return ""
    import torch
    model, tok = _load()
    ids = tok(PREFIX + transcript, return_tensors="pt",
              truncation=True, max_length=96).input_ids
    with torch.no_grad():
        out = model.generate(ids, max_length=48, num_beams=beams or BEAMS)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def warmup():
    """Pre-load the parser model AND run one tiny parse, so the first real request
    pays neither the load NOR torch/transformers' first-inference setup (the
    generate loop is lazily built on the first call) -- which is part of why the
    first mic press felt slow even with the weights resident."""
    if available():
        _load()
        try:
            parse("warm up")
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    print("ckpt:", CKPT, "available:", available())
    if len(sys.argv) > 1:
        print("DSL:", parse(" ".join(sys.argv[1:])))
