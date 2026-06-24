"""Local CPU text-to-speech for the sim (Piper / onnxruntime, no GPU).

Two voices live in tts/voices/:
    controller.onnx (+.json)  -- controller_1 fine-tune  (ATC voice)
    pilot.onnx      (+.json)  -- controller_2 fine-tune  (pilot voice)

`controller_voice` (bool) switches the checkpoint. Voices are loaded once and
cached, so in a long-running process (Flask) only the first call pays load cost.

Output is a 22050 Hz mono 16-bit WAV (bytes) -- directly playable in a browser
via an <audio>/Audio() blob, which is how the frontend consumes it.

CLI (writes a wav so you can listen):
    python tts/synth.py "Delta three twenty nine, descend and maintain four thousand."
    python tts/synth.py --pilot --raw "Turn right heading 030, ..., DL329"
"""
from __future__ import annotations

import io
import os
import sys
import wave
from pathlib import Path

import onnxruntime as ort
from piper import PiperVoice

# Make `normalize` importable whether this is run as a script or imported as a
# package module (Flask: `from tts.synth import ...`).
sys.path.insert(0, os.path.dirname(__file__))
from normalize import to_spoken  # noqa: E402

VOICES_DIR = Path(__file__).parent / "voices"
_CACHE: dict[str, PiperVoice] = {}

# Piper's default onnxruntime session uses ALL cores AND spin-waits (busy-loops)
# them, which starves the 1s sim tick (/step) during synthesis and makes the
# radar stutter. Cap threads (~1/3 of cores) and disable spinning so plenty of
# cores stay free for /step -- even if a controller + pilot synth briefly overlap.
_SYNTH_THREADS = max(2, (os.cpu_count() or 4) // 3)


def _build_session(onnx_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = _SYNTH_THREADS
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def _voice(controller_voice: bool) -> PiperVoice:
    key = "controller" if controller_voice else "pilot"
    if key not in _CACHE:
        onnx = VOICES_DIR / f"{key}.onnx"
        cfg = VOICES_DIR / f"{key}.onnx.json"
        if not onnx.exists():
            raise FileNotFoundError(f"missing voice {onnx}")
        voice = PiperVoice.load(str(onnx), config_path=str(cfg))
        voice.session = _build_session(str(onnx))   # CPU-friendly session
        _CACHE[key] = voice
    return _CACHE[key]


def synthesize_wav_bytes(text: str, controller_voice: bool) -> bytes:
    """Synthesize already-spoken text -> WAV bytes. `text` should be the spelled
    output of normalize.to_spoken (e.g. 'Delta three twenty nine, ...')."""
    voice = _voice(controller_voice)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    return buf.getvalue()


def render(text: str, is_controller: bool, controller_voice: bool | None = None) -> bytes:
    """Full path used by the frontend: raw sim string -> normalize -> WAV bytes.

    is_controller selects callsign placement for normalization; controller_voice
    selects the checkpoint (defaults to is_controller: ATC line -> controller voice,
    pilot readback -> pilot voice)."""
    if controller_voice is None:
        controller_voice = is_controller
    spoken = to_spoken(text, is_controller)
    return synthesize_wav_bytes(spoken, controller_voice)


def warmup() -> None:
    """Pre-load both voices (call at server start to avoid first-request latency)."""
    _voice(True)
    _voice(False)


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Synthesize a line to a wav file.")
    ap.add_argument("text", help="text to speak")
    ap.add_argument("--pilot", action="store_true",
                    help="use pilot voice / pilot callsign placement")
    ap.add_argument("--raw", action="store_true",
                    help="text is a raw sim string -> run normalize first")
    ap.add_argument("-o", "--out", default=str(Path(__file__).parent / "_synth_test.wav"))
    a = ap.parse_args()

    controller_voice = not a.pilot
    if a.raw:
        spoken = to_spoken(a.text, is_controller=controller_voice)
        print("SPOKEN:", spoken)
    else:
        spoken = a.text
    t0 = time.time()
    wav = synthesize_wav_bytes(spoken, controller_voice)
    Path(a.out).write_bytes(wav)
    print(f"wrote {a.out} ({len(wav)} bytes) in {time.time()-t0:.2f}s "
          f"[{'controller' if controller_voice else 'pilot'} voice]")
