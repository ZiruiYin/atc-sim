"""Local CPU speech-to-text for the sim via Moonshine (ONNX runtime, no GPU).

One of two interchangeable STT backends (the other is stt/whisper_asr.py);
stt/__init__.py picks between them via ATC_STT_BACKEND. Part 1 of the voice
pipeline: controller *audio* -> *raw text*.

Moonshine is built for low-latency voice commands: its compute scales with the
*actual* audio length (Whisper always pads to 30s internally), so short clips
transcribe much faster -- ~0.3s for a few-second clip here vs ~1.2s for Whisper
base.en.

IMPORTANT: load the model ONCE and reuse it. The convenience function
moonshine_onnx.transcribe() rebuilds the ONNX sessions on every call (~4-5s);
reusing a single MoonshineOnnxModel instance is what gets the ~0.3s. Models
(`moonshine/base` default, `moonshine/tiny` smaller) download once from the HF
hub; override with ATC_STT_MODEL.

Audio in is whatever the browser's MediaRecorder produced (webm/opus); we decode
it to 16 kHz mono float32 with faster-whisper's av-based decoder (already a dep),
so no separate ffmpeg/librosa is needed (Moonshine's own loader pulls numba,
which we deliberately skip on Python 3.9).

CLI (transcribe a file so you can sanity-check):
    python stt/moonshine_asr.py path/to/clip.webm
"""
from __future__ import annotations

import io
import os

# onnxruntime busy-waits (spins) its threads by default, which can blip the 1s
# radar tick. PASSIVE keeps idle threads from burning cores between the short
# bursts of STT work. Must be set before onnxruntime is imported.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
from faster_whisper.audio import decode_audio

MODEL_NAME = os.environ.get("ATC_STT_MODEL", "moonshine/base")

# Below this RMS the clip is treated as silence -> "". Moonshine has no
# no-speech score (unlike Whisper), so we gate on energy to avoid hallucinating
# a transcript on an empty mic press. Tune up if quiet speech gets dropped.
_SILENCE_RMS = 0.006

_MODEL = None
_TOKENIZER = None


def _load():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        _MODEL = MoonshineOnnxModel(model_name=MODEL_NAME)
        _TOKENIZER = load_tokenizer()
    return _MODEL, _TOKENIZER


def transcribe(audio: bytes) -> str:
    """Transcribe encoded audio bytes (webm/opus/wav/...) -> raw text.

    Returns the stripped transcript, or "" if the clip was empty/silent.
    """
    if not audio:
        return ""
    samples = decode_audio(io.BytesIO(audio), sampling_rate=16000).astype(np.float32)
    if samples.size == 0 or float(np.sqrt(np.mean(samples ** 2))) < _SILENCE_RMS:
        return ""
    model, tok = _load()
    tokens = model.generate(samples[np.newaxis, :])
    return tok.decode_batch(tokens)[0].strip()


def _warm_clip() -> bytes:
    """0.5s of a quiet tone as 16 kHz mono 16-bit WAV bytes. Above the silence-RMS
    gate so warmup actually runs the model (silence would be dropped early)."""
    import math
    import struct
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(2000 * math.sin(i * 0.1)))
                               for i in range(8000)))
    return buf.getvalue()


def warmup() -> None:
    """Pre-load the model AND run one tiny decode, so the first real request pays
    neither the load NOR onnxruntime's first-call setup."""
    _load()
    try:
        transcribe(_warm_clip())
    except Exception:
        pass


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("usage: python stt/moonshine_asr.py <audio-file>", file=sys.stderr)
        sys.exit(1)
    data = open(sys.argv[1], "rb").read()
    t0 = time.time()
    text = transcribe(data)
    print(f"[{MODEL_NAME}] {time.time()-t0:.2f}s")
    print("TRANSCRIPT:", text)
