"""Speech-to-text for the sim: controller audio -> raw text (part 1).

Two interchangeable CPU backends, selected by the ATC_STT_BACKEND env var:
  * "whisper" (default)  -- stt/whisper_asr.py, faster-whisper / CTranslate2
  * "moonshine"          -- stt/moonshine_asr.py, low-latency ONNX (~0.3s/clip)

Within a backend, ATC_STT_MODEL overrides the model (e.g. "small.en" for Whisper,
or "moonshine/tiny"). The downstream step (raw text -> sim command DSL) is part 2
and lives elsewhere.
"""
import os

BACKEND = os.environ.get("ATC_STT_BACKEND", "whisper").strip().lower()
if BACKEND == "moonshine":
    from stt.moonshine_asr import transcribe, warmup
else:
    BACKEND = "whisper"
    from stt.whisper_asr import transcribe, warmup

__all__ = ["transcribe", "warmup", "BACKEND"]
