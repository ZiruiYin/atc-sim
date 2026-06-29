"""Local CPU speech-to-text for the sim (faster-whisper / CTranslate2, no GPU).

One of two interchangeable STT backends (the other is stt/moonshine_asr.py);
stt/__init__.py picks between them via ATC_STT_BACKEND. Part 1 of the voice
pipeline: controller *audio* -> *raw text*. The raw text is later parsed into a
sim command string (part 2, text->DSL) -- that step lives elsewhere; this module
stops at the transcript.

Model: Whisper `base.en` quantized to int8 (CTranslate2). ~140 MB, downloaded
once from the HF hub on first use and cached under ~/.cache/huggingface. `base.en`
is the accuracy/latency sweet spot for short push-to-talk clips on a 2-vCPU host;
override with ATC_STT_MODEL (e.g. "tiny.en" for less latency, "small.en" for more
accuracy). The model is loaded once and cached, so in a long-running process
(Flask) only the first call pays the load cost -- call warmup() at server start.

Audio in is whatever the browser's MediaRecorder produced (webm/opus); PyAV (av),
a faster-whisper dependency, decodes it from raw bytes -- no ffmpeg binary needed.

CLI (transcribe a file so you can sanity-check):
    python stt/whisper_asr.py path/to/clip.webm
"""
from __future__ import annotations

import io
import os

from faster_whisper import WhisperModel

MODEL_NAME = os.environ.get("ATC_STT_MODEL", "base.en")

# CTranslate2 defaults to ALL cores. The TTS path caps threads hard because it
# synthesizes on every command; STT is burstier (one ~1s transcription per mic
# press), so it can safely use more cores for lower latency without making the
# radar tick limp. Half the cores is a good balance.
_STT_THREADS = max(4, (os.cpu_count() or 4) // 2)

# Seed the decoder with this sim's phraseology so callsign telephony and the
# fixed command vocabulary bias the output (Whisper's initial_prompt is just a
# textual hint -- it is NOT a hard constraint). Kept short and limited to the
# airline telephony names + core verbs, which measurably helped callsign/heading
# recognition; rarer tail phrases (go around / hold / expedite) were dropped
# because the decoder would sometimes hallucinate them into otherwise-clean text.
_BIAS_PROMPT = (
    "Air traffic control. Delta, Speedbird, Air France, Lufthansa, United, "
    "American. Turn left heading, turn right heading, climb and maintain, "
    "descend and maintain, reduce speed, increase speed, cleared ILS runway approach."
)

_MODEL: WhisperModel | None = None


def _model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = WhisperModel(
            MODEL_NAME,
            device="cpu",
            compute_type="int8",
            cpu_threads=_STT_THREADS,
            num_workers=1,
        )
    return _MODEL


def transcribe(audio: bytes) -> str:
    """Transcribe encoded audio bytes (webm/opus/wav/...) -> raw text.

    Returns the stripped transcript (joined across segments), or "" if the clip
    was empty/silent. Uses beam search (beam_size=5) for accuracy on these short
    single-utterance clips.
    """
    if not audio:
        return ""
    segments, _info = _model().transcribe(
        io.BytesIO(audio),
        language="en",
        # Beam search (beam_size=5) for accuracy; no timestamp tokens to keep the
        # decode lean. The clips are short single utterances.
        beam_size=3,
        without_timestamps=True,
        # No VAD: the mic flow is click-to-stop (little dead air), and the Silero
        # VAD will occasionally discard a whole short clip -> empty transcript.
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=_BIAS_PROMPT,
    )
    # On silence / no real speech Whisper tends to echo the initial_prompt back
    # ("Air traffic control...") or emit low-confidence junk. Drop any segment the
    # model itself flags as probably-silence or low-confidence, so an empty press
    # returns "" instead of a hallucinated transcript.
    parts = [
        seg.text.strip() for seg in segments
        if seg.no_speech_prob < 0.6 and seg.avg_logprob > -1.0 and seg.text.strip()
    ]
    return " ".join(parts).strip()


def _warm_clip() -> bytes:
    """0.5s of a quiet tone as 16 kHz mono 16-bit WAV bytes -- a representative
    input to exercise the full decode path during warmup."""
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
    neither the load NOR CTranslate2's first-call setup (lazy buffer allocation /
    kernel selection) -- which is part of why the first mic press felt slow."""
    _model()
    try:
        transcribe(_warm_clip())
    except Exception:
        pass


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("usage: python stt/whisper_asr.py <audio-file>", file=sys.stderr)
        sys.exit(1)
    data = open(sys.argv[1], "rb").read()
    t0 = time.time()
    text = transcribe(data)
    print(f"[{MODEL_NAME}] {time.time()-t0:.2f}s")
    print("TRANSCRIPT:", text)
