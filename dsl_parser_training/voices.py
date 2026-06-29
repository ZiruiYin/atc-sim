"""Base-Piper TTS voices for synthesizing controller phraseology into speech.

Uses the seven clean *American* base Piper voices (NOT the radio fine-tunes in
tts/voices -- those carry band-limited radio ambiance). Each row of the training
set picks a random voice, plus light speed/noise augmentation, so the downstream
STT (and hence the parser's text input) sees acoustic variety.

Voices download once from the HF hub (rhasspy/piper-voices) and are cached.
"""
import io
import wave

import numpy as np
from huggingface_hub import hf_hub_download
from piper import PiperVoice, SynthesisConfig

# Seven clean American voices kept after the temp_script audition (norman/john
# and the UK accents were dropped). kathleen ships only at 'low' quality.
AMERICAN_VOICES = {
    "lessac":     "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "ryan":       "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
    "amy":        "en/en_US/amy/medium/en_US-amy-medium.onnx",
    "hfc_female": "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx",
    "joe":        "en/en_US/joe/medium/en_US-joe-medium.onnx",
    "kristin":    "en/en_US/kristin/medium/en_US-kristin-medium.onnx",
    "kathleen":   "en/en_US/kathleen/low/en_US-kathleen-low.onnx",
}

_CACHE = {}


def get_voice(name):
    if name not in _CACHE:
        path = AMERICAN_VOICES[name]
        onnx = hf_hub_download("rhasspy/piper-voices", path)
        hf_hub_download("rhasspy/piper-voices", path + ".json")
        _CACHE[name] = PiperVoice.load(onnx, config_path=onnx + ".json")
    return _CACHE[name]


def random_voice(rng):
    """Return (name, PiperVoice) for a random American voice."""
    name = rng.choice(list(AMERICAN_VOICES))
    return name, get_voice(name)


def warmup():
    """Pre-download + load all voices (call once before a big batch)."""
    for name in AMERICAN_VOICES:
        get_voice(name)


def synth_wav_bytes(voice, spoken, rng):
    """Synthesize spoken text -> WAV bytes, faster (length_scale ~0.65) with very
    light hiss. Speed/noise are jittered per call for acoustic augmentation."""
    length_scale = rng.uniform(0.62, 0.72)            # ~28-38% faster than default
    snr_db = rng.choice([None, 26, 28, 30, 32, 34])   # None = clean; else light hiss

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(spoken, wf, syn_config=SynthesisConfig(length_scale=length_scale))
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        p = wf.getparams()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)

    if snr_db is not None:
        rms = np.sqrt(np.mean(samples ** 2)) + 1e-9
        samples = samples + np.random.normal(0, rms / (10 ** (snr_db / 20)), samples.shape)
    samples = np.clip(samples, -32768, 32767).astype(np.int16)

    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(p.nchannels)
        wf.setsampwidth(p.sampwidth)
        wf.setframerate(p.framerate)
        wf.writeframes(samples.tobytes())
    return out.getvalue()
