# tts_training — Piper fine-tune for ATC controller voices

Fine-tune [Piper](https://github.com/OHF-Voice/piper1-gpl) (VITS) on the two cleaned
controller datasets so we can synthesize ATC-style speech and plug it into `atc-sim`.
Training runs on **Modal GPU**; inference exports to **ONNX** and runs on CPU.

## Layout
```
tts_training/
  prepare_data.py     # tts_model cleaned clips -> data/ (Piper layout + QA filter)
  modal_app.py        # Modal app: upload, fetch base ckpt, load/verify, train, export
  data/
    controller_1/{metadata.csv, wavs/*.wav}   # 156 clips, 10.2 min
    controller_2/{metadata.csv, wavs/*.wav}   # 291 clips, 16.3 min
```
`metadata.csv` is Piper single-speaker format: `<file>.wav|<transcript>`.
Audio is already 22050 Hz / mono / 16-bit (Piper-native) — **no resampling needed**.
The `wavs/` are git-ignored (they live in the Modal volume); regenerate with
`python prepare_data.py`.

## Base checkpoint
We fine-tune from **`en_US-lessac-medium`** (`epoch=2164-step=1355540.ckpt`) from
`rhasspy/piper-checkpoints`. Medium quality is the only tier Piper fine-tunes from
without extra config, and 22050 Hz matches our clips.

## Run
```bash
# local atc-sim env only needs `modal` (already installed + authed as profile jerryyin)
python tts_training/prepare_data.py                                  # populate data/
modal run tts_training/modal_app.py::upload_data                     # data -> Volume
modal run tts_training/modal_app.py::fetch_base                      # base ckpt -> Volume
modal run tts_training/modal_app.py::load_dataset --controller controller_1   # verify
modal run tts_training/modal_app.py::train --controller controller_1
modal run tts_training/modal_app.py::export --controller controller_1
# repeat train/export for controller_2 -> two separate voices
```

## Notes
- Two single-speaker fine-tunes (one per controller) = two distinct voices (goal b).
  If controller_1 (10 min) is too thin for fidelity, fall back to one merged voice.
- The "on-frequency" radio character (band-limited / squelch / compression) is a
  cheap DSP post-filter applied at inference — **not** something the model learns.
