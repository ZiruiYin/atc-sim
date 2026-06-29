# dsl_parser_training — text → DSL parser training

Trains the **dsl_parser**: raw ASR transcript → sim command DSL (e.g.
`"Delta 377, turn left heading 240, descend maintain three thousand, cleared ILS
two seven"` → `DL377 C 240;L C 3 L 27`). The trained model plugs into the runtime
`dsl_parser` (parser → validator → execute / error).

## Model
**t5-small** (~60M, encoder-decoder Transformer), token-level cross-entropy
(teacher forced). Small = low-latency CPU inference, and the DSL output space is
tiny so it learns fast. ByT5-small is the byte-level fallback if subword
tokenization of digits/callsigns is brittle (≈5× bigger/slower).

## Pipeline
```
random valid command            generate_commands.random_command
  -> Aircraft.process_command   VALIDATOR FILTER (keep ok=True) -> ATC phraseology
  -> canonicalize_target        DSL label made recoverable from the spoken line
  -> tts.normalize.to_spoken    spoken text
  -> voices.synth_wav_bytes     base-Piper American voice (random) + speed/noise aug
  -> stt.transcribe             Whisper/Moonshine  -> transcript
=> {"input": transcript, "target": "<CALLSIGN> <DSL>"}
```
Each run draws a **fresh random batch**. The validator is the valid-token filter;
empty-transcript rows are dropped.

## Commands
```bash
# fresh batch (local: needs Piper voices + STT)
python -m dsl_parser_training.build_dataset --n 8000 --out dsl_parser_training/data/batch.jsonl

# quick verify: 50 rows + sample_commands.txt + sample_pairs.txt
python -m dsl_parser_training.build_dataset --n 50 --verify --seed 7

# train locally (slow) ...
python -m dsl_parser_training.train --data dsl_parser_training/data/batch.jsonl --out dsl_parser_training/runs/v0
# ... or on Modal GPU
modal run dsl_parser_training/modal_app.py::upload_cli --local dsl_parser_training/data/batch.jsonl
modal run dsl_parser_training/modal_app.py::train --epochs 8
modal run dsl_parser_training/modal_app.py::download --dest dsl_parser_training/runs/v0
```

## Spec ranges
altitude 1000–8000 (`C 1`..`C 8`), heading 010–360 (3-digit), speed 140–240 (tens),
plus direct-to / hold over the 23 RNAV waypoints and ILS to runway 09/27.

## Voices
Seven clean **American** base Piper voices (lessac, ryan, amy, hfc_female, joe,
kristin, kathleen) — NOT the radio fine-tunes in `tts/voices`.
