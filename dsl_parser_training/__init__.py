"""Training pipeline for the dsl_parser (raw ASR text -> sim command DSL).

Modules:
  generate_commands  -- random valid-command sampler + target canonicalization
  voices             -- base-Piper American TTS voices (random per row + augmentation)
  build_dataset      -- command -> validator -> phraseology -> TTS -> STT -> (input, target)
  train              -- t5-small seq2seq fine-tune (local)
  modal_app          -- same training on Modal GPU
"""
