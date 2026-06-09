"""Evaluation harness for BC actors.

`runner.py` runs N cases per STAR in parallel, captures per-rollout outcomes
(LANDED / IMPROPER_EXIT / TIMEOUT), and writes per-STAR + overall success
rates to `eval/` at repo root.
"""
