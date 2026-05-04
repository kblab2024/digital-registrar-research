"""Non-nested (scalar) field evaluation subcommand.

Scores categorical, boolean, ordinal, and continuous single-value
fields. Three-way outcome classification (correct / wrong / missing)
with Wilson CI on every rate, Cohen's κ + MCC + per-class P/R/F1,
schema-conformance, refusal calibration, and multi-primary
stratification.

Entry point: ``python -m scripts.eval.cli non_nested ...``
"""
