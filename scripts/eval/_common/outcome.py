"""Three-way outcome classifier: ``correct / wrong / missing``.

This is the foundation of every model-vs-gold scoring step. Each
``(case, field, run)`` tuple is classified into one of three buckets.
Missing further decomposes into ``parse_error`` (whole-case failure) vs
``field_missing`` (case loaded but specific field null/absent).

The classifier reuses :func:`digital_registrar_research.benchmarks.eval.metrics.field_correct`
for the correct/wrong decision, so existing tolerance rules (±2 mm on
``tumor_size``, etc.) carry over unchanged. The new bit is the explicit
``missing`` flag, which today's binary correctness scoring conflates
with "wrong."

Why this matters: missing fields usually indicate runtime errors —
parse failure, schema-shy model, context-window pressure — and those
are categorically different failure modes from "model tried and got it
wrong." The ablation argument (modular DSPy vs raw_json) hinges on this
distinction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from digital_registrar_research.benchmarks.eval.metrics import (
    field_correct,
    is_attempted,
)
from digital_registrar_research.benchmarks.eval.scope import get_field_value

from .loaders import ErrorMode, LoadOutcome


OutcomeKind = Literal["correct", "wrong", "field_missing", "parse_error", "ineligible"]


@dataclass(frozen=True)
class Outcome:
    """Per-(case, field) outcome.

    Flags are mutually exclusive and exhaustive within ``eligible``
    cases. ``ineligible`` is a separate bucket for cases the field
    doesn't apply to (e.g. breast biomarkers when cancer is colorectal).

    Conventions:
        - ``parse_error``: case-level load failed — every field of this
          case is parse_error. ``error_mode`` is set on the case load
          and propagates here.
        - ``field_missing``: case loaded but this field is absent or
          null on the prediction side, while gold has a non-null value.
        - ``attempted``: the model produced a value (even if wrong).
        - ``correct``: ``attempted AND value matches gold``.
        - ``wrong``: ``attempted AND not correct``.

    Sum invariant: across an eligible cohort,
        ``correct + wrong + field_missing + parse_error == n_eligible``.
    """

    kind: OutcomeKind
    gold_present: bool
    parse_error: bool
    field_missing: bool
    attempted: bool
    correct: bool
    wrong: bool
    error_mode: ErrorMode | None = None
    pred_value: Any = None
    gold_value: Any = None

    def as_row(self) -> dict[str, Any]:
        """Flat dict for DataFrame ingestion."""
        return {
            "outcome": self.kind,
            "gold_present": self.gold_present,
            "parse_error": self.parse_error,
            "field_missing": self.field_missing,
            "attempted": self.attempted,
            "correct": self.correct,
            "wrong": self.wrong,
            "error_mode": self.error_mode,
        }


@dataclass(frozen=True)
class CaseLoad:
    """Captures how a single case loaded.

    Used by :func:`classify_outcome` to short-circuit field-level
    classification when the case-level load failed.
    """

    ok: bool
    pred: dict[str, Any] | None
    error_mode: ErrorMode | None
    error_message: str | None = None

    @classmethod
    def from_load_outcome(cls, lo: LoadOutcome) -> "CaseLoad":
        return cls(
            ok=lo.ok,
            pred=lo.data if lo.ok else None,
            error_mode=lo.error_mode,
            error_message=lo.error_message,
        )

    @classmethod
    def ok_load(cls, pred: dict[str, Any]) -> "CaseLoad":
        return cls(ok=True, pred=pred, error_mode=None, error_message=None)

    @classmethod
    def parse_failed(cls, mode: ErrorMode, message: str | None = None) -> "CaseLoad":
        return cls(ok=False, pred=None, error_mode=mode, error_message=message)


# --- Eligibility -------------------------------------------------------------

def is_field_eligible(gold: dict[str, Any], field_name: str,
                      *, organ: str | None = None) -> bool:
    """Return whether a field is eligible for scoring on a given gold case.

    Today's policy: a field is eligible if gold has the key (value or
    null). Breast biomarkers are eligible only when ``cancer_category ==
    "breast"``. Other organ-specific fields are filtered upstream by
    the caller selecting which fields to score per organ.
    """
    if field_name.startswith("biomarker_"):
        # Breast biomarkers are conditional.
        from digital_registrar_research.benchmarks.eval.metrics import normalize
        return normalize(gold.get("cancer_category")) == "breast"
    # Default: any field declared by the gold annotation is eligible.
    # Returns True even for null gold values — callers may further
    # restrict to ``gold_present`` based on their metric definition.
    if field_name in gold:
        return True
    cd = gold.get("cancer_data") or {}
    return field_name in cd


# --- Classifier --------------------------------------------------------------

def classify_outcome(
    gold: dict[str, Any],
    case_load: CaseLoad,
    field_name: str,
) -> Outcome:
    """Classify one ``(case, field)`` outcome.

    ``case_load`` summarises how the prediction file loaded — if it
    failed at the case level, every field becomes ``parse_error``.
    """
    g_value = get_field_value(gold, field_name)
    gold_present = g_value is not None

    if not case_load.ok:
        return Outcome(
            kind="parse_error",
            gold_present=gold_present,
            parse_error=True,
            field_missing=False,
            attempted=False,
            correct=False,
            wrong=False,
            error_mode=case_load.error_mode,
            pred_value=None,
            gold_value=g_value,
        )

    pred = case_load.pred or {}
    attempted_flag = is_attempted(pred, field_name)
    if not attempted_flag:
        return Outcome(
            kind="field_missing",
            gold_present=gold_present,
            parse_error=False,
            field_missing=True,
            attempted=False,
            correct=False,
            wrong=False,
            error_mode=None,
            pred_value=None,
            gold_value=g_value,
        )

    p_value = get_field_value(pred, field_name)
    correct_flag = field_correct(gold, pred, field_name)
    # ``field_correct`` returns None when not attempted; we already
    # handled that branch.
    if correct_flag is True:
        return Outcome(
            kind="correct",
            gold_present=gold_present,
            parse_error=False,
            field_missing=False,
            attempted=True,
            correct=True,
            wrong=False,
            error_mode=None,
            pred_value=p_value,
            gold_value=g_value,
        )
    return Outcome(
        kind="wrong",
        gold_present=gold_present,
        parse_error=False,
        field_missing=False,
        attempted=True,
        correct=False,
        wrong=True,
        error_mode=None,
        pred_value=p_value,
        gold_value=g_value,
    )


__all__ = [
    "Outcome",
    "OutcomeKind",
    "CaseLoad",
    "is_field_eligible",
    "classify_outcome",
]
