"""Biomarker scorer for the cascade.

Re-exports :func:`scripts.eval.nested.biomarkers.score_biomarkers`
applied through the cascade's biomarker whitelist filter so any direct
caller is also clean.

Usage from cascade reductions:

    from scripts.eval.cascade.biomarkers import score_biomarkers_filtered
    stats = score_biomarkers_filtered(gold, pred, organ="colorectal")
"""
from __future__ import annotations

from digital_registrar_research.benchmarks.eval.metrics import (
    _filter_biomarkers_by_whitelist,
    normalize,
)
from digital_registrar_research.benchmarks.eval.scope import biomarkers_for_organ

# Re-export the per-case biomarker scorer for direct cascade use; the
# whitelist filter is applied above its bipartite logic so out-of-
# whitelist categories are invisible on both sides.
from scripts.eval.nested.biomarkers import score_biomarkers


def score_biomarkers_filtered(
    gold: dict, pred: dict, *, organ: str | None,
) -> dict:
    """Per-case biomarker metrics with whitelist filtering applied.

    Wraps :func:`score_biomarkers` so callers don't have to know about
    the whitelist. Wraps both gold and pred in a synthetic shape
    matching :func:`get_field_value`.
    """
    whitelist = biomarkers_for_organ(organ)
    if not whitelist:
        # Organ doesn't have a biomarker scope (e.g. lung, thyroid);
        # return a zero-record empty result.
        return {
            "biomarker_tp": 0,
            "biomarker_fp": 0,
            "biomarker_fn": 0,
            "biomarker_matched": 0,
            "biomarker_expression_correct": 0,
            "biomarker_percentage_correct_tol": 0,
            "biomarker_score_correct": 0,
            "biomarker_n_gold": 0,
            "biomarker_n_pred": 0,
        }

    g_bio = (gold.get("cancer_data") or {}).get("biomarkers") or gold.get("biomarkers")
    p_bio = (pred.get("cancer_data") or {}).get("biomarkers") or pred.get("biomarkers")
    g_filtered = _filter_biomarkers_by_whitelist(g_bio or [], organ)
    p_filtered = _filter_biomarkers_by_whitelist(p_bio or [], organ)

    gold_wrap = {"cancer_data": {"biomarkers": g_filtered}}
    pred_wrap = {"cancer_data": {"biomarkers": p_filtered}}
    return score_biomarkers(gold_wrap, pred_wrap)


__all__ = ["score_biomarkers", "score_biomarkers_filtered"]
