"""Per-case scorer for the ``biomarkers`` nested-list field.

Currently populated only for breast and colorectal in the canonical
schemas. Other organs return ``None`` (skipped). Each biomarker is
keyed by ``biomarker_category`` (er, pr, her2, ki67, msi, ...).

Returns a flat dict in the same idiom as ``nested_metrics.score_lymph_nodes``
so the nested orchestrator's reduce step is uniform across fields.
"""
from __future__ import annotations

from typing import Any

from digital_registrar_research.benchmarks.eval.metrics import normalize
from digital_registrar_research.benchmarks.eval.scope import get_field_value


def _safe_list(v: Any) -> list[dict]:
    return v if isinstance(v, list) else []


def _bm_similarity(g: dict, p: dict) -> float:
    """Score a (gold, pred) biomarker pair.

    Primary key: biomarker_category — exact match required for any
    similarity > 0. Tie-break on expression match.
    """
    score = 0.0
    gc = normalize(g.get("biomarker_category"))
    pc = normalize(p.get("biomarker_category"))
    if gc is None or pc is None or gc != pc:
        return 0.0
    score += 3.0
    g_exp = normalize(g.get("expression"))
    p_exp = normalize(p.get("expression"))
    if g_exp is not None and p_exp is not None and g_exp == p_exp:
        score += 1.0
    g_pct = g.get("percentage")
    p_pct = p.get("percentage")
    if (isinstance(g_pct, (int, float)) and isinstance(p_pct, (int, float))
            and abs(g_pct - p_pct) <= 5):
        score += 0.5
    return score


def score_biomarkers(gold: dict, pred: dict) -> dict:
    """Per-case biomarker metrics.

    Keys:
        biomarker_tp/fp/fn/matched
        biomarker_category_correct (matched cases with same category — by
        construction always 1.0 if matched at all)
        biomarker_expression_correct (out of matched, exact expression match)
        biomarker_percentage_correct_tol (within ±5 percentage points)
        biomarker_score_correct (exact match on score field)
    """
    from digital_registrar_research.benchmarks.eval.nested_metrics import _greedy_match

    g_list = _safe_list(get_field_value(gold, "biomarkers"))
    p_list = _safe_list(get_field_value(pred, "biomarkers"))

    matched, unm_g, unm_p = _greedy_match(g_list, p_list, _bm_similarity)
    tp = len(matched)
    fp = len(unm_p)
    fn = len(unm_g)

    expression_ok = percentage_ok = score_ok = 0
    for gi, pi in matched:
        g, p = g_list[gi], p_list[pi]
        if normalize(g.get("expression")) == normalize(p.get("expression")):
            expression_ok += 1
        gp, pp = g.get("percentage"), p.get("percentage")
        if gp is None and pp is None:
            percentage_ok += 1
        elif (isinstance(gp, (int, float)) and isinstance(pp, (int, float))
                and abs(gp - pp) <= 5):
            percentage_ok += 1
        if normalize(g.get("score")) == normalize(p.get("score")):
            score_ok += 1

    return {
        "biomarker_tp": tp,
        "biomarker_fp": fp,
        "biomarker_fn": fn,
        "biomarker_matched": tp,
        "biomarker_expression_correct": expression_ok,
        "biomarker_percentage_correct_tol": percentage_ok,
        "biomarker_score_correct": score_ok,
        "biomarker_n_gold": len(g_list),
        "biomarker_n_pred": len(p_list),
    }


__all__ = ["score_biomarkers"]
