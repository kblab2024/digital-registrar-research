"""
Per-field metrics for the two nested-list extraction targets
(`regional_lymph_node`, `margins`) that the flat-scoped `metrics.py`
reduces to a single F1.

These are reported against generative methods only — `rules.py`,
`clinicalbert_cls.py` and `clinicalbert_qa.py` do not emit list-of-dicts
output by design.

Two axes per field:
  - Case-level clinical summary (total counts, any-positive, closest
    margin distance) — what actually drives a treatment decision.
  - Per-item structural accuracy — did we identify the right
    stations / margins, and are their inner values correct.

Matching uses greedy bipartite descent on a similarity score, with a
composite key fallback (so a null `station_name` or `"others"` margin
category doesn't collapse to position-based zip the way
`metrics.match_nested_list` does).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .metrics import NUMERIC_TOLERANCE_MM, normalize
from .scope import get_field_value

LN_COUNT_TOLERANCE = 1  # +/- 1 node on examined / involved counts


# --- Token utilities for margin description Jaccard --------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str | None) -> set[str]:
    if not s:
        return set()
    return set(_TOKEN_RE.findall(s.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- Greedy bipartite matching -----------------------------------------------

def _greedy_match(gold_list: list[dict], pred_list: list[dict],
                  sim_fn) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Returns (matched_pairs, unmatched_gold_idx, unmatched_pred_idx).

    Iteratively picks the (g, p) pair with the highest positive similarity,
    removes both from the pool, repeats until nothing positive remains.
    Quadratic but fine for the ~1-10 items per case we see in practice.
    """
    if not gold_list or not pred_list:
        return [], list(range(len(gold_list))), list(range(len(pred_list)))

    sims = [
        [sim_fn(g, p) for p in pred_list]
        for g in gold_list
    ]
    gold_avail = set(range(len(gold_list)))
    pred_avail = set(range(len(pred_list)))
    matched: list[tuple[int, int]] = []

    while gold_avail and pred_avail:
        best = (0.0, -1, -1)
        for gi in gold_avail:
            for pi in pred_avail:
                s = sims[gi][pi]
                if s > best[0]:
                    best = (s, gi, pi)
        if best[1] < 0:
            break
        _, gi, pi = best
        matched.append((gi, pi))
        gold_avail.remove(gi)
        pred_avail.remove(pi)

    return matched, sorted(gold_avail), sorted(pred_avail)


# --- Similarity functions ----------------------------------------------------

def _ln_similarity(g: dict, p: dict) -> float:
    """Score a (gold, pred) lymph-node pair. Higher = better match.

    Primary-key components (all three optional in schema):
        station_name          +3 if both non-null and equal
        lymph_node_category   +2 if both non-null and equal (excluding "others")
        lymph_node_side       +1 if both non-null and equal

    Tie-break: +0.5 each if involved / examined are within tolerance.
    """
    score = 0.0
    gs, ps = normalize(g.get("station_name")), normalize(p.get("station_name"))
    if gs is not None and ps is not None and gs == ps:
        score += 3.0
    gc, pc = normalize(g.get("lymph_node_category")), normalize(p.get("lymph_node_category"))
    if gc is not None and pc is not None and gc == pc and gc != "others":
        score += 2.0
    gside, pside = normalize(g.get("lymph_node_side")), normalize(p.get("lymph_node_side"))
    if gside is not None and pside is not None and gside == pside:
        score += 1.0
    gi, pi = g.get("involved"), p.get("involved")
    if isinstance(gi, (int, float)) and isinstance(pi, (int, float)) \
            and abs(gi - pi) <= LN_COUNT_TOLERANCE:
        score += 0.5
    ge, pe = g.get("examined"), p.get("examined")
    if isinstance(ge, (int, float)) and isinstance(pe, (int, float)) \
            and abs(ge - pe) <= LN_COUNT_TOLERANCE:
        score += 0.5
    return score


def _margin_similarity(g: dict, p: dict) -> float:
    """Score a (gold, pred) margin pair.

    margin_category        +3 if both equal and not in {None, "others"}
    description Jaccard    +0..3 scaled by token overlap
    margin_involved        +1 if agree (tie-break only)
    """
    score = 0.0
    gc, pc = normalize(g.get("margin_category")), normalize(p.get("margin_category"))
    if gc is not None and pc is not None and gc == pc and gc != "others":
        score += 3.0
    j = _jaccard(_tokens(g.get("description")), _tokens(p.get("description")))
    score += 3.0 * j
    if g.get("margin_involved") is not None and p.get("margin_involved") is not None \
            and bool(g["margin_involved"]) == bool(p["margin_involved"]):
        score += 1.0
    return score


# --- Per-case scoring --------------------------------------------------------

def _safe_list(v) -> list[dict]:
    return v if isinstance(v, list) else []


def score_lymph_nodes(gold: dict, pred: dict) -> dict:
    """Returns a flat dict of per-case metrics for `regional_lymph_node`.

    Keys are suitable for aggregation:
      ln_examined_total_{gold,pred,abs_err,correct_tol}
      ln_involved_total_{gold,pred,abs_err,correct_tol}
      ln_any_positive_{gold,pred,correct}
      ln_station_{tp,fp,fn,matched,
                  involved_correct, examined_correct,
                  category_correct, side_correct}
    """
    g_list = _safe_list(get_field_value(gold, "regional_lymph_node"))
    p_list = _safe_list(get_field_value(pred, "regional_lymph_node"))

    def _sum(items, key):
        total = 0
        for x in items:
            v = x.get(key)
            if isinstance(v, (int, float)):
                total += int(v)
        return total

    g_exam, p_exam = _sum(g_list, "examined"), _sum(p_list, "examined")
    g_inv, p_inv = _sum(g_list, "involved"), _sum(p_list, "involved")
    g_any = g_inv > 0
    p_any = p_inv > 0

    matched, unm_g, unm_p = _greedy_match(g_list, p_list, _ln_similarity)
    tp = len(matched)
    fp = len(unm_p)
    fn = len(unm_g)

    involved_ok = examined_ok = category_ok = side_ok = 0
    for gi, pi in matched:
        g, p = g_list[gi], p_list[pi]
        gi_v, pi_v = g.get("involved"), p.get("involved")
        if isinstance(gi_v, (int, float)) and isinstance(pi_v, (int, float)) \
                and abs(gi_v - pi_v) <= LN_COUNT_TOLERANCE:
            involved_ok += 1
        ge_v, pe_v = g.get("examined"), p.get("examined")
        if isinstance(ge_v, (int, float)) and isinstance(pe_v, (int, float)) \
                and abs(ge_v - pe_v) <= LN_COUNT_TOLERANCE:
            examined_ok += 1
        if normalize(g.get("lymph_node_category")) == normalize(p.get("lymph_node_category")):
            category_ok += 1
        if normalize(g.get("lymph_node_side")) == normalize(p.get("lymph_node_side")):
            side_ok += 1

    return {
        "ln_examined_total_gold": g_exam,
        "ln_examined_total_pred": p_exam,
        "ln_examined_total_abs_err": abs(g_exam - p_exam),
        "ln_examined_total_correct_tol": int(abs(g_exam - p_exam) <= LN_COUNT_TOLERANCE),
        "ln_involved_total_gold": g_inv,
        "ln_involved_total_pred": p_inv,
        "ln_involved_total_abs_err": abs(g_inv - p_inv),
        "ln_involved_total_correct_tol": int(abs(g_inv - p_inv) <= LN_COUNT_TOLERANCE),
        "ln_any_positive_gold": int(g_any),
        "ln_any_positive_pred": int(p_any),
        "ln_any_positive_correct": int(g_any == p_any),
        "ln_station_tp": tp,
        "ln_station_fp": fp,
        "ln_station_fn": fn,
        "ln_station_matched": tp,
        "ln_station_involved_correct": involved_ok,
        "ln_station_examined_correct": examined_ok,
        "ln_station_category_correct": category_ok,
        "ln_station_side_correct": side_ok,
    }


def score_margins(gold: dict, pred: dict) -> dict:
    """Returns a flat dict of per-case metrics for `margins`.

    Case-level:
      margin_any_involved_{gold,pred,correct}
      margin_closest_distance_{gold,pred,abs_err,correct_tol,has_both}
    Per-item:
      margin_{tp,fp,fn,matched,
              status_correct, distance_correct, category_correct}
    """
    g_list = _safe_list(get_field_value(gold, "margins"))
    p_list = _safe_list(get_field_value(pred, "margins"))

    def _any_involved(items):
        return any(bool(x.get("margin_involved")) for x in items)

    def _closest(items):
        vals = []
        for x in items:
            if x.get("margin_involved"):
                continue
            d = x.get("distance")
            if isinstance(d, (int, float)):
                vals.append(int(d))
        return min(vals) if vals else None

    g_any = _any_involved(g_list)
    p_any = _any_involved(p_list)
    g_close = _closest(g_list)
    p_close = _closest(p_list)

    has_both = g_close is not None and p_close is not None
    abs_err = abs(g_close - p_close) if has_both else None
    dist_correct_tol = int(abs_err <= NUMERIC_TOLERANCE_MM) if has_both else None

    matched, unm_g, unm_p = _greedy_match(g_list, p_list, _margin_similarity)
    tp = len(matched)
    fp = len(unm_p)
    fn = len(unm_g)

    status_ok = distance_ok = category_ok = 0
    for gi, pi in matched:
        g, p = g_list[gi], p_list[pi]
        if bool(g.get("margin_involved")) == bool(p.get("margin_involved")):
            status_ok += 1
        gd, pd_ = g.get("distance"), p.get("distance")
        if gd is None and pd_ is None:
            distance_ok += 1
        elif isinstance(gd, (int, float)) and isinstance(pd_, (int, float)) \
                and abs(gd - pd_) <= NUMERIC_TOLERANCE_MM:
            distance_ok += 1
        if normalize(g.get("margin_category")) == normalize(p.get("margin_category")):
            category_ok += 1

    return {
        "margin_any_involved_gold": int(g_any),
        "margin_any_involved_pred": int(p_any),
        "margin_any_involved_correct": int(g_any == p_any),
        "margin_closest_distance_gold": g_close,
        "margin_closest_distance_pred": p_close,
        "margin_closest_distance_abs_err": abs_err,
        "margin_closest_distance_correct_tol": dist_correct_tol,
        "margin_closest_distance_has_both": int(has_both),
        "margin_tp": tp,
        "margin_fp": fp,
        "margin_fn": fn,
        "margin_matched": tp,
        "margin_status_correct": status_ok,
        "margin_distance_correct": distance_ok,
        "margin_category_correct": category_ok,
    }


# --- Aggregation & summary ---------------------------------------------------

GENERATIVE_METHODS = {"digital_registrar", "gpt4_dspy"}


def _iter_test_cases(splits_path: Path):
    with splits_path.open(encoding="utf-8") as f:
        return json.load(f)["test"]


def _attempted(pred: dict, field: str) -> bool:
    if field in pred:
        return True
    cd = pred.get("cancer_data") or {}
    return field in cd


def _aggregate(method_to_preds: dict[str, Path],
               splits_path: Path,
               out_csv: Path,
               *,
               field: str,
               scorer) -> pd.DataFrame:
    cases = _iter_test_cases(splits_path)
    rows = []
    for method, preds_dir in method_to_preds.items():
        for case in cases:
            cid = case["id"]
            gold_path = Path(case["annotation_path"])
            pred_path = preds_dir / f"{cid}.json"
            row: dict = {"method": method, "case_id": cid, "attempted": False}
            if not pred_path.exists() or not gold_path.exists():
                rows.append(row)
                continue
            with gold_path.open(encoding="utf-8") as f:
                gold = json.load(f)
            with pred_path.open(encoding="utf-8") as f:
                pred = json.load(f)
            if not _attempted(pred, field):
                rows.append(row)
                continue
            row["attempted"] = True
            row.update(scorer(gold, pred))
            rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def aggregate_ln_to_csv(method_to_preds: dict[str, Path],
                        splits_path: Path,
                        out_csv: Path) -> pd.DataFrame:
    return _aggregate(method_to_preds, splits_path, out_csv,
                      field="regional_lymph_node", scorer=score_lymph_nodes)


def aggregate_margin_to_csv(method_to_preds: dict[str, Path],
                            splits_path: Path,
                            out_csv: Path) -> pd.DataFrame:
    return _aggregate(method_to_preds, splits_path, out_csv,
                      field="margins", scorer=score_margins)


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _bool_f1(df: pd.DataFrame, gold_col: str, pred_col: str) -> float:
    g = df[gold_col].fillna(0).astype(int)
    p = df[pred_col].fillna(0).astype(int)
    tp = int(((g == 1) & (p == 1)).sum())
    fp = int(((g == 0) & (p == 1)).sum())
    fn = int(((g == 1) & (p == 0)).sum())
    _, _, f1 = _prf(tp, fp, fn)
    return f1


def summarize_ln(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method summary of LN metrics."""
    attempted = df[df["attempted"]]
    if attempted.empty:
        return pd.DataFrame()

    def _per_method(sub):
        tp, fp, fn = sub["ln_station_tp"].sum(), sub["ln_station_fp"].sum(), sub["ln_station_fn"].sum()
        prec, rec, f1 = _prf(tp, fp, fn)
        matched = sub["ln_station_matched"].sum()
        return pd.Series({
            "coverage": len(sub) / len(df.loc[df["method"] == sub.name]),
            "cases_scored": len(sub),
            "examined_mae": sub["ln_examined_total_abs_err"].mean(),
            "examined_acc_tol1": sub["ln_examined_total_correct_tol"].mean(),
            "involved_mae": sub["ln_involved_total_abs_err"].mean(),
            "involved_acc_tol1": sub["ln_involved_total_correct_tol"].mean(),
            "any_positive_acc": sub["ln_any_positive_correct"].mean(),
            "any_positive_f1": _bool_f1(sub, "ln_any_positive_gold", "ln_any_positive_pred"),
            "station_precision": prec,
            "station_recall": rec,
            "station_f1": f1,
            "matched_involved_acc": (sub["ln_station_involved_correct"].sum() / matched) if matched else float("nan"),
            "matched_examined_acc": (sub["ln_station_examined_correct"].sum() / matched) if matched else float("nan"),
            "matched_category_acc": (sub["ln_station_category_correct"].sum() / matched) if matched else float("nan"),
            "matched_side_acc": (sub["ln_station_side_correct"].sum() / matched) if matched else float("nan"),
        })

    return attempted.groupby("method").apply(_per_method, include_groups=False).reset_index()


def summarize_margins(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method summary of margin metrics."""
    attempted = df[df["attempted"]]
    if attempted.empty:
        return pd.DataFrame()

    def _per_method(sub):
        tp, fp, fn = sub["margin_tp"].sum(), sub["margin_fp"].sum(), sub["margin_fn"].sum()
        prec, rec, f1 = _prf(tp, fp, fn)
        matched = sub["margin_matched"].sum()
        both = sub[sub["margin_closest_distance_has_both"] == 1]
        return pd.Series({
            "coverage": len(sub) / len(df.loc[df["method"] == sub.name]),
            "cases_scored": len(sub),
            "any_involved_acc": sub["margin_any_involved_correct"].mean(),
            "any_involved_f1": _bool_f1(sub, "margin_any_involved_gold", "margin_any_involved_pred"),
            "closest_dist_mae": both["margin_closest_distance_abs_err"].mean() if not both.empty else float("nan"),
            "closest_dist_acc_tol2": both["margin_closest_distance_correct_tol"].mean() if not both.empty else float("nan"),
            "closest_dist_both_rate": sub["margin_closest_distance_has_both"].mean(),
            "margin_precision": prec,
            "margin_recall": rec,
            "margin_f1": f1,
            "matched_status_acc": (sub["margin_status_correct"].sum() / matched) if matched else float("nan"),
            "matched_distance_acc": (sub["margin_distance_correct"].sum() / matched) if matched else float("nan"),
            "matched_category_acc": (sub["margin_category_correct"].sum() / matched) if matched else float("nan"),
        })

    return attempted.groupby("method").apply(_per_method, include_groups=False).reset_index()


__all__ = [
    "GENERATIVE_METHODS",
    "LN_COUNT_TOLERANCE",
    "score_lymph_nodes",
    "score_margins",
    "aggregate_ln_to_csv",
    "aggregate_margin_to_csv",
    "summarize_ln",
    "summarize_margins",
]
