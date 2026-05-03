"""
Shared scoring harness for all four methods.

Every method's predictions must live under a per-method folder as
`<method>/<case_id>.json`, with the SAME flat structure as the gold
annotations (top-level `cancer_category`, `cancer_data`, ...).

Core primitives:
    normalize(value)            — canonicalise strings/bools/ints for equality
    field_correct(g, p, field)  — per-field scoring with tolerance rules
    match_nested_list(g, p, k)  — bipartite greedy match of list-of-dicts fields
    score_case(gold, pred)      — returns {field: bool/None} across fair scope
    aggregate_cases_to_df(cases, method_to_preds, scope) — long-form correctness DataFrame

Coverage rule
-------------
A field is "attempted" if the prediction dict for that case has an
explicit entry (value OR null). Missing keys → coverage = 0 for that
field. This separates "wrong answer" from "didn't try", which is what
makes the ClinicalBERT / rules comparison honest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .scope import (
    BREAST_BIOMARKERS,
    FAIR_SCOPE,
    NESTED_LIST_FIELDS,
    get_field_value,
)

ScopeArg = Iterable[str] | Callable[[str | None], Iterable[str]] | None

NUMERIC_TOLERANCE_MM = 2  # ±2 mm


# --- Primitives ---------------------------------------------------------------

def normalize(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    return str(v).strip().lower()


def is_attempted(pred_annotation: dict, field: str) -> bool:
    """Did the method produce any value (including explicit null) for the field?"""
    if field in pred_annotation:
        return True
    cd = pred_annotation.get("cancer_data") or {}
    return field in cd


def field_correct(gold: dict, pred: dict, field: str) -> bool | None:
    """Returns True/False on attempted predictions, None on non-attempts."""
    if not is_attempted(pred, field):
        return None
    g = get_field_value(gold, field)
    p = get_field_value(pred, field)
    if field == "tumor_size" and isinstance(g, (int, float)) and isinstance(p, (int, float)):
        return abs(g - p) <= NUMERIC_TOLERANCE_MM
    return normalize(g) == normalize(p)


# --- Nested-list bipartite match ---------------------------------------------

NESTED_KEY = {
    "margins": "margin_category",
    "biomarkers": "biomarker_category",
    "regional_lymph_node": "station_name",
}


def _item_eq(a, b) -> int:
    """Count how many inner fields match (for ranking bipartite candidates).

    Defensive against non-dict items: if either side isn't a dict, fall
    back to a single-value comparison (returns 1 if equal under
    ``normalize``, else 0).
    """
    if not (a and b):
        return 0
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return 1 if normalize(a) == normalize(b) else 0
    return sum(1 for k in a if k in b and normalize(a[k]) == normalize(b[k]))


def match_nested_list(gold_annotation: dict, pred_annotation: dict,
                      field: str) -> dict:
    """Greedy bipartite match on the primary key, then per-inner-field F1.
    Returns {'tp': ..., 'fp': ..., 'fn': ..., 'f1': ...}.
    """
    key = NESTED_KEY.get(field, None)
    gold_list = get_field_value(gold_annotation, field) or []
    pred_list = get_field_value(pred_annotation, field) or []

    tp = fp = fn = 0
    if key is None:
        # Fallback: just match by position. Lists may differ in length — the
        # loop covers the overlap and the two sum() calls below account for the
        # leftover on each side; strict=False is required here.
        for g, p in zip(gold_list, pred_list, strict=False):
            tp += _item_eq(g, p)
            fp += len(p) - _item_eq(g, p)
            fn += len(g) - _item_eq(g, p)
        fp += sum(len(p) for p in pred_list[len(gold_list):])
        fn += sum(len(g) for g in gold_list[len(pred_list):])
    else:
        unmatched_pred = list(pred_list)
        for g in gold_list:
            gk = normalize(g.get(key))
            match = None
            for p in unmatched_pred:
                if normalize(p.get(key)) == gk:
                    match = p
                    break
            if match is None:
                fn += 1
            else:
                unmatched_pred.remove(match)
                # Score inner fields.
                inner_fields = set(g.keys()) | set(match.keys())
                for k in inner_fields:
                    if normalize(g.get(k)) == normalize(match.get(k)):
                        tp += 1
                    else:
                        fp += 1  # treat mismatches as FP to keep it strict
        fp += len(unmatched_pred)

    if tp + fp == 0 or tp + fn == 0:
        f1 = 0.0
    else:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1}


# --- Per-case + aggregation --------------------------------------------------

def _resolve_scope(scope: ScopeArg, gold: dict) -> list[str]:
    if scope is None:
        return list(FAIR_SCOPE)
    if callable(scope):
        return list(scope(gold.get("cancer_category")))
    return list(scope)


def score_case(gold: dict, pred: dict, scope: ScopeArg = None) -> dict:
    """{field: True/False/None, '_nested': {field: f1dict}}

    With ``scope=None`` (default) scores ``FAIR_SCOPE`` plus the
    breast-biomarker columns and nested-list F1s — the original
    publication scope. Pass a flat iterable of field names or a
    ``(cancer_category) -> set[str]`` callable (e.g. ``bert_scope_for_organ``)
    to score a custom subset; in that mode breast biomarkers and nested
    fields are skipped (they are out of scope for the BERT comparison).
    """
    out: dict = {"_nested": {}}
    fields = _resolve_scope(scope, gold)
    for field in fields:
        out[field] = field_correct(gold, pred, field)

    if scope is None:
        # Breast biomarkers: only score if gold cancer_category == breast.
        if normalize(gold.get("cancer_category")) == "breast":
            gold_bm = {
                normalize(b.get("biomarker_category")): b
                for b in (get_field_value(gold, "biomarkers") or [])
            }
            pred_bm = {
                normalize(b.get("biomarker_category")): b
                for b in (get_field_value(pred, "biomarkers") or [])
            }
            for cat in BREAST_BIOMARKERS:
                g = gold_bm.get(cat)
                p = pred_bm.get(cat)
                if p is None:
                    out[f"biomarker_{cat}"] = None
                elif g is None:
                    out[f"biomarker_{cat}"] = False
                else:
                    out[f"biomarker_{cat}"] = (
                        normalize(g.get("expression")) == normalize(p.get("expression"))
                    )

        # Nested fields: f1 scores (generative-only — N/A for classifier/rules
        # is indicated by absent key).
        for field in NESTED_LIST_FIELDS:
            if is_attempted(pred, field):
                out["_nested"][field] = match_nested_list(gold, pred, field)
    return out


def _resolve_pred_path(pred_root: Path, case: dict) -> Path | None:
    """Look for predictions in both layouts: per-dataset subdir or flat."""
    cid = case["id"]
    ds = case.get("dataset")
    if ds is not None:
        ds_path = pred_root / ds / f"{cid}.json"
        if ds_path.exists():
            return ds_path
    flat = pred_root / f"{cid}.json"
    return flat if flat.exists() else None


def aggregate_cases_to_df(
    cases: list[dict],
    method_to_preds: dict[str, Path],
    scope: ScopeArg = None,
) -> pd.DataFrame:
    """Score every (method, case, field) into long-form, with dataset / organ columns.

    Inputs:
        cases: list of case dicts already loaded via ``baselines._data.load_cases``
            (each carries ``id, dataset, cancer_category, annotation_path``).
        method_to_preds: ``{method_name: Path}`` where Path is the prediction
            root. The lookup tries ``<root>/<dataset>/<id>.json`` first, then
            falls back to ``<root>/<id>.json``.
        scope: passed straight through to ``score_case``.

    Output: DataFrame with columns
        ``method, dataset, organ, case_id, field, correct, attempted``.
    """
    rows = []
    for method, preds_root in method_to_preds.items():
        for case in cases:
            cid = case["id"]
            organ = case.get("cancer_category")
            ds = case.get("dataset")
            with open(case["annotation_path"], encoding="utf-8") as f:
                gold = json.load(f)

            pred_path = _resolve_pred_path(preds_root, case)
            fields = _resolve_scope(scope, gold)
            if pred_path is None:
                # Method didn't emit anything for this case — coverage 0.
                for field in fields:
                    rows.append({"method": method, "dataset": ds, "organ": organ,
                                 "case_id": cid, "field": field,
                                 "correct": None, "attempted": False})
                continue

            with pred_path.open(encoding="utf-8") as f:
                pred = json.load(f)
            result = score_case(gold, pred, scope=scope)

            for field in fields:
                if field not in result:
                    continue
                correct = result[field]
                rows.append({
                    "method": method, "dataset": ds, "organ": organ,
                    "case_id": cid, "field": field,
                    "correct": (bool(correct) if correct is not None else None),
                    "attempted": correct is not None,
                })
            for field, f1d in result["_nested"].items():
                rows.append({
                    "method": method, "dataset": ds, "organ": organ,
                    "case_id": cid, "field": field,
                    "correct": f1d["f1"], "attempted": True,
                })
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Accuracy + coverage grouped by ``by`` (default ``["method", "field"]``).

    Pass ``by=["method", "dataset", "field"]`` for the dataset-stratified
    table when the input came from ``aggregate_cases_to_df``.
    """
    if by is None:
        by = ["method", "field"]

    def _agg(group):
        attempted = group["attempted"].sum()
        total = len(group)
        numeric = group["correct"].dropna()
        if numeric.empty:
            acc = float("nan")
        else:
            acc = float(pd.to_numeric(numeric, errors="coerce").mean())
        return pd.Series({
            "attempted": attempted,
            "total": total,
            "coverage": attempted / total if total else 0.0,
            "accuracy_attempted": acc,
        })

    return df.groupby(by).apply(_agg).reset_index()
