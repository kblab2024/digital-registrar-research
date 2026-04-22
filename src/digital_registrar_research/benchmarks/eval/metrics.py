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
    aggregate_to_csv(method_to_preds) — writes results/by_method.csv

Coverage rule
-------------
A field is "attempted" if the prediction dict for that case has an
explicit entry (value OR null). Missing keys → coverage = 0 for that
field. This separates "wrong answer" from "didn't try", which is what
makes the ClinicalBERT / rules comparison honest.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .scope import (
    BREAST_BIOMARKERS,
    FAIR_SCOPE,
    NESTED_LIST_FIELDS,
    get_field_value,
)

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


def _item_eq(a: dict, b: dict) -> int:
    """Count how many inner fields match (for ranking bipartite candidates)."""
    if not (a and b):
        return 0
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
        # Fallback: just match by position.
        for g, p in zip(gold_list, pred_list):
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

def score_case(gold: dict, pred: dict) -> dict:
    """{field: True/False/None, '_nested': {field: f1dict}}"""
    out: dict = {"_nested": {}}
    for field in FAIR_SCOPE:
        out[field] = field_correct(gold, pred, field)

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


def aggregate_to_csv(method_to_preds: dict[str, Path],
                     gold_root: Path,
                     splits_path: Path,
                     out_csv: Path) -> pd.DataFrame:
    """
    method_to_preds: {"digital_registrar": Path(".../dr"), "gpt4_dspy": Path(...), ...}
    gold_root: folder of gold annotations (path resolved via splits.json entries)
    splits_path: data/splits.json
    out_csv: results/by_method.csv
    """
    with splits_path.open(encoding="utf-8") as f:
        split = json.load(f)
    test_cases = split["test"]

    # Long-form: one row per (method, case_id, field).
    rows = []
    for method, preds_dir in method_to_preds.items():
        for case in test_cases:
            cid = case["id"]
            gold_path = Path(case["annotation_path"])
            pred_path = preds_dir / f"{cid}.json"
            if not pred_path.exists():
                # Method didn't emit anything for this case — coverage 0.
                for field in FAIR_SCOPE:
                    rows.append({"method": method, "case_id": cid,
                                 "field": field, "correct": None,
                                 "attempted": False})
                continue

            with gold_path.open(encoding="utf-8") as f:
                gold = json.load(f)
            with pred_path.open(encoding="utf-8") as f:
                pred = json.load(f)
            result = score_case(gold, pred)

            for field in FAIR_SCOPE + [f"biomarker_{b}" for b in BREAST_BIOMARKERS]:
                if field not in result:
                    continue
                correct = result[field]
                rows.append({
                    "method": method, "case_id": cid, "field": field,
                    "correct": (bool(correct) if correct is not None else None),
                    "attempted": correct is not None,
                })

            for field, f1d in result["_nested"].items():
                rows.append({
                    "method": method, "case_id": cid, "field": field,
                    "correct": f1d["f1"],  # stored as float, not bool
                    "attempted": True,
                })

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method × per-field accuracy + coverage."""
    def _agg(group):
        attempted = group["attempted"].sum()
        total = len(group)
        # For scalar fields `correct` is bool-as-object; for nested it's float.
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

    return df.groupby(["method", "field"]).apply(_agg).reset_index()
