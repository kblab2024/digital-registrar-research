"""
Defines the "fair scope" — which fields each method can legitimately
produce, and the allowed-value lists for the categorical ones.

This is the single source of truth for both the benchmark code and the
results-table column definitions. The `FAIR_SCOPE` list is the
head-to-head comparison set (fields that ALL four methods populate).

Per-organ option lists live in `scope_organs.py` (`ORGAN_CATEGORICAL`,
`ORGAN_BOOL`, `ORGAN_SPAN`, `ORGAN_NESTED_LIST`), derived from the
canonical schemas under `digital_registrar_research.schemas.data`.

Accessors:
    get_allowed_values(field, organ=None)  — enum list for a categorical field
    get_categorical_fields(organ)          — all enum fields for an organ
    get_bool_fields(organ)                 — all bool fields for an organ
    get_span_fields(organ)                 — all int span fields for an organ
    get_nested_list_fields(organ)          — all array fields for an organ
    get_field_value(annotation, field)     — read a field from a gold record
"""
from __future__ import annotations

from functools import cache

from .scope_organs import (
    ORGAN_BOOL,
    ORGAN_CATEGORICAL,
    ORGAN_NESTED_LIST,
    ORGAN_SPAN,
)

# --- Top-level cancer category (drives organ-gated field scoring) -------------

CANCER_CATEGORIES: list[str] = [
    "breast", "lung", "colorectal", "prostate", "esophagus",
    "pancreas", "thyroid", "cervix", "liver", "stomach", "others",
]

# Organs that have a populated schema (excludes "others" and the
# placeholder "bladder" entry in schemas/data).
IMPLEMENTED_ORGANS: list[str] = sorted(ORGAN_CATEGORICAL.keys())


# --- Shared categorical option lists -----------------------------------------
#
# Options are lowercased to match the gold-annotation normalisation rule
# applied by `eval/metrics.py:normalize`. Where a field's enum differs by
# organ (`pt_category`, `pn_category`, `pm_category`, `grade`, `histology`,
# `procedure`, stage-group fields), the value below is the UNION across
# all organs — a superset useful for a single cross-organ classifier head.
# For organ-specific scoring use `get_allowed_values(field, organ)`.

def _union_field_values(field: str) -> list[str]:
    seen: list[str] = []
    for organ_map in ORGAN_CATEGORICAL.values():
        for v in organ_map.get(field, []):
            if v not in seen:
                seen.append(v)
    return seen


def _union_bool_fields() -> list[str]:
    seen: list[str] = []
    for fields in ORGAN_BOOL.values():
        for f in sorted(fields):
            if f not in seen:
                seen.append(f)
    return seen


CATEGORICAL_FIELDS: dict[str, list[str]] = {
    "tnm_descriptor":          _union_field_values("tnm_descriptor"),
    "pt_category":             _union_field_values("pt_category"),
    "pn_category":             _union_field_values("pn_category"),
    "pm_category":             _union_field_values("pm_category"),
    "stage_group":             _union_field_values("stage_group"),
    "overall_stage":           _union_field_values("overall_stage"),
    "pathologic_stage_group":  _union_field_values("pathologic_stage_group"),
    "anatomic_stage_group":    _union_field_values("anatomic_stage_group"),
    "grade":                   _union_field_values("grade"),
    "nuclear_grade":           ["1", "2", "3"],
    "tubule_formation":        ["1", "2", "3"],
    "mitotic_rate":            ["1", "2", "3"],
    "total_score":             ["3", "4", "5", "6", "7", "8", "9"],
    "histology":               _union_field_values("histology"),
    "procedure":               _union_field_values("procedure"),
    "surgical_technique":      _union_field_values("surgical_technique"),
    "cancer_primary_site":     _union_field_values("cancer_primary_site"),
    "tumor_focality":          _union_field_values("tumor_focality"),
    "tumor_site":              _union_field_values("tumor_site"),
    "tumor_extent":            _union_field_values("tumor_extent"),
    # Boolean / 3-way {true, false, null} fields kept in CATEGORICAL_FIELDS
    # for the flat-head classifier API (clinicalbert_cls.py) that treats
    # everything in this dict as a head.
    **{f: ["true", "false"] for f in _union_bool_fields()},
}


# --- Span / numeric fields (go to ClinicalBERT-QA head) ----------------------
#
# Cross-organ union of integer fields. Grading sub-scores (`grade`,
# `nuclear_grade`, `tubule_formation`, `mitotic_rate`, `total_score`) are
# listed here as well because in some organs (colorectal, lung) `grade` is
# stored as a free integer rather than a Literal, so the QA head is
# responsible for it.

SPAN_FIELDS: set[str] = set()
for _s in ORGAN_SPAN.values():
    SPAN_FIELDS |= _s
SPAN_FIELDS |= {
    "grade", "nuclear_grade", "tubule_formation",
    "mitotic_rate", "total_score",
}


# --- Nested-list fields — reported N/A for ClinicalBERT and rules. -----------

NESTED_LIST_FIELDS: set[str] = set()
for _n in ORGAN_NESTED_LIST.values():
    NESTED_LIST_FIELDS |= _n


# --- Fair-scope whitelist (head-to-head comparison table) --------------------
#
# Fields that EVERY method can legitimately populate. Nested-list fields
# (margins, biomarkers, regional_lymph_node) are NOT in this list — they
# appear in the supplementary coverage table instead.

FAIR_SCOPE: list[str] = [
    "cancer_category",
    "cancer_excision_report",
    "pt_category", "pn_category", "pm_category",
    "grade",
    "lymphovascular_invasion",
    "perineural_invasion",
    "tumor_size",
    # Plus breast-specific biomarkers when cancer_category == "breast":
    # these are scored conditionally in metrics.py via BREAST_BIOMARKERS.
]

BREAST_BIOMARKERS: list[str] = ["er", "pr", "her2"]


# --- Accessors ---------------------------------------------------------------

@cache
def get_allowed_values(field: str, organ: str | None = None) -> list[str] | None:
    """Return the allowed option list for a categorical field.

    If `organ` is given and the field has an organ-specific enum, return
    that. Otherwise fall back to the cross-organ union in
    `CATEGORICAL_FIELDS`. Returns `None` if the field is not categorical.
    """
    if organ is not None:
        organ_map = ORGAN_CATEGORICAL.get(organ, {})
        if field in organ_map:
            return list(organ_map[field])
        bools = ORGAN_BOOL.get(organ, set())
        if field in bools:
            return ["true", "false"]
    return list(CATEGORICAL_FIELDS.get(field, [])) or None


def get_categorical_fields(organ: str) -> dict[str, list[str]]:
    """All categorical (enum) fields for `organ`, with their option lists."""
    return dict(ORGAN_CATEGORICAL.get(organ, {}))


def get_bool_fields(organ: str) -> set[str]:
    """All boolean fields for `organ` (scored 3-way: true/false/null)."""
    return set(ORGAN_BOOL.get(organ, set()))


def get_span_fields(organ: str) -> set[str]:
    """All integer span fields for `organ` (ClinicalBERT-QA head)."""
    return set(ORGAN_SPAN.get(organ, set()))


def get_nested_list_fields(organ: str) -> set[str]:
    """All nested-list fields for `organ` (supplementary coverage table)."""
    return set(ORGAN_NESTED_LIST.get(organ, set()))


def get_field_value(annotation: dict, field: str):
    """Read a field from a flat gold annotation, handling the nested
    `cancer_data` container the project uses."""
    if field in annotation:
        return annotation[field]
    data = annotation.get("cancer_data") or {}
    return data.get(field)


__all__ = [
    "CANCER_CATEGORIES",
    "IMPLEMENTED_ORGANS",
    "CATEGORICAL_FIELDS",
    "ORGAN_CATEGORICAL",
    "ORGAN_BOOL",
    "ORGAN_SPAN",
    "ORGAN_NESTED_LIST",
    "SPAN_FIELDS",
    "NESTED_LIST_FIELDS",
    "FAIR_SCOPE",
    "BREAST_BIOMARKERS",
    "get_allowed_values",
    "get_categorical_fields",
    "get_bool_fields",
    "get_span_fields",
    "get_nested_list_fields",
    "get_field_value",
]
