"""
Defines the "fair scope" — which fields each method can legitimately
produce, and the allowed-value lists for the categorical ones.

This is the single source of truth for both the benchmark code and the
results-table column definitions. The `FAIR_SCOPE` list is the
head-to-head comparison set (fields that ALL four methods populate).

Extend CATEGORICAL_FIELDS / SPAN_FIELDS from the full DSPy signatures
in ../../digitalregistrar/models/*.py as you broaden coverage. This
scaffold ships with the core set that every organ shares; per-organ
extensions go in ORGAN_CATEGORICAL below.
"""
from __future__ import annotations

# --- Top-level cancer category (drives organ-gated field scoring) -------------

CANCER_CATEGORIES: list[str] = [
    "breast", "lung", "colorectal", "prostate", "esophagus",
    "pancreas", "thyroid", "cervix", "liver", "stomach", "others",
]


# --- Shared categorical fields (Literal[...] in DSPy signatures) --------------
#
# Options are lowercased to match the gold-annotation normalisation rule
# applied by eval/metrics.py.
# Fields marked with a trailing "_boolean" are derived from `bool|None`
# DSPy fields and get a 3-way {True, False, null} treatment at eval time.

CATEGORICAL_FIELDS: dict[str, list[str]] = {
    # Staging (shared across all organs via StagingSignature children)
    "tnm_descriptor":         ["y", "r", "m"],
    "pt_category":            ["tx", "tis", "t1mi", "t1a", "t1b", "t1c",
                               "t2", "t2a", "t2b", "t3", "t4", "t4a", "t4b", "t4c"],
    "pn_category":            ["nx", "n0", "n1mi", "n1", "n1a", "n1b", "n1c",
                               "n2", "n2a", "n2b", "n3", "n3a", "n3b", "n3c"],
    "pm_category":            ["mx", "m0", "m1"],
    "pathologic_stage_group": ["0", "ia", "ib", "iia", "iib",
                               "iiia", "iiib", "iiic", "iv"],
    "anatomic_stage_group":   ["0", "ia", "ib", "iia", "iib",
                               "iiia", "iiib", "iiic", "iv"],

    # Grading (breast, but present under same names on most organs)
    "grade":                  ["1", "2", "3"],
    "nuclear_grade":          ["1", "2", "3"],
    "tubule_formation":       ["1", "2", "3"],
    "mitotic_rate":           ["1", "2", "3"],

    # Invasion flags — 3-way {true, false, null}
    "lymphovascular_invasion": ["true", "false"],
    "perineural_invasion":     ["true", "false"],
    "distant_metastasis":      ["true", "false"],
    "extranodal_extension":    ["true", "false"],
    "dcis_present":            ["true", "false"],
    "dcis_comedo_necrosis":    ["true", "false"],
}


# --- Organ-specific categorical fields (extend in follow-up passes) ----------
# TODO: fill in from models/*.py Literal[...] declarations; this scaffold
# ships with a representative subset so the pipeline runs end-to-end.
ORGAN_CATEGORICAL: dict[str, dict[str, list[str]]] = {
    "breast": {
        "procedure": [
            "partial_mastectomy", "simple_mastectomy",
            "breast_conserving_surgery", "modified_radical_mastectomy",
            "total_mastectomy", "wide_excision", "others",
        ],
        "cancer_quadrant": [
            "upper_outer_quadrant", "upper_inner_quadrant",
            "lower_outer_quadrant", "lower_inner_quadrant",
            "nipple", "others",
        ],
        "cancer_laterality": ["right", "left", "bilateral"],
        "histology": [
            "invasive_carcinoma_no_special_type",
            "invasive_lobular_carcinoma",
            "mixed_ductal_and_lobular_carcinoma",
            "tubular_adenocarcinoma", "mucinous_adenocarcinoma",
            "encapsulated_papillary_carcinoma",
            "solid_papillary_carcinoma", "inflammatory_carcinoma",
            "other_special_types",
        ],
    },
    "lung": {
        "procedure": [
            "wedge_resection", "segmentectomy", "lobectomy",
            "bilobectomy", "pneumonectomy", "others",
        ],
        "surgical_technique": ["open", "thoracoscopic", "robotic", "others"],
        "sideness": ["right", "left", "midline"],
    },
    # TODO: colorectal, prostate, esophagus, pancreas, thyroid, cervix,
    # liver, stomach — extend by reading ../digitalregistrar/models/*.py
}


# --- Span / numeric fields (go to ClinicalBERT-QA head) ----------------------

SPAN_FIELDS: set[str] = {
    "tumor_size", "dcis_size", "maximal_ln_size",
    "grade", "nuclear_grade", "tubule_formation",
    "mitotic_rate", "total_score",
}


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

# Nested-list fields — reported N/A for ClinicalBERT and rules.
NESTED_LIST_FIELDS: set[str] = {
    "margins", "biomarkers", "regional_lymph_node",
}


# --- Accessor ----------------------------------------------------------------

def get_field_value(annotation: dict, field: str):
    """Read a field from a flat gold annotation, handling the nested
    `cancer_data` container the project uses."""
    if field in annotation:
        return annotation[field]
    data = annotation.get("cancer_data") or {}
    return data.get(field)
