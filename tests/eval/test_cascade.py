"""Tests for the cascade redesign.

Covers:
    * Cascade gating in :func:`score_case` (Stage A halts B and C;
      Stage B mismatch halts C; "others" diverts).
    * Biomarker whitelist (colon ``her2`` is invisible; breast
      ``ki67`` is scored).
    * Inner-key exclusions (margins ``description`` not in match
      output; LN ``station_name`` not used in matching).
    * LN category-aggregation scoring.
    * scope.py exclusion structures are well-formed.
"""
from __future__ import annotations

import pytest

from digital_registrar_research.benchmarks.eval.metrics import (
    match_nested_list_filtered,
    score_case,
)
from digital_registrar_research.benchmarks.eval.nested_metrics import (
    score_lymph_nodes,
)
from digital_registrar_research.benchmarks.eval.scope import (
    BIOMARKER_WHITELIST,
    BREAST_BIOMARKERS,
    EVAL_EXCLUDED_FIELDS,
    EVAL_EXCLUDED_NESTED_INNER_KEYS,
    EXCLUDED_INNER_KEYS_BY_FIELD,
    biomarkers_for_organ,
)


# --- Scope structures --------------------------------------------------

def test_excluded_fields_contain_ajcc_version_and_treatment_effect():
    assert "ajcc_version" in EVAL_EXCLUDED_FIELDS
    assert "treatment_effect" in EVAL_EXCLUDED_FIELDS


def test_excluded_inner_keys_resolve_correctly():
    assert "description" in EXCLUDED_INNER_KEYS_BY_FIELD["margins"]
    assert "station_name" in EXCLUDED_INNER_KEYS_BY_FIELD["regional_lymph_node"]


def test_excluded_inner_keys_dict_has_human_readable_paths():
    assert "cancer_data.margins[*].description" in EVAL_EXCLUDED_NESTED_INNER_KEYS
    assert ("cancer_data.regional_lymph_node[*].station_name"
            in EVAL_EXCLUDED_NESTED_INNER_KEYS)


def test_biomarker_whitelist_breast():
    assert biomarkers_for_organ("breast") == frozenset({"er", "pr", "her2", "ki67"})
    assert "ki67" in BREAST_BIOMARKERS


def test_biomarker_whitelist_colon():
    assert biomarkers_for_organ("colorectal") == frozenset(
        {"msh2", "msh6", "pms2", "mlh1"}
    )


def test_biomarker_whitelist_empty_for_unrelated_organ():
    assert biomarkers_for_organ("lung") == frozenset()
    assert biomarkers_for_organ("others") == frozenset()
    assert biomarkers_for_organ(None) == frozenset()


# --- Cascade gating ----------------------------------------------------

def _gold_passing():
    return {
        "cancer_excision_report": True,
        "cancer_category": "breast",
        "cancer_data": {
            "pt_category": "t1c",
            "pn_category": "n0",
            "ajcc_version": "8",
            "treatment_effect": "complete_response",
            "biomarkers": [
                {"biomarker_category": "er", "expression": True},
                {"biomarker_category": "pr", "expression": True},
                {"biomarker_category": "her2", "expression": False},
                {"biomarker_category": "ki67", "expression": False},
            ],
            "margins": [
                {"margin_category": "deep", "margin_involved": False,
                 "distance": 5, "description": "negative free-text"},
            ],
            "regional_lymph_node": [],
        },
    }


def test_score_case_passes_all_stages_on_identical_pred():
    g = _gold_passing()
    p = _gold_passing()
    result = score_case(g, p)
    assert result["stage_a"]["correct"] is True
    assert result["stage_b"]["correct"] is True
    assert result["stage_c_eligible"] is True
    assert result["others_disposition"] == "none"


def test_score_case_excludes_ajcc_version_and_treatment_effect():
    g = _gold_passing()
    p = _gold_passing()
    result = score_case(g, p)
    assert "ajcc_version" not in result
    assert "treatment_effect" not in result


def test_score_case_halts_on_stage_a_mismatch():
    g = _gold_passing()
    p = _gold_passing()
    p["cancer_excision_report"] = False
    result = score_case(g, p)
    assert result["stage_a"]["correct"] is False
    assert result["stage_c_eligible"] is False
    assert "stage_b" not in result


def test_score_case_halts_on_stage_b_mismatch():
    g = _gold_passing()
    p = _gold_passing()
    p["cancer_category"] = "lung"
    result = score_case(g, p)
    assert result["stage_a"]["correct"] is True
    assert result["stage_b"]["correct"] is False
    assert result["stage_c_eligible"] is False
    assert "pt_category" not in result


def test_score_case_others_disposition_pred():
    g = _gold_passing()
    p = _gold_passing()
    p["cancer_category"] = "others"
    result = score_case(g, p)
    assert result["others_disposition"] == "pred_others"
    assert result["stage_c_eligible"] is False


def test_score_case_others_disposition_gold():
    g = _gold_passing()
    g["cancer_category"] = "others"
    p = _gold_passing()
    p["cancer_category"] = "others"
    result = score_case(g, p)
    assert result["others_disposition"] == "both_others"
    # Both-others: Stage B matches but Stage C doesn't run.
    assert result["stage_b"]["correct"] is True
    assert result["stage_c_eligible"] is False


def test_score_case_breast_biomarkers_include_ki67():
    g = _gold_passing()
    p = _gold_passing()
    result = score_case(g, p)
    # ki67 should appear in Stage C scoring for breast.
    assert "biomarker_ki67" in result
    assert result["biomarker_ki67"] is True


# --- Biomarker whitelist filtering ------------------------------------

def test_biomarker_whitelist_drops_out_of_scope_categories():
    """Colon biomarkers should only score msh2/msh6/pms2/mlh1.

    Adding a non-whitelisted category to either side must not change
    the F1 (whitelist filters before matching).
    """
    g = {"cancer_data": {"biomarkers": [
        {"biomarker_category": "msh2", "expression": True},
        {"biomarker_category": "her2", "expression": True},  # filtered
    ]}}
    p_clean = {"cancer_data": {"biomarkers": [
        {"biomarker_category": "msh2", "expression": True},
    ]}}
    p_extras = {"cancer_data": {"biomarkers": [
        {"biomarker_category": "msh2", "expression": True},
        {"biomarker_category": "her2", "expression": False},  # filtered
        {"biomarker_category": "p53", "expression": True},    # filtered
    ]}}
    r_clean = match_nested_list_filtered(g, p_clean, "biomarkers", organ="colorectal")
    r_extras = match_nested_list_filtered(g, p_extras, "biomarkers", organ="colorectal")
    assert r_clean["f1"] == r_extras["f1"]
    assert r_clean["fp"] == r_extras["fp"]
    assert r_clean["fn"] == r_extras["fn"]


def test_match_nested_list_filtered_rejects_lymph_node_field():
    with pytest.raises(ValueError, match="regional_lymph_node"):
        match_nested_list_filtered({}, {}, "regional_lymph_node")


# --- Lymph-node category-aggregation scoring --------------------------

def test_ln_aggregates_split_rows_to_single_group():
    """Two gold rows summing to (4 examined, 0 involved) match a single
    pred row of (4, 0). The gold split into multiple textual rows
    should not penalise a model that produced the correct totals."""
    g = {"cancer_data": {"regional_lymph_node": [
        {"lymph_node_side": "right", "lymph_node_category": "nonsentinel",
         "examined": 2, "involved": 0, "station_name": "x #1"},
        {"lymph_node_side": "right", "lymph_node_category": "nonsentinel",
         "examined": 2, "involved": 0, "station_name": "x #2"},
    ]}}
    p = {"cancer_data": {"regional_lymph_node": [
        {"lymph_node_side": "right", "lymph_node_category": "nonsentinel",
         "examined": 4, "involved": 0, "station_name": "any_label"},
    ]}}
    r = score_lymph_nodes(g, p)
    assert r["ln_examined_total_gold"] == 4
    assert r["ln_examined_total_pred"] == 4
    assert r["ln_examined_total_correct_tol"] == 1
    assert r["ln_n_groups_gold"] == 1
    assert r["ln_n_groups_pred"] == 1
    assert r["ln_group_recall"] == 1.0
    assert r["ln_group_precision"] == 1.0


def test_ln_hallucinated_group_drops_precision():
    """Pred adds a hallucinated mesenteric group; precision drops."""
    g = {"cancer_data": {"regional_lymph_node": [
        {"lymph_node_side": "right", "lymph_node_category": "nonsentinel",
         "examined": 4, "involved": 0, "station_name": None},
    ]}}
    p = {"cancer_data": {"regional_lymph_node": [
        {"lymph_node_side": "right", "lymph_node_category": "nonsentinel",
         "examined": 4, "involved": 0, "station_name": None},
        {"lymph_node_side": None, "lymph_node_category": "mesenteric",
         "examined": 5, "involved": 0, "station_name": None},
    ]}}
    r = score_lymph_nodes(g, p)
    assert r["ln_n_groups_gold"] == 1
    assert r["ln_n_groups_pred"] == 2
    assert r["ln_group_precision"] == 0.5
    assert r["ln_group_recall"] == 1.0


# --- Margin description exclusion -------------------------------------

def test_margin_description_excluded_from_filtered_match():
    """Two cases identical in everything except `description` should
    bipartite-match perfectly (description is stripped before matching).
    """
    g = {"cancer_data": {"margins": [
        {"margin_category": "deep", "margin_involved": False,
         "distance": 5, "description": "free-text gold"},
    ]}}
    p = {"cancer_data": {"margins": [
        {"margin_category": "deep", "margin_involved": False,
         "distance": 5, "description": "wholly different free text"},
    ]}}
    r = match_nested_list_filtered(g, p, "margins")
    assert r["fp"] == 0
    assert r["fn"] == 0
    assert r["tp"] >= 3  # margin_category + margin_involved + distance match
    assert r["f1"] == 1.0
