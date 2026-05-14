"""Regression test for the free_text_regex extractor's KeyError on ki67.

The historical bug: ``BREAST_BIOMARKERS = ["er", "pr", "her2", "ki67"]``
in `benchmarks.eval.scope`, but the extractor's ``_BIOMARKER_RES`` dict
only had patterns for er/pr/her2. The loop did
``_BIOMARKER_RES[cat].search(text)`` for every category, so the ki67
iteration raised KeyError for every case → 100% pipeline_error in the
B6 cell.

The fix: defensive ``_BIOMARKER_RES.get(cat)`` with skip-on-None. The
correct ablation signal for a missing pattern is "missing key" (cascade
treats the field as not extracted), NOT a crash.
"""
from __future__ import annotations

from digital_registrar_research.ablations.extractors.regex_per_field import (
    RegexExtractor,
)


def test_extract_does_not_keyerror_on_missing_biomarker_pattern():
    """Reproducer: an arbitrary breast-style summary that mentions Ki-67
    used to raise ``KeyError: 'ki67'`` because no regex pattern was
    registered. After the fix it returns a payload, possibly without
    a ki67 entry — that's the correct ablation signal."""
    text = (
        "Patient with invasive ductal carcinoma of the breast. "
        "Tumor measures 22 mm. Histologic grade 2. "
        "ER positive, PR positive, HER2 negative. "
        "Ki-67 of 20%. No lymphovascular invasion identified."
    )
    out = RegexExtractor().extract(text)  # must not raise
    assert isinstance(out, dict)
    assert out["cancer_category"] == "breast"
    biomarkers = (out.get("cancer_data") or {}).get("biomarkers") or []
    cats = {b["biomarker_category"] for b in biomarkers}
    # er, pr, her2 should all match; ki67 may or may not match
    # depending on whether a pattern is registered. We do NOT assert
    # ki67 is present — the ablation principle is that this baseline
    # is allowed to miss fields.
    assert "er" in cats
    assert "pr" in cats
    assert "her2" in cats


def test_extract_on_minimal_text_doesnt_keyerror():
    """Even with no biomarker mentions in the text, the loop must
    complete without referencing a missing key in _BIOMARKER_RES."""
    text = "Pancreatic adenocarcinoma. Pancreaticoduodenectomy specimen."
    out = RegexExtractor().extract(text)
    assert isinstance(out, dict)


def test_extract_completes_when_biomarker_res_empty(monkeypatch):
    """Belt-and-suspenders: simulate every biomarker pattern being
    removed. The extractor must still return a valid payload."""
    from digital_registrar_research.ablations.extractors import (
        regex_per_field as mod,
    )
    monkeypatch.setattr(mod, "_BIOMARKER_RES", {})
    text = "Breast cancer with ER positive HER2 negative."
    out = RegexExtractor().extract(text)
    assert isinstance(out, dict)
    assert out["cancer_category"] == "breast"
    biomarkers = (out.get("cancer_data") or {}).get("biomarkers") or []
    assert biomarkers == []
