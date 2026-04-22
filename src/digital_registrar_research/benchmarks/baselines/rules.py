"""
Rule-based baseline extractor.

Pure-regex + lexicon. Targets the "fair-scope" subset of fields that all
four methods (Digital Registrar, GPT-4, ClinicalBERT, rules) can populate.
Returns a dict in the same shape as the gold annotations: top-level
`cancer_excision_report`, `cancer_category`, `cancer_category_others_description`,
plus `cancer_data` with the extracted scalars.

Not expected to perform well on narrative or nuanced fields — that is
the point: it establishes the floor that LLM-based methods have to beat.

Usage:
    python baselines/rules.py < report.txt
    # or
    from baselines.rules import extract
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


# --- Categorical / regular fields ---------------------------------------------

PATTERNS = {
    "pt_category": re.compile(
        r"\bp?T\s*([0-4](?:mi|a|b|c)?|is|x)\b", re.I),
    "pn_category": re.compile(
        r"\bp?N\s*([0-3](?:mi|a|b|c)?|x)\b", re.I),
    "pm_category": re.compile(
        r"\bp?M\s*([01x])\b", re.I),
    "grade": re.compile(
        r"\b(?:nottingham\s*grade|histologic(?:al)?\s*grade|grade)\s*[:=]?\s*"
        r"(?:g\s*)?([1-3])\b", re.I),
    "nuclear_grade": re.compile(
        r"\bnuclear\s*grade\s*[:=]?\s*([1-3])\b", re.I),
    "total_score": re.compile(
        r"\bnottingham\s*score\s*[:=]?\s*(\d)\s*/\s*9\b", re.I),
    "margins_status": re.compile(
        r"\bmargins?\b[^\.]*?(negative|positive|involved|clear|free|uninvolved)",
        re.I),
    "lymphovascular_invasion": re.compile(
        r"(lymph[o\-]?vascular|lymphovascular|venous\s*/\s*lymphatic)\s*invasion"
        r"[^\.]*?(present|absent|positive|negative|identified|not\s*identified|"
        r"no\s*evidence)", re.I),
    "perineural_invasion": re.compile(
        r"\bperineural\s*invasion[^\.]*?"
        r"(present|absent|positive|negative|identified|not\s*identified|"
        r"no\s*evidence)", re.I),
    "er_status": re.compile(
        r"\b(?:estrogen\s*receptor|ER)\b[^\.]{0,50}?"
        r"(positive|negative)", re.I),
    "pr_status": re.compile(
        r"\b(?:progesterone\s*receptor|PR)\b[^\.]{0,50}?"
        r"(positive|negative)", re.I),
    "her2_status": re.compile(
        r"\bHER[\s\-]?2\b[^\.]{0,80}?"
        r"(positive|negative|equivocal)", re.I),
    "tumor_size_cm": re.compile(
        r"\btumou?r\s*size[^\.]{0,60}?"
        r"(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)", re.I),
    "tumor_size_mm": re.compile(
        r"\btumou?r\s*size[^\.]{0,60}?"
        r"(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)", re.I),
}

NEGATIVE_LEXICON = {"absent", "negative", "not identified", "no evidence"}
POSITIVE_LEXICON = {"present", "positive", "identified"}

# --- Cancer category lexicon --------------------------------------------------

CANCER_LEXICON: dict[str, list[str]] = {
    "breast":      ["breast", "mastectomy", "lumpectomy",
                    "ductal carcinoma", "lobular carcinoma", "nipple"],
    "lung":        ["lung", "lobectomy", "pulmonary", "bronchial",
                    "pleural", "alveolar"],
    "colorectal":  ["colon", "rectal", "rectum", "colectomy",
                    "sigmoid", "colonic", "colorectal"],
    "prostate":    ["prostate", "prostatectomy", "seminal vesicle",
                    "gleason"],
    "esophagus":   ["esophagus", "esophageal", "esophagectomy"],
    "pancreas":    ["pancreas", "pancreatic", "whipple",
                    "pancreaticoduodenectomy"],
    "thyroid":     ["thyroid", "thyroidectomy", "papillary thyroid",
                    "follicular thyroid"],
    "cervix":      ["cervix", "cervical", "cervical carcinoma",
                    "uterine cervix"],
    "liver":       ["liver", "hepatic", "hepatocellular",
                    "hepatectomy"],
    "stomach":     ["stomach", "gastric", "gastrectomy"],
}


def classify_organ(report: str) -> str | None:
    """Simple majority-vote lexicon match."""
    text = report.lower()
    scores: dict[str, int] = {}
    for organ, words in CANCER_LEXICON.items():
        scores[organ] = sum(text.count(w) for w in words)
    if not any(scores.values()):
        return None
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else None


def _norm_yes_no(match_group: str) -> bool | None:
    g = match_group.lower().strip()
    if g in NEGATIVE_LEXICON or "not" in g or "absent" in g or "negative" in g:
        return False
    if g in POSITIVE_LEXICON or "present" in g or "positive" in g or "identified" in g:
        return True
    return None


def _norm_margins(match_group: str) -> bool | None:
    """margin_involved (bool) from the match text."""
    g = match_group.lower().strip()
    if g in {"negative", "clear", "free", "uninvolved"}:
        return False
    if g in {"positive", "involved"}:
        return True
    return None


def extract(report: str) -> dict:
    """Extract a flat dict matching the gold annotation structure."""
    out_cancer_data: dict = {}

    # TNM
    if (m := PATTERNS["pt_category"].search(report)):
        out_cancer_data["pt_category"] = f"t{m.group(1).lower()}"
    if (m := PATTERNS["pn_category"].search(report)):
        out_cancer_data["pn_category"] = f"n{m.group(1).lower()}"
    if (m := PATTERNS["pm_category"].search(report)):
        out_cancer_data["pm_category"] = f"m{m.group(1).lower()}"

    # Grade
    if (m := PATTERNS["grade"].search(report)):
        out_cancer_data["grade"] = int(m.group(1))
    if (m := PATTERNS["nuclear_grade"].search(report)):
        out_cancer_data["nuclear_grade"] = int(m.group(1))
    if (m := PATTERNS["total_score"].search(report)):
        out_cancer_data["total_score"] = int(m.group(1))

    # Invasion
    if (m := PATTERNS["lymphovascular_invasion"].search(report)):
        out_cancer_data["lymphovascular_invasion"] = _norm_yes_no(m.group(2))
    if (m := PATTERNS["perineural_invasion"].search(report)):
        out_cancer_data["perineural_invasion"] = _norm_yes_no(m.group(1))

    # Margins (flattened: derive margin_involved from status keyword)
    if (m := PATTERNS["margins_status"].search(report)):
        involved = _norm_margins(m.group(1))
        if involved is not None:
            # Emit as single-element margins list to match gold schema shape.
            out_cancer_data["margins"] = [
                {"margin_category": "others",
                 "margin_involved": involved,
                 "distance": None,
                 "description": "rule-based: unspecified location"}
            ]

    # Tumor size (normalize to mm to match gold)
    if (m := PATTERNS["tumor_size_mm"].search(report)):
        out_cancer_data["tumor_size"] = int(round(float(m.group(1))))
    elif (m := PATTERNS["tumor_size_cm"].search(report)):
        out_cancer_data["tumor_size"] = int(round(float(m.group(1)) * 10))

    # Breast biomarkers (emit as list-of-dicts matching gold)
    bm = []
    if (m := PATTERNS["er_status"].search(report)):
        bm.append({
            "biomarker_category": "er",
            "expression": m.group(1).lower() == "positive",
            "percentage": None, "score": None,
            "biomarker_name": "estrogen receptor",
        })
    if (m := PATTERNS["pr_status"].search(report)):
        bm.append({
            "biomarker_category": "pr",
            "expression": m.group(1).lower() == "positive",
            "percentage": None, "score": None,
            "biomarker_name": "progesterone receptor",
        })
    if (m := PATTERNS["her2_status"].search(report)):
        status = m.group(1).lower()
        bm.append({
            "biomarker_category": "her2",
            "expression": None if status == "equivocal" else (status == "positive"),
            "percentage": None,
            "score": 2 if status == "equivocal" else (3 if status == "positive" else 0),
            "biomarker_name": "human epidermal growth factor receptor 2",
        })
    if bm:
        out_cancer_data["biomarkers"] = bm

    cancer_category = classify_organ(report)
    return {
        "cancer_excision_report": cancer_category is not None,
        "cancer_category": cancer_category,
        "cancer_category_others_description": None,
        "cancer_data": out_cancer_data,
    }


def main() -> None:
    if len(sys.argv) > 1:
        report = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        report = sys.stdin.read()
    print(json.dumps(extract(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
