"""Shared dataset loader for the ClinicalBERT and gpt4 baselines.

Cases are discovered by walking the gold-annotation tree under
``<root>/data/<dataset>/annotations/gold/<organ_n>/*.json``. The matching
report path is derived by convention as
``<root>/data/<dataset>/reports/<organ_n>/<case_id>.txt``.

There are no train/test splits within a corpus: the cross-corpus baseline
uses every case of CMUH for training and every case of TCGA for prediction.
Disjointness is guaranteed by the dataset boundary (CMUH and TCGA are
disjoint corpora) and enforced at predict time by a dataset-disjointness
check on the checkpoint metadata.

Returns a uniform list of case dicts:
    {id, dataset, organ_n, cancer_category, report_path, annotation_path}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_CASE_ID_RE = re.compile(r"^([a-z]+)(\d+)_(\d+)$")


def _organ_n(case_id: str, dataset: str) -> str:
    m = re.match(rf"{re.escape(dataset)}(\d+)_", case_id)
    if not m:
        raise ValueError(
            f"could not parse organ_n from id {case_id!r} for dataset {dataset!r}"
        )
    return m.group(1)


def _walk_dataset(dataset: str, root: Path) -> list[dict]:
    """Walk ``<root>/data/<dataset>/annotations/gold/`` and return case dicts."""
    gold_root = root / "data" / dataset / "annotations" / "gold"
    reports_root = root / "data" / dataset / "reports"
    if not gold_root.is_dir():
        print(f"[warn] no gold annotations under {gold_root}")
        return []

    cases: list[dict] = []
    for ann_path in sorted(gold_root.rglob("*.json")):
        organ_n = ann_path.parent.name
        case_id = ann_path.stem
        report_path = reports_root / organ_n / f"{case_id}.txt"
        try:
            with ann_path.open(encoding="utf-8") as f:
                ann = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[warn] skipping malformed annotation {ann_path}: {e}")
            continue
        cases.append({
            "id": case_id,
            "dataset": dataset,
            "organ_n": organ_n,
            "cancer_category": ann.get("cancer_category"),
            "report_path": str(report_path),
            "annotation_path": str(ann_path),
        })
    return cases


def load_cases(
    datasets: list[str],
    root: Path,
    organs: set[str] | None = None,
    included_only: bool = False,
) -> list[dict]:
    """Pool gold-annotation cases across `datasets` and filter.

    Args:
        datasets: dataset names to include (e.g. ["cmuh", "tcga"]).
        root: experiment root containing ``data/<dataset>/`` subtrees
            (use ``dummy`` for dev, ``workspace`` for production).
        organs: keep only cases whose cancer_category is in this set.
        included_only: drop cases whose cancer_excision_report is False.
    """
    out: list[dict] = []
    for ds in datasets:
        out.extend(_walk_dataset(ds, root))

    if organs is not None:
        out = [c for c in out if c.get("cancer_category") in organs]

    if included_only:
        kept = []
        for c in out:
            with open(c["annotation_path"], encoding="utf-8") as f:
                if json.load(f).get("cancer_excision_report"):
                    kept.append(c)
        out = kept

    return out


def per_dataset_counts(cases: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in cases:
        counts[c["dataset"]] = counts.get(c["dataset"], 0) + 1
    return counts
