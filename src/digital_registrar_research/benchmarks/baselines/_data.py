"""Shared dataset loader for the ClinicalBERT baselines.

Resolves train/test cases from either layout the project uses:

- Dummy fixture (id-only splits): ``<root>/data/<dataset>/splits.json``
  containing ``{"train": [bare_id, ...], "test": [...]}``. Report and
  gold annotation paths are derived by convention from the dataset and
  the digit prefix in the id (``cmuh3_17`` → organ folder ``3``).

- Production data (full case dicts): the packaged
  ``benchmarks/data/splits.json`` carries Mac-absolute paths for the
  TCGA gold set; we remap them onto the local
  ``data/tcga_{dataset,annotation}_*`` folders via ``RAW_REPORTS`` /
  ``GOLD_ANNOTATIONS`` from ``paths.py``.

Returns a uniform list of case dicts:
    {id, dataset, organ_n, cancer_category, report_path, annotation_path}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ...paths import GOLD_ANNOTATIONS, RAW_REPORTS, SPLITS_JSON


def _organ_n(case_id: str, dataset: str) -> str:
    m = re.match(rf"{re.escape(dataset)}(\d+)_", case_id)
    if not m:
        raise ValueError(
            f"could not parse organ_n from id {case_id!r} for dataset {dataset!r}"
        )
    return m.group(1)


def _read_cancer_category(annotation_path: Path) -> str | None:
    with annotation_path.open(encoding="utf-8") as f:
        return json.load(f).get("cancer_category")


def _resolve_dummy(root: Path, dataset: str, case_id: str) -> dict:
    organ_n = _organ_n(case_id, dataset)
    ann = root / "data" / dataset / "annotations" / "gold" / organ_n / f"{case_id}.json"
    rep = root / "data" / dataset / "reports" / organ_n / f"{case_id}.txt"
    return {
        "id": case_id,
        "dataset": dataset,
        "organ_n": organ_n,
        "cancer_category": _read_cancer_category(ann) if ann.exists() else None,
        "report_path": str(rep),
        "annotation_path": str(ann),
    }


def _resolve_production(case: dict, dataset: str) -> dict:
    """Remap Mac-absolute paths from packaged splits onto local layout."""
    case_id = case["id"]
    report_src = Path(case["report_path"])
    ann_src = Path(case["annotation_path"])
    return {
        "id": case_id,
        "dataset": dataset,
        "organ_n": _organ_n(case_id, dataset),
        "cancer_category": case.get("cancer_category"),
        "report_path": str(RAW_REPORTS / report_src.parent.name / report_src.name),
        "annotation_path": str(GOLD_ANNOTATIONS / ann_src.parent.name / ann_src.name),
    }


def _load_one_dataset(dataset: str, split: str, root: Path) -> list[dict]:
    """Try dummy layout first, fall back to packaged TCGA splits."""
    dummy_splits = root / "data" / dataset / "splits.json"
    if dummy_splits.exists():
        with dummy_splits.open(encoding="utf-8") as f:
            data = json.load(f)
        return [_resolve_dummy(root, dataset, cid) for cid in data[split]]

    if dataset == "tcga" and SPLITS_JSON.exists():
        with SPLITS_JSON.open(encoding="utf-8") as f:
            data = json.load(f)
        return [_resolve_production(c, dataset) for c in data[split]]

    print(f"[warn] no splits.json for dataset={dataset!r} under {root}")
    return []


def load_cases(
    datasets: list[str],
    split: str,
    root: Path,
    organs: set[str] | None = None,
    included_only: bool = False,
) -> list[dict]:
    """Pool cases across `datasets` and filter by organ / excision-report.

    Args:
        datasets: dataset names to include (e.g. ["cmuh", "tcga"]).
        split: "train" or "test".
        root: data root containing ``data/<dataset>/`` subtrees. Use the
            repo root for production data (TCGA falls back to packaged
            splits when no per-dataset splits.json exists).
        organs: keep only cases whose cancer_category is in this set.
        included_only: drop cases whose cancer_excision_report is False.
    """
    out: list[dict] = []
    for ds in datasets:
        out.extend(_load_one_dataset(ds, split, root))

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
