"""Single source of truth for per-dataset organ-n mappings.

Loads `configs/organ_code.yaml` once at import. The two datasets number
their organs differently and that file is the only place that truth lives:

    TCGA (5 organs):  1=breast, 2=colorectal, 3=esophagus, 4=stomach, 5=liver
    CMUH (10 organs): 1=pancreas, 2=breast, 3=cervix, 4=colorectal,
                      5=esophagus, 6=liver, 7=lung, 8=prostate, 9=stomach,
                      10=thyroid

Every site that converts `organ_n <-> organ_name`, parses a case_id, or
needs the cross-corpus organ scope must use these helpers.
"""
from __future__ import annotations

import re
from functools import lru_cache

import yaml

from ..paths import REPO_ROOT

ORGAN_CODE_YAML = REPO_ROOT / "configs" / "organ_code.yaml"


@lru_cache(maxsize=1)
def _load_yaml() -> dict[str, dict[int, str]]:
    with ORGAN_CODE_YAML.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    block = raw.get("organ_code") or {}
    out: dict[str, dict[int, str]] = {}
    for ds, mapping in block.items():
        if not isinstance(mapping, dict):
            raise ValueError(
                f"organ_code.yaml: dataset {ds!r} mapping is not a dict"
            )
        out[ds] = {int(k): str(v) for k, v in mapping.items()}
    if not out:
        raise ValueError(
            f"organ_code.yaml at {ORGAN_CODE_YAML} has no `organ_code:` block"
        )
    return out


def all_datasets() -> tuple[str, ...]:
    """Datasets defined in organ_code.yaml (sorted)."""
    return tuple(sorted(_load_yaml().keys()))


def organs_for(dataset: str) -> dict[int, str]:
    """{organ_n: organ_name} for one dataset."""
    data = _load_yaml()
    if dataset not in data:
        raise KeyError(
            f"unknown dataset {dataset!r}; known: {sorted(data.keys())}"
        )
    return dict(data[dataset])


def organ_name(dataset: str, organ_n: int) -> str:
    """Lookup organ name for a (dataset, organ_n). KeyError if unknown."""
    mapping = organs_for(dataset)
    if organ_n not in mapping:
        raise KeyError(
            f"dataset {dataset!r} has no organ_n={organ_n}; "
            f"valid: {sorted(mapping.keys())}"
        )
    return mapping[organ_n]


def organ_n_for(dataset: str, organ_name: str) -> int:
    """Reverse lookup: organ_n for a (dataset, organ_name). KeyError if unknown."""
    name = organ_name.strip().lower()
    for n, label in organs_for(dataset).items():
        if label.lower() == name:
            return n
    raise KeyError(
        f"dataset {dataset!r} has no organ named {organ_name!r}; "
        f"valid: {sorted(organs_for(dataset).values())}"
    )


def common_organs(*datasets: str) -> tuple[str, ...]:
    """Intersection of organ-name sets across datasets, sorted alphabetically.

    `common_organs('cmuh', 'tcga')` returns the 5 organs both cover, which is
    the canonical scope for the cross-corpus train-CMUH/test-TCGA baseline.
    """
    if not datasets:
        datasets = all_datasets()
    sets = [set(organs_for(ds).values()) for ds in datasets]
    return tuple(sorted(set.intersection(*sets)))


def union_organs(*datasets: str) -> tuple[str, ...]:
    """Union of organ-name sets across datasets, sorted alphabetically."""
    if not datasets:
        datasets = all_datasets()
    sets = [set(organs_for(ds).values()) for ds in datasets]
    return tuple(sorted(set.union(*sets)))


_CASE_ID_RE = re.compile(r"^([a-z]+)(\d+)_(\d+)$")


def parse_case_id(case_id: str) -> tuple[str, int, int]:
    """Decode `{dataset}{organ_n}_{case_num}`, validating against the yaml.

    Examples:
        parse_case_id('cmuh1_42')   -> ('cmuh', 1, 42)   # CMUH 1 = pancreas
        parse_case_id('tcga3_99')   -> ('tcga', 3, 99)   # TCGA 3 = esophagus
        parse_case_id('tcga6_1')    -> ValueError         # TCGA has no organ 6

    Raises ValueError if the id is malformed, the dataset is unknown, or
    the organ_n is not defined for that dataset.
    """
    m = _CASE_ID_RE.match(case_id)
    if not m:
        raise ValueError(f"malformed case id: {case_id!r}")
    dataset, organ_str, num_str = m.group(1), m.group(2), m.group(3)
    organs = _load_yaml().get(dataset)
    if organs is None:
        raise ValueError(
            f"unknown dataset prefix in case id {case_id!r}; "
            f"known datasets: {sorted(_load_yaml().keys())}"
        )
    organ_n = int(organ_str)
    if organ_n not in organs:
        raise ValueError(
            f"case id {case_id!r}: dataset {dataset!r} has no organ_n={organ_n}; "
            f"valid: {sorted(organs.keys())}"
        )
    return dataset, organ_n, int(num_str)


# Convenience constants — re-evaluated lazily via the cached loader.
def CMUH_ORGANS() -> dict[int, str]:  # noqa: N802 (constant-style accessor)
    return organs_for("cmuh")


def TCGA_ORGANS() -> dict[int, str]:  # noqa: N802
    return organs_for("tcga")


def COMMON_ORGANS() -> tuple[str, ...]:  # noqa: N802
    return common_organs("cmuh", "tcga")


__all__ = [
    "ORGAN_CODE_YAML",
    "all_datasets",
    "organs_for",
    "organ_name",
    "organ_n_for",
    "common_organs",
    "union_organs",
    "parse_case_id",
    "CMUH_ORGANS",
    "TCGA_ORGANS",
    "COMMON_ORGANS",
]
