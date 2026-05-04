"""Dataset-aware organ_n <-> organ_name lookups.

Single source of truth for converting numeric organ folder codes to
organ names. TCGA and CMUH use different numbering (historical artifact
of the data ingest pipelines):

    TCGA: 1=breast, 2=colorectal, 3=thyroid, 4=stomach, 5=liver
    CMUH: 1=pancreas, 2=breast, 3=cervix, 4=colorectal, 5=esophagus,
          6=liver, 7=lung, 8=prostate, 9=stomach, 10=thyroid

The mapping is loaded from configs/organ_code.yaml. Do not duplicate
this mapping in code — call into this module instead.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ORGAN_CODE_YAML = _REPO_ROOT / "configs" / "organ_code.yaml"


@cache
def load_organ_code() -> dict[str, dict[int, str]]:
    """Load configs/organ_code.yaml. Returns ``{dataset: {organ_n: name}}``."""
    if not _ORGAN_CODE_YAML.is_file():
        raise FileNotFoundError(
            f"organ_code.yaml not found at {_ORGAN_CODE_YAML}")
    doc = yaml.safe_load(_ORGAN_CODE_YAML.read_text(encoding="utf-8")) or {}
    raw = doc.get("organ_code") or {}
    out: dict[str, dict[int, str]] = {}
    for dataset, mapping in raw.items():
        out[dataset] = {int(k): str(v) for k, v in mapping.items()}
    return out


def organ_n_to_name(dataset: str, organ_n: str | int) -> str | None:
    """Resolve a folder code to an organ name. Returns None on miss."""
    try:
        idx = int(organ_n)
    except (TypeError, ValueError):
        return None
    return load_organ_code().get(dataset, {}).get(idx)


def organ_name_to_n(dataset: str, name: str) -> str | None:
    """Inverse lookup: organ name -> folder code (as string). None on miss."""
    inv = {v: k for k, v in load_organ_code().get(dataset, {}).items()}
    n = inv.get(name)
    return str(n) if n is not None else None


def dataset_organs(dataset: str) -> list[str]:
    """Ordered list of organ names for a dataset (sorted by folder number)."""
    mapping = load_organ_code().get(dataset, {})
    return [mapping[k] for k in sorted(mapping)]
