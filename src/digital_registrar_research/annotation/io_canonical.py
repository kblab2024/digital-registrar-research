"""Folder discovery and sample indexing for the canonical dataset layout.

Canonical layout (as produced by ``scripts/gen_dummy_skeleton.py``)::

    <base_dir>/
    └── data/
        └── <dataset>/                 # e.g. cmuh, tcga
            ├── reports/<n>/<case_id>.txt
            ├── preannotation/<model>/<n>/<case_id>.json
            └── annotations/<annotator>_<mode>/<n>/<case_id>.json

Pre-annotation model is fixed to ``gpt_oss_20b`` for now (the only model
present in ``dummy/``). JSON I/O helpers are re-exported from :mod:`.io`
so the save payload format stays identical across both UIs.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from .io import (  # noqa: F401 — re-exported for app_canonical
    build_save_payload,
    load_json,
    load_report_text,
    save_annotation,
    strip_meta,
)

# TODO: make this config-driven once we support multiple pre-annotation models.
PREANNOTATION_MODEL = "gpt_oss_20b"

MODES = ("with_preann", "without_preann")


@dataclass
class SampleRef:
    sample_id: str            # e.g. "tcga1_37"
    stem: str                 # e.g. "tcga1"
    n: str                    # e.g. "1" (organ folder name)
    report_path: str          # {reports_dir}/{n}/{sample_id}.txt
    preannotation_path: str   # {preannotation_dir}/{n}/{sample_id}.json  (may not exist)
    annotation_path: str      # {annotations_dir}/{annotator}_{mode}/{n}/{sample_id}.json
    annotator_suffix: str
    mode: str


@dataclass
class WorkspaceSet:
    base_dir: str            # e.g. .../workspace
    dataset: str             # e.g. "tcga"
    dataset_root: str        # {base_dir}/data/{dataset}
    reports_dir: str         # {dataset_root}/reports
    preannotation_dir: str   # {dataset_root}/preannotation/{PREANNOTATION_MODEL}
    annotations_dir: str     # {dataset_root}/annotations


def list_datasets(base_dir: str) -> list[str]:
    """Return sorted names of datasets under ``{base_dir}/data/``.

    A directory counts as a dataset iff it has a ``reports/`` subdir (so
    we don't trip on stray metadata folders).
    """
    if not base_dir:
        return []
    data_root = os.path.join(base_dir, "data")
    if not os.path.isdir(data_root):
        return []
    out: list[str] = []
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, "reports")):
            out.append(entry)
    return out


def load_workspace(base_dir: str, dataset: str) -> WorkspaceSet | None:
    if not base_dir or not dataset:
        return None
    dataset_root = os.path.join(base_dir, "data", dataset)
    reports_dir = os.path.join(dataset_root, "reports")
    if not os.path.isdir(reports_dir):
        return None
    return WorkspaceSet(
        base_dir=base_dir,
        dataset=dataset,
        dataset_root=dataset_root,
        reports_dir=reports_dir,
        preannotation_dir=os.path.join(dataset_root, "preannotation", PREANNOTATION_MODEL),
        annotations_dir=os.path.join(dataset_root, "annotations"),
    )


def list_samples(ws: WorkspaceSet, annotator_suffix: str, mode: str) -> list[SampleRef]:
    """Walk ``reports/<n>/<case_id>.txt`` and build SampleRefs.

    We intentionally list reports (not pre-annotations) so that samples with no
    pre-annotation still show up in ``without_preann`` mode.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}.")
    samples: list[SampleRef] = []
    pattern = os.path.join(ws.reports_dir, "*", "*.txt")
    for path in glob.glob(pattern):
        filename = os.path.basename(path)
        sample_id, _ = os.path.splitext(filename)
        n = os.path.basename(os.path.dirname(path))
        # stem = the alphabetic+digit prefix before the last "_<idx>"
        stem = sample_id.rsplit("_", 1)[0] if "_" in sample_id else sample_id
        samples.append(SampleRef(
            sample_id=sample_id,
            stem=stem,
            n=n,
            report_path=path,
            preannotation_path=os.path.join(ws.preannotation_dir, n, f"{sample_id}.json"),
            annotation_path=os.path.join(
                ws.annotations_dir, f"{annotator_suffix}_{mode}", n, f"{sample_id}.json"
            ),
            annotator_suffix=annotator_suffix,
            mode=mode,
        ))
    samples.sort(key=_sample_sort_key)
    return samples


def _sample_sort_key(s: SampleRef) -> tuple:
    import re
    m = re.match(r"([A-Za-z]+)(\d+)_(\d+)$", s.sample_id)
    if m:
        return (m.group(1), int(m.group(2)), int(m.group(3)))
    return (s.sample_id, 0, 0)
