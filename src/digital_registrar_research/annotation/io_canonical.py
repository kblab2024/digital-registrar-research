"""Folder discovery and sample indexing for the canonical dataset layout.

Canonical layout (as produced by ``scripts/gen_dummy_skeleton.py`` and
``scripts/select_without_preann_subset.py``)::

    <base_dir>/
    └── data/<dataset>/              # e.g. cmuh, tcga
        ├── reports/<n>/<case_id>.txt                   # used by with_preann mode
        ├── reports_without_preann/<n>/<case_id>.txt    # curated subset for without_preann mode
        ├── preannotation/<model>/<n>/<case_id>.json
        └── annotations/<annotator>_<mode>/<n>/<case_id>.json

``with_preann`` mode reads from ``reports/`` (the full dataset);
``without_preann`` mode reads from ``reports_without_preann/`` (a
curated subset, see ``scripts/select_without_preann_subset.py``). Both
modes share ``preannotation/`` (loaded only in ``with_preann``) and
write annotations to ``annotations/<annotator>_<mode>/`` so the two
modes' outputs stay separate per annotator. Pre-annotation model is
fixed to ``gpt_oss_20b`` for now. JSON I/O helpers are re-exported from
:mod:`.io` so the save payload format stays identical across both UIs.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from .io import (  # noqa: F401 — re-exported for app_canonical
    NA_SENTINEL,
    build_save_payload,
    load_json,
    load_report_text,
    rehydrate_sentinels,
    save_annotation,
    strip_meta,
)

# TODO: make this config-driven once we support multiple pre-annotation models.
PREANNOTATION_MODEL = "gpt_oss_20b"

MODES = ("with_preann", "without_preann")


def _reports_subdir(mode: str) -> str:
    return "reports_without_preann" if mode == "without_preann" else "reports"


@dataclass
class SampleRef:
    sample_id: str            # e.g. "tcga1_37"
    stem: str                 # e.g. "tcga1"
    n: str                    # e.g. "1" (organ folder name)
    report_path: str          # {reports_dir}/{n}/{sample_id}.txt
    preannotation_path: str   # {preannotation_dir}/{n}/{sample_id}.json  (may not exist)
    annotation_path: str      # {annotations_dir}/{annotator}/{n}/{sample_id}.json
    annotator_suffix: str


@dataclass
class WorkspaceSet:
    base_dir: str            # e.g. .../workspace
    mode: str                # "with_preann" | "without_preann"
    dataset: str             # e.g. "tcga"
    dataset_root: str        # {base_dir}/{mode}/data/{dataset}
    reports_dir: str         # {dataset_root}/reports
    preannotation_dir: str   # {dataset_root}/preannotation/{PREANNOTATION_MODEL}
    annotations_dir: str     # {dataset_root}/annotations


def list_datasets(base_dir: str, mode: str) -> list[str]:
    """Return sorted names of datasets under ``{base_dir}/data/``.

    A directory counts as a dataset iff it has the per-mode reports
    subdir (``reports/`` for ``with_preann``, ``reports_without_preann/``
    for ``without_preann``), so we don't trip on stray metadata folders
    and we hide datasets that haven't been curated for the active mode.
    """
    if not base_dir or not mode:
        return []
    data_root = os.path.join(base_dir, "data")
    if not os.path.isdir(data_root):
        return []
    sub = _reports_subdir(mode)
    out: list[str] = []
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, sub)):
            out.append(entry)
    return out


def load_workspace(base_dir: str, dataset: str, mode: str) -> WorkspaceSet | None:
    if not base_dir or not dataset or not mode:
        return None
    dataset_root = os.path.join(base_dir, "data", dataset)
    reports_dir = os.path.join(dataset_root, _reports_subdir(mode))
    if not os.path.isdir(reports_dir):
        return None
    return WorkspaceSet(
        base_dir=base_dir,
        mode=mode,
        dataset=dataset,
        dataset_root=dataset_root,
        reports_dir=reports_dir,
        preannotation_dir=os.path.join(dataset_root, "preannotation", PREANNOTATION_MODEL),
        annotations_dir=os.path.join(dataset_root, "annotations"),
    )


def list_samples(ws: WorkspaceSet, annotator_suffix: str) -> list[SampleRef]:
    """Walk ``reports/<n>/<case_id>.txt`` and build SampleRefs.

    We intentionally list reports (not pre-annotations) so that samples with no
    pre-annotation still show up in ``without_preann`` mode.
    """
    samples: list[SampleRef] = []
    pattern = os.path.join(ws.reports_dir, "*", "*.txt")
    annotator_dir = f"{annotator_suffix}_{ws.mode}"
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
                ws.annotations_dir, annotator_dir, n, f"{sample_id}.json"
            ),
            annotator_suffix=annotator_suffix,
        ))
    samples.sort(key=_sample_sort_key)
    return samples


def _sample_sort_key(s: SampleRef) -> tuple:
    import re
    m = re.match(r"([A-Za-z]+)(\d+)_(\d+)$", s.sample_id)
    if m:
        return (m.group(1), int(m.group(2)), int(m.group(3)))
    return (s.sample_id, 0, 0)


# ── Canonical path helpers (used by compare_app_canonical) ────────────────────

def gold_path(ws: WorkspaceSet, n: str, case_id: str) -> str:
    """Path to the consensus-gold annotation under the canonical layout.

    Eval scripts read gold from a flat ``annotations/gold/`` folder — no
    ``_with_preann`` / ``_without_preann`` mode suffix — so the consensus
    output is shared across both annotation modes.
    """
    return os.path.join(ws.annotations_dir, "gold", n, f"{case_id}.json")


def annotator_path(ws: WorkspaceSet, annotator_suffix: str, n: str, case_id: str) -> str:
    """Path to a per-annotator annotation file under the canonical layout.

    Mirrors ``list_samples``' ``{suffix}_{mode}/{n}/{case_id}.json`` rule
    but is callable for the "B" annotator in compare/consensus mode.
    Gold is special-cased to the flat ``gold/`` folder so eval can find it.
    """
    if annotator_suffix == "gold":
        return gold_path(ws, n, case_id)
    return os.path.join(
        ws.annotations_dir, f"{annotator_suffix}_{ws.mode}", n, f"{case_id}.json"
    )


def prediction_path(
    ws_root: str,
    dataset: str,
    method: str,
    model: str | None,
    run_id: str | None,
    n: str,
    case_id: str,
) -> str:
    """Path to a model prediction file under ``results/predictions/``.

    Mirrors ``scripts/eval/_common/paths.py:Paths.prediction``. Duplicated
    here because ``scripts/`` is not on the package import path.
    """
    base = os.path.join(ws_root, "results", "predictions", dataset)
    if method == "llm":
        if not model or not run_id:
            raise ValueError("llm predictions require model and run_id")
        return os.path.join(base, "llm", model, run_id, n, f"{case_id}.json")
    if method == "clinicalbert":
        if not model:
            raise ValueError("clinicalbert predictions require model")
        if run_id:
            return os.path.join(base, "clinicalbert", model, run_id, n, f"{case_id}.json")
        return os.path.join(base, "clinicalbert", model, n, f"{case_id}.json")
    if method == "rule_based":
        return os.path.join(base, "rule_based", n, f"{case_id}.json")
    raise ValueError(f"unknown method: {method!r}")


def list_predictions(ws_root: str, dataset: str) -> list[tuple[str, str | None, str | None]]:
    """Discover available predictions under ``{ws_root}/results/predictions/{dataset}/``.

    Returns a sorted list of ``(method, model, run_id)`` triples. ``model``
    is ``None`` for ``rule_based`` and ``run_id`` is ``None`` when the
    method/model combination has no run-id sub-folders (e.g. some
    clinicalbert single-run layouts).
    """
    out: list[tuple[str, str | None, str | None]] = []
    base = os.path.join(ws_root, "results", "predictions", dataset)
    if not os.path.isdir(base):
        return out

    # rule_based: leaf is rule_based/{n}/{case}.json
    rb_dir = os.path.join(base, "rule_based")
    if os.path.isdir(rb_dir):
        out.append(("rule_based", None, None))

    # llm: llm/{model}/{run_id}/{n}/{case}.json
    llm_dir = os.path.join(base, "llm")
    if os.path.isdir(llm_dir):
        for model in sorted(os.listdir(llm_dir)):
            mdir = os.path.join(llm_dir, model)
            if not os.path.isdir(mdir):
                continue
            for run_id in sorted(os.listdir(mdir)):
                rdir = os.path.join(mdir, run_id)
                if os.path.isdir(rdir):
                    out.append(("llm", model, run_id))

    # clinicalbert: clinicalbert/{model}[/{run_id}]/{n}/{case}.json
    cb_dir = os.path.join(base, "clinicalbert")
    if os.path.isdir(cb_dir):
        for model in sorted(os.listdir(cb_dir)):
            mdir = os.path.join(cb_dir, model)
            if not os.path.isdir(mdir):
                continue
            run_subdirs = []
            has_organ_subdir = False
            for entry in sorted(os.listdir(mdir)):
                full = os.path.join(mdir, entry)
                if not os.path.isdir(full):
                    continue
                # An organ subdir (numeric) means there's no run-id layer.
                if entry.isdigit():
                    has_organ_subdir = True
                else:
                    run_subdirs.append(entry)
            if has_organ_subdir and not run_subdirs:
                out.append(("clinicalbert", model, None))
            else:
                for run_id in run_subdirs:
                    out.append(("clinicalbert", model, run_id))

    return out
