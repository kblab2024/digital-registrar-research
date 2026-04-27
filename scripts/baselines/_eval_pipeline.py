"""Shared helpers for the eval_*_vs_*.py convenience wrappers.

Each wrapper script (eval_rule_vs_llm, eval_bert_vs_llm,
eval_rule_bert_llm) runs ``scripts.eval.cli non_nested`` for each
(method, dataset) pair, then concatenates the per-dataset
``correctness_table.parquet`` outputs into a single per-method parquet
(with a ``dataset`` column added), then calls
``scripts.eval.compare.run_compare`` to join across methods.

Defaults (cross-corpus baseline contract)
------------------------------------------
- ``--folder workspace`` (override to ``dummy`` or absolute path).
- ``--datasets tcga`` — TCGA is the LLM-comparable evaluation corpus
  (privacy: OpenAI API can't see CMUH). BERT trains on CMUH only, so
  TCGA is fully held out.
- ``--split all`` — score every TCGA case (no test-fold filter needed
  since TCGA was never in training).

For intra-corpus ablations (e.g. CMUH-only benchmark), pass
``--datasets cmuh --split test`` so the leakage guard passes.

Split-aware filtering
---------------------
When ``--split`` is ``test`` or ``train``, the wrapper resolves
``splits.json`` per (folder, dataset), writes
``cases_<split>_<dataset>.txt``, and passes ``--cases @<file>`` to the
underlying ``non_nested`` call so all methods are scored on the same
case set.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from baselines._split_helpers import (
    SplitName, per_organ_counts, resolve_case_allowlist, write_allowlist_file,
)
from _config_loader import resolve_folder

logger = logging.getLogger("scripts.baselines.eval_pipeline")

DEFAULT_DATASETS = ("tcga",)  # Cross-corpus default: TCGA is the LLM-comparable
                               # held-out corpus. Override with `cmuh` or
                               # `cmuh tcga` for ablations.
ALL_DATASETS = ("cmuh", "tcga")


class MethodSpec:
    """One method to evaluate: name + invocation args."""

    def __init__(self, label: str, method: str, model: str | None,
                 run_ids: list[str] | None = None):
        self.label = label
        self.method = method  # "rule_based" | "clinicalbert" | "llm"
        self.model = model
        self.run_ids = run_ids or []

    def non_nested_args(self, root: str, dataset: str, out: Path,
                         organs: list[str] | None,
                         cases_file: Path | None) -> list[str]:
        cmd = [
            sys.executable, "-m", "scripts.eval.cli", "non_nested",
            "--root", root, "--dataset", dataset,
            "--method", self.method,
            "--annotator", "gold",
            "--out", str(out),
        ]
        if self.model:
            cmd += ["--model", self.model]
        if self.run_ids:
            cmd += ["--run-ids", *self.run_ids]
        if organs:
            cmd += ["--organs", *organs]
        if cases_file is not None:
            cmd += ["--cases", f"@{cases_file}"]
        return cmd


def run_non_nested(spec: MethodSpec, *, root: str, dataset: str, out_dir: Path,
                    organs: list[str] | None,
                    cases_file: Path | None) -> Path:
    """Invoke non_nested for one (method, dataset). Returns the parquet path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = spec.non_nested_args(
        root=root, dataset=dataset, out=out_dir, organs=organs,
        cases_file=cases_file,
    )
    logger.info("[%s/%s] %s", spec.label, dataset, " ".join(cmd))
    subprocess.run(cmd, check=True)
    parquet = out_dir / "correctness_table.parquet"
    if not parquet.is_file():
        raise SystemExit(
            f"non_nested for {spec.label}/{dataset} produced no parquet at {parquet}"
        )
    return parquet


def concat_per_dataset_parquets(
    per_dataset: dict[str, Path], combined_path: Path,
) -> Path:
    """Read each per-dataset parquet, add a ``dataset`` column, concat, write.

    Returns the path to the combined parquet (which run_compare consumes).
    """
    frames = []
    for dataset, parquet in per_dataset.items():
        df = pd.read_parquet(parquet)
        if "dataset" not in df.columns:
            df = df.assign(dataset=dataset)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(combined_path, index=False)
    return combined_path


def run_compare(specs: list[MethodSpec], non_nested_dirs: dict[str, Path],
                out_dir: Path, n_boot: int, seed: int) -> None:
    """Invoke scripts.eval.compare.run_compare with the joined inputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = [f"{s.label}:{non_nested_dirs[s.label]}" for s in specs]
    cmd = [
        sys.executable, "-m", "scripts.eval.compare.run_compare",
        "--inputs", *inputs,
        "--out", str(out_dir),
        "--n-boot", str(n_boot),
        "--seed", str(seed),
    ]
    logger.info("[compare] %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--folder", default="workspace",
                    help="Experiment root (default: workspace; dummy / abs path "
                         "also accepted). Passed to scripts.eval.cli non_nested as --root.")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                    choices=ALL_DATASETS,
                    help="Dataset(s) to evaluate on (default: tcga — the LLM-"
                         "comparable held-out corpus). Pass 'cmuh' or 'cmuh tcga' "
                         "for intra-corpus ablations (then use --split test).")
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Restrict to organ indices (1..10) or names.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output root. Subdirs non_nested_<label>/ and compare/ "
                         "are created under it.")
    ap.add_argument("--split", default="all", choices=("test", "train", "all"),
                    help="Which split to score (default: all, since the default "
                         "eval corpus is TCGA which is held out from CMUH-only "
                         "training). Use 'test' for intra-corpus ablations.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")


def _resolve_cases_per_dataset(
    args: argparse.Namespace,
) -> dict[str, Path | None]:
    """Per-dataset case allowlist files (or None when split=='all').

    Writes ``<out>/cases_<split>_<dataset>.txt`` for each dataset that
    has a non-empty allowlist; raises if any dataset's split is empty
    when a non-'all' filter is requested.
    """
    out: dict[str, Path | None] = {}
    if args.split == "all":
        for ds in args.datasets:
            out[ds] = None
        logger.info("split=all: scoring every gold case (no --cases filter)")
        return out
    folder = resolve_folder(args.folder)
    for ds in args.datasets:
        case_ids = resolve_case_allowlist(folder, ds, args.split)
        if not case_ids:
            raise SystemExit(
                f"split={args.split!r} resolved to an empty case list for "
                f"({args.folder}, {ds}). Refusing to run an eval on zero cases."
            )
        args.out.mkdir(parents=True, exist_ok=True)
        out_file = args.out / f"cases_{args.split}_{ds}.txt"
        write_allowlist_file(case_ids, out_file)
        counts = per_organ_counts(case_ids)
        pretty_counts = ", ".join(
            f"organ {organ}: {n}" for organ, n in sorted(counts.items())
        )
        logger.info(
            "[%s] split=%s: %d cases (%s) → %s",
            ds, args.split, len(case_ids), pretty_counts, out_file,
        )
        out[ds] = out_file
    return out


def run_pipeline(specs: list[MethodSpec], args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cases_files = _resolve_cases_per_dataset(args)

    # Per method × dataset: run non_nested into a per-dataset subdir.
    # Then concatenate per-dataset parquets per method into a single combined
    # parquet (with a `dataset` column) that run_compare consumes.
    method_combined: dict[str, Path] = {}
    for spec in specs:
        method_root = args.out / f"non_nested_{spec.label}"
        per_ds_parquets: dict[str, Path] = {}
        for ds in args.datasets:
            ds_out = method_root / ds
            parquet = run_non_nested(
                spec, root=args.folder, dataset=ds, out_dir=ds_out,
                organs=args.organs, cases_file=cases_files.get(ds),
            )
            per_ds_parquets[ds] = parquet
        # Combined parquet at the method root (sibling of <dataset>/ subdirs).
        combined = method_root / "correctness_table.parquet"
        concat_per_dataset_parquets(per_ds_parquets, combined)
        logger.info(
            "[%s] combined %d datasets into %s",
            spec.label, len(per_ds_parquets), combined,
        )
        method_combined[spec.label] = method_root

    compare_out = args.out / "compare"
    run_compare(specs, method_combined, compare_out,
                n_boot=args.n_boot, seed=args.seed)
    print(f"compare output: {compare_out}")
    return 0
