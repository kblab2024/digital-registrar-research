"""Shared helpers for the eval_*_vs_*.py convenience wrappers.

Each wrapper script (eval_rule_vs_llm, eval_bert_vs_llm,
eval_rule_bert_llm) runs ``scripts.eval.cli non_nested`` for each
method, then ``scripts.eval.compare.run_compare`` to join the outputs.

Split-aware by default: every wrapper resolves the train/test split
from ``{folder}/data/{dataset}/splits.json`` (or the packaged TCGA
fallback) and restricts every ``non_nested`` call to the test set via
``--cases @<test_ids.txt>``. This is **load-bearing** when BERT is
involved — BERT is trained on the train split, so scoring it on
training cases would be in-sample memorization. Restricting all
methods to the same test set keeps coverage comparable.

Override with ``--split all`` to disable the filter (e.g. for rule-vs-
LLM only, where there's no leakage concern).
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from baselines._split_helpers import (
    SplitName, per_organ_counts, resolve_case_allowlist, write_allowlist_file,
)
from _config_loader import resolve_folder

logger = logging.getLogger("scripts.baselines.eval_pipeline")


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
    """Invoke scripts.eval.cli non_nested for a single method.

    Returns the output directory containing correctness_table.parquet.
    Raises ``CalledProcessError`` on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = spec.non_nested_args(
        root=root, dataset=dataset, out=out_dir, organs=organs,
        cases_file=cases_file,
    )
    logger.info("[%s] %s", spec.label, " ".join(cmd))
    subprocess.run(cmd, check=True)
    parquet = out_dir / "correctness_table.parquet"
    if not parquet.is_file():
        raise SystemExit(f"non_nested for {spec.label} produced no parquet at {parquet}")
    return out_dir


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
    ap.add_argument("--folder", required=True,
                    help="Experiment root: dummy / workspace / abs path. Passed to "
                         "scripts.eval.cli non_nested as --root.")
    ap.add_argument("--dataset", required=True, choices=("cmuh", "tcga"))
    ap.add_argument("--organs", nargs="*", default=None,
                    help="Restrict to organ indices (1..10) or names.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output root. Subdirs non_nested_<label>/ and compare/ "
                         "are created under it.")
    ap.add_argument("--split", default="test", choices=("test", "train", "all"),
                    help="Which split of cases to score (default: test). "
                         "BERT is trained on the train split, so default 'test' "
                         "prevents in-sample scoring. Use 'all' to disable the "
                         "filter (rule-vs-LLM only — no leakage concern).")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")


def _resolve_cases_file(args: argparse.Namespace) -> Path | None:
    """If args.split != 'all', resolve splits.json and write a case allowlist file.

    Returns the path to ``<out>/cases_<split>.txt``, or ``None`` if no
    filter is applied.
    """
    split: SplitName = args.split
    if split == "all":
        logger.info("split=all: scoring every gold case (no --cases filter)")
        return None
    folder = resolve_folder(args.folder)
    case_ids = resolve_case_allowlist(folder, args.dataset, split)
    if not case_ids:
        raise SystemExit(
            f"split={split!r} resolved to an empty case list for "
            f"({args.folder}, {args.dataset}). Refusing to run an eval "
            f"on zero cases."
        )
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"cases_{split}.txt"
    write_allowlist_file(case_ids, out_file)
    counts = per_organ_counts(case_ids)
    pretty_counts = ", ".join(
        f"organ {organ}: {n}" for organ, n in sorted(counts.items())
    )
    logger.info(
        "split=%s: %d cases (%s) → %s",
        split, len(case_ids), pretty_counts, out_file,
    )
    return out_file


def run_pipeline(specs: list[MethodSpec], args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cases_file = _resolve_cases_file(args)
    non_nested_dirs: dict[str, Path] = {}
    for spec in specs:
        out = args.out / f"non_nested_{spec.label}"
        run_non_nested(
            spec, root=args.folder, dataset=args.dataset, out_dir=out,
            organs=args.organs, cases_file=cases_file,
        )
        non_nested_dirs[spec.label] = out

    compare_out = args.out / "compare"
    run_compare(specs, non_nested_dirs, compare_out,
                n_boot=args.n_boot, seed=args.seed)
    print(f"compare output: {compare_out}")
    return 0
