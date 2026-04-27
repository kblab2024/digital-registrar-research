"""Shared helpers for the eval_*_vs_*.py convenience wrappers.

Each wrapper script (eval_rule_vs_llm, eval_bert_vs_llm,
eval_rule_bert_llm) runs ``scripts.eval.cli non_nested`` for each
method, then ``scripts.eval.compare.run_compare`` to join the outputs.
This module factors out the orchestration so the wrappers stay tiny.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

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
                         organs: list[str] | None) -> list[str]:
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
        return cmd


def run_non_nested(spec: MethodSpec, *, root: str, dataset: str, out_dir: Path,
                    organs: list[str] | None) -> Path:
    """Invoke scripts.eval.cli non_nested for a single method.

    Returns the output directory containing correctness_table.parquet.
    Raises ``CalledProcessError`` on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = spec.non_nested_args(root=root, dataset=dataset, out=out_dir, organs=organs)
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
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")


def run_pipeline(specs: list[MethodSpec], args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    non_nested_dirs: dict[str, Path] = {}
    for spec in specs:
        out = args.out / f"non_nested_{spec.label}"
        run_non_nested(
            spec, root=args.folder, dataset=args.dataset, out_dir=out,
            organs=args.organs,
        )
        non_nested_dirs[spec.label] = out

    compare_out = args.out / "compare"
    run_compare(specs, non_nested_dirs, compare_out,
                n_boot=args.n_boot, seed=args.seed)
    print(f"compare output: {compare_out}")
    return 0
