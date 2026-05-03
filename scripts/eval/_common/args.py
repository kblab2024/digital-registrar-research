"""Shared CLI argument schema for eval subcommands.

Every subcommand assembles its parser by calling :func:`add_common_args`
and :func:`add_model_args` etc. as appropriate. This keeps the argument
surface uniform across subcommands so users can re-run different
analyses on the same prediction tree without re-learning a new flag set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .paths import KNOWN_METHODS, parse_run_id_to_path_segment
from .stratify import all_organ_indices, parse_organ_arg

KNOWN_ANNOTATORS: tuple[str, ...] = (
    "gold",
    "nhc_with_preann",
    "nhc_without_preann",
    "kpc_with_preann",
    "kpc_without_preann",
)


def add_common_args(parser: argparse.ArgumentParser, *, subcommand: str) -> None:
    """Add ``--root``, ``--dataset``, ``--organs``, ``--cases``, ``--out``,
    ``--seed``, ``--alpha``, ``--n-boot``, ``--n-jobs``, ``-v``.

    Every subcommand needs all of these. ``subcommand`` is the name of
    the calling subcommand (e.g. ``"non_nested"``); it is used to build
    the default ``--out`` directory ``workspace/results/eval/<subcommand>``.
    """
    parser.add_argument(
        "--root", required=True,
        help="Root tree to read from. 'dummy', 'workspace', or an absolute path.",
    )
    parser.add_argument(
        "--dataset", required=True, choices=("cmuh", "tcga"),
        help="Dataset name under <root>/data/.",
    )
    parser.add_argument(
        "--organs", nargs="+", default=None,
        help="Restrict to organ indices or names valid for --dataset (e.g. for "
             "tcga: 1..5 or breast/colorectal/esophagus/stomach/liver). Default: "
             "all organs defined for the dataset in configs/organ_code.yaml.",
    )
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Optional case-id allowlist. Pass either inline IDs or @path/to/list.txt.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("workspace") / "results" / "eval" / subcommand,
        help="Output directory (default: %(default)s). "
             "A manifest.json will be stamped here.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for bootstrap and resampling (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Confidence-interval coverage (default: %(default)s).",
    )
    parser.add_argument(
        "--n-boot", type=int, default=2000,
        help="Bootstrap replicates (default: %(default)s).",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel workers for bootstrap loops (default: %(default)s; "
             "set >1 with care — memory cost scales with n_jobs).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )


def add_model_args(parser: argparse.ArgumentParser, *, require_runs: bool = True) -> None:
    """Add ``--method``, ``--model``, ``--run-ids``, ``--annotator``.

    Used by the ``non_nested``, ``nested``, ``completeness`` subcommands
    that score model predictions against an annotator. ``require_runs``
    is False for subcommands that allow empty run lists (e.g. methods
    without runs).
    """
    parser.add_argument(
        "--method", default="llm", choices=KNOWN_METHODS,
        help="Prediction backend (default: %(default)s).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (required for method=llm/clinicalbert).",
    )
    parser.add_argument(
        "--run-ids", nargs="*", default=None,
        help="Run IDs as strings (run01, run02-alpha, ...). Default: discover all "
             "run* under the model's prediction tree. Single-run = one ID.",
    )
    parser.add_argument(
        "--annotator", default="gold", choices=KNOWN_ANNOTATORS,
        help="Annotator subdir to score against (default: %(default)s).",
    )


def add_iaa_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--annotators`` and ``--preann-model`` for IAA-style commands."""
    parser.add_argument(
        "--annotators", nargs="+", choices=KNOWN_ANNOTATORS,
        default=list(KNOWN_ANNOTATORS),
        help="Annotators to compare pairwise (default: all five).",
    )
    parser.add_argument(
        "--preann-model", default="gpt_oss_20b",
        help="Pre-annotation model used for the with_preann variants "
             "(default: %(default)s). Required for preann-effect analysis.",
    )


# --- Validation / normalisation ---------------------------------------------

def parse_run_ids(args: argparse.Namespace) -> list[str]:
    """Validate and return ``args.run_ids`` (or empty list).

    Each ID must match the ``runNN[-slug]`` form. Raises
    :class:`ValueError` on bad input.
    """
    raw = args.run_ids or []
    return [parse_run_id_to_path_segment(r) for r in raw]


def parse_organs(args: argparse.Namespace) -> list[int]:
    """Return validated organ indices for ``args.dataset``, or the dataset's
    full organ-index set if ``--organs`` was omitted."""
    if not args.organs:
        return list(all_organ_indices(args.dataset))
    return parse_organ_arg(args.organs, args.dataset)


def parse_cases(args: argparse.Namespace) -> set[str] | None:
    """Return a case-id allowlist set, or ``None`` for "all cases."""
    if not args.cases:
        return None
    out: set[str] = set()
    for entry in args.cases:
        if entry.startswith("@"):
            list_path = Path(entry[1:])
            if not list_path.is_file():
                raise FileNotFoundError(f"--cases list file missing: {list_path}")
            with list_path.open(encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        out.add(s)
        else:
            out.add(entry)
    return out


def require_model(args: argparse.Namespace) -> None:
    """For subcommands where ``--method`` ≠ rule_based, fail early if
    ``--model`` is missing."""
    if args.method != "rule_based" and not args.model:
        raise SystemExit(
            f"--model is required for method={args.method!r} "
            f"(rule_based has no model name)."
        )


__all__ = [
    "KNOWN_ANNOTATORS",
    "add_common_args",
    "add_model_args",
    "add_iaa_args",
    "parse_run_ids",
    "parse_organs",
    "parse_cases",
    "require_model",
]
