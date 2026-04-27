"""Subcommand dispatcher for evaluation scripts.

Usage:
    python -m scripts.eval.cli <subcommand> [options]

Subcommands:
    non_nested   — accuracy + missingness for scalar (non-nested) fields
    nested       — bipartite F1 + missingness for nested-list fields
                   (lymph nodes, margins, biomarkers)
    iaa          — inter-annotator agreement + preann effect
    completeness — detailed missingness analysis across methods
    diagnostics  — source-of-error decomposition, difficulty tiers, worst cases
    cross_dataset — per-field Δ between datasets + distribution shift
    headline     — joint forest-plot CSV combining IAA + accuracy

All subcommands share a common argument schema (see _common/args.py).
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Callable

logger = logging.getLogger("scripts.eval.cli")

# Subcommand registry. Each entry maps name → (help_text, builder).
# Builders return (parser, main_callable). Lazy-imported so the CLI can
# load even if a subcommand has missing optional dependencies.

SubcommandBuilder = Callable[[argparse._SubParsersAction], None]


def _register_non_nested(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.non_nested.run_non_nested import register
    register(sub)


def _register_nested(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.nested.run_nested import register
    register(sub)


def _register_iaa(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.iaa.run_iaa import register
    register(sub)


def _register_completeness(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.completeness.run_completeness import register
    register(sub)


def _register_diagnostics(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.diagnostics.run_diagnostics import register
    register(sub)


def _register_cross_dataset(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.cross_dataset.run_cross_dataset import register
    register(sub)


def _register_headline(sub: argparse._SubParsersAction) -> None:
    from scripts.eval.joint.headline_forest import register
    register(sub)


REGISTRARS: dict[str, SubcommandBuilder] = {
    "non_nested": _register_non_nested,
    "nested": _register_nested,
    "iaa": _register_iaa,
    "completeness": _register_completeness,
    "diagnostics": _register_diagnostics,
    "cross_dataset": _register_cross_dataset,
    "headline": _register_headline,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.eval.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    for name, register in REGISTRARS.items():
        try:
            register(sub)
        except ImportError as e:
            # Subcommand has missing dependency — register a stub so the
            # CLI lists it but errors usefully at invocation.
            stub = sub.add_parser(name, help=f"(unavailable: {e})")
            stub.set_defaults(_unavailable=str(e))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "_unavailable", None):
        logger.error(
            "subcommand %r is unavailable: %s",
            args.subcommand, args._unavailable,
        )
        return 2
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.error(f"subcommand {args.subcommand!r} did not register a handler")
    return int(handler(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
