"""
A3 — DSPy + monolithic, no `ReportJsonize`.

Drops the intermediate `ReportJsonize` step relative to A2 (`dspy_monolithic`).
A separate cell so its predictions land under

    {root}/results/ablations/{dataset}/dspy_monolithic_no_jsonize/{model_slug}/{run}/...

instead of colliding with A2's output. The actual pipeline is
``MonolithicPipeline(skip_jsonize=True)`` from the A2 runner — no logic
duplicated.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss [--run runNN]
"""
from __future__ import annotations

import argparse

from . import _base
from .dspy_monolithic import MonolithicPipeline

CELL_ID = "dspy_monolithic_no_jsonize"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _base.add_canonical_args(ap)
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    paths, organs, run_name = _base.resolve_run_paths(args, CELL_ID)
    overrides = _base.load_decoding_overrides(args.model)
    lm_kwargs = _base.setup_dspy_lm(args.model, overrides=overrides)

    logger = _base.make_logger("dspy_monolithic_no_jsonize",
                               paths.run_dir(run_name), args.verbose)
    logger.info("cell=%s model=%s slug=%s run=%s organs=%s",
                CELL_ID, args.model, paths.model_slug, run_name,
                [o[0] for o in organs])

    pipe = MonolithicPipeline(skip_jsonize=True)

    def _predict(report_text: str, organ: str, case_id: str) -> dict:
        return pipe(report=report_text, logger=logger, fname=case_id)

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding=lm_kwargs,
        manifest_extra={"skip_jsonize": True},
        extra_meta={"skip_jsonize": True,
                    "dspy_lm_kwargs": lm_kwargs},
    )

    print(f"OK={summary.n_ok} ERR={summary.n_pipeline_error} "
          f"CACHED={summary.n_cached} N={summary.n_cases} "
          f"NOT_CANCER={summary.n_skipped_not_cancer} "
          f"UNKNOWN_ORGAN={summary.n_skipped_unknown_organ} "
          f"DOWNSTREAM={summary.n_downstream_called} "
          f"WALL={summary.wall_time_s:.1f}s")
    print(f"run dir: {paths.run_dir(run_name)}")

    if summary.n_pipeline_error > 0 and not args.tolerate_errors:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    import sys
    sys.exit(main())
