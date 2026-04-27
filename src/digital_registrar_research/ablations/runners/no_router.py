"""
A4 — Monolithic DSPy without the ``is_cancer`` router.

Same as Cell B (monolithic) but skips the routing call. The router
decides BOTH whether-to-extract AND which organ; the cleanest way to
remove it is to use the gold organ derived from the report's organ
subdirectory (the canonical layout encodes organ in the path:
``data/{dataset}/reports/{organ_n}/{case_id}.txt``).

The numeric organ index is mapped to its canonical name via
:data:`scope.IMPLEMENTED_ORGANS`. This is an upper-bound estimate of
router contribution: the cell sees the right organ for free.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss [--run runNN]
"""
from __future__ import annotations

import argparse
import logging

import dspy

from ...benchmarks.eval.scope import IMPLEMENTED_ORGANS
from ...util.predictiondump import dump_prediction_plain
from ..signatures.monolithic import get_monolithic_signature
from . import _base

CELL_ID = "no_router"


def _organ_from_index(organ_n: str) -> str | None:
    """Map a numeric organ subdir name (``"1"``, ``"2"``, ...) to the
    organ key used by the per-organ signatures."""
    try:
        idx = int(organ_n)
    except ValueError:
        return None
    if 1 <= idx <= len(IMPLEMENTED_ORGANS):
        return IMPLEMENTED_ORGANS[idx - 1]
    return None


class NoRouterPipeline(dspy.Module):
    """Monolithic pipeline minus the is_cancer router stage."""

    def __init__(self, skip_jsonize: bool = True):
        super().__init__()
        self.skip_jsonize = skip_jsonize
        self._organ_predictors: dict[str, dspy.Predict] = {}
        if not skip_jsonize:
            from ...models.common import ReportJsonize
            self.jsonize = dspy.Predict(ReportJsonize)

    def _get_organ_predictor(self, organ: str) -> dspy.Predict:
        if organ not in self._organ_predictors:
            sig = get_monolithic_signature(organ)
            self._organ_predictors[organ] = dspy.Predict(sig)
        return self._organ_predictors[organ]

    def forward(self, report: str, organ: str, logger: logging.Logger,
                fname: str = "") -> dict:
        logger.debug("[no_router] %s organ=%s", fname, organ)
        paragraphs = [p.strip() for p in report.split("\n\n") if p.strip()]
        out: dict = {
            "cancer_excision_report": True,
            "cancer_category": organ,
            "cancer_data": {},
            "_router_skipped": True,
        }
        report_jsonized: dict = {}
        if not self.skip_jsonize:
            try:
                jr = self.jsonize(report=paragraphs, cancer_category=organ)
                report_jsonized = jr.output or {}
            except Exception as e:
                logger.warning("jsonize failed for %s: %s", fname, e)
        try:
            predictor = self._get_organ_predictor(organ)
            organ_response = predictor(
                report=paragraphs, report_jsonized=report_jsonized)
            out["cancer_data"] = dump_prediction_plain(organ_response)
        except Exception as e:
            logger.error("no_router %s failed for %s: %s", organ, fname, e)
            out["_error"] = str(e)
        return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _base.add_canonical_args(ap)
    ap.add_argument("--include-jsonize", action="store_true",
                    help="keep the upstream ReportJsonize step (default: skip)")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    paths, organs, run_name = _base.resolve_run_paths(args, CELL_ID)
    overrides = _base.load_decoding_overrides(args.model)
    lm_kwargs = _base.setup_dspy_lm(args.model, overrides=overrides)

    logger = _base.make_logger("no_router", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s run=%s organs=%s",
                CELL_ID, args.model, paths.model_slug, run_name,
                [o[0] for o in organs])

    skip_jsonize = not args.include_jsonize
    pipe = NoRouterPipeline(skip_jsonize=skip_jsonize)

    def _predict(report_text: str, organ_n: str, case_id: str) -> dict:
        organ = _organ_from_index(organ_n)
        if organ is None:
            return {"cancer_excision_report": True, "cancer_category": None,
                    "cancer_data": {}, "_router_skipped": True,
                    "_error": f"cannot infer organ from organ_n={organ_n!r}"}
        return pipe(report=report_text, organ=organ,
                    logger=logger, fname=case_id)

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding=lm_kwargs,
        manifest_extra={"router_skipped": True, "skip_jsonize": skip_jsonize},
        extra_meta={"router_skipped": True, "skip_jsonize": skip_jsonize,
                    "dspy_lm_kwargs": lm_kwargs},
    )

    print(f"OK={summary.n_ok} ERR={summary.n_pipeline_error} "
          f"CACHED={summary.n_cached} N={summary.n_cases} "
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
