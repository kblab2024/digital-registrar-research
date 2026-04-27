"""
B2 — DSPy with ``str`` outputs + post-hoc parser.

Strips every ``Literal[...]`` and numeric output annotation on the
monolithic signature down to ``str | None``, runs the same pipeline
through DSPy, then projects the resulting strings back into the typed
field space using :mod:`extractors.post_hoc_parser`.

Canonical layout:
    --folder dummy --dataset tcga --model gptoss [--run runNN]
"""
from __future__ import annotations

import argparse
import logging

import dspy

from ...models.common import ReportJsonize, is_cancer
from ...models.modellist import organmodels
from ...util.predictiondump import dump_prediction_plain
from ..extractors.post_hoc_parser import parse_cancer_data
from ..signatures.str_outputs import get_str_signature
from . import _base

CELL_ID = "str_outputs"


class StrOutputsPipeline(dspy.Module):
    def __init__(self, skip_jsonize: bool = False):
        super().__init__()
        self.skip_jsonize = skip_jsonize
        self.analyzer_is_cancer = dspy.Predict(is_cancer)
        self.jsonize = dspy.Predict(ReportJsonize)
        self._organ_predictors: dict[str, dspy.Predict] = {}

    def _get_organ_predictor(self, organ: str) -> dspy.Predict:
        if organ not in self._organ_predictors:
            sig = get_str_signature(organ)
            self._organ_predictors[organ] = dspy.Predict(sig)
        return self._organ_predictors[organ]

    def forward(self, report: str, logger: logging.Logger,
                fname: str = "") -> dict:
        logger.debug("[str_outputs] %s", fname)
        paragraphs = [p.strip() for p in report.split("\n\n") if p.strip()]
        context_response = self.analyzer_is_cancer(report=paragraphs)
        if not context_response.cancer_excision_report:
            return {"cancer_excision_report": False,
                    "cancer_category": None, "cancer_data": {}}
        organ = context_response.cancer_category
        out: dict = {
            "cancer_excision_report": True,
            "cancer_category": organ,
            "cancer_category_others_description":
                context_response.cancer_category_others_description,
            "cancer_data": {},
        }
        if organ not in organmodels:
            return out
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
            raw_data = dump_prediction_plain(organ_response)
            parsed, parse_errs = parse_cancer_data(raw_data, organ)
            out["cancer_data"] = parsed
            if parse_errs:
                out["_b2_parse_errors"] = parse_errs
            out["_b2_raw"] = raw_data
        except Exception as e:
            logger.error("str_outputs %s failed for %s: %s", organ, fname, e)
            out["_error"] = str(e)
        return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _base.add_canonical_args(ap)
    ap.add_argument("--skip-jsonize", action="store_true")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    paths, organs, run_name = _base.resolve_run_paths(args, CELL_ID)
    overrides = _base.load_decoding_overrides(args.model)
    lm_kwargs = _base.setup_dspy_lm(args.model, overrides=overrides)

    logger = _base.make_logger("str_outputs", paths.run_dir(run_name),
                               args.verbose)
    logger.info("cell=%s model=%s slug=%s run=%s organs=%s",
                CELL_ID, args.model, paths.model_slug, run_name,
                [o[0] for o in organs])

    pipe = StrOutputsPipeline(skip_jsonize=args.skip_jsonize)

    def _predict(report_text: str, organ_n: str, case_id: str) -> dict:
        return pipe(report=report_text, logger=logger, fname=case_id)

    summary = _base.run_loop(
        paths, organs, run_name, model_alias=args.model,
        predict=_predict, args=args, logger=logger,
        decoding=lm_kwargs,
        manifest_extra={"skip_jsonize": args.skip_jsonize,
                        "discipline": "str_outputs"},
        extra_meta={"skip_jsonize": args.skip_jsonize,
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
