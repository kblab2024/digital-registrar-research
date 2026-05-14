"""
pipelines/pipeline_v2.py

End-to-end v2 inference pipeline with secondary-primary support.

Mirrors ``digital_registrar_research.pipeline.CancerPipeline`` but uses the
v2 signatures (``is_cancer_v2``, ``ReportJsonize_v2``) and dispatches the
organ-specific extractors PER ROLE — once for the primary cancer and once
more for the secondary primary cancer (when present).

The external output shape is a superset of v1: it carries the same
top-level eligibility / category fields plus three new ``secondary_*``
fields, and ``cancer_data`` becomes a role-keyed dict instead of a flat
merge.
"""
from __future__ import annotations

import json
import logging
import time

import dspy

# Star-imports mirror pipeline.py so dynamic class lookup via
# globals().get(name) against organmodels keeps working for both roles.
from ..models.breast import *  # noqa: F401, F403
from ..models.cervix import *  # noqa: F401, F403
from ..models.colon import *  # noqa: F401, F403
from ..models.common import *  # noqa: F401, F403
from ..models.common_v2 import ReportJsonize_v2, is_cancer_v2
from ..models.esophagus import *  # noqa: F401, F403
from ..models.liver import *  # noqa: F401, F403
from ..models.lung import *  # noqa: F401, F403
from ..models.modellist import organmodels
from ..models.pancreas import *  # noqa: F401, F403
from ..models.prostate import *  # noqa: F401, F403
from ..models.stomach import *  # noqa: F401, F403
from ..models.thyroid import *  # noqa: F401, F403
from ..pipeline import run_pipeline, setup_pipeline, timeit
from ..util.predictiondump import dump_prediction_plain


def _split_paragraphs(report: str | list[str]) -> list[str]:
    if isinstance(report, list):
        return [p.strip() for p in report if isinstance(p, str) and p.strip()]
    return [p.strip() for p in report.split('\n\n') if p.strip()]


class CancerPipelineV2(dspy.Module):
    """v2 pipeline: eligibility → role-keyed Jsonize → per-role organ extraction."""

    def __init__(self):
        super().__init__()
        self.analyzer_is_cancer = dspy.Predict(is_cancer_v2)
        self.jsonize = dspy.Predict(ReportJsonize_v2)

    def _run_organ_extractors(
        self,
        paragraphs: list[str],
        category: str,
        role_json: dict,
        role: str,
        fname: str,
        logger: logging.Logger,
    ) -> dict:
        role_output: dict = {}
        for items in organmodels.get(category, []):
            cls = globals().get(items)
            if cls is None:
                logger.error(f"[v2] Model class {items} not found (role={role}).")
                continue
            logger.info(
                f"[v2] Processing organ-specific model: {cls.__name__} at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} for {category} cancer "
                f"(role={role}) for {fname}"
            )
            organ_analyzer = dspy.Predict(cls)
            try:
                organ_response = organ_analyzer(report=paragraphs, report_jsonized=role_json)
                organ_data = dump_prediction_plain(organ_response)
                role_output.update(organ_data)
            except Exception as e:
                logger.error(f"[v2] Error processing {cls.__name__} (role={role}): {e}")
                continue
        return role_output

    def forward(
        self,
        report: str | list[str],
        logger: logging.Logger,
        fname: str = "",
    ) -> dict:
        print(f"[v2] Processing report: {fname}")
        logger.info(f"[v2] Processing report: {fname}")

        paragraphs = _split_paragraphs(report)
        context_response = self.analyzer_is_cancer(report=paragraphs)

        if not context_response.cancer_excision_report:
            logger.info("[v2] This is NOT a cancer excision report.")
            output_report = {
                "cancer_excision_report": False,
                "cancer_category": None,
                "cancer_category_others_description": None,
                "secondary_primary_present": False,
                "secondary_cancer_category": None,
                "secondary_cancer_category_others_description": None,
                "cancer_data": {},
            }
            print(json.dumps(output_report, indent=2, ensure_ascii=False))
            return output_report

        primary_category = context_response.cancer_category
        secondary_present = bool(context_response.secondary_primary_present)
        secondary_category = (
            context_response.secondary_cancer_category if secondary_present else None
        )

        output_report = {
            "cancer_excision_report": True,
            "cancer_category": primary_category,
            "cancer_category_others_description": context_response.cancer_category_others_description,
            "secondary_primary_present": secondary_present,
            "secondary_cancer_category": secondary_category,
            "secondary_cancer_category_others_description": (
                context_response.secondary_cancer_category_others_description
                if secondary_present else None
            ),
            "cancer_data": {},
        }

        logger.info("[v2] This is a cancer excision report.")
        if primary_category == 'others':
            logger.info(
                f"[v2] Primary cancer category is "
                f"{context_response.cancer_category_others_description}, "
                "not implemented."
            )
        elif primary_category:
            logger.info(f"[v2] Primary cancer category is {primary_category}.")

        if secondary_present:
            if secondary_category == 'others':
                logger.info(
                    f"[v2] Secondary primary category is "
                    f"{context_response.secondary_cancer_category_others_description}, "
                    "not implemented."
                )
            elif secondary_category:
                logger.info(f"[v2] Secondary primary category is {secondary_category}.")

        # Categories the Jsonize_v2 signature can route on (not 'others', not None).
        jsonize_primary = primary_category if primary_category in organmodels else None
        jsonize_secondary = secondary_category if secondary_category in organmodels else None

        try:
            json_response = self.jsonize(
                report=paragraphs,
                primary_category=jsonize_primary,
                secondary_cancer_category=jsonize_secondary,
            )
            role_jsons = json_response.output if isinstance(json_response.output, dict) else {}
        except Exception as e:
            logger.error(f"[v2] Error during ReportJsonize_v2: {e}")
            role_jsons = {}

        if jsonize_primary is not None:
            primary_json = role_jsons.get("primary", {}) if isinstance(role_jsons, dict) else {}
            output_report["cancer_data"]["primary"] = self._run_organ_extractors(
                paragraphs=paragraphs,
                category=jsonize_primary,
                role_json=primary_json if isinstance(primary_json, dict) else {},
                role="primary",
                fname=fname,
                logger=logger,
            )

        if jsonize_secondary is not None:
            secondary_json = (
                role_jsons.get("secondary_primary", {}) if isinstance(role_jsons, dict) else {}
            )
            output_report["cancer_data"]["secondary_primary"] = self._run_organ_extractors(
                paragraphs=paragraphs,
                category=jsonize_secondary,
                role_json=secondary_json if isinstance(secondary_json, dict) else {},
                role="secondary_primary",
                fname=fname,
                logger=logger,
            )

        return output_report


def run_cancer_pipeline_v2(report: str | list[str], fname: str = "") -> tuple[dict, str]:
    """Convenience runner mirroring ``run_cancer_pipeline`` for v2."""
    pipeline_v2 = CancerPipelineV2()
    response, timing = run_pipeline(pipeline_v2, report=report, fname=fname)
    return response, timing


if __name__ == "__main__":
    setup_pipeline("gpt")
    print("[v2] Pipeline is ready for processing pathology reports.")
