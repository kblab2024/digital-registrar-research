"""
pipelines/eligibility_only.py

Category-only testing pipeline for the v2 eligibility signature. Runs ONLY
``is_cancer_v2`` and returns its raw outputs as a plain dict — no
``ReportJsonize_v2``, no organ-specific extractors. Use this to validate
the v2 eligibility-step shape and prompt behavior (including the
secondary-primary detection rules) on real reports before paying the cost
of full extraction.
"""
from __future__ import annotations

import logging

import dspy

from ..models.common_v2 import is_cancer_v2
from ..util.predictiondump import dump_prediction_plain


class EligibilityOnlyPipelineV2(dspy.Module):
    """Minimal v2 pipeline: only the eligibility / category signature."""

    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(is_cancer_v2)

    def forward(
        self,
        report: str | list[str],
        logger: logging.Logger,
        fname: str = "",
    ) -> dict:
        logger.info(f"[v2-eligibility-only] processing: {fname}")
        if isinstance(report, list):
            paragraphs = [p.strip() for p in report if isinstance(p, str) and p.strip()]
        else:
            paragraphs = [p.strip() for p in report.split('\n\n') if p.strip()]

        prediction = self.analyzer(report=paragraphs)
        return dump_prediction_plain(prediction)
