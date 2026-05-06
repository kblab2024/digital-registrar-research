#!/usr/bin/env python3
"""Compare rule_based vs ClinicalBERT vs one-or-more LLMs side-by-side.

Runs ``scripts.eval.cli cascade`` for each method (rule, bert, and each
LLM passed via ``--llm-models``), then joins the outputs with
``scripts.eval.cli compare`` into chapter1-5 comparison folders +
pairwise paired-bootstrap deltas across every method pair.

The comparison is naturally **scope-restricted** to the chapters that
ClinicalBERT and the rule-based extractor can produce (eligibility,
organ classification, scalar fields). For the full cascade including
the nested margins / lymph-node / biomarker chapters, run a
``compare``-only sweep across the LLM cascades directly.

Usage
-----
    # Original 3-way (rule + bert + one LLM)
    python scripts/baselines/eval_rule_bert_llm.py \\
        --folder workspace --datasets tcga \\
        --bert-head merged \\
        --llm-models gpt_oss_20b \\
        --out workspace/results/eval/rule_bert_llm

    # 4-way for the rebuttal: rule + bert + local LLM + hosted (OpenAI) LLM
    python scripts/baselines/eval_rule_bert_llm.py \\
        --folder workspace --datasets tcga \\
        --bert-head merged \\
        --llm-models gpt_oss_20b gpt_5_4_mini \\
        --out workspace/results/eval/rule_bert_locallm_apillm

Prerequisites: predictions for every method must exist under
``{folder}/results/predictions/{dataset}/...``. ``--llm-runs`` (if
given) applies to every LLM; the default (auto-discover) works for
K-seed sweeps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from baselines._eval_pipeline import (  # noqa: E402
    MethodSpec, add_common_args, run_pipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--bert-head", default="merged",
                    choices=("cls", "qa", "merged"),
                    help="Which ClinicalBERT head to compare (default: merged).")
    ap.add_argument("--llm-models", required=True, nargs="+",
                    help="One or more LLM model slugs to include, e.g. "
                         "'gpt_oss_20b' for a 3-way compare or "
                         "'gpt_oss_20b gpt_5_4_mini' for a 4-way "
                         "(rule + bert + local LLM + hosted LLM).")
    ap.add_argument("--llm-runs", nargs="*", default=None,
                    help="LLM run IDs (default: auto-discover). Applied to "
                         "every model in --llm-models.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs: list[MethodSpec] = [
        MethodSpec(label="rule_based", method="rule_based", model=None),
        MethodSpec(label=f"bert_{args.bert_head}",
                   method="clinicalbert", model=args.bert_head),
    ]
    for llm_slug in args.llm_models:
        specs.append(MethodSpec(
            label=f"llm_{llm_slug}",
            method="llm", model=llm_slug,
            run_ids=args.llm_runs,
        ))
    return run_pipeline(specs, args)


if __name__ == "__main__":
    sys.exit(main())
