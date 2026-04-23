"""
GPT-4 baseline — DSPy model swap.

Reuses the parent project's `CancerPipeline` verbatim; only swaps the
LM backend from local Ollama gpt-oss:20b to openai/gpt-4-turbo. This
isolates the contribution of model capacity from pipeline design: the
signatures, prompts, and post-processing are all held constant.

Requires:
    pip install dspy-ai openai
    set OPENAI_API_KEY=sk-...

Usage:
    python baselines/gpt4_dspy.py --split test --out ../results/gpt4_dspy
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import dspy

from ...paths import SPLITS_JSON
from ...pipeline import CancerPipeline

DEFAULT_MODEL = "openai/gpt-4-turbo"  # swap to "openai/gpt-4o" if preferred
SPLITS_PATH = SPLITS_JSON


def setup_gpt4(model_id: str = DEFAULT_MODEL) -> CancerPipeline:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Provision the key before running.")
    lm = dspy.LM(
        model=model_id,
        api_key=api_key,
        max_tokens=4000,
        temperature=0.0,
    )
    dspy.configure(lm=lm)
    return CancerPipeline()


def run_on_split(split_name: str, out_dir: Path, limit: int | None = None,
                 model_id: str = DEFAULT_MODEL) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with SPLITS_PATH.open(encoding="utf-8") as f:
        split = json.load(f)
    cases = split[split_name]
    if limit is not None:
        cases = cases[:limit]

    pipe = setup_gpt4(model_id)
    logger = logging.getLogger("gpt4_dspy")
    logger.setLevel(logging.INFO)

    cost_ledger = {"model": model_id, "runs": []}
    for case in cases:
        report = Path(case["report_path"]).read_text(encoding="utf-8")
        t0 = time.perf_counter()
        try:
            result = pipe(report=report, logger=logger, fname=case["id"])
        except Exception as e:
            logger.error(f"{case['id']}: {e}")
            result = {"error": str(e)}
        elapsed = time.perf_counter() - t0

        with (out_dir / f"{case['id']}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        cost_ledger["runs"].append({"id": case["id"], "elapsed_s": elapsed})
        print(f"  [{case['id']}] {elapsed:.1f}s")

    with (out_dir / "_cost_ledger.json").open("w", encoding="utf-8") as f:
        json.dump(cost_ledger, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", default="../results/gpt4_dspy",
                    help="output directory (relative to this file)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    out_dir = (Path(__file__).parent / args.out).resolve()
    run_on_split(args.split, out_dir, args.limit, args.model)


if __name__ == "__main__":
    main()
