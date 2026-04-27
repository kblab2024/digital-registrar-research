#!/usr/bin/env python3
"""Cell-C wrapper — raw OpenAI-compatible JSON mode (no DSPy).

The whole pipeline is replaced with two raw chat-completion calls per
case (classify, then extract) using ``response_format={"type":
"json_object"}``. Schema is inlined as text in the system prompt; no
DSPy, no Pydantic-typed signatures at the LM boundary.

Backed by ``digital_registrar_research.ablations.runners.raw_json``.

Usage
-----
    # Cell C × gpt-oss:20b (local Ollama)
    python scripts/ablations/run_cell_c.py --model gpt-oss:20b \\
        --out workspace/results/ablations/raw_json_gpt-oss

    # Cell C × gpt-4-turbo
    OPENAI_API_KEY=... python scripts/ablations/run_cell_c.py \\
        --model gpt-4-turbo \\
        --out workspace/results/ablations/raw_json_gpt4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402, F401

from digital_registrar_research.ablations.runners.raw_json import (  # noqa: E402
    run as run_cell_c,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help="model id (e.g. 'gpt-4-turbo' or 'gpt-oss:20b')")
    ap.add_argument("--api-base", default=None,
                    help="override API base; auto-set to local Ollama "
                         "(http://localhost:11434/v1) for gpt-oss/gemma/"
                         "qwen/phi/llama models")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_cell_c(args)


if __name__ == "__main__":
    main()
