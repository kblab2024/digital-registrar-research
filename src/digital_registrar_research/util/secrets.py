"""Secret loading for hosted-LLM runners.

The OpenAI API key lives at ``<repo_root>/researchtemp/env/.env``. The
``researchtemp/`` tree is fully git-ignored (only ``.gitkeep`` is
tracked), so the file is never committed or shipped in a tarball, while
keeping the secret co-located with the code that needs it.

Resolution order for ``OPENAI_API_KEY`` (first non-empty wins):
  1. ``OPENAI_API_KEY`` already in process env (CI / one-shot override).
  2. ``OPENAI_API_KEY=...`` line in ``<repo_root>/researchtemp/env/.env``.

The loader never prints the key. Callers should also avoid logging it.

This is a deliberately tiny parser (no python-dotenv dependency): we
only support ``KEY=VALUE`` lines, optional surrounding whitespace, and
optional ``"`` / ``'`` quoting around the value. Comments (``#``) and
blank lines are skipped.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..paths import REPO_ROOT

API_KEY_FILE: Path = REPO_ROOT / "researchtemp" / "env" / ".env"


def _parse_env_file(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (len(value) >= 2 and value[0] == value[-1]
                and value[0] in ("'", '"')):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_openai_key() -> str:
    """Return the OpenAI API key, reading from env or the in-repo secrets file.

    Caches the value in ``os.environ["OPENAI_API_KEY"]`` after first read so
    repeated calls within one process do not re-parse the file. Raises
    ``RuntimeError`` with an actionable message (no key contents) if the key
    cannot be found.
    """
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    if not API_KEY_FILE.is_file():
        raise RuntimeError(
            "OPENAI_API_KEY is not set and no secrets file was found at "
            f"{API_KEY_FILE}. Create it with one line:\n"
            "    OPENAI_API_KEY=sk-...\n"
            "or export OPENAI_API_KEY in your shell before running.")

    parsed = _parse_env_file(API_KEY_FILE.read_text(encoding="utf-8"))
    key = parsed.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            f"OPENAI_API_KEY not found in {API_KEY_FILE}. The file exists "
            "but does not contain a non-empty 'OPENAI_API_KEY=...' line.")

    os.environ["OPENAI_API_KEY"] = key
    return key


__all__ = ["load_openai_key", "API_KEY_FILE"]
