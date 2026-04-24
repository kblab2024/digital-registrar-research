"""Auto-loader for per-model decoding-override YAMLs.

The consolidated runners ``scripts/run_dspy_ollama_{single,smoke}.py`` read an
optional YAML of the form ``configs/dspy_ollama_{model}.yaml`` so decoding
params (``temperature``, ``top_p``, ``num_ctx``, ...) and smoke defaults
(``smoke.n``, ``smoke.seed``) can be tuned without editing code. If the
config file is absent, or every override is null, the per-model defaults in
``models.common.MODEL_PROFILES`` are used unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

DECODING_KEYS = (
    "temperature", "top_p", "top_k", "num_ctx", "max_tokens",
    "repeat_penalty", "keep_alive", "cache", "seed",
)


def model_config_path(model_alias: str) -> Path:
    """Return the conventional config-file path for a model alias."""
    return CONFIGS_DIR / f"dspy_ollama_{model_alias}.yaml"


def load_model_config(model_alias: str) -> dict[str, Any]:
    """Load ``configs/dspy_ollama_{model_alias}.yaml``, or return ``{}`` if
    it does not exist. The config is optional; missing = fall back to
    MODEL_PROFILES in models.common."""
    cfg_path = model_config_path(model_alias)
    if not cfg_path.is_file():
        return {}
    import yaml
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def split_decoding_overrides(decoding_section: dict[str, Any] | None) -> dict[str, Any]:
    """Filter a config ``decoding:`` block down to the non-null overrides.

    Special value for ``seed``: the string ``"random"`` (or ``"auto"``) is
    replaced with a freshly drawn 31-bit integer. The chosen seed is still
    baked into the LM kwargs, so it gets logged into ``_run_meta.json`` and
    the model-level ``_manifest.yaml`` — runs remain individually
    reproducible, only the starting point is non-deterministic.
    """
    if not decoding_section:
        return {}
    out = {k: decoding_section[k] for k in DECODING_KEYS
           if k in decoding_section and decoding_section[k] is not None}
    seed = out.get("seed")
    if isinstance(seed, str):
        token = seed.strip().lower()
        if token in ("random", "auto"):
            import secrets
            out["seed"] = secrets.randbelow(2**31)
        else:
            raise ValueError(
                f"decoding.seed must be an int, null, or 'random'; got {seed!r}"
            )
    return out


def resolve_folder(raw: str | Path) -> Path:
    """Resolve a ``--folder`` CLI arg against the repo root when relative.

    "dummy" and "workspace" are plain relative paths — they resolve to
    ``<repo_root>/dummy`` and ``<repo_root>/workspace`` respectively. Any
    absolute path is used verbatim; any other relative path is resolved
    against the repo root (not the caller's cwd) for reproducibility."""
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()
