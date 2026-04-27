"""Auto-loader for per-model decoding-override YAMLs.

The consolidated runners ``scripts/pipeline/run_dspy_ollama_{single,smoke}.py``
read an optional YAML of the form ``configs/dspy_ollama_{model}.yaml`` so
decoding params (``temperature``, ``top_p``, ``num_ctx``, ...) and smoke
defaults (``smoke.n``, ``smoke.seed``) can be tuned without editing code. If
the config file is absent, or every override is null, the per-model defaults
in ``models.common.MODEL_PROFILES`` are used unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
LOCAL_CONFIGS_DIR = CONFIGS_DIR / "local"

DECODING_KEYS = (
    "temperature", "top_p", "top_k", "num_ctx", "max_tokens",
    "repeat_penalty", "keep_alive", "cache", "seed",
)


def model_config_path(model_alias: str) -> Path:
    """Return the config-file path for a model alias, preferring local/ override.

    File-level resolution (used when callers want a single canonical path,
    e.g. for logging). Per-key merging is handled in ``load_model_config``."""
    local = LOCAL_CONFIGS_DIR / f"dspy_ollama_{model_alias}.yaml"
    if local.is_file():
        return local
    return CONFIGS_DIR / f"dspy_ollama_{model_alias}.yaml"


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *over* on top of *base*. Nested dicts merge key-by-key;
    everything else (lists, scalars) is replaced wholesale by *over*."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import yaml
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_config(model_alias: str) -> dict[str, Any]:
    """Load the config for *model_alias* with per-key local-over-shared merge.

    Loads ``configs/dspy_ollama_{alias}.yaml`` as the base and overlays
    ``configs/local/dspy_ollama_{alias}.yaml`` on top. Each key is resolved
    independently, so a local file can hold a partial override (e.g. just
    ``decoding.num_ctx``) without redeclaring every other knob. Returns
    ``{}`` if neither file exists."""
    shared = _load_yaml(CONFIGS_DIR / f"dspy_ollama_{model_alias}.yaml")
    local = _load_yaml(LOCAL_CONFIGS_DIR / f"dspy_ollama_{model_alias}.yaml")
    return _deep_merge(shared, local)


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
