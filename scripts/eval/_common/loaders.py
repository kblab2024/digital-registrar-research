"""JSON loading + parse-error / runtime-error detection for prediction files.

Three load paths:

- :func:`load_json` returns the parsed dict or raises ``ParseError`` if
  the file is missing or malformed. Used for gold annotations where any
  failure is a hard error.
- :func:`load_prediction` is *defensive*: it returns a structured
  outcome (``ok`` + dict, or ``parse_error`` + error_mode + raw text) so
  the eval pipeline can record the failure mode rather than crashing.
- :func:`load_log_entry` reads optional ``_log.jsonl`` next to a run
  directory to enrich parse-error decomposition with timeout / refusal /
  schema-invalid signals.

Error-mode taxonomy:
    json_parse       — raised :class:`json.JSONDecodeError`
    schema_invalid   — file parsed but failed pydantic validation
    timeout          — log entry indicates the model didn't return
    refusal          — log entry indicates the model refused
    file_missing     — the file isn't on disk
    other            — unrecognised failure
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ErrorMode = Literal[
    "json_parse",
    "schema_invalid",
    "timeout",
    "refusal",
    "file_missing",
    "other",
]


class ParseError(Exception):
    """Hard failure loading an annotation/prediction file."""


@dataclass(frozen=True)
class LoadOutcome:
    """Result of a defensive load attempt."""

    ok: bool
    data: dict[str, Any] | None
    error_mode: ErrorMode | None
    error_message: str | None
    path: Path

    def __bool__(self) -> bool:
        return self.ok


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return the parsed dict.

    Raises :class:`ParseError` on missing or malformed files. Use this
    for *gold* loads where missingness is a fatal error in the dataset.
    """
    if not path.exists():
        raise ParseError(f"file missing: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ParseError(f"json parse failed for {path}: {e}") from e
    if not isinstance(data, dict):
        raise ParseError(f"expected JSON object at {path}, got {type(data).__name__}")
    return data


def load_prediction(path: Path) -> LoadOutcome:
    """Defensively load a prediction file.

    Returns an :class:`LoadOutcome` with ``ok=False`` and an
    :data:`ErrorMode` rather than raising on parse failure. Honours an
    explicit ``_parse_error`` marker inside the JSON (some upstream
    runners write a sentinel object on failure).
    """
    if not path.exists():
        return LoadOutcome(
            ok=False, data=None, error_mode="file_missing",
            error_message=f"file missing: {path}", path=path,
        )
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return LoadOutcome(
            ok=False, data=None, error_mode="json_parse",
            error_message=str(e), path=path,
        )
    except OSError as e:
        return LoadOutcome(
            ok=False, data=None, error_mode="other",
            error_message=str(e), path=path,
        )
    if not isinstance(data, dict):
        return LoadOutcome(
            ok=False, data=None, error_mode="schema_invalid",
            error_message=f"expected dict, got {type(data).__name__}",
            path=path,
        )
    if data.get("_parse_error"):
        # Upstream runner explicitly flagged this case as failed.
        return LoadOutcome(
            ok=False, data=data, error_mode="schema_invalid",
            error_message=str(data.get("_parse_error_message") or "parse_error sentinel"),
            path=path,
        )
    return LoadOutcome(
        ok=True, data=data, error_mode=None,
        error_message=None, path=path,
    )


def extract_meta(annotation: dict[str, Any]) -> dict[str, Any]:
    """Pull the ``_meta`` block from a human annotation, or return ``{}``.

    Human annotations carry ``_meta = {annotator, mode, annotated_at}``
    for IAA/preann analysis. Gold and model predictions don't have it.
    """
    meta = annotation.get("_meta")
    return dict(meta) if isinstance(meta, dict) else {}


def index_runs(run_logs: dict[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """Read ``_log.jsonl`` files and return ``{run_id: {case_id: log_entry}}``.

    Optional — missing or malformed log files are silently skipped. The
    returned mapping is consulted by the outcome classifier to enrich
    parse-error decomposition with ``timeout``/``refusal`` signals.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for run_id, run_dir in run_logs.items():
        log_path = Path(run_dir) / "_log.jsonl"
        if not log_path.is_file():
            out[run_id] = {}
            continue
        entries: dict[str, dict[str, Any]] = {}
        try:
            with log_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cid = entry.get("case_id") or entry.get("case")
                    if cid:
                        entries[cid] = entry
        except OSError:
            pass
        out[run_id] = entries
    return out


def classify_log_error(log_entry: dict[str, Any] | None) -> ErrorMode | None:
    """Map a log entry to a refined error mode.

    Returns ``None`` if the entry doesn't indicate a failure — caller
    falls back to ``json_parse`` / ``schema_invalid`` from the file
    inspection. Heuristic field names match the conventions in
    ``scripts/run_*.py`` runners.
    """
    if not log_entry:
        return None
    if log_entry.get("timeout"):
        return "timeout"
    err = log_entry.get("error") or log_entry.get("error_type")
    if isinstance(err, str):
        err_l = err.lower()
        if "timeout" in err_l:
            return "timeout"
        if "refus" in err_l or "policy" in err_l:
            return "refusal"
        if "schema" in err_l or "validation" in err_l:
            return "schema_invalid"
        if "json" in err_l or "parse" in err_l:
            return "json_parse"
        return "other"
    return None


__all__ = [
    "ParseError",
    "LoadOutcome",
    "ErrorMode",
    "load_json",
    "load_prediction",
    "extract_meta",
    "index_runs",
    "classify_log_error",
]
