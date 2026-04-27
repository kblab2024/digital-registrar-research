"""DSPy signatures with `str` outputs — the B2 ablation.

The default monolithic signature uses ``Literal[...]`` enums and typed
numerics so DSPy's adapters can enforce schema conformance at generation
time. B2 strips every output annotation down to ``str | None`` to
measure how much accuracy is lost when type discipline is removed.

Implementation: clone :func:`get_monolithic_signature(organ)`, walk its
``__annotations__``, and rewrite each non-input annotation to
``str | None`` while keeping the ``dspy.OutputField`` descriptor (so the
field's natural-language description still flows into the prompt).
"""
from __future__ import annotations

from functools import cache

import dspy

from .monolithic import (
    INPUT_FIELD_NAMES,
    MONOLITHIC_DOCSTRING,
    get_monolithic_signature,
)

STR_DOCSTRING_SUFFIX = (
    " For each field, return a free-text answer using the values found "
    "in the report — do not pick from a fixed set; we will normalise "
    "your answers afterwards."
)


@cache
def get_str_signature(organ: str) -> type[dspy.Signature]:
    """Build a DSPy signature for ``organ`` with every output typed as ``str``."""
    base = get_monolithic_signature(organ)

    merged_annotations: dict[str, object] = {}
    merged_attrs: dict[str, object] = {}

    for name, type_hint in getattr(base, "__annotations__", {}).items():
        descriptor = base.__dict__.get(name)
        if descriptor is None:
            continue
        if name in INPUT_FIELD_NAMES:
            merged_annotations[name] = type_hint
            merged_attrs[name] = descriptor
            continue
        # Output field — coerce to str | None.
        merged_annotations[name] = str | None
        merged_attrs[name] = descriptor

    merged_attrs["__annotations__"] = merged_annotations
    merged_attrs["__doc__"] = (
        MONOLITHIC_DOCSTRING.format(organ=organ) + STR_DOCSTRING_SUFFIX
    )

    cls_name = f"{organ.title()}CancerStrOutputs"
    sig_cls = type(cls_name, (dspy.Signature,), merged_attrs)
    return sig_cls


__all__ = ["get_str_signature"]
