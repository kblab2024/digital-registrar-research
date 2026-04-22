"""Parse JSON Schema 2020-12 files into flat Python data structures for Streamlit rendering."""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import SCHEMAS_DATA as SCHEMAS_DIR

DISPLAY_NAMES = {
    "Nonnested": "General",
    "Staging": "Staging",
    "Margins": "Margins",
    "LN": "Lymph Nodes",
    "Biomarkers": "Biomarkers",
    "Grading": "Grading",
    "Othernested": "Patterns",
    "DCIS": "DCIS",
}

CANCER_TO_FILE = {
    "stomach": "stomach.json",
    "colorectal": "colorectal.json",
    "breast": "breast.json",
    "esophagus": "esophagus.json",
    "lung": "lung.json",
    "prostate": "prostate.json",
    "thyroid": "thyroid.json",
    "pancreas": "pancreas.json",
    "cervix": "cervix.json",
    "liver": "liver.json",
    "bladder": "bladder.json",
    "others": None,
}

CANCER_CATEGORIES = [
    None,
    "stomach",
    "colorectal",
    "breast",
    "esophagus",
    "lung",
    "prostate",
    "thyroid",
    "pancreas",
    "cervix",
    "liver",
    "bladder",
    "others",
]


@dataclass
class FieldSpec:
    name: str
    title: str
    description: str
    field_type: str  # enum|int_enum|bool|int|string|array_of_objects|array_of_strings_enum
    enum_values: list = field(default_factory=list)
    item_fields: list = field(default_factory=list)  # list[FieldSpec] for array_of_objects
    required: bool = False


@dataclass
class SectionSpec:
    name: str
    display_name: str
    flat_fields: list  # list[FieldSpec]
    array_field_name: str | None = None
    array_item_fields: list = field(default_factory=list)  # list[FieldSpec]
    array_item_enum_values: list = field(default_factory=list)  # for array_of_strings_enum
    array_field_type: str | None = None  # array_of_objects | array_of_strings_enum


def _detect_field_type(prop: dict) -> tuple[str, list, list]:
    """Returns (field_type, enum_values, item_fields)."""
    ao = prop.get("anyOf")
    if ao:
        non_null = [s for s in ao if s.get("type") != "null"]
        if not non_null:
            return "string", [], []
        core = non_null[0]
    elif "type" in prop:
        # Non-nullable field (e.g. required boolean in common.json)
        core = prop
    else:
        return "string", [], []

    if "enum" in core:
        if core.get("type") == "integer":
            return "int_enum", core["enum"], []
        return "enum", core["enum"], []
    t = core.get("type")
    if t == "boolean":
        return "bool", [], []
    if t == "integer":
        return "int", [], []
    if t == "array":
        items = core.get("items", {})
        if "$ref" in items:
            return "array_of_objects", [], []  # item_fields resolved separately
        if "enum" in items:
            return "array_of_strings_enum", items["enum"], []
        return "array_of_strings_enum", [], []
    return "string", [], []


def _parse_properties(properties: dict, defs: dict, required_fields: list) -> list[FieldSpec]:
    result = []
    for fname, fprop in properties.items():
        ftype, enum_vals, _ = _detect_field_type(fprop)
        item_fields: list[FieldSpec] = []
        if ftype == "array_of_objects":
            ao = fprop.get("anyOf", [])
            non_null = [s for s in ao if s.get("type") != "null"]
            if non_null:
                ref = non_null[0].get("items", {}).get("$ref", "")
                ref_name = ref.split("/")[-1] if ref else ""
                if ref_name and ref_name in defs:
                    item_def = defs[ref_name]
                    item_required = item_def.get("required", [])
                    item_fields = _parse_properties(
                        item_def.get("properties", {}), defs, item_required
                    )
        result.append(
            FieldSpec(
                name=fname,
                title=fprop.get("title", fname.replace("_", " ").title()),
                description=fprop.get("description", ""),
                field_type=ftype,
                enum_values=enum_vals,
                item_fields=item_fields,
                required=fname in required_fields,
            )
        )
    return result


def _derive_display_name(section_name: str) -> str:
    # Strip leading cancer-type prefix, e.g. "ColonCancer" from "ColonCancerLN"
    # Pattern: one or more Title-cased words followed by "Cancer" (possibly none) then the suffix
    suffix = re.sub(r"^[A-Za-z]+Cancer", "", section_name)
    if not suffix:
        suffix = section_name
    return DISPLAY_NAMES.get(suffix, suffix)


def parse_cancer_schema(cancer_type: str) -> list[SectionSpec]:
    """Parse a cancer schema file and return a list of SectionSpecs."""
    schema_file = CANCER_TO_FILE.get(cancer_type)
    if not schema_file:
        return []
    path = SCHEMAS_DIR / schema_file
    if not path.exists():
        return []
    with open(path) as f:
        schema = json.load(f)

    defs = schema.get("$defs", {})
    sections: list[SectionSpec] = []

    for section_name, section_schema in schema.get("properties", {}).items():
        if not isinstance(section_schema, dict):
            continue
        props = section_schema.get("properties", {})
        required_in_section = section_schema.get("required", [])

        flat_fields: list[FieldSpec] = []
        array_field_name: str | None = None
        array_item_fields: list[FieldSpec] = []
        array_item_enum_values: list = []
        array_field_type: str | None = None

        for fname, fprop in props.items():
            ftype, enum_vals, _ = _detect_field_type(fprop)
            if ftype in ("array_of_objects", "array_of_strings_enum"):
                array_field_name = fname
                array_field_type = ftype
                if ftype == "array_of_objects":
                    ao = fprop.get("anyOf", [])
                    non_null = [s for s in ao if s.get("type") != "null"]
                    if non_null:
                        ref = non_null[0].get("items", {}).get("$ref", "")
                        ref_name = ref.split("/")[-1] if ref else ""
                        if ref_name and ref_name in defs:
                            item_def = defs[ref_name]
                            item_required = item_def.get("required", [])
                            array_item_fields = _parse_properties(
                                item_def.get("properties", {}), defs, item_required
                            )
                else:
                    array_item_enum_values = enum_vals
            else:
                item_fields_for_flat: list[FieldSpec] = []
                flat_fields.append(
                    FieldSpec(
                        name=fname,
                        title=fprop.get("title", fname.replace("_", " ").title()),
                        description=fprop.get("description", ""),
                        field_type=ftype,
                        enum_values=enum_vals,
                        item_fields=item_fields_for_flat,
                        required=fname in required_in_section,
                    )
                )

        sections.append(
            SectionSpec(
                name=section_name,
                display_name=_derive_display_name(section_name),
                flat_fields=flat_fields,
                array_field_name=array_field_name,
                array_item_fields=array_item_fields,
                array_item_enum_values=array_item_enum_values,
                array_field_type=array_field_type,
            )
        )

    return sections


if __name__ == "__main__":
    for cancer in CANCER_TO_FILE:
        if cancer == "others":
            continue
        sections = parse_cancer_schema(cancer)
        print(f"\n=== {cancer.upper()} ===")
        for s in sections:
            flat = [f.name for f in s.flat_fields]
            arr = f"{s.array_field_name}({s.array_field_type})" if s.array_field_name else "-"
            print(f"  {s.display_name:15} flat={flat}  array={arr}")
