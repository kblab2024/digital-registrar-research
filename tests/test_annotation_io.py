"""Smoke tests for the annotation IO layer (three-folder discovery + flat-JSON contract)."""
from digital_registrar_research.annotation import io as ann_io
from digital_registrar_research.paths import DATA_ROOT


def test_module_exposes_public_helpers():
    for name in ("FolderSet", "SampleRef", "build_save_payload",
                 "discover_folders", "list_samples", "load_json",
                 "load_report_text", "save_annotation", "strip_meta"):
        assert hasattr(ann_io, name), f"annotation.io missing {name!r}"


def test_discover_folders_on_packaged_data():
    """The shipped TCGA example folders should be discoverable."""
    from pathlib import Path
    fs = ann_io.discover_folders(str(DATA_ROOT))
    assert fs is not None
    # Three-folder convention: {prefix}_dataset_{date}/ + _result_/ + _annotation_/.
    assert Path(fs.dataset_dir).exists()
    assert Path(fs.result_dir).exists()
    assert Path(fs.annotation_dir).exists()
    assert fs.prefix == "tcga"
