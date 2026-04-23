#!/usr/bin/env bash
# Build a portable Windows ZIP of the annotation tool.
#
# Output: packaging/dist/digital-registrar-annotator-windows.zip
#
# The ZIP contains:
#   python/     Windows embedded Python 3.12.x
#   app/        slim digital_registrar_research package (annotation subtree only)
#   dummy/      bundled dummy dataset
#   run.bat     double-click launcher
#   README.txt  annotator-facing instructions
#
# Build host: macOS or Linux. We cross-install win_amd64 wheels with pip.
# The annotator's Windows machine needs nothing but the OS.
#
# Usage (from repo root, with dev venv active):
#     bash packaging/build_windows_zip.sh

set -euo pipefail

# ── Pins ──────────────────────────────────────────────────────────────────────
PYTHON_VERSION="3.12.8"
PYTHON_TAG="cp312"
PYTHON_PLATFORM="win_amd64"
RUNTIME_REQS=(
    "streamlit>=1.35,<2"
    "pydantic>=2,<3"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG_DIR="$REPO_ROOT/packaging"
DL_DIR="$PKG_DIR/_downloads"
BUILD_DIR="$PKG_DIR/build"
DIST_DIR="$PKG_DIR/dist"
BUNDLE_NAME="digital-registrar-annotator-windows"
BUNDLE_DIR="$BUILD_DIR/$BUNDLE_NAME"

EMBED_ZIP_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-embed-amd64.zip"
EMBED_ZIP_PATH="$DL_DIR/python-${PYTHON_VERSION}-embed-amd64.zip"

PY_HOST="${PYTHON:-python3}"

# ── Helpers ───────────────────────────────────────────────────────────────────
say()  { printf "\033[1;36m==>\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31mERR:\033[0m %s\n" "$*" >&2; exit 1; }

# ── Preconditions ─────────────────────────────────────────────────────────────
command -v "$PY_HOST" >/dev/null || die "host python not found ($PY_HOST)"
command -v curl      >/dev/null || die "curl is required"
command -v unzip     >/dev/null || die "unzip is required"
command -v zip       >/dev/null || die "zip is required"

[ -d "$REPO_ROOT/src/digital_registrar_research/annotation" ] \
    || die "expected to run from the repo root — annotation package not found"
[ -d "$REPO_ROOT/dummy" ] \
    || die "dummy/ not found; nothing to bundle"

"$PY_HOST" -c "
import digital_registrar_research.models.modellist
import digital_registrar_research.schemas.pydantic._builder
" >/dev/null 2>&1 \
    || die "host python cannot import dspy-backed modules — activate dev venv first"

# ── Clean build dir (keep downloads cached) ───────────────────────────────────
say "Preparing build tree"
rm -rf "$BUILD_DIR"
mkdir -p "$DL_DIR" "$DIST_DIR" "$BUNDLE_DIR"

# ── Download + extract embeddable Python ──────────────────────────────────────
if [ ! -f "$EMBED_ZIP_PATH" ]; then
    say "Downloading Windows embeddable Python $PYTHON_VERSION"
    curl -fL --retry 3 -o "$EMBED_ZIP_PATH.part" "$EMBED_ZIP_URL"
    mv "$EMBED_ZIP_PATH.part" "$EMBED_ZIP_PATH"
else
    say "Using cached $EMBED_ZIP_PATH"
fi

say "Extracting embedded Python"
PYTHON_DIR="$BUNDLE_DIR/python"
mkdir -p "$PYTHON_DIR"
unzip -q -o "$EMBED_ZIP_PATH" -d "$PYTHON_DIR"

# Enable `import site` so pip-installed packages are picked up.
PTH_FILE="$(ls "$PYTHON_DIR"/python*._pth | head -1)"
[ -n "$PTH_FILE" ] || die "python*._pth not found in embedded distribution"

say "Patching $(basename "$PTH_FILE")"
cat > "$PTH_FILE" <<EOF
python312.zip
.
Lib\\site-packages
import site
EOF

# ── Cross-install runtime wheels for Windows ──────────────────────────────────
SITE_PACKAGES="$PYTHON_DIR/Lib/site-packages"
mkdir -p "$SITE_PACKAGES"

say "Installing Windows wheels into embedded site-packages"
"$PY_HOST" -m pip install \
    --disable-pip-version-check \
    --target "$SITE_PACKAGES" \
    --platform "$PYTHON_PLATFORM" \
    --python-version "${PYTHON_VERSION%.*}" \
    --implementation cp \
    --abi "$PYTHON_TAG" \
    --only-binary=:all: \
    --upgrade \
    "${RUNTIME_REQS[@]}"

# ── Trim size: drop tests/ and __pycache__ from site-packages ─────────────────
say "Trimming site-packages"
find "$SITE_PACKAGES" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$SITE_PACKAGES" -type d -name "tests" -prune -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -type d -name "test"  -prune -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -type f -name "*.pyc" -delete
# pip --target creates a unix `bin/` of console shims; useless on Windows.
rm -rf "$SITE_PACKAGES/bin"

# ── Assemble slim digital_registrar_research package ─────────────────────────
say "Copying annotation package"
APP_ROOT="$BUNDLE_DIR/app"
DRR_DEST="$APP_ROOT/digital_registrar_research"
mkdir -p "$DRR_DEST"

# Minimal package __init__ (the source one is already tiny — copy verbatim)
cp "$REPO_ROOT/src/digital_registrar_research/__init__.py" "$DRR_DEST/__init__.py"
cp "$REPO_ROOT/src/digital_registrar_research/paths.py"    "$DRR_DEST/paths.py"

# annotation subtree — exclude dev-only generators and fixtures
mkdir -p "$DRR_DEST/annotation"
rsync -a \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude "generate_dummy_data.py" \
    --exclude "dummy_data" \
    "$REPO_ROOT/src/digital_registrar_research/annotation/" \
    "$DRR_DEST/annotation/"

# schemas subtree — only the data/*.json files + a stub package marker
mkdir -p "$DRR_DEST/schemas/data"
touch "$DRR_DEST/schemas/__init__.py"
rsync -a --exclude "__pycache__" --exclude "*.pyc" \
    "$REPO_ROOT/src/digital_registrar_research/schemas/data/" \
    "$DRR_DEST/schemas/data/"

# Generate the precomputed section-groups JSON (removes DSPy at runtime)
say "Generating precomputed section-groups map"
"$PY_HOST" "$PKG_DIR/precompute_section_groups.py" \
    "$DRR_DEST/annotation/_section_groups.json"

# Wipe annotators.json so the Windows bundle ships with the defaults.
# (The source tree's annotators.json may have locally-added names we don't
# want to leak to external collaborators.)
if [ -f "$DRR_DEST/annotation/annotators.json" ]; then
    rm "$DRR_DEST/annotation/annotators.json"
fi

# ── Copy dummy dataset ────────────────────────────────────────────────────────
say "Copying dummy dataset"
rsync -a \
    --exclude ".DS_Store" \
    --exclude "__pycache__" \
    --exclude "results" \
    "$REPO_ROOT/dummy/" \
    "$BUNDLE_DIR/dummy/"

# ── Launcher + README ─────────────────────────────────────────────────────────
say "Copying launcher + README"
cp "$PKG_DIR/run.bat"    "$BUNDLE_DIR/run.bat"
cp "$PKG_DIR/README.txt" "$BUNDLE_DIR/README.txt"

# ── Final pycache sweep + size report ─────────────────────────────────────────
find "$BUNDLE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLE_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true

BUNDLE_SIZE="$(du -sh "$BUNDLE_DIR" | awk '{print $1}')"
say "Bundle tree assembled: $BUNDLE_DIR ($BUNDLE_SIZE)"

# ── Zip ───────────────────────────────────────────────────────────────────────
say "Zipping"
ZIP_PATH="$DIST_DIR/${BUNDLE_NAME}.zip"
rm -f "$ZIP_PATH"
(
    cd "$BUILD_DIR"
    zip -qr "$ZIP_PATH" "$BUNDLE_NAME"
)
ZIP_SIZE="$(du -sh "$ZIP_PATH" | awk '{print $1}')"

say "DONE"
printf "   bundle: %s  (%s unpacked)\n" "$ZIP_PATH" "$BUNDLE_SIZE"
printf "   zipped: %s\n" "$ZIP_SIZE"
printf "\nNext: copy %s to the Windows machine, unzip, double-click run.bat.\n" "$ZIP_PATH"
