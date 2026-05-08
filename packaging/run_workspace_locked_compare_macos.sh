#!/usr/bin/env bash
# Launch the Compare/Consensus tool on macOS (Apple Silicon).
# Locked to NHC vs KPC, with_preann, Consensus mode -> gold.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# macOS Gatekeeper marks files extracted from a downloaded archive with the
# com.apple.quarantine xattr. The bundled python binary is not notarized and
# will be blocked unless we strip it. Safe to run repeatedly.
xattr -dr com.apple.quarantine "$HERE/python" 2>/dev/null || true

# Skip Streamlit's one-time "please enter your email" prompt on first launch.
CRED="$HOME/.streamlit/credentials.toml"
if [ ! -f "$CRED" ]; then
    mkdir -p "$HOME/.streamlit"
    printf '[general]\nemail = ""\n' > "$CRED"
fi

export REGISTRAR_ANNOTATE_BASE_DIR="$HERE/workspace"
export REGISTRAR_COMPARE_LOCKED=1
export REGISTRAR_COMPARE_SUFFIX_A=nhc
export REGISTRAR_COMPARE_SUFFIX_B=kpc
export PYTHONPATH="$HERE/app"
export PYTHONDONTWRITEBYTECODE=1
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

cat <<'BANNER'
================================================================
  Digital Registrar - Compare / Consensus (workspace, locked)
================================================================
  Data root  : workspace/
  Mode       : Consensus (NHC vs KPC -> Gold), with_preann
  Output     : workspace/data/<dataset>/annotations/gold/<n>/<case_id>.json
  URL        : http://localhost:8501

  Keep this terminal open while annotating.
  Press Ctrl+C to stop the server.
================================================================

BANNER

exec "$HERE/python/bin/python3" -m streamlit run \
    "$HERE/app/digital_registrar_research/annotation/compare_app_canonical.py" \
    --server.address=localhost \
    --server.port=8501
