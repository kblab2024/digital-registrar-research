#!/usr/bin/env bash
# Launch the Compare/Consensus tool on macOS (Apple Silicon).
# Unlocked demo against the empty dummy/ skeleton; full UI for QA.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# macOS Gatekeeper: strip quarantine from the bundled python tree.
xattr -dr com.apple.quarantine "$HERE/python" 2>/dev/null || true

CRED="$HOME/.streamlit/credentials.toml"
if [ ! -f "$CRED" ]; then
    mkdir -p "$HOME/.streamlit"
    printf '[general]\nemail = ""\n' > "$CRED"
fi

export REGISTRAR_ANNOTATE_BASE_DIR="$HERE/dummy"
export PYTHONPATH="$HERE/app"
export PYTHONDONTWRITEBYTECODE=1
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

cat <<'BANNER'
================================================================
  Digital Registrar - Compare / Consensus (dummy, unlocked)
================================================================
  Data root : dummy/
  Mode      : full UI (mode/annotator/dataset pickers visible)
  URL       : http://localhost:8501

  Use this launcher to explore the full Compare/Consensus UI
  against the dummy skeleton tree (no real data).
  Press Ctrl+C to stop the server.
================================================================

BANNER

exec "$HERE/python/bin/python3" -m streamlit run \
    "$HERE/app/digital_registrar_research/annotation/compare_app_canonical.py" \
    --server.address=localhost \
    --server.port=8501
