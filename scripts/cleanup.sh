#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# cleanup.sh  –  Remove generated artefacts, virtual environment, and caches
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "🧹  Cleaning up project directory …"

# Stop running instance first
if [ -f ".app.pid" ]; then
  bash "$SCRIPT_DIR/stop.sh" || true
fi

# Remove Python virtual environment
if [ -d ".venv" ]; then
  echo "  → Removing .venv"
  rm -rf .venv
fi

# Remove Python cache directories
find . -type d -name "__pycache__" ! -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" ! -path "./.git/*" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" ! -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" ! -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
echo "  → Python caches removed"

# Remove log and pid files
rm -f .app.pid .app.log
echo "  → Log/PID files removed"

echo ""
echo "✅  Cleanup complete."
echo "    Note: HuggingFace model cache (~/.cache/huggingface) was NOT removed."
echo "    To remove model weights: rm -rf ~/.cache/huggingface/hub/models--ibm-granite--granite-speech-5.0-2b"
