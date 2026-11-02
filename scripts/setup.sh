#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh  –  One-time project setup: create venv and install dependencies
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "⚙️   Setting up Granite Speech 5.0-2B demo environment …"
echo ""

# ── Create virtual environment with uv ────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "❌  'uv' is not installed."
  echo "    Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "→ Creating virtual environment with uv …"
uv venv --python 3.11 2>/dev/null || uv venv

echo "→ Installing dependencies …"
uv pip install -r requirements.txt

# ── Create .env from example if not present ────────────────────────────────────
if [ ! -f "$PROJECT_DIR/.env" ] && [ -f "$PROJECT_DIR/.env.example" ]; then
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
  echo ""
  echo "✔  Created .env from .env.example"
  echo "   Edit .env to set your API keys and LLM configuration."
fi

# ── Create output directory ────────────────────────────────────────────────────
mkdir -p "$PROJECT_DIR/output"

echo ""
echo "✅  Setup complete!"
echo ""
echo "    Next steps:"
echo "    1. Edit .env with your API keys (optional – Ollama works without keys)"
echo "    2. Edit config/config.yaml to select your LLM backend"
echo "    3. Run: bash scripts/start.sh"
