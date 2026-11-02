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

# ── Create output / input directories ─────────────────────────────────────────
mkdir -p "$PROJECT_DIR/output"
mkdir -p "$PROJECT_DIR/input"

# ── Generate self-signed TLS cert for localhost HTTPS (enables microphone) ────
CERT_DIR="$PROJECT_DIR/certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

mkdir -p "$CERT_DIR"

if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
  if command -v openssl &>/dev/null; then
    echo "→ Generating self-signed TLS certificate for localhost …"
    openssl req -x509 -newkey rsa:2048 -nodes \
      -keyout "$KEY_FILE" \
      -out    "$CERT_FILE" \
      -days   3650 \
      -subj   "/CN=localhost" \
      -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" 2>/dev/null
    echo "✔  TLS cert generated at certs/cert.pem (key: certs/key.pem)"
    echo "   On first browser visit accept the self-signed cert warning."
  else
    echo "⚠  openssl not found — skipping TLS cert generation."
    echo "   Microphone recording will not work without HTTPS."
    echo "   Install openssl and re-run this script, or generate manually:"
    echo "     openssl req -x509 -newkey rsa:2048 -nodes \\"
    echo "       -keyout certs/key.pem -out certs/cert.pem -days 3650 \\"
    echo "       -subj '/CN=localhost' \\"
    echo "       -addext 'subjectAltName=DNS:localhost,IP:127.0.0.1'"
  fi
else
  echo "✔  TLS cert already exists (certs/cert.pem)"
fi

echo ""
echo "✅  Setup complete!"
echo ""
echo "    Next steps:"
echo "    1. Edit .env with your API keys (optional – Ollama works without keys)"
echo "    2. Edit config/config.yaml to select your LLM backend"
echo "    3. Run: bash scripts/start.sh"
echo "    4. Open https://localhost:7860 (accept the self-signed cert warning)"
