#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# start.sh  –  Launch the Granite Speech 5.0-2B demo application (detached)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
PIDFILE="$PROJECT_DIR/.app.pid"
LOGFILE="$PROJECT_DIR/.app.log"

cd "$PROJECT_DIR"

# ── Activate virtual environment ──────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "❌  Virtual environment not found at $VENV_DIR"
  echo "    Run: uv venv && uv pip install -r requirements.txt"
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ── Load .env if present ──────────────────────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_DIR/.env"
  set +a
  echo "✔  Loaded .env"
fi

# ── Read port from config (default 7860) ─────────────────────────────────────
APP_PORT="${APP_PORT:-7860}"

# Kill any stale instance first
if [ -f "$PIDFILE" ]; then
  OLD_PID=$(cat "$PIDFILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "⚠  Stopping previous instance (PID $OLD_PID) …"
    kill "$OLD_PID" || true
    sleep 1
  fi
  rm -f "$PIDFILE"
fi

# ── Start the app in background ───────────────────────────────────────────────
echo "🚀  Starting Granite Speech demo …"
nohup python "$PROJECT_DIR/app.py" >> "$LOGFILE" 2>&1 &
APP_PID=$!
echo "$APP_PID" > "$PIDFILE"

# Give the server a moment to bind the port
sleep 3
if kill -0 "$APP_PID" 2>/dev/null; then
  echo "✅  Application started  (PID $APP_PID)"
  echo ""
  echo "    ┌──────────────────────────────────────────────┐"
  echo "    │  🌐  http://localhost:${APP_PORT}                    │"
  echo "    └──────────────────────────────────────────────┘"
  echo ""
  echo "    Logs → $LOGFILE"
  echo "    Stop → bash scripts/stop.sh"
else
  echo "❌  Application failed to start. Check logs: $LOGFILE"
  exit 1
fi
