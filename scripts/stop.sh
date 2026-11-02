#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# stop.sh  –  Gracefully stop the Granite Speech demo application
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PIDFILE="$PROJECT_DIR/.app.pid"

if [ ! -f "$PIDFILE" ]; then
  echo "ℹ  No PID file found ($PIDFILE). Application may not be running."
  exit 0
fi

PID=$(cat "$PIDFILE")

if kill -0 "$PID" 2>/dev/null; then
  echo "🛑  Stopping application (PID $PID) …"
  kill -TERM "$PID"
  # Wait up to 10 s for graceful shutdown
  for i in $(seq 1 10); do
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$PID" 2>/dev/null; then
    echo "⚠  Process did not exit; sending SIGKILL …"
    kill -KILL "$PID" || true
  fi
  echo "✅  Application stopped."
else
  echo "ℹ  No process found with PID $PID (already stopped?)."
fi

rm -f "$PIDFILE"
