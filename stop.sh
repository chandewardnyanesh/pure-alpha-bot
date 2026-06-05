#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  PureAlpha Bot — stop script
# ─────────────────────────────────────────────────────────────────

BOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Stopping PureAlpha Bot..."

for proc in "bot.py" "dashboard_server.py" "scheduler.py"; do
    PIDS=$(pgrep -f "$BOT_DIR/$proc" 2>/dev/null)
    if [ -n "$PIDS" ]; then
        kill $PIDS 2>/dev/null
        echo "  Stopped $proc (PIDs: $PIDS)"
    fi
done

echo "Done."
