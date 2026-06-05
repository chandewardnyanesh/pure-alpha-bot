#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  PureAlpha Bot — start script
#  Starts bot + dashboard in background from the correct directory.
#  Safe to run multiple times — detects if already running.
# ─────────────────────────────────────────────────────────────────

BOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(which python3)"
LOG_DIR="$BOT_DIR/logs"
mkdir -p "$LOG_DIR" "$BOT_DIR/data" "$BOT_DIR/models"

echo "======================================"
echo "  PureAlpha Bot  |  Paper Mode"
echo "  Dir: $BOT_DIR"
echo "  Dashboard: http://127.0.0.1:5002"
echo "======================================"

# ── Bot ──────────────────────────────────────────────────────────
if pgrep -f "$BOT_DIR/bot.py" > /dev/null; then
    echo "⚠️  Bot already running (PID: $(pgrep -f "$BOT_DIR/bot.py"))"
else
    cd "$BOT_DIR"
    nohup "$PYTHON" bot.py --paper >> "$LOG_DIR/purealpha.log" 2>&1 &
    echo "✅ Bot started  (PID: $!)"
fi

# ── Dashboard ─────────────────────────────────────────────────────
if pgrep -f "$BOT_DIR/dashboard_server.py" > /dev/null; then
    echo "⚠️  Dashboard already running (PID: $(pgrep -f "$BOT_DIR/dashboard_server.py"))"
else
    cd "$BOT_DIR"
    nohup "$PYTHON" dashboard_server.py >> "$LOG_DIR/purealpha_dashboard.log" 2>&1 &
    echo "✅ Dashboard started  (PID: $!) → http://127.0.0.1:5002"
fi

echo ""
echo "📄 Logs:"
echo "   tail -f $LOG_DIR/purealpha.log"
echo "   tail -f $LOG_DIR/purealpha_dashboard.log"
