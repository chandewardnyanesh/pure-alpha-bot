#!/bin/bash
# One-time setup script for the AI Trading Bot

set -e
echo "=== AI Trading Bot Setup ==="

# Create directories
mkdir -p data logs models

# Install dependencies
pip3 install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo ""
echo "NEXT STEPS:"
echo "  1. Get Zerodha Kite API credentials from https://developers.kite.trade/"
echo "  2. Export your credentials:"
echo "       export KITE_API_KEY='your_api_key'"
echo "       export KITE_API_SECRET='your_api_secret'"
echo ""
echo "  3. (Optional) Run backtest first to validate strategies:"
echo "       python backtest.py --all --days 60"
echo ""
echo "  4. Run in PAPER TRADE mode first (no real money):"
echo "       python bot.py --paper"
echo ""
echo "  5. Once satisfied, run live:"
echo "       python bot.py"
echo ""
echo "  6. Open dashboard in another terminal:"
echo "       python dashboard.py"
