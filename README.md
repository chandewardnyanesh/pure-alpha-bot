# AI Options Trading Bot — Nifty / BankNifty (Zerodha Kite)

An intraday **self-learning** options trading bot for Indian markets.  
Buys CE/PE options on Nifty and BankNifty based on technical signals, manages risk automatically, and retrains its ML model from closed trade outcomes.

> **Capital goal**: ₹39,430 → ₹10,00,000  
> **Exchange**: NSE/NFO via Zerodha Kite

---

## Architecture

```
bot.py              ← Main loop (scan every 60s)
├── data_fetcher.py ← Kite historical API + LTP
├── indicators.py   ← EMA, RSI, MACD, SuperTrend, VWAP, ATR, BB
├── strategies.py   ← 5 signal strategies + ensemble blender
├── options_manager.py ← NFO contract selection (strike, expiry)
├── risk_manager.py ← Position sizing, SL/target/trail, daily limits
├── ml_model.py     ← RandomForest self-learning layer
├── trade_logger.py ← SQLite trade journal
├── broker.py       ← KiteConnect order placement
├── login.py        ← OAuth login helper
├── backtest.py     ← Historical strategy simulation
├── dashboard.py    ← Terminal P&L display
└── config.py       ← All tuneable parameters
```

---

## Strategy Stack

| Strategy | Weight | Signal |
|---|---|---|
| SuperTrend flip | 0.45 | Trend reversal on index |
| EMA Crossover + RSI | 0.30 | Fast/slow EMA cross |
| VWAP Extreme Reversion | 0.15 | >2σ VWAP deviation |
| Volume Breakout | 0.10 | N-bar high/low with surge |
| Opening Range Breakout | 0.15 | First 15-min range (09:30–10:30) |

Signals are **ensemble-blended** (un-normalized weighted sum) and gated by an ML success-probability score.

---

## Risk Management

- Max capital per trade: **40%** (~₹15,772 → 1 lot NIFTY)
- Stop-loss: **−40%** of option premium
- Target: **+80%** of option premium
- Trailing stop: activates at **+35%**, locks **+15%** floor
- Midday cut: if down **30%** after 13:00, exit (no EOD bagholding)
- Hard daily loss limit: **5%** of capital
- Square-off all positions by **15:10**

---

## ML Self-Learning

Every closed trade is fed back to a `RandomForestClassifier` (or `GradientBoostingClassifier`, whichever cross-validates better).  
After every **5 trades** (minimum **20 samples**), the model retrains and updates strategy weights from feature importances.  
Model persists to `models/rf_model.pkl`.

---

## Setup

### Prerequisites
```bash
pip3 install -r requirements.txt
```

Or run the setup script:
```bash
bash setup.sh
```

### Zerodha API credentials
Create `data/credentials.txt` (**never commit this file**):
```
api_key=YOUR_API_KEY
api_secret=YOUR_API_SECRET
```

### Daily login (required every trading day)
Kite access tokens expire daily. Run the login helper:

```bash
python3 login.py
# Opens browser → log in → copy the request_token from the redirect URL
python3 login.py --token YOUR_REQUEST_TOKEN
```

---

## Running

### Paper trade (no real orders)
```bash
python3 bot.py --paper
```

### Live trade
```bash
python3 bot.py
```

### Backtest (requires valid Kite session)
```bash
python3 backtest.py --all --days 60
```

### Dashboard
```bash
python3 dashboard.py
```

---

## Configuration

All parameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `INITIAL_CAPITAL` | 39,430 | Starting capital (INR) |
| `MAX_CAPITAL_PER_TRADE_PCT` | 40.0 | % of capital per position |
| `SIGNAL_THRESHOLD` | 0.28 | Min blended score to enter |
| `STRONG_SIGNAL_SCORE` | 0.72 | Use 2-OTM instead of 3-OTM |
| `OPTION_SL_MULT` | 0.40 | Exit if premium drops 40% |
| `OPTION_TARGET_MULT` | 0.80 | Exit if premium gains 80% |
| `OPTION_TRAIL_START` | 0.35 | Activate trail at +35% |
| `NO_ENTRY_AFTER` | 14:00 | No new positions after 2 PM |
| `SQUARE_OFF_TIME` | 15:10 | Mandatory EOD exit |

---

## Instruments

| Underlying | Lot Size | Strike Step |
|---|---|---|
| NIFTY | 65 | 50 |
| BANKNIFTY | 30 | 100 |

*(Lot sizes verified from live NFO instrument dump, May 2026 SEBI revision)*

---

## Important Notes

- **Never commit** `data/credentials.txt`, `data/access_token.txt`, or `data/trades.db`
- Bot only **buys** options (CE for bullish, PE for bearish) — no naked selling
- Kite access tokens are **single-use per day** — run `login.py` fresh each morning
- Index volume from Kite API is 0 for spot indices — the bot uses a volatility proxy internally

---

## Capital Scaling Guide

| Capital | Strategy |
|---|---|
| ₹39,430 | 1 lot NIFTY only |
| ₹60,000 | 1 lot NIFTY + BANKNIFTY |
| ₹1,00,000 | 2 lots NIFTY per underlying |
| ₹5,00,000 | 3 lots, tighter OTM offsets |
| ₹10,00,000 | **Target achieved** 🎯 |

---

## Disclaimer

This bot is for **educational and personal use only**.  
Options trading involves significant risk of loss. Past performance does not guarantee future results.  
Always paper-trade first and understand the risks before using real capital.
