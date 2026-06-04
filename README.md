# PureAlpha Bot 📈

**Dalio-Inspired Systematic NSE Options Trading Bot**

Named after Ray Dalio's *Pure Alpha* strategy at Bridgewater Associates — the world's largest hedge fund.
Dalio's core principle: *diversified, uncorrelated signals aggregated into a single conviction score before taking action.*
This bot applies that exact framework to intraday NSE options (Nifty & BankNifty).

---

## Architecture

```
LAYER 1: DATA LAYER          5-min OHLCV + 15-min HTF + India VIX
         ↓
LAYER 2: SIGNAL LAYER        6 Independent Voters
         ↓                   Trend · Reversion · Breakout · Kronos · ML · HTF-Alignment
LAYER 3: FILTER LAYER        VIX Regime · Session Dead Zone · ATR/ADX/Volume
         ↓
LAYER 4: CONSENSUS           4/6 votes minimum (Dalio: no single signal, always aggregate)
         ↓
LAYER 5: POSITION SIZING     4/6=15% · 5/6=22% · 6/6=35% of capital
         ↓
LAYER 6: RISK MANAGEMENT     ATR Stop + Partial Exit at 1R + Trailing Stop to 2R
```

---

## Target Performance

| Metric | Target |
|--------|--------|
| Win Rate | 55–65% |
| Risk:Reward | 1:2 |
| Expectancy | +0.65 |

---

## Key Improvements (v2)

- **HTF Voter** — 15-minute timeframe alignment as 6th independent voter
- **Session Filter** — blocks entries during 11:30–13:00 NSE dead zone
- **India VIX Gate** — size reduction when VIX > 20, hard block > 28
- **Partial Exit** — 50% exit at 1R (+40% premium), breakeven SL, rest runs to 2R
- **MACD Divergence** — divergence detection in trend voter adds/removes conviction
- **Tightened Reversion** — RSI 35/65 thresholds (reduced noise vs 38/62)
- **Per-Voter WR Tracking** — SQLite voter_stats table, Dalio-style attribution
- **Rolling Expectancy** — live 20-trade rolling expectancy vs +0.65 target

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Paper trade (no real orders)
python bot.py --paper

# Live dashboard
python dashboard_server.py
# → http://127.0.0.1:5001

# Backtest
python backtest.py
```

---

## Dalio Principles Applied

> *"He who lives by the crystal ball will eat shattered glass."*

| Dalio Principle | Implementation |
|----------------|----------------|
| Diversify signal sources | 6 uncorrelated voters: technical, ML, DL forecast, HTF |
| Never bet on single outcome | 4/6 consensus required — no single voter can force a trade |
| Systematic, not discretionary | Zero manual intervention — pure rules-based |
| All-weather sizing | VIX-adjusted position size adapts to regime |
| Risk parity | Partial exit locks 1R, lets 2R run — asymmetric payoff |

---

## Progress Log

| Date | Update |
|------|--------|
| 2026-06-04 | v2: 6-voter architecture, VIX filter, session gate, partial exits, MACD divergence |
| 2026-05-xx | v1: 7-layer architecture, Kronos + ML ensemble |

---

*Capital: ₹39,430 → Target: ₹10,00,000*
