"""
Options strategy backtest — simulates buying CE/PE on signal, tracks premium P&L.

Premium simulation:
  We can't perfectly replay option prices historically (chain data not free),
  so we use a Black-Scholes approximation to estimate ATM premium from spot + ATR,
  then simulate P&L based on underlying move and DTE decay.

Usage:
    python backtest.py --underlying NIFTY --days 60
    python backtest.py --all --days 30
"""

import argparse
import logging
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    OPTION_UNDERLYINGS, SIGNAL_THRESHOLD, STRATEGY_WEIGHTS,
    OPTION_SL_MULT, OPTION_TARGET_MULT,
    BROKERAGE_PER_ORDER, INITIAL_CAPITAL,
)
from indicators import compute_all
from strategies import blend_signals

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("backtest")

UNDERLYING_MAP = {u["name"]: u for u in OPTION_UNDERLYINGS}


# ─── Simplified Black-Scholes ATM premium estimate ───────────────────────────

def bs_atm_premium(spot: float, days_to_expiry: int,
                   iv_annual: float = 0.18) -> float:
    """
    Approximate ATM straddle (CE or PE) premium using simplified BS.
    iv_annual: implied volatility estimate (18% default for Nifty).
    """
    t = max(days_to_expiry, 1) / 365.0
    return round(spot * iv_annual * math.sqrt(t) * 0.4, 2)


# ─── Backtester ───────────────────────────────────────────────────────────────

class OptionsBacktester:
    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.initial_capital = capital

    def run(self, df: pd.DataFrame, underlying_name: str,
            lot_size: int, iv_annual: float = 0.18) -> dict:

        df     = compute_all(df)
        cap    = self.initial_capital
        trades = []
        pos    = None   # active simulated option position

        for i in range(55, len(df)):
            window = df.iloc[:i + 1]
            row    = df.iloc[i]
            close  = row["close"]

            # ── Manage open option ────────────────────────────────────────────
            if pos:
                # Estimate current premium: BS based on current move
                days_left   = max(pos["dte"] - (i - pos["entry_bar"]) // 2, 1)
                move        = (close - pos["entry_spot"]) / pos["entry_spot"]
                delta       = 0.5                    # ATM delta approximation
                prem_change = move * pos["entry_spot"] * delta * (1 if pos["type"] == "CE" else -1)
                theta_decay = pos["entry_prem"] * (1 - (days_left / pos["dte"]) ** 0.5)
                current_prem = max(pos["entry_prem"] + prem_change - theta_decay * 0.3, 0.5)

                sl_prem  = pos["entry_prem"] * (1 - OPTION_SL_MULT)
                tgt_prem = pos["entry_prem"] * (1 + OPTION_TARGET_MULT)
                exit_r   = None

                if current_prem <= sl_prem:
                    exit_r = "stop_loss"
                elif current_prem >= tgt_prem:
                    exit_r = "target"
                elif i == len(df) - 1:
                    exit_r = "eod"

                if exit_r:
                    pnl  = (current_prem - pos["entry_prem"]) * pos["qty"]
                    net  = pnl - BROKERAGE_PER_ORDER * 2
                    cap += net
                    trades.append({
                        "underlying":   underlying_name,
                        "type":         pos["type"],
                        "entry_prem":   pos["entry_prem"],
                        "exit_prem":    round(current_prem, 2),
                        "qty":          pos["qty"],
                        "pnl":          round(pnl, 2),
                        "net_pnl":      round(net, 2),
                        "profitable":   net > 0,
                        "exit_reason":  exit_r,
                        "bar_index":    i,
                    })
                    pos = None
                continue

            # ── Look for entry ────────────────────────────────────────────────
            sig = blend_signals(window, STRATEGY_WEIGHTS)
            if sig["action"] == "HOLD" or sig["score"] < SIGNAL_THRESHOLD:
                continue

            dte      = 5             # weekly option, ~5 trading days
            entry_p  = bs_atm_premium(close, dte, iv_annual)
            if entry_p <= 0:
                continue

            budget  = cap * 0.25    # 25% of capital per trade
            lots    = max(1, int(budget / (entry_p * lot_size)))
            qty     = lots * lot_size
            cost    = qty * entry_p

            if cost > cap * 0.30:
                lots = max(1, int(cap * 0.30 / (entry_p * lot_size)))
                qty  = lots * lot_size

            pos = {
                "type":        "CE" if sig["action"] == "BUY" else "PE",
                "entry_prem":  entry_p,
                "entry_spot":  close,
                "entry_bar":   i,
                "qty":         qty,
                "lots":        lots,
                "dte":         dte,
            }

        return self._summarise(trades, underlying_name, self.initial_capital)

    @staticmethod
    def _summarise(trades: list, name: str, start_cap: float) -> dict:
        if not trades:
            return {"underlying": name, "total": 0, "message": "No trades"}

        pnls   = [t["net_pnl"] for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        wrate = len(wins) / len(pnls) if pnls else 0
        avg_w = sum(wins)   / len(wins)   if wins   else 0
        avg_l = sum(losses) / len(losses) if losses else 0
        pf    = sum(wins) / abs(sum(losses)) if losses and sum(losses) else float("inf")

        cum    = np.cumsum(pnls)
        run_mx = np.maximum.accumulate(cum)
        max_dd = float((run_mx - cum).max()) if len(cum) else 0

        return {
            "underlying":   name,
            "total_trades": len(trades),
            "wins":         len(wins),
            "losses":       len(losses),
            "win_rate":     round(wrate, 4),
            "gross_pnl":    round(sum(pnls), 2),
            "best_trade":   round(max(pnls), 2),
            "worst_trade":  round(min(pnls), 2),
            "avg_win":      round(avg_w, 2),
            "avg_loss":     round(avg_l, 2),
            "profit_factor": round(pf, 2),
            "max_drawdown": round(max_dd, 2),
            "expectancy":   round(wrate * avg_w + (1 - wrate) * avg_l, 2),
            "trades":       trades,
        }


def print_report(r: dict):
    print(f"\n{'='*58}")
    print(f"  BACKTEST  —  {r['underlying']}")
    print(f"{'='*58}")
    print(f"  Trades        : {r.get('total_trades', 0)}")
    print(f"  Win rate      : {r.get('win_rate', 0):.1%}")
    print(f"  Gross PnL     : ₹{r.get('gross_pnl', 0):,.2f}")
    print(f"  Profit factor : {r.get('profit_factor', 0):.2f}")
    print(f"  Max drawdown  : ₹{r.get('max_drawdown', 0):,.2f}")
    print(f"  Expectancy    : ₹{r.get('expectancy', 0):,.2f} / trade")
    print(f"  Best / Worst  : ₹{r.get('best_trade', 0):,.2f} / ₹{r.get('worst_trade', 0):,.2f}")
    print(f"{'='*58}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--underlying", default="NIFTY",
                        help="NIFTY, BANKNIFTY, or SENSEX")
    parser.add_argument("--days",   type=int, default=60)
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--iv",     type=float, default=0.18,
                        help="Annual IV for premium simulation (default 0.18=18%)")
    args = parser.parse_args()

    from broker       import Broker
    from data_fetcher import DataFetcher

    broker  = Broker()
    fetcher = DataFetcher(broker.kite)
    bt      = OptionsBacktester()

    targets = OPTION_UNDERLYINGS if args.all else [
        u for u in OPTION_UNDERLYINGS if u["name"] == args.underlying
    ]

    total_pnl = 0
    for cfg in targets:
        logger.info(f"Backtesting {cfg['name']} over {args.days} days ...")
        try:
            df = fetcher.fetch_history(cfg["index"], days=args.days)
            if df.empty:
                logger.warning(f"No data for {cfg['index']}")
                continue
            result = bt.run(df, cfg["name"], cfg["lot_size"], args.iv)
            print_report(result)
            total_pnl += result.get("gross_pnl", 0)
        except Exception as e:
            logger.error(f"Backtest error for {cfg['name']}: {e}")

    if args.all:
        print(f"\n  COMBINED PnL : ₹{total_pnl:,.2f}")
