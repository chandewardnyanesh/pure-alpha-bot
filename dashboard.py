"""
Live terminal dashboard — shows P&L, open positions, ML stats, trade log.
Run in a separate terminal alongside bot.py:
    python dashboard.py
Refreshes every 5 seconds.
"""

import time
import os
import sys
from datetime import datetime

import trade_logger as tl
from config import INITIAL_CAPITAL

REFRESH = 5


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def color(text, code):
    return f"\033[{code}m{text}\033[0m"


def green(t):  return color(t, "92")
def red(t):    return color(t, "91")
def yellow(t): return color(t, "93")
def bold(t):   return color(t, "1")
def cyan(t):   return color(t, "96")


def pnl_color(v):
    return green(f"₹{v:+,.2f}") if v >= 0 else red(f"₹{v:+,.2f}")


def render():
    tl.init_db()
    today  = tl.get_today_trades()
    closed = [t for t in today if t["exit_price"] is not None]
    open_  = [t for t in today if t["exit_price"] is None]
    stats  = tl.get_performance_summary()

    today_pnl   = sum(t["net_pnl"] for t in closed if t["net_pnl"])
    capital_now = INITIAL_CAPITAL + sum(t["net_pnl"] for t in tl.get_all_closed_trades() if t["net_pnl"])
    progress    = (capital_now / 1_000_000) * 100

    w = 60
    print(bold("═" * w))
    print(bold(f"  AI TRADING BOT  |  {datetime.now().strftime('%d %b %Y  %H:%M:%S')}"))
    print(bold("═" * w))

    print(f"\n  {bold('CAPITAL JOURNEY')}")
    print(f"  Start  : ₹{INITIAL_CAPITAL:>10,.0f}")
    print(f"  Current: {cyan(f'₹{capital_now:>10,.0f}')}")
    print(f"  Target : ₹{1_000_000:>10,.0f}")
    bar_len  = 40
    filled   = int(min(progress, 100) / 100 * bar_len)
    bar      = green("█" * filled) + "░" * (bar_len - filled)
    print(f"  [{bar}] {progress:.2f}%")

    print(f"\n  {bold('TODAY')}  ({datetime.now().strftime('%d %b %Y')})")
    print(f"  Trades : {len(closed):>4}  open: {len(open_):>2}")
    print(f"  PnL    : {pnl_color(today_pnl)}")

    if stats.get("total", 0) > 0:
        print(f"\n  {bold('ALL-TIME PERFORMANCE')}")
        wr = stats.get("win_rate", 0)
        wr_str = green(f"{wr:.1%}") if wr >= 0.5 else red(f"{wr:.1%}")
        print(f"  Trades      : {stats['total']:>5}")
        print(f"  Win rate    : {wr_str}")
        print(f"  Profit fact : {stats.get('profit_factor', 0):>6.2f}")
        print(f"  Expectancy  : {pnl_color(stats.get('expectancy', 0))} / trade")
        print(f"  Gross PnL   : {pnl_color(stats.get('gross_pnl', 0))}")

    if open_:
        print(f"\n  {bold('OPEN POSITIONS')}")
        for t in open_:
            print(f"  {t['symbol']:<15} {t['action']:<5} qty={t['qty']:<4} "
                  f"entry=₹{t['entry_price']:.2f}  sl=₹{t.get('sl_price',0):.2f}")

    if closed:
        print(f"\n  {bold('TODAY CLOSED TRADES (last 5)')}")
        for t in closed[-5:]:
            pnl   = t.get("net_pnl", 0)
            sym   = t["symbol"][:12]
            rstr  = (t.get("exit_reason") or "")[:10]
            print(f"  {sym:<13} {t['action']:<5} {pnl_color(pnl)}  ({rstr})")

    print(f"\n{bold('═' * w)}")
    print(f"  Press Ctrl+C to exit dashboard")
    print(bold("═" * w))


def main():
    print("Starting dashboard ... (Ctrl+C to stop)")
    while True:
        try:
            clear()
            render()
        except Exception as e:
            print(f"Dashboard error: {e}")
        try:
            time.sleep(REFRESH)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
            break


if __name__ == "__main__":
    main()
