"""
AI Options Trading Bot — Nifty / BankNifty / Sensex weekly options.
Zerodha Kite, intraday (NRML, squared off before 15:10).

Loop (every 60 seconds during market hours):
  1. Fetch latest index candles (spot).
  2. Compute all indicators.
  3. Blend strategy signals (EMA, VWAP, SuperTrend, Breakout, ORB).
  4. ML model predicts success probability → scales final score.
  5. Score > SIGNAL_THRESHOLD → find the right CE/PE contract.
  6. Size position, place buy order.
  7. Monitor open options: trail, SL, target, EOD exit.
  8. Feed every closed trade back to ML → self-improvement.

Run: python bot.py [--paper]
"""

import os
import sys
import time
import signal
import logging
import argparse
from datetime import datetime, date

from config import (
    OPTION_UNDERLYINGS, SCAN_INTERVAL_SECS,
    MARKET_OPEN, MARKET_CLOSE, SQUARE_OFF_TIME,
    NO_ENTRY_AFTER, NO_ENTRY_BEFORE,
    SIGNAL_THRESHOLD, INITIAL_CAPITAL,
    LOG_PATH, LOG_LEVEL, BROKERAGE_PER_ORDER,
    OPTION_MIDDAY_SL,
)
from broker          import Broker
from data_fetcher    import DataFetcher
from indicators      import compute_all, extract_features
from strategies      import blend_signals
from risk_manager    import RiskManager
from ml_model        import TradingModel
from options_manager import OptionsManager
import trade_logger as tl

# ─── Logging ──────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level   = getattr(logging, LOG_LEVEL),
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("bot")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")

def is_market_open() -> bool:
    return MARKET_OPEN <= now_hhmm() <= MARKET_CLOSE

def entry_allowed() -> bool:
    t = now_hhmm()
    return NO_ENTRY_BEFORE <= t <= NO_ENTRY_AFTER

def squareoff_due() -> bool:
    return now_hhmm() >= SQUARE_OFF_TIME


# ─── Trading Bot ──────────────────────────────────────────────────────────────

class OptionsBot:
    def __init__(self, paper_trade: bool = False):
        self.paper  = paper_trade
        logger.info("=" * 65)
        logger.info(f"  AI OPTIONS BOT  |  paper={paper_trade}  |  {date.today()}")
        logger.info(f"  Capital: ₹{INITIAL_CAPITAL:,.0f}   →   Target: ₹10,00,000")
        logger.info("=" * 65)

        tl.init_db()

        self.broker   = Broker()
        self.fetcher  = DataFetcher(self.broker.kite)
        self.opts_mgr = OptionsManager(self.broker.kite)
        self.risk_mgr = RiskManager()
        self.model    = TradingModel()

        # Active option positions: tradingsymbol → {pos dict, db_trade_id}
        self.active: dict[str, dict] = {}
        self.capital_at_open = INITIAL_CAPITAL
        self.squaredoff      = False
        self.running         = True

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, *_):
        logger.info("Shutdown requested.")
        self.running = False

    # ─── Startup ──────────────────────────────────────────────────────────────

    def startup(self):
        logger.info("Loading instrument list and historical data ...")
        self.opts_mgr.refresh_instruments()
        self.fetcher.warmup()

        try:
            cap = self.broker.get_available_capital()
            if cap > 0:
                self.risk_mgr.update_capital(cap)
                self.capital_at_open = cap
                logger.info(f"Available capital: ₹{cap:,.2f}")
        except Exception as e:
            logger.warning(f"Capital fetch failed: {e} — using config value")

        logger.info("Startup complete. Entering main loop.")

    # ─── Process one underlying ───────────────────────────────────────────────

    def process_underlying(self, cfg: dict):
        """
        cfg: one entry from OPTION_UNDERLYINGS (name, index, exchange, lot_size, strike_step)
        """
        index_symbol = cfg["index"]
        name         = cfg["name"]

        # 1. Fetch and compute indicators on the index
        try:
            df = self.fetcher.refresh(index_symbol)
        except Exception as e:
            logger.warning(f"{name}: data fetch error — {e}")
            return

        if len(df) < 55:
            return

        df = compute_all(df)
        row = df.iloc[-1]
        if df.isnull().iloc[-1][["rsi", "macd_hist", "ema_fast"]].any():
            return

        # 2. Strategy signal
        weights = self.model.get_weights()
        sig     = blend_signals(df, weights)

        if sig["action"] != "HOLD":
            logger.info(f"[SIGNAL] {name} {sig['action']} score={sig['score']:.3f} | {sig['reasons']}")

        if sig["action"] == "HOLD":
            return

        # 3. ML probability
        features = extract_features(row)
        ml_proba = self.model.predict_success_proba(features)
        final    = sig["score"] * ml_proba

        logger.info(f"[EVAL]   {name} | {sig['action']} score={sig['score']:.2f} "
                    f"× ml={ml_proba:.2f} = final={final:.2f} "
                    f"(threshold={SIGNAL_THRESHOLD})")

        # 4. Monitor open positions for this underlying (exit check)
        for sym in list(self.active.keys()):
            if self.active[sym]["pos"]["underlying"] == name:
                self._monitor(sym)

        # 5. Entry gate
        if final < SIGNAL_THRESHOLD or not entry_allowed():
            return

        # Already have a position on this underlying?
        underlying_open = any(
            v["pos"]["underlying"] == name for v in self.active.values()
        )
        if underlying_open:
            return

        can, reason = self.risk_mgr.can_take_new_trade()
        if not can:
            logger.info(f"Skipping {name}: {reason}")
            return

        # 6. Find and enter the option
        contract = self.opts_mgr.select_option_contract(
            underlying_name = name,
            signal_action   = sig["action"],
            signal_score    = sig["score"],
            capital         = self.risk_mgr.current_capital,
        )
        if not contract:
            return

        pos = self.risk_mgr.calculate_option_position(contract)
        self._enter(pos, features, sig, ml_proba)

    # ─── Enter trade ──────────────────────────────────────────────────────────

    def _enter(self, pos: dict, features, sig: dict, ml_proba: float):
        symbol   = pos["tradingsymbol"]
        exchange = pos["exchange"]

        if not self.paper:
            try:
                self.broker.buy_option(
                    tradingsymbol = symbol,
                    exchange      = exchange,
                    qty           = pos["qty"],
                    tag           = "BOT_BUY",
                )
            except Exception as e:
                logger.error(f"Entry failed for {symbol}: {e}")
                return
        else:
            logger.info(
                f"[PAPER] BUY {symbol} lots={pos['lots']} qty={pos['qty']} "
                f"@ ₹{pos['entry_premium']:.2f}  cost=₹{pos['total_cost']:.0f} "
                f"SL=₹{pos['sl_premium']:.2f}  TGT=₹{pos['target_premium']:.2f}"
            )

        pos["entry_time"] = datetime.now().isoformat()
        trade_id = tl.log_entry({
            "symbol":          symbol,
            "action":          "BUY",
            "qty":             pos["qty"],
            "entry_price":     pos["entry_premium"],
            "sl_price":        pos["sl_premium"],
            "target_price":    pos["target_premium"],
            "entry_time":      pos["entry_time"],
            "features":        features.tolist(),
            "strategy_scores": {r: 1 for r in sig["reasons"]},
            "ml_proba":        ml_proba,
        })

        self.active[symbol] = {"pos": pos, "db_id": trade_id}
        self.risk_mgr.add_position(pos)

        logger.info(
            f"ENTERED  {symbol} | lots={pos['lots']} | prem=₹{pos['entry_premium']:.2f} "
            f"| cost=₹{pos['total_cost']:.0f} | SL=₹{pos['sl_premium']:.2f} "
            f"| TGT=₹{pos['target_premium']:.2f} | signal={sig['score']:.2f} ml={ml_proba:.2f}"
        )

    # ─── Monitor open option ──────────────────────────────────────────────────

    def _monitor(self, symbol: str):
        entry = self.active.get(symbol)
        if not entry:
            return

        pos      = entry["pos"]
        exchange = pos["exchange"]

        current_premium = self.fetcher.get_option_ltp(symbol, exchange)
        if current_premium <= 0:
            return

        pos = self.risk_mgr.update_option_trail(pos, current_premium)
        exit_, reason = self.risk_mgr.should_exit_option(pos, current_premium)

        # Midday cut: if down OPTION_MIDDAY_SL% after 13:00, don't baghold into EOD
        if (not exit_ and now_hhmm() >= "13:00"
                and current_premium <= pos["entry_premium"] * (1 - OPTION_MIDDAY_SL)):
            exit_, reason = True, "midday_cut"

        pnl_now = (current_premium - pos["entry_premium"]) * pos["qty"]
        pnl_pct = (current_premium / pos["entry_premium"] - 1) * 100
        logger.debug(f"  {symbol} prem=₹{current_premium:.2f} "
                     f"({pnl_pct:+.1f}%) trail={pos.get('trailing_sl')}")

        if exit_:
            self._exit(symbol, current_premium, reason)

    # ─── Exit trade ───────────────────────────────────────────────────────────

    def _exit(self, symbol: str, exit_premium: float, reason: str):
        entry = self.active.get(symbol)
        if not entry:
            return

        pos      = entry["pos"]
        db_id    = entry["db_id"]
        exchange = pos["exchange"]

        if not self.paper:
            try:
                self.broker.sell_option(
                    tradingsymbol = symbol,
                    exchange      = exchange,
                    qty           = pos["qty"],
                    tag           = "BOT_EXIT",
                )
            except Exception as e:
                logger.error(f"Exit failed for {symbol}: {e}")
                return
        else:
            pnl_paper = (exit_premium - pos["entry_premium"]) * pos["qty"]
            logger.info(f"[PAPER] SELL {symbol} @ ₹{exit_premium:.2f} — "
                        f"{reason} — pnl=₹{pnl_paper:+.2f}")

        charges = pos.get("charges", BROKERAGE_PER_ORDER * 2)
        result  = tl.log_exit(db_id, exit_premium, reason, charges)

        if result:
            self.risk_mgr.record_pnl(result["net_pnl"])

            # Feed outcome to ML
            import numpy as np
            feats = pos.get("features") or entry.get("features")
            if feats is not None:
                self.model.record_trade_result(
                    np.array(feats, dtype=np.float32),
                    result["profitable"],
                )

        self.risk_mgr.remove_position(symbol)
        self.active.pop(symbol, None)

        pnl_str = f"₹{result['net_pnl']:+.2f}" if result else "?"
        logger.info(f"EXITED   {symbol} @ ₹{exit_premium:.2f} | {reason} | pnl={pnl_str}")

    # ─── EOD square-off ───────────────────────────────────────────────────────

    def eod_squareoff(self):
        logger.info("─── EOD SQUARE-OFF ───")
        for symbol in list(self.active.keys()):
            pos = self.active[symbol]["pos"]
            ltp = self.fetcher.get_option_ltp(symbol, pos["exchange"])
            if ltp <= 0:
                ltp = pos["entry_premium"]   # fallback
            self._exit(symbol, ltp, "eod")

        if not self.paper:
            self.broker.square_off_all_options("NFO")
            self.broker.square_off_all_options("BFO")

        try:
            cap = self.broker.get_available_capital()
        except Exception:
            cap = self.risk_mgr.current_capital

        tl.save_daily_summary(self.capital_at_open, cap)
        stats = tl.get_performance_summary()

        logger.info(f"Day PnL  : ₹{self.risk_mgr.daily_pnl:+,.2f}")
        logger.info(f"Capital  : ₹{self.risk_mgr.current_capital:,.2f}")
        logger.info(f"All-time : {stats}")
        logger.info(f"ML       : {self.model.summary()}")
        self.squaredoff = True

    # ─── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        self.startup()

        while self.running:
            if not is_market_open():
                logger.debug("Market closed — waiting ...")
                self.squaredoff = False     # reset for next day
                self.risk_mgr.reset_daily()
                time.sleep(60)
                continue

            if squareoff_due() and not self.squaredoff:
                self.eod_squareoff()
                time.sleep(60)
                continue

            if self.squaredoff:
                time.sleep(60)
                continue

            if self.risk_mgr.is_daily_loss_breached():
                logger.warning("Daily loss limit hit — sitting out rest of day.")
                time.sleep(300)
                continue

            # Process each underlying
            for cfg in OPTION_UNDERLYINGS:
                if not self.running:
                    break
                self.process_underlying(cfg)

            # Heartbeat
            st = self.risk_mgr.status()
            open_syms = list(self.active.keys())
            logger.info(
                f"[SCAN {now_hhmm()}] cap=₹{st['capital']:,.0f} | "
                f"pnl={st['daily_pnl']:+.0f} | open={open_syms or 'none'} | "
                f"dd={st['drawdown_pct']:.1f}%"
            )

            time.sleep(SCAN_INTERVAL_SECS)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Options Bot — Zerodha Kite")
    parser.add_argument("--paper", action="store_true",
                        help="Paper trade mode — no real orders placed")
    args = parser.parse_args()

    bot = OptionsBot(paper_trade=args.paper)
    bot.run()
