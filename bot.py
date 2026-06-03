"""
AI Options Bot — 7-Layer Architecture
======================================
Layer 1: Indicators  — SuperTrend, RSI, MACD, Volume, ATR, ADX
Layer 2: Forecasting — Kronos (NeoQuasar DL price forecast)
Layer 3: ML Ensemble — XGBoost + RandomForest + LightGBM
Layer 4: Consensus   — 4/5 voters must agree (PDF key finding: 5/5=67% WR)
Layer 5: Mkt Filter  — ATR ratio + ADX + Volume gate
Layer 6: Sizing      — Confidence-based: 4/5=20%, 5/5=35% of capital
Layer 7: Risk        — ATR stop on underlying + ATR trailing stop

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
    INITIAL_CAPITAL, LOG_PATH, LOG_LEVEL, BROKERAGE_PER_ORDER,
    OPTION_MIDDAY_SL, MIDDAY_CUT_HOUR,
    COUNTER_SIGNAL_EXIT, COUNTER_SIGNAL_THRESH,
    USE_KRONOS_FILTER, KRONOS_LOOKBACK,
    CONSENSUS_VOTES_REQUIRED,
    STATUS_FILE,
)
from broker          import Broker
from data_fetcher    import DataFetcher
from indicators      import compute_all, extract_features
from signal_layers   import run_all_voters, VOTE_BUY, VOTE_SELL
from consensus       import run_consensus
from market_filter   import check_market_filter, filter_status
from ml_ensemble     import MLEnsemble
from options_manager import OptionsManager
from risk_manager    import RiskManager
from kronos_filter   import KronosFilter
from dashboard_server import write_status
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
        self.paper = paper_trade
        logger.info("=" * 70)
        logger.info(f"  AI OPTIONS BOT  |  7-Layer Architecture  |  paper={paper_trade}")
        logger.info(f"  {date.today()}   Capital: ₹{INITIAL_CAPITAL:,.0f} → Target: ₹10,00,000")
        logger.info("=" * 70)
        logger.info("  L1:Indicators  L2:Kronos  L3:ML-Ensemble  L4:Consensus(4/5)")
        logger.info("  L5:MktFilter   L6:Sizing  L7:ATR-Risk")
        logger.info("=" * 70)

        tl.init_db()

        self.broker   = Broker()
        self.fetcher  = DataFetcher(self.broker.kite)
        self.opts_mgr = OptionsManager(self.broker.kite)
        self.risk_mgr = RiskManager()
        self.ml       = MLEnsemble()
        self.kronos   = KronosFilter() if USE_KRONOS_FILTER else None

        self.active: dict[str, dict] = {}
        self.capital_at_open = INITIAL_CAPITAL
        self.squaredoff      = False
        self.running         = True
        self._signal_log     = []   # for dashboard
        self._last_voters    = []
        self._last_consensus = {}
        self._last_filter    = {}

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, *_):
        logger.info("Shutdown requested.")
        self.running = False

    # ─── Startup ──────────────────────────────────────────────────────────────

    def startup(self):
        logger.info("Loading instruments and historical data...")
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

    # ─── Process one underlying (7-layer pipeline) ────────────────────────────

    def process_underlying(self, cfg: dict):
        index_symbol = cfg["index"]
        name         = cfg["name"]

        # ── Layer 1: Fetch + compute indicators ───────────────────────────────
        try:
            df = self.fetcher.refresh(index_symbol)
        except Exception as e:
            logger.warning(f"{name}: data fetch error — {e}")
            return

        if len(df) < 60:
            return

        df  = compute_all(df)
        row = df.iloc[-1]
        if df.isnull().iloc[-1][["rsi", "macd_hist", "ema_fast"]].any():
            return

        # ── Layer 5: Market filter (early exit to avoid unnecessary computation)
        flt = filter_status(row)
        self._last_filter = flt

        # ── Determine proposed direction from strongest technical signal ───────
        # We use trend + breakout voter to get initial direction hypothesis
        from signal_layers import trend_voter, breakout_voter
        tv = trend_voter(df)
        bv = breakout_voter(df)

        proposed = None
        if tv["vote"] != "ABSTAIN":
            proposed = tv["vote"]
        elif bv["vote"] != "ABSTAIN":
            proposed = bv["vote"]

        if proposed is None:
            # No directional signal — only monitor open positions
            for sym in list(self.active.keys()):
                if self.active[sym]["pos"]["underlying"] == name:
                    self._monitor(sym, current_spot=float(row["close"]))
            return

        # ── Layer 2: Kronos forecast ──────────────────────────────────────────
        # (handled inside kronos_voter via signal_layers)

        # ── Layer 3: ML Ensemble probability ─────────────────────────────────
        extra    = {"agreement_count": 0, "kronos_score": 0.5}
        features = extract_features(row, extra=extra)
        ml_proba = self.ml.predict_success_proba(features)

        # ── Layers 1-3: Run all 5 voters ─────────────────────────────────────
        logger.info(f"[VOTE]  {name} proposed={proposed} — running 5 voters...")
        voter_results = run_all_voters(
            df              = df,
            proposed_action = proposed,
            kronos_filter   = self.kronos,
            ml_ensemble     = self.ml,
            features        = features,
        )
        self._last_voters = voter_results["voters"]

        # ── Layer 4: Consensus (need 4/5) ────────────────────────────────────
        consensus = run_consensus(voter_results, proposed)
        self._last_consensus = consensus

        # Monitor open positions (always, regardless of consensus)
        # Pass consensus as counter-signal check
        for sym in list(self.active.keys()):
            if self.active[sym]["pos"]["underlying"] == name:
                self._monitor(
                    sym,
                    current_spot = float(row["close"]),
                    counter_action = proposed if not consensus["approved"] else None,
                    counter_votes  = voter_results,
                )

        if not consensus["approved"]:
            logger.debug(f"{name}: consensus not reached ({consensus['votes_for']}/5)")
            return

        # ── Layer 5: Market filter gate ───────────────────────────────────────
        if not flt["allowed"]:
            logger.info(f"[L5 BLOCK] {name}: {flt['reason']}")
            return

        if not entry_allowed():
            logger.info(f"{name}: outside entry window ({NO_ENTRY_BEFORE}–{NO_ENTRY_AFTER})")
            return

        # Already have a position?
        if any(v["pos"]["underlying"] == name for v in self.active.values()):
            return

        can, reason = self.risk_mgr.can_take_new_trade()
        if not can:
            logger.info(f"Skipping {name}: {reason}")
            return

        # ── Layer 6: Confidence-based position sizing ─────────────────────────
        position_pct = consensus["position_pct"]
        logger.info(
            f"[CONSENSUS ✅] {name} {consensus['direction']} | "
            f"{consensus['votes_for']}/5 voters | "
            f"{'UNANIMOUS' if consensus['unanimity'] else ''} "
            f"sizing={position_pct:.0f}% | conf={consensus['confidence']:.2f}"
        )

        # Select contract (unanimity → tighter OTM strike for better delta)
        signal_score = 0.72 if consensus["unanimity"] else 0.50
        contract = self.opts_mgr.select_option_contract(
            underlying_name = name,
            signal_action   = consensus["direction"],
            signal_score    = signal_score,
            capital         = self.risk_mgr.current_capital,
        )
        if not contract:
            return

        # Inject Layer 6 position sizing + Layer 7 ATR data into contract
        contract["position_pct"] = position_pct
        contract["entry_atr"]    = float(row.get("atr", 0) or 0)

        # ── Layer 7: Calculate position with ATR-based SL ────────────────────
        pos = self.risk_mgr.calculate_option_position(contract)
        if not pos:
            return

        # Enrich features with actual agreement count for ML logging
        extra    = {"agreement_count": consensus["votes_for"], "kronos_score": 0.5}
        features = extract_features(row, extra=extra)

        self._enter(pos, features, consensus, ml_proba)

        # Log to signal log for dashboard
        self._signal_log.append({
            "time":   datetime.now().strftime("%H:%M:%S"),
            "action": consensus["direction"],
            "symbol": contract.get("tradingsymbol", name),
            "reason": f"{consensus['votes_for']}/5 votes | conf={consensus['confidence']:.2f}",
        })
        if len(self._signal_log) > 50:
            self._signal_log = self._signal_log[-50:]

    # ─── Enter trade ──────────────────────────────────────────────────────────

    def _enter(self, pos: dict, features, consensus: dict, ml_proba: float):
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
                f"SL=₹{pos['sl_premium']:.2f}  ATR_stop_spot={pos.get('atr_stop_spot','—')}"
                f"  sizing={pos.get('position_pct',0):.0f}%"
            )

        pos["entry_time"]    = datetime.now().isoformat()
        pos["signal_action"] = consensus["direction"]
        pos["votes_for"]     = consensus["votes_for"]
        pos["unanimity"]     = consensus["unanimity"]

        trade_id = tl.log_entry({
            "symbol":          symbol,
            "action":          "BUY",
            "qty":             pos["qty"],
            "entry_price":     pos["entry_premium"],
            "sl_price":        pos["sl_premium"],
            "target_price":    pos["target_premium"],
            "entry_time":      pos["entry_time"],
            "features":        features.tolist(),
            "strategy_scores": {v["name"]: v["vote"] for v in consensus.get("voters", [])},
            "ml_proba":        ml_proba,
            "votes_for":       consensus["votes_for"],
            "unanimity":       consensus["unanimity"],
        })

        self.active[symbol] = {"pos": pos, "db_id": trade_id, "features": features}
        self.risk_mgr.add_position(pos)

        logger.info(
            f"ENTERED  {symbol} | lots={pos['lots']} | prem=₹{pos['entry_premium']:.2f} "
            f"| cost=₹{pos['total_cost']:.0f} | SL_prem=₹{pos['sl_premium']:.2f} "
            f"| ATR_stop={pos.get('atr_stop_spot','—')} "
            f"| {consensus['votes_for']}/5 votes | size={pos.get('position_pct',0):.0f}%"
        )

    # ─── Monitor open option ──────────────────────────────────────────────────

    def _monitor(self, symbol: str, current_spot: float = 0,
                 counter_action: str = None, counter_votes: dict = None):
        entry = self.active.get(symbol)
        if not entry:
            return

        pos      = entry["pos"]
        exchange = pos["exchange"]

        current_premium = self.fetcher.get_option_ltp(symbol, exchange)
        if current_premium <= 0:
            return

        # Premium-based trail and SL
        pos = self.risk_mgr.update_option_trail(pos, current_premium)
        exit_, reason = self.risk_mgr.should_exit_option(pos, current_premium)

        # Layer 7: ATR-based stop on underlying spot
        if not exit_ and current_spot > 0:
            exit_, reason = self.risk_mgr.check_atr_stop(pos, current_spot)

        # Midday cut
        if (not exit_ and now_hhmm() >= MIDDAY_CUT_HOUR
                and current_premium <= pos["entry_premium"] * (1 - OPTION_MIDDAY_SL)):
            exit_, reason = True, "midday_cut"

        # Counter-signal exit (if consensus failed = opposing direction gaining votes)
        if not exit_ and COUNTER_SIGNAL_EXIT and counter_votes:
            pos_action = pos.get("signal_action", "BUY")
            opp_action = VOTE_SELL if pos_action == VOTE_BUY else VOTE_BUY
            opp_votes  = counter_votes.get(
                "sell_votes" if opp_action == VOTE_SELL else "buy_votes", 0
            )
            if opp_votes >= COUNTER_SIGNAL_THRESH * 5:  # e.g. ≥ 1.4 → 2+ opposing
                exit_  = True
                reason = f"counter_signal({opp_action} {opp_votes}/5 votes)"
                logger.info(f"[COUNTER] {symbol}: {opp_action} getting {opp_votes}/5 votes — exiting")

        pnl_pct = (current_premium / pos["entry_premium"] - 1) * 100
        logger.debug(
            f"  {symbol} prem=₹{current_premium:.2f} ({pnl_pct:+.1f}%) "
            f"spot={current_spot:.0f} atr_stop={pos.get('atr_stop_spot','—')}"
        )

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
            import numpy as np
            feats = entry.get("features")
            if feats is not None:
                self.ml.record_trade_result(
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
                ltp = pos["entry_premium"]
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
        logger.info(f"ML       : {self.ml.summary()}")
        self.squaredoff = True

    # ─── Dashboard status writer ──────────────────────────────────────────────

    def _write_dashboard(self):
        st = self.risk_mgr.status()
        perf = tl.get_performance_summary()

        open_pos = []
        for sym, entry in self.active.items():
            pos = entry["pos"]
            ltp = self.fetcher.get_option_ltp(sym, pos["exchange"])
            open_pos.append({
                "symbol":  sym,
                "entry":   pos["entry_premium"],
                "current": round(ltp, 2) if ltp > 0 else pos["entry_premium"],
                "sl":      pos["sl_premium"],
                "qty":     pos["qty"],
            })

        write_status({
            "capital":        st["capital"],
            "day_pnl":        st["daily_pnl"],
            "open_trades":    len(self.active),
            "open_positions": open_pos,
            "win_rate":       perf.get("win_rate", 0),
            "total_trades":   perf.get("total", 0),
            "wins":           perf.get("wins", 0),
            "losses":         perf.get("losses", 0),
            "signal_log":     self._signal_log[-20:],
            "last_voters":    self._last_voters,
            "last_consensus": self._last_consensus,
            "market_filter":  self._last_filter,
            "scan_time":      now_hhmm(),
        })

    # ─── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        self.startup()

        while self.running:
            if not is_market_open():
                logger.debug("Market closed — waiting...")
                self.squaredoff = False
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
                logger.warning("Daily loss limit — sitting out rest of day.")
                time.sleep(300)
                continue

            for cfg in OPTION_UNDERLYINGS:
                if not self.running:
                    break
                self.process_underlying(cfg)

            st = self.risk_mgr.status()
            logger.info(
                f"[SCAN {now_hhmm()}] cap=₹{st['capital']:,.0f} | "
                f"pnl={st['daily_pnl']:+.0f} | open={list(self.active.keys()) or 'none'} | "
                f"dd={st['drawdown_pct']:.1f}% | ml={self.ml.summary()['training_samples']} samples"
            )

            # Write to dashboard
            try:
                self._write_dashboard()
            except Exception:
                pass

            time.sleep(SCAN_INTERVAL_SECS)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Options Bot — 7-Layer Architecture")
    parser.add_argument("--paper", action="store_true",
                        help="Paper trade mode — no real orders placed")
    args = parser.parse_args()

    bot = OptionsBot(paper_trade=args.paper)
    bot.run()
