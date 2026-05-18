"""
Options-specific risk management.

For option BUYS:
  - Max loss is defined: the premium paid (no stop needed at broker level).
  - We track premium in real-time and exit if it drops ≥ SL_MULT of paid premium.
  - Trail: once premium doubles TRAIL_START, lock in TRAIL_LOCK gain.
"""

import logging
from config import (
    MAX_CAPITAL_PER_TRADE_PCT, MAX_OPEN_POSITIONS,
    MAX_DAILY_LOSS_PCT, BROKERAGE_PER_ORDER,
    OPTION_TARGET_MULT, OPTION_SL_MULT,
    OPTION_TRAIL_START, OPTION_TRAIL_LOCK,
    INITIAL_CAPITAL,
)

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self):
        self.current_capital   = INITIAL_CAPITAL
        self.peak_capital      = INITIAL_CAPITAL
        self.daily_pnl         = 0.0
        self.open_positions    = {}    # symbol → position dict
        self.trade_count_today = 0

    # ─── Capital ──────────────────────────────────────────────────────────────

    def update_capital(self, capital: float):
        self.current_capital = capital
        if capital > self.peak_capital:
            self.peak_capital = capital

    def record_pnl(self, pnl: float):
        self.daily_pnl       += pnl
        self.current_capital += pnl
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

    def reset_daily(self):
        self.daily_pnl         = 0.0
        self.trade_count_today = 0

    # ─── Guards ───────────────────────────────────────────────────────────────

    def is_daily_loss_breached(self) -> bool:
        limit = -self.current_capital * MAX_DAILY_LOSS_PCT / 100
        if self.daily_pnl <= limit:
            logger.warning(f"Daily loss limit hit: ₹{self.daily_pnl:.2f}")
            return True
        return False

    def can_take_new_trade(self) -> tuple[bool, str]:
        if self.is_daily_loss_breached():
            return False, "daily loss limit hit"
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return False, f"max {MAX_OPEN_POSITIONS} open positions reached"
        return True, "ok"

    # ─── Options Position Sizing ──────────────────────────────────────────────

    def calculate_option_position(self, contract: dict) -> dict:
        """
        contract: output from OptionsManager.select_option_contract()
        Returns a full position dict ready for execution.
        """
        premium   = contract["premium"]
        lot_size  = contract["lot_size"]
        lot_cost  = premium * lot_size
        max_lots  = contract["max_lots"]

        # How much capital are we willing to deploy?
        budget    = self.current_capital * MAX_CAPITAL_PER_TRADE_PCT / 100
        lots      = min(max_lots, max(1, int(budget / lot_cost)))

        total_cost = lots * lot_cost
        charges    = BROKERAGE_PER_ORDER * 2    # entry + exit

        # Exit levels based on premium paid
        sl_premium     = round(premium * (1 - OPTION_SL_MULT),   2)
        target_premium = round(premium * (1 + OPTION_TARGET_MULT), 2)
        trail_trigger  = round(premium * (1 + OPTION_TRAIL_START), 2)
        trail_lock     = round(premium * (1 + OPTION_TRAIL_LOCK),  2)

        return {
            "tradingsymbol":    contract["tradingsymbol"],
            "exchange":         contract["exchange"],
            "instrument_token": contract["instrument_token"],
            "underlying":       contract["underlying_name"],
            "option_type":      contract["option_type"],
            "strike":           contract["strike"],
            "expiry":           str(contract["expiry"]),
            "lot_size":         lot_size,
            "lots":             lots,
            "qty":              lots * lot_size,
            "entry_premium":    premium,
            "sl_premium":       sl_premium,
            "target_premium":   target_premium,
            "trail_trigger":    trail_trigger,
            "trail_lock":       trail_lock,
            "trailing_sl":      None,
            "total_cost":       round(total_cost, 2),
            "charges":          charges,
            "max_loss":         round(total_cost + charges, 2),
            "signal_action":    contract["signal_action"],
            "spot_at_entry":    contract["spot_price"],
            "days_to_expiry":   contract["days_to_expiry"],
        }

    # ─── Options Exit Logic ───────────────────────────────────────────────────

    def update_option_trail(self, pos: dict, current_premium: float) -> dict:
        """Activate and ratchet trailing stop once premium crosses trail_trigger."""
        if current_premium >= pos["trail_trigger"]:
            new_trail = max(pos["trail_lock"], current_premium * (1 - OPTION_TRAIL_LOCK))
            if pos["trailing_sl"] is None:
                pos["trailing_sl"] = new_trail
                logger.info(f"Trail activated for {pos['tradingsymbol']} @ ₹{current_premium:.2f}")
            else:
                pos["trailing_sl"] = max(pos["trailing_sl"], new_trail)
        return pos

    def should_exit_option(self, pos: dict, current_premium: float) -> tuple[bool, str]:
        if current_premium <= pos["sl_premium"]:
            return True, "stop_loss"
        if pos["trailing_sl"] and current_premium <= pos["trailing_sl"]:
            return True, "trailing_stop"
        if current_premium >= pos["target_premium"]:
            return True, "target"
        return False, ""

    # ─── Position Registry ────────────────────────────────────────────────────

    def add_position(self, pos: dict):
        self.open_positions[pos["tradingsymbol"]] = pos
        self.trade_count_today += 1
        logger.info(
            f"OPTION POSITION OPENED: {pos['tradingsymbol']} "
            f"lots={pos['lots']} qty={pos['qty']} "
            f"prem=₹{pos['entry_premium']} "
            f"SL=₹{pos['sl_premium']} TGT=₹{pos['target_premium']} "
            f"max_loss=₹{pos['max_loss']:.0f}"
        )

    def remove_position(self, symbol: str):
        self.open_positions.pop(symbol, None)

    def status(self) -> dict:
        return {
            "capital":        round(self.current_capital, 2),
            "daily_pnl":      round(self.daily_pnl, 2),
            "open_positions": len(self.open_positions),
            "trade_count":    self.trade_count_today,
            "drawdown_pct":   round(
                (self.peak_capital - self.current_capital) / self.peak_capital * 100, 2
            ) if self.peak_capital else 0,
        }
