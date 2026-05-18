"""
Options contract discovery and management.

Responsibilities:
  - Find the right CE/PE contract (symbol, expiry, strike) for a given signal.
  - Fetch the option chain to pick the most liquid strike.
  - Track Greeks proxies (moneyness, days-to-expiry).
  - Detect and handle expiry-day rules.
"""

import logging
import math
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
from kiteconnect import KiteConnect

from config import (
    OPTION_UNDERLYINGS, STRIKE_OFFSET_WEAK, STRIKE_OFFSET_STRONG,
    STRONG_SIGNAL_SCORE, SKIP_EXPIRY_DAYS_BEFORE,
)

logger = logging.getLogger(__name__)

# Map underlying name → config dict
UNDERLYING_MAP = {u["name"]: u for u in OPTION_UNDERLYINGS}


class OptionsManager:
    def __init__(self, kite: KiteConnect):
        self.kite = kite
        # Cache: exchange → list of instrument dicts for NFO/BFO
        self._instruments: dict[str, list] = {}

    # ─── Instrument List ──────────────────────────────────────────────────────

    def _load_instruments(self, exchange: str):
        if exchange not in self._instruments:
            logger.info(f"Loading {exchange} instruments ...")
            self._instruments[exchange] = self.kite.instruments(exchange)
            logger.info(f"  {len(self._instruments[exchange])} instruments loaded.")
        return self._instruments[exchange]

    def refresh_instruments(self):
        """Call once per day at startup to get fresh contract list."""
        for cfg in OPTION_UNDERLYINGS:
            self._instruments.pop(cfg["exchange"], None)
            self._load_instruments(cfg["exchange"])

    # ─── Spot Price ───────────────────────────────────────────────────────────

    def get_spot(self, underlying_name: str) -> float:
        cfg   = UNDERLYING_MAP[underlying_name]
        index = cfg["index"]
        try:
            data = self.kite.ltp([index])
            return float(data[index]["last_price"])
        except Exception as e:
            logger.error(f"Spot fetch failed for {underlying_name}: {e}")
            return 0.0

    # ─── Strike Rounding ──────────────────────────────────────────────────────

    @staticmethod
    def nearest_strike(spot: float, step: int, offset: int, direction: str) -> int:
        """
        Round spot to nearest strike, then offset by `offset` strikes
        in the OTM direction.
          direction = 'CE' → OTM is above spot
          direction = 'PE' → OTM is below spot
        """
        atm = round(spot / step) * step
        if direction == "CE":
            return int(atm + offset * step)
        else:
            return int(atm - offset * step)

    # ─── Expiry Selection ─────────────────────────────────────────────────────

    def get_expiries(self, underlying_name: str, exchange: str) -> list[date]:
        """Return sorted list of upcoming expiry dates for this underlying."""
        instruments = self._load_instruments(exchange)
        today = date.today()
        expiries = set()
        for inst in instruments:
            if (inst.get("name") == underlying_name
                    and inst.get("instrument_type") in ("CE", "PE")):
                exp = inst.get("expiry")
                if exp and isinstance(exp, date) and exp >= today:
                    expiries.add(exp)
        return sorted(expiries)

    def select_expiry(self, underlying_name: str, exchange: str) -> Optional[date]:
        """
        Pick the nearest weekly expiry that is at least SKIP_EXPIRY_DAYS_BEFORE away.
        Falls back to next expiry if too close.
        """
        expiries = self.get_expiries(underlying_name, exchange)
        today    = date.today()
        cutoff   = today + timedelta(days=SKIP_EXPIRY_DAYS_BEFORE)

        for exp in expiries:
            if exp > cutoff:
                return exp

        return expiries[0] if expiries else None

    # ─── Option Symbol Lookup ─────────────────────────────────────────────────

    def find_option(self, underlying_name: str, expiry: date,
                    strike: int, option_type: str) -> Optional[dict]:
        """
        Returns the instrument dict for a specific option contract.
        option_type: 'CE' or 'PE'
        """
        cfg          = UNDERLYING_MAP[underlying_name]
        exchange     = cfg["exchange"]
        instruments  = self._load_instruments(exchange)

        for inst in instruments:
            if (inst.get("name")            == underlying_name
                    and inst.get("expiry")       == expiry
                    and inst.get("strike")        == strike
                    and inst.get("instrument_type") == option_type):
                return inst

        logger.warning(f"Option not found: {underlying_name} {expiry} {strike}{option_type}")
        return None

    # ─── Premium Fetch ────────────────────────────────────────────────────────

    def get_premium(self, tradingsymbol: str, exchange: str) -> float:
        """Fetch last traded price of an option contract."""
        key = f"{exchange}:{tradingsymbol}"
        try:
            data = self.kite.ltp([key])
            return float(data[key]["last_price"])
        except Exception as e:
            logger.error(f"Premium fetch failed for {key}: {e}")
            return 0.0

    def get_option_quote(self, tradingsymbol: str, exchange: str) -> dict:
        """Full quote including OI, volume, bid/ask."""
        key = f"{exchange}:{tradingsymbol}"
        try:
            return self.kite.quote([key]).get(key, {})
        except Exception as e:
            logger.error(f"Quote failed for {key}: {e}")
            return {}

    # ─── Main Selection Function ──────────────────────────────────────────────

    def select_option_contract(
        self,
        underlying_name: str,
        signal_action: str,          # 'BUY' or 'SELL' (direction of underlying)
        signal_score:  float,
        capital:       float,
    ) -> Optional[dict]:
        """
        Given a directional signal on the underlying, find the best option to buy.

        Returns a dict with:
          tradingsymbol, exchange, instrument_token, lot_size,
          strike, expiry, option_type, premium, lot_cost,
          max_lots, underlying_name, spot_price
        """
        cfg          = UNDERLYING_MAP[underlying_name]
        exchange     = cfg["exchange"]
        lot_size     = cfg["lot_size"]
        strike_step  = cfg["strike_step"]

        option_type  = "CE" if signal_action == "BUY" else "PE"

        # 1. Spot price
        spot = self.get_spot(underlying_name)
        if spot <= 0:
            return None

        # 2. Expiry
        expiry = self.select_expiry(underlying_name, exchange)
        if not expiry:
            logger.warning(f"No valid expiry for {underlying_name}")
            return None

        days_to_expiry = (expiry - date.today()).days
        logger.info(f"{underlying_name} spot={spot:.0f} expiry={expiry} "
                    f"DTE={days_to_expiry} type={option_type}")

        # 3. Strike offset — use ATM for strong signals, 1-OTM for weak
        offset = (STRIKE_OFFSET_STRONG if signal_score >= STRONG_SIGNAL_SCORE
                  else STRIKE_OFFSET_WEAK)

        # Try ATM first, fallback to ±1 strike if not found
        for try_offset in [offset, offset + 1, max(0, offset - 1)]:
            strike = self.nearest_strike(spot, strike_step, try_offset, option_type)
            inst   = self.find_option(underlying_name, expiry, strike, option_type)
            if inst:
                break
        else:
            logger.warning(f"Could not find a valid strike for {underlying_name}")
            return None

        # 4. Premium
        premium = self.get_premium(inst["tradingsymbol"], exchange)
        if premium <= 0:
            premium = self.get_option_quote(inst["tradingsymbol"], exchange).get(
                "last_price", 0
            )
        if premium <= 0:
            logger.warning(f"Zero premium for {inst['tradingsymbol']}")
            return None

        # 5. Lot sizing — how many lots can we afford?
        from config import MAX_CAPITAL_PER_TRADE_PCT
        max_spend = capital * MAX_CAPITAL_PER_TRADE_PCT / 100
        lot_cost  = premium * lot_size
        max_lots  = max(1, int(max_spend / lot_cost))

        # Hard cap: never spend more than 30% of capital on one trade
        if lot_cost * max_lots > capital * 0.30:
            max_lots = max(1, int(capital * 0.30 / lot_cost))

        logger.info(f"Selected: {inst['tradingsymbol']} prem=₹{premium:.2f} "
                    f"lot_cost=₹{lot_cost:.0f} max_lots={max_lots} DTE={days_to_expiry}")

        return {
            "tradingsymbol":    inst["tradingsymbol"],
            "exchange":         exchange,
            "instrument_token": inst["instrument_token"],
            "lot_size":         lot_size,
            "strike":           strike,
            "expiry":           expiry,
            "option_type":      option_type,
            "premium":          premium,
            "lot_cost":         lot_cost,
            "max_lots":         max_lots,
            "underlying_name":  underlying_name,
            "spot_price":       spot,
            "days_to_expiry":   days_to_expiry,
            "signal_action":    signal_action,
        }

    # ─── Moneyness Helper ─────────────────────────────────────────────────────

    @staticmethod
    def moneyness(spot: float, strike: int, option_type: str) -> str:
        diff_pct = (spot - strike) / spot * 100
        if option_type == "CE":
            if diff_pct > 0.5:  return "ITM"
            if diff_pct < -0.5: return "OTM"
            return "ATM"
        else:
            if diff_pct < -0.5: return "ITM"
            if diff_pct > 0.5:  return "OTM"
            return "ATM"
