"""
PureAlpha Bot — Options Contract Management
============================================
Two implementations:

PaperOptionsManager  — no Kite needed
    Synthetic contracts + Black-Scholes premium estimates.
    Used automatically when running --paper mode.

OptionsManager       — requires live Kite session
    Uses real NFO instrument list + live LTP from Kite.
    Used in live trading mode.
"""

import logging
import math
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd

from config import (
    OPTION_UNDERLYINGS, STRIKE_OFFSET_WEAK, STRIKE_OFFSET_STRONG,
    STRONG_SIGNAL_SCORE, SKIP_EXPIRY_DAYS_BEFORE, MAX_CAPITAL_PER_TRADE_PCT,
)
from data_fetcher import get_bs_premium

logger = logging.getLogger(__name__)

# Map underlying name → config dict
UNDERLYING_MAP = {u["name"]: u for u in OPTION_UNDERLYINGS}


# ─── Paper Options Manager (no Kite needed) ────────────────────────────────────

class PaperOptionsManager:
    """
    Synthetic option contract selection for paper trading.
    Generates standard NSE weekly option tradingsymbols and prices them
    via Black-Scholes using the current spot fetched from YFinance.

    Fully offline — zero Kite dependency.
    """

    def __init__(self, fetcher=None):
        self.fetcher = fetcher   # DataFetcherYF instance

    def refresh_instruments(self):
        pass   # no instrument list needed

    def _get_spot(self, underlying_name: str) -> float:
        cfg    = UNDERLYING_MAP[underlying_name]
        symbol = cfg["index"]
        if self.fetcher and hasattr(self.fetcher, "get_current_spot"):
            return self.fetcher.get_current_spot(symbol)
        return 0.0

    @staticmethod
    def nearest_strike(spot: float, step: int, offset: int,
                       option_type: str) -> int:
        atm = round(spot / step) * step
        if option_type == "CE":
            return int(atm + offset * step)
        return int(atm - offset * step)

    def _next_expiry(self) -> date:
        """Next weekly Thursday expiry at least SKIP_EXPIRY_DAYS_BEFORE away."""
        today  = date.today()
        cutoff = today + timedelta(days=SKIP_EXPIRY_DAYS_BEFORE)
        d      = cutoff
        while d.weekday() != 3:   # Thursday = 3
            d += timedelta(days=1)
        return d

    @staticmethod
    def _make_symbol(underlying: str, expiry: date,
                     strike: int, option_type: str) -> str:
        """NSE format: NIFTY25JUN23500CE"""
        MONTHS = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
                  7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}
        yr  = str(expiry.year)[2:]
        mon = MONTHS[expiry.month]
        return f"{underlying}{yr}{mon}{strike}{option_type}"

    def select_option_contract(
        self,
        underlying_name: str,
        signal_action: str,
        signal_score: float,
        capital: float,
    ) -> Optional[dict]:
        cfg         = UNDERLYING_MAP[underlying_name]
        lot_size    = cfg["lot_size"]
        strike_step = cfg["strike_step"]
        exchange    = cfg["exchange"]
        option_type = "CE" if signal_action == "BUY" else "PE"

        spot = self._get_spot(underlying_name)
        if spot <= 0:
            logger.warning(f"[PAPER] {underlying_name}: spot=0 — skip")
            return None

        expiry         = self._next_expiry()
        days_to_expiry = max((expiry - date.today()).days, 1)
        offset         = (STRIKE_OFFSET_STRONG if signal_score >= STRONG_SIGNAL_SCORE
                          else STRIKE_OFFSET_WEAK)
        strike         = self.nearest_strike(spot, strike_step, offset, option_type)

        premium  = get_bs_premium(underlying_name, spot, strike,
                                  option_type, days_to_expiry)
        if premium <= 0:
            return None

        symbol   = self._make_symbol(underlying_name, expiry, strike, option_type)
        lot_cost = premium * lot_size
        max_lots = max(1, int(capital * MAX_CAPITAL_PER_TRADE_PCT / 100 / lot_cost))

        logger.info(
            f"[PAPER] {symbol} spot={spot:.0f} K={strike} "
            f"prem=₹{premium:.2f} DTE={days_to_expiry} (BS)"
        )

        return {
            "tradingsymbol":    symbol,
            "exchange":         exchange,
            "instrument_token": 0,
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
            "entry_atr":        0,
        }


class OptionsManager:
    def __init__(self, kite):
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
