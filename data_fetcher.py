"""
PureAlpha Bot — Market Data
============================
Primary  : Yahoo Finance (yfinance) — free, no token, auto-starts without login
Fallback : Zerodha Kite — used for option LTP + live orders only

Paper mode  → 100% YFinance, no Kite needed at all
Live mode   → YFinance for OHLCV candles, Kite for option LTP + order placement

Black-Scholes pricing is used in paper mode to estimate option premiums
so the entire paper-trading loop runs without any Kite dependency.
"""

import logging
import math
import pandas as pd
from datetime import datetime, timedelta

from config import (
    LIVE_INTERVAL, HTF_INTERVAL, HISTORICAL_DAYS, OPTION_UNDERLYINGS,
    USE_YFINANCE_PRIMARY, YFINANCE_SYMBOLS,
    BS_IV_NIFTY, BS_IV_BANKNIFTY, BS_RISK_FREE,
)

logger = logging.getLogger(__name__)


# ─── Black-Scholes (paper-mode option pricing) ───────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_option_price(S: float, K: float, T: float, r: float,
                    sigma: float, option_type: str = "CE") -> float:
    """Standard Black-Scholes price. S=spot K=strike T=years r=rate sigma=IV."""
    if T <= 0:
        return max(S - K, 0.0) if option_type == "CE" else max(K - S, 0.0)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "CE":
            return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    except Exception:
        return max(S - K, 0.0) if option_type == "CE" else max(K - S, 0.0)


def get_bs_premium(underlying: str, spot: float, strike: float,
                   option_type: str, days_to_expiry: int) -> float:
    """Approximate option LTP via BS — used in paper mode instead of Kite LTP."""
    T     = max(days_to_expiry, 0.5) / 365.0
    sigma = BS_IV_BANKNIFTY if "BANK" in underlying.upper() else BS_IV_NIFTY
    price = bs_option_price(spot, strike, T, BS_RISK_FREE, sigma, option_type)
    return max(round(price, 2), 5.0)   # floor at ₹5 — avoid penny premiums


# ─── YFinance data fetcher ────────────────────────────────────────────────────

class DataFetcherYF:
    """
    Full data source using Yahoo Finance only.
    No Kite token required — allows the bot to auto-start at 09:15 without
    the user manually running the Kite login flow.

    Used directly in paper mode.
    Used as the candle source in live mode (Kite only for option LTP/orders).
    """

    # YF interval strings for each bot interval config
    _YF_INTERVAL_MAP = {
        "5minute":  "5m",
        "15minute": "15m",
        "30minute": "30m",
        "60minute": "60m",
        "day":      "1d",
    }

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}   # keyed by "symbol|interval"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _yf_sym(self, index_symbol: str) -> str:
        sym = YFINANCE_SYMBOLS.get(index_symbol)
        if not sym:
            raise ValueError(f"No yfinance mapping for {index_symbol!r} — "
                             f"add it to YFINANCE_SYMBOLS in config.py")
        return sym

    @staticmethod
    def _clean(raw: pd.DataFrame) -> pd.DataFrame:
        """Standardise a yfinance DataFrame → our OHLCV schema."""
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        # Convert timezone to IST and strip tz info
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        # Flatten multi-level columns that newer yfinance versions produce
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        # Keep only OHLCV
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[cols].astype(float)
        df.index.name = "timestamp"
        # Trim to NSE market hours
        try:
            df = df.between_time("09:15", "15:30")
        except Exception:
            pass
        return DataFetcherYF._fix_volume(df)

    @staticmethod
    def _fix_volume(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing / zero volume with a volatility proxy.
        NSE spot indices often have zero volume in the YF feed.
        """
        if df.empty or "volume" not in df.columns:
            return df
        if df["volume"].sum() == 0 or (df["volume"] == 0).mean() > 0.5:
            proxy = df["close"].pct_change().abs().fillna(0) * 1_000_000
            median_nz = proxy[proxy > 0].median() if (proxy > 0).any() else 1.0
            df = df.copy()
            df["volume"] = proxy.replace(0, median_nz)
        return df

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def fetch_history(self, symbol: str, interval: str = LIVE_INTERVAL,
                      days: int = HISTORICAL_DAYS) -> pd.DataFrame:
        import yfinance as yf
        yf_sym   = self._yf_sym(symbol)
        yf_int   = self._YF_INTERVAL_MAP.get(interval, "5m")
        # yfinance only keeps 60 days of intraday data
        period   = f"{min(days, 59)}d"
        raw      = yf.Ticker(yf_sym).history(period=period, interval=yf_int,
                                              auto_adjust=True)
        df       = self._clean(raw)
        cache_key = f"{symbol}|{interval}"
        self._cache[cache_key] = df
        logger.info(f"YF history: {symbol} ({yf_int}) → {len(df)} candles")
        return df

    def refresh(self, symbol: str, interval: str = LIVE_INTERVAL) -> pd.DataFrame:
        """Fetch today's candles and merge with cached history."""
        import yfinance as yf
        yf_sym    = self._yf_sym(symbol)
        yf_int    = self._YF_INTERVAL_MAP.get(interval, "5m")
        cache_key = f"{symbol}|{interval}"

        raw   = yf.Ticker(yf_sym).history(period="2d", interval=yf_int,
                                           auto_adjust=True)
        today = self._clean(raw)

        cached = self._cache.get(cache_key)
        if cached is None or cached.empty:
            full = self.fetch_history(symbol, interval)
        else:
            full = pd.concat([cached, today])

        full = self._fix_volume(full)
        full = full[~full.index.duplicated(keep="last")].sort_index()
        self._cache[cache_key] = full
        return full

    def refresh_htf(self, symbol: str) -> pd.DataFrame:
        """15-minute candles for HTF alignment voter."""
        return self.refresh(symbol, interval=HTF_INTERVAL)

    # ── Spot price ────────────────────────────────────────────────────────────

    def get_current_spot(self, symbol: str) -> float:
        """Latest spot price from YF (1-min bar). Used for paper LTP calc."""
        try:
            import yfinance as yf
            yf_sym = self._yf_sym(symbol)
            df     = yf.Ticker(yf_sym).history(period="1d", interval="1m",
                                                auto_adjust=True)
            if not df.empty:
                close_col = "Close" if "Close" in df.columns else "close"
                return float(df[close_col].iloc[-1])
        except Exception as e:
            logger.warning(f"YF spot {symbol}: {e}")
        # Fallback: last cached candle
        for interval in (LIVE_INTERVAL, HTF_INTERVAL):
            cached = self._cache.get(f"{symbol}|{interval}")
            if cached is not None and not cached.empty:
                return float(cached["close"].iloc[-1])
        return 0.0

    # ── Option pricing (paper mode) ───────────────────────────────────────────

    def get_option_ltp(self, tradingsymbol: str, exchange: str,
                       spot: float = 0, strike: float = 0,
                       option_type: str = "CE", days_to_expiry: int = 7,
                       underlying: str = "NIFTY") -> float:
        """
        Paper mode: Black-Scholes estimate of option premium.
        Call signature is compatible with DataFetcher.get_option_ltp so
        bot.py can call either without branching.
        """
        if spot <= 0:
            return 0.0
        return get_bs_premium(underlying, spot, strike, option_type, days_to_expiry)

    def get_option_quote(self, *args, **kwargs) -> dict:
        return {}

    # ── VIX ───────────────────────────────────────────────────────────────────

    def get_vix(self) -> float:
        """Fetch India VIX from Yahoo Finance."""
        try:
            import yfinance as yf
            df = yf.Ticker("^INDIAVIX").history(period="1d", interval="1m",
                                                  auto_adjust=True)
            if not df.empty:
                close_col = "Close" if "Close" in df.columns else "close"
                return float(df[close_col].iloc[-1])
        except Exception as e:
            logger.debug(f"YF VIX: {e}")
        return 0.0

    # ── Warmup ────────────────────────────────────────────────────────────────

    def warmup(self):
        for cfg in OPTION_UNDERLYINGS:
            try:
                df = self.fetch_history(cfg["index"])
                logger.info(f"YF warmup: {cfg['name']} → {len(df)} candles")
            except Exception as e:
                logger.warning(f"YF warmup failed {cfg['index']}: {e}")


# ─── Kite + YF hybrid fetcher (live mode) ────────────────────────────────────

class DataFetcher:
    """
    Live-mode fetcher:
    • OHLCV candles   → YFinance (primary) with Kite as fallback
    • Option LTP      → Kite (only Kite has real option chain prices)
    • Order placement → Kite

    kite=None is allowed (e.g. when running --paper and Kite is unavailable).
    """

    def __init__(self, kite=None):
        self.kite          = kite
        self._yf           = DataFetcherYF()
        self._token_cache: dict[str, int] = {}

    # ── Token (Kite only) ─────────────────────────────────────────────────────

    def _get_token(self, symbol: str) -> int:
        if symbol in self._token_cache:
            return self._token_cache[symbol]
        exchange, name = symbol.split(":", 1)
        instruments    = self.kite.instruments(exchange)
        for inst in instruments:
            if inst["tradingsymbol"] == name or inst["name"] == name:
                self._token_cache[symbol] = inst["instrument_token"]
                return inst["instrument_token"]
        raise ValueError(f"Token not found for {symbol}")

    # ── OHLCV — YF primary, Kite fallback ─────────────────────────────────────

    def fetch_history(self, symbol: str, interval: str = LIVE_INTERVAL,
                      days: int = HISTORICAL_DAYS) -> pd.DataFrame:
        if USE_YFINANCE_PRIMARY:
            try:
                return self._yf.fetch_history(symbol, interval, days)
            except Exception as e:
                logger.warning(f"YF history failed ({e}), trying Kite...")

        if self.kite is None:
            logger.error("No Kite client and YF also failed — returning empty")
            return pd.DataFrame()

        return self._kite_history(symbol, interval, days)

    def _kite_history(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        token   = self._get_token(symbol)
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=days)
        raw     = self.kite.historical_data(
            instrument_token = token,
            from_date        = from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            to_date          = to_dt.strftime("%Y-%m-%d %H:%M:%S"),
            interval         = interval,
            continuous       = False,
            oi               = False,
        )
        df = pd.DataFrame(raw)
        if df.empty:
            return df
        df.rename(columns={"date": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = DataFetcherYF._fix_volume(df)
        logger.info(f"Kite history: {symbol} — {len(df)} candles")
        return df

    def refresh(self, symbol: str, interval: str = LIVE_INTERVAL) -> pd.DataFrame:
        """Always prefer YF refresh — no token cost, faster."""
        try:
            return self._yf.refresh(symbol, interval)
        except Exception as e:
            logger.warning(f"YF refresh failed ({e}), trying Kite...")
            if self.kite:
                return self._kite_history(symbol, interval, days=1)
            return pd.DataFrame()

    def refresh_htf(self, symbol: str) -> pd.DataFrame:
        """15-minute HTF candles via YFinance."""
        return self._yf.refresh_htf(symbol)

    def get_current_spot(self, symbol: str) -> float:
        return self._yf.get_current_spot(symbol)

    # ── Option LTP — Kite only (real prices) ─────────────────────────────────

    def get_option_ltp(self, tradingsymbol: str, exchange: str) -> float:
        if self.kite is None:
            return 0.0
        key = f"{exchange}:{tradingsymbol}"
        try:
            data = self.kite.ltp([key])
            return float(data[key]["last_price"])
        except Exception as e:
            logger.warning(f"Kite option LTP error {key}: {e}")
            return 0.0

    def get_option_quote(self, tradingsymbol: str, exchange: str) -> dict:
        if self.kite is None:
            return {}
        key = f"{exchange}:{tradingsymbol}"
        try:
            return self.kite.quote([key]).get(key, {})
        except Exception as e:
            logger.warning(f"Option quote error {key}: {e}")
            return {}

    def get_vix(self) -> float:
        """India VIX via YFinance (no Kite needed)."""
        return self._yf.get_vix()

    # ── Warmup ────────────────────────────────────────────────────────────────

    def warmup(self):
        for cfg in OPTION_UNDERLYINGS:
            try:
                df = self.fetch_history(cfg["index"])
                logger.info(f"Warmup: {cfg['name']} — {len(df)} candles")
            except Exception as e:
                logger.warning(f"Warmup failed {cfg['index']}: {e}")
