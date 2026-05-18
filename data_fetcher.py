"""
Market data for OPTIONS trading.

Fetches candles on the UNDERLYING INDEX (Nifty spot / BankNifty spot) —
that's what we run strategies on.
Also provides live option premium via ltp().
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from config import LIVE_INTERVAL, HISTORICAL_DAYS, OPTION_UNDERLYINGS

logger = logging.getLogger(__name__)

# Map index feed symbol → instrument token (cached)
INDEX_TOKENS = {}


class DataFetcher:
    def __init__(self, kite: KiteConnect):
        self.kite   = kite
        self._cache = {}   # index_symbol → DataFrame of OHLCV candles

    # ─── Token helpers ────────────────────────────────────────────────────────

    def _get_token(self, symbol: str) -> int:
        """
        symbol e.g. 'NSE:NIFTY 50' or 'NSE:NIFTY BANK'.
        Kite provides index tokens via instruments('NSE').
        """
        if symbol in INDEX_TOKENS:
            return INDEX_TOKENS[symbol]

        exchange, name = symbol.split(":", 1)
        instruments = self.kite.instruments(exchange)
        for inst in instruments:
            if inst["tradingsymbol"] == name or inst["name"] == name:
                INDEX_TOKENS[symbol] = inst["instrument_token"]
                logger.debug(f"Token for {symbol}: {inst['instrument_token']}")
                return inst["instrument_token"]

        raise ValueError(f"Token not found for {symbol}")

    # ─── Historical OHLCV ─────────────────────────────────────────────────────

    def fetch_history(self, symbol: str, interval: str = LIVE_INTERVAL,
                      days: int = HISTORICAL_DAYS) -> pd.DataFrame:
        token   = self._get_token(symbol)
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=days)

        raw = self.kite.historical_data(
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
        df = self._fix_index_volume(df)
        self._cache[symbol] = df
        logger.info(f"History loaded: {symbol} {len(df)} candles ({interval})")
        return df

    # ─── Today's session ──────────────────────────────────────────────────────

    def fetch_today(self, symbol: str, interval: str = LIVE_INTERVAL) -> pd.DataFrame:
        token = self._get_token(symbol)
        today = datetime.now().strftime("%Y-%m-%d")

        raw = self.kite.historical_data(
            instrument_token = token,
            from_date        = f"{today} 09:15:00",
            to_date          = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            interval         = interval,
            continuous       = False,
        )
        df = pd.DataFrame(raw)
        if df.empty:
            return df

        df.rename(columns={"date": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return self._fix_index_volume(df)

    # ─── Volume proxy for spot indices (Kite returns volume=0 for indices) ────

    @staticmethod
    def _fix_index_volume(df: pd.DataFrame) -> pd.DataFrame:
        """Replace zero volume with volatility proxy so vol_ratio works."""
        if df["volume"].sum() == 0:
            proxy = df["close"].pct_change().abs().fillna(0) * 1_000_000
            median_nz = proxy[proxy > 0].median() if (proxy > 0).any() else 1.0
            df = df.copy()
            df["volume"] = proxy.replace(0, median_nz)
        return df

    # ─── Refresh (incremental) ────────────────────────────────────────────────

    def refresh(self, symbol: str, interval: str = LIVE_INTERVAL) -> pd.DataFrame:
        today_df = self.fetch_today(symbol, interval)
        cached   = self._cache.get(symbol)

        if cached is None or cached.empty:
            full = self.fetch_history(symbol, interval)
        else:
            full = pd.concat([cached, today_df])

        full = full[~full.index.duplicated(keep="last")].sort_index()
        self._cache[symbol] = full
        return full

    # ─── Option premium (LTP) ─────────────────────────────────────────────────

    def get_option_ltp(self, tradingsymbol: str, exchange: str) -> float:
        key = f"{exchange}:{tradingsymbol}"
        try:
            data = self.kite.ltp([key])
            return float(data[key]["last_price"])
        except Exception as e:
            logger.warning(f"Option LTP error {key}: {e}")
            return 0.0

    def get_option_quote(self, tradingsymbol: str, exchange: str) -> dict:
        key = f"{exchange}:{tradingsymbol}"
        try:
            return self.kite.quote([key]).get(key, {})
        except Exception as e:
            logger.warning(f"Option quote error {key}: {e}")
            return {}

    # ─── Warmup all underlyings ───────────────────────────────────────────────

    def warmup(self):
        for cfg in OPTION_UNDERLYINGS:
            try:
                df = self.fetch_history(cfg["index"])
                logger.info(f"Warmup: {cfg['index']} — {len(df)} candles")
            except Exception as e:
                logger.warning(f"Warmup failed for {cfg['index']}: {e}")
