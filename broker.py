"""
Broker abstraction over KiteConnect for OPTIONS trading.
All option orders use NRML product on NFO/BFO exchange.
"""

import os
import logging
import webbrowser
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

logger = logging.getLogger(__name__)

_BASE        = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE   = os.path.join(_BASE, "data", "access_token.txt")
CREDS_FILE   = os.path.join(_BASE, "data", "credentials.txt")


def _load_creds() -> tuple[str, str]:
    """Read api_key and api_secret saved by login.py."""
    if os.path.exists(CREDS_FILE):
        with open(CREDS_FILE) as f:
            lines = f.read().strip().splitlines()
        creds = {l.split("=", 1)[0]: l.split("=", 1)[1] for l in lines if "=" in l}
        return creds.get("api_key", ""), creds.get("api_secret", "")
    return "", ""


class Broker:
    def __init__(self, api_key: str = None, api_secret: str = None,
                 access_token: str = None):
        file_key, file_secret = _load_creds()
        self.api_key    = api_key    or os.getenv("KITE_API_KEY")    or file_key
        self.api_secret = api_secret or os.getenv("KITE_API_SECRET") or file_secret
        self._access_token = access_token or os.getenv("KITE_ACCESS_TOKEN")

        if not self.api_key:
            raise ValueError(
                "KITE_API_KEY not set. Run python3 login.py first."
            )

        self.kite = KiteConnect(api_key=self.api_key)

        if self._access_token:
            self.kite.set_access_token(self._access_token)
            logger.info("Broker: access token loaded from env.")
        else:
            self._login()

    # ─── Auth ─────────────────────────────────────────────────────────────────

    def _login(self):
        saved = self._load_token()
        if saved:
            try:
                self.kite.set_access_token(saved)
                self.kite.profile()
                self._access_token = saved
                logger.info("Broker: reused saved access token.")
                return
            except Exception:
                logger.info("Saved token invalid — re-authenticating.")

        login_url = self.kite.login_url()
        print(f"\n{'='*60}")
        print("ZERODHA LOGIN REQUIRED")
        print(f"URL: {login_url}")
        print("Paste the 'request_token' from the redirect URL below.")
        print("="*60)
        try:
            webbrowser.open(login_url)
        except Exception:
            pass

        request_token = input("request_token: ").strip()
        session       = self.kite.generate_session(request_token, self.api_secret)
        token         = session["access_token"]
        self.kite.set_access_token(token)
        self._access_token = token
        self._save_token(token)
        logger.info("Broker: new session — token saved.")

    def _save_token(self, token: str):
        os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
        with open(TOKEN_FILE, "w") as f:
            f.write(token)

    def _load_token(self) -> str:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE) as f:
                return f.read().strip()
        return ""

    # ─── Account Info ─────────────────────────────────────────────────────────

    def get_available_capital(self) -> float:
        """Equity segment available cash."""
        try:
            margins = self.kite.margins(segment="equity")
            return float(margins.get("net", 0))
        except Exception as e:
            logger.error(f"Margin fetch error: {e}")
            return 0.0

    def get_positions(self) -> list:
        """All open positions (day + net)."""
        try:
            pos = self.kite.positions()
            return pos.get("net", [])
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return []

    def get_orders(self) -> list:
        try:
            return self.kite.orders()
        except Exception as e:
            logger.error(f"Orders error: {e}")
            return []

    # ─── Options Order Placement ──────────────────────────────────────────────

    def buy_option(self, tradingsymbol: str, exchange: str,
                   qty: int, tag: str = "BOT_CE/PE") -> str:
        """
        Buy (open) an option contract.
        Product = NRML for options (MIS is also allowed intraday but NRML is standard).
        Since we're buying only and will square off same day, NRML is fine.
        """
        try:
            order_id = self.kite.place_order(
                variety          = KiteConnect.VARIETY_REGULAR,
                exchange         = exchange,
                tradingsymbol    = tradingsymbol,
                transaction_type = KiteConnect.TRANSACTION_TYPE_BUY,
                quantity         = qty,
                product          = KiteConnect.PRODUCT_NRML,
                order_type       = KiteConnect.ORDER_TYPE_MARKET,
                validity         = KiteConnect.VALIDITY_DAY,
                tag              = tag[:20],
            )
            logger.info(f"BUY option: {tradingsymbol} qty={qty} → order {order_id}")
            return str(order_id)
        except KiteException as e:
            logger.error(f"Buy option failed: {tradingsymbol} — {e}")
            raise

    def sell_option(self, tradingsymbol: str, exchange: str,
                    qty: int, tag: str = "BOT_EXIT") -> str:
        """Sell (close) an option we hold."""
        try:
            order_id = self.kite.place_order(
                variety          = KiteConnect.VARIETY_REGULAR,
                exchange         = exchange,
                tradingsymbol    = tradingsymbol,
                transaction_type = KiteConnect.TRANSACTION_TYPE_SELL,
                quantity         = qty,
                product          = KiteConnect.PRODUCT_NRML,
                order_type       = KiteConnect.ORDER_TYPE_MARKET,
                validity         = KiteConnect.VALIDITY_DAY,
                tag              = tag[:20],
            )
            logger.info(f"SELL option: {tradingsymbol} qty={qty} → order {order_id}")
            return str(order_id)
        except KiteException as e:
            logger.error(f"Sell option failed: {tradingsymbol} — {e}")
            raise

    def sell_option_limit(self, tradingsymbol: str, exchange: str,
                          qty: int, price: float, tag: str = "BOT_TGT") -> str:
        """Limit sell (target exit) for an option."""
        try:
            order_id = self.kite.place_order(
                variety          = KiteConnect.VARIETY_REGULAR,
                exchange         = exchange,
                tradingsymbol    = tradingsymbol,
                transaction_type = KiteConnect.TRANSACTION_TYPE_SELL,
                quantity         = qty,
                product          = KiteConnect.PRODUCT_NRML,
                order_type       = KiteConnect.ORDER_TYPE_LIMIT,
                price            = price,
                validity         = KiteConnect.VALIDITY_DAY,
                tag              = tag[:20],
            )
            logger.info(f"LIMIT SELL option: {tradingsymbol} @ ₹{price} qty={qty} → {order_id}")
            return str(order_id)
        except KiteException as e:
            logger.error(f"Limit sell failed: {tradingsymbol} — {e}")
            raise

    def cancel_order(self, order_id: str):
        try:
            self.kite.cancel_order(KiteConnect.VARIETY_REGULAR, order_id)
            logger.info(f"Order {order_id} cancelled.")
        except Exception as e:
            logger.warning(f"Cancel failed {order_id}: {e}")

    def get_ltp(self, instrument_key: str) -> float:
        """e.g. 'NFO:NIFTY24MAY24000CE'"""
        try:
            data = self.kite.ltp([instrument_key])
            return float(data[instrument_key]["last_price"])
        except Exception:
            return 0.0

    # ─── EOD Square-off ───────────────────────────────────────────────────────

    def square_off_all_options(self, exchange: str = "NFO"):
        """Close all open option positions for safety."""
        positions = self.get_positions()
        for pos in positions:
            qty = pos.get("quantity", 0)
            if qty == 0 or pos.get("exchange") != exchange:
                continue
            try:
                self.sell_option(
                    tradingsymbol = pos["tradingsymbol"],
                    exchange      = pos["exchange"],
                    qty           = abs(qty),
                    tag           = "EOD_SQ",
                )
                logger.info(f"EOD square-off: {pos['tradingsymbol']} qty={abs(qty)}")
            except Exception as e:
                logger.error(f"EOD square-off failed: {pos['tradingsymbol']} — {e}")
