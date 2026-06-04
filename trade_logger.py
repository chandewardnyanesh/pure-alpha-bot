"""
SQLite-backed trade log.
Stores every trade with entry/exit details, indicators, PnL, and outcome.
Provides analytics used for ML retraining and performance dashboard.
"""

import sqlite3
import logging
import json
import os
from datetime import date, datetime
from contextlib import contextmanager
from config import DB_PATH

logger = logging.getLogger(__name__)


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    action          TEXT    NOT NULL,   -- BUY / SELL
    qty             INTEGER NOT NULL,
    entry_price     REAL    NOT NULL,
    exit_price      REAL,
    sl_price        REAL,
    target_price    REAL,
    entry_time      TEXT    NOT NULL,
    exit_time       TEXT,
    exit_reason     TEXT,               -- stop_loss / target / trailing_stop / eod / manual
    pnl             REAL,
    charges         REAL    DEFAULT 0,
    net_pnl         REAL,
    profitable      INTEGER,            -- 1 / 0
    features_json   TEXT,               -- JSON of indicator features at entry
    strategy_scores TEXT,               -- JSON: voter name → vote at entry
    ml_proba        REAL,
    votes_for       INTEGER DEFAULT 0,  -- consensus votes at entry
    unanimity       INTEGER DEFAULT 0,  -- was it unanimous (1/0)
    session_bucket  INTEGER DEFAULT -1, -- 1=morning 2=deadzone 3=afternoon 4=power_hour
    trade_date      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_summary (
    trade_date      TEXT PRIMARY KEY,
    capital_start   REAL,
    capital_end     REAL,
    total_trades    INTEGER,
    wins            INTEGER,
    losses          INTEGER,
    gross_pnl       REAL,
    net_pnl         REAL,
    win_rate        REAL,
    best_trade      REAL,
    worst_trade     REAL
);

CREATE TABLE IF NOT EXISTS voter_stats (
    voter_name      TEXT NOT NULL,
    vote_direction  TEXT NOT NULL,   -- BUY / SELL / ABSTAIN
    outcome         INTEGER,         -- 1=trade won, 0=trade lost, NULL=abstained
    trade_id        INTEGER,
    trade_date      TEXT
);
"""


@contextmanager
def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.executescript(CREATE_SQL)
    logger.info(f"DB initialised at {DB_PATH}")


# ─── Write ────────────────────────────────────────────────────────────────────

def log_entry(trade: dict) -> int:
    """Insert a new trade row at entry time. Returns row id."""
    from datetime import datetime as _dt
    now_min = _dt.now().hour * 60 + _dt.now().minute
    if now_min < 9 * 60 + 25:
        sb = 0
    elif now_min < 11 * 60 + 30:
        sb = 1
    elif now_min < 13 * 60:
        sb = 2
    elif now_min < 14 * 60 + 30:
        sb = 3
    else:
        sb = 4

    with _conn() as con:
        cur = con.execute("""
            INSERT INTO trades
              (symbol, action, qty, entry_price, sl_price, target_price,
               entry_time, features_json, strategy_scores, ml_proba,
               votes_for, unanimity, session_bucket, trade_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            trade["symbol"],
            trade["action"],
            trade["qty"],
            trade["entry_price"],
            trade.get("sl_price"),
            trade.get("target_price"),
            trade.get("entry_time", datetime.now().isoformat()),
            json.dumps(trade.get("features", [])),
            json.dumps(trade.get("strategy_scores", {})),
            trade.get("ml_proba"),
            trade.get("votes_for", 0),
            int(trade.get("unanimity", False)),
            sb,
            str(date.today()),
        ))
        trade_id = cur.lastrowid

        # Record per-voter entry votes for post-hoc analysis
        voters = trade.get("strategy_scores", {})
        if isinstance(voters, dict):
            for voter_name, vote in voters.items():
                con.execute("""
                    INSERT INTO voter_stats (voter_name, vote_direction, outcome, trade_id, trade_date)
                    VALUES (?,?,NULL,?,?)
                """, (voter_name, str(vote), trade_id, str(date.today())))

        return trade_id


def log_exit(trade_id: int, exit_price: float, exit_reason: str,
             charges: float = 0.0):
    """Update the trade row when position is closed."""
    with _conn() as con:
        row = con.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        if not row:
            logger.error(f"Trade id {trade_id} not found")
            return

        pnl = (exit_price - row["entry_price"]) * row["qty"]
        if row["action"] == "SELL":
            pnl = -pnl
        net_pnl    = pnl - charges
        profitable = int(net_pnl > 0)

        con.execute("""
            UPDATE trades SET
                exit_price=?, exit_time=?, exit_reason=?,
                pnl=?, charges=?, net_pnl=?, profitable=?
            WHERE id=?
        """, (
            exit_price,
            datetime.now().isoformat(),
            exit_reason,
            round(pnl, 2),
            round(charges, 2),
            round(net_pnl, 2),
            profitable,
            trade_id,
        ))
        # Update voter_stats outcome for this trade
        con.execute("""
            UPDATE voter_stats SET outcome=? WHERE trade_id=?
        """, (profitable, trade_id))

        logger.info(f"Trade {trade_id} closed — pnl=₹{net_pnl:.2f} ({exit_reason})")
        return {
            "pnl": pnl, "net_pnl": net_pnl, "profitable": bool(profitable)
        }


# ─── Read / Analytics ─────────────────────────────────────────────────────────

def get_today_trades() -> list:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM trades WHERE trade_date=?", (str(date.today()),)
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_closed_trades() -> list:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM trades WHERE exit_price IS NOT NULL ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def get_voter_win_rates() -> dict:
    """
    Returns per-voter win rate based on all closed trades.
    Useful for identifying which voters add alpha vs noise.

    Returns dict: { voter_name: { "votes": N, "wins": W, "win_rate": 0.xx } }
    """
    with _conn() as con:
        rows = con.execute("""
            SELECT voter_name, vote_direction, outcome
            FROM voter_stats
            WHERE outcome IS NOT NULL
        """).fetchall()

    stats: dict[str, dict] = {}
    for r in rows:
        name = r["voter_name"]
        if name not in stats:
            stats[name] = {"votes": 0, "wins": 0}
        stats[name]["votes"] += 1
        if r["outcome"] == 1:
            stats[name]["wins"] += 1

    result = {}
    for name, s in stats.items():
        result[name] = {
            "votes":    s["votes"],
            "wins":     s["wins"],
            "win_rate": round(s["wins"] / s["votes"], 4) if s["votes"] else 0.0,
        }
    return result


def get_rolling_expectancy(window: int = 20) -> dict:
    """
    Rolling expectancy over the last `window` closed trades.
    Expectancy = WR × avg_win + (1 - WR) × avg_loss
    Target: +0.65 (in ₹ per ₹ risked, or positive absolute ₹)
    """
    trades = get_all_closed_trades()
    recent = [t for t in trades if t["net_pnl"] is not None][-window:]
    if not recent:
        return {"window": window, "trades": 0, "expectancy": 0.0}

    pnls = [t["net_pnl"] for t in recent]
    wins = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p <= 0]
    wr   = len(wins) / len(pnls) if pnls else 0
    avg_w = sum(wins) / len(wins) if wins else 0
    avg_l = sum(loss) / len(loss) if loss else 0
    exp   = wr * avg_w + (1 - wr) * avg_l

    return {
        "window":       window,
        "trades":       len(recent),
        "win_rate":     round(wr, 4),
        "avg_win":      round(avg_w, 2),
        "avg_loss":     round(avg_l, 2),
        "expectancy":   round(exp, 2),
        "on_target":    exp >= 0.65,
    }


def get_session_breakdown() -> dict:
    """Win rate and expectancy broken down by session bucket."""
    trades = get_all_closed_trades()
    closed = [t for t in trades if t["net_pnl"] is not None]

    bucket_names = {0: "pre", 1: "morning", 2: "deadzone", 3: "afternoon", 4: "power_hour"}
    result = {}
    for bucket, label in bucket_names.items():
        subset = [t for t in closed if t.get("session_bucket") == bucket]
        if not subset:
            continue
        pnls = [t["net_pnl"] for t in subset]
        wins = [p for p in pnls if p > 0]
        result[label] = {
            "trades":   len(subset),
            "wins":     len(wins),
            "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0,
            "total_pnl": round(sum(pnls), 2),
        }
    return result


def get_performance_summary() -> dict:
    trades = get_all_closed_trades()
    if not trades:
        return {"total": 0}

    pnls  = [t["net_pnl"] for t in trades if t["net_pnl"] is not None]
    wins  = [p for p in pnls if p > 0]
    loss  = [p for p in pnls if p <= 0]

    win_rate   = len(wins) / len(pnls) if pnls else 0
    avg_win    = sum(wins) / len(wins)  if wins else 0
    avg_loss   = sum(loss) / len(loss)  if loss else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "total":        len(trades),
        "wins":         len(wins),
        "losses":       len(loss),
        "win_rate":     round(win_rate, 4),
        "gross_pnl":    round(sum(pnls), 2),
        "best_trade":   round(max(pnls), 2)  if pnls else 0,
        "worst_trade":  round(min(pnls), 2)  if pnls else 0,
        "avg_win":      round(avg_win, 2),
        "avg_loss":     round(avg_loss, 2),
        "expectancy":   round(expectancy, 2),
        "profit_factor": round(sum(wins) / abs(sum(loss)), 2) if loss and sum(loss) != 0 else float("inf"),
    }


def save_daily_summary(capital_start: float, capital_end: float):
    today  = str(date.today())
    trades = get_today_trades()
    closed = [t for t in trades if t["exit_price"] is not None]
    pnls   = [t["net_pnl"] for t in closed if t["net_pnl"] is not None]
    wins   = [p for p in pnls if p > 0]
    loss   = [p for p in pnls if p <= 0]

    with _conn() as con:
        con.execute("""
            INSERT OR REPLACE INTO daily_summary VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            today,
            round(capital_start, 2),
            round(capital_end, 2),
            len(closed),
            len(wins),
            len(loss),
            round(sum(p for t in closed for p in [t.get("pnl", 0)] if p), 2),
            round(sum(pnls), 2),
            round(len(wins) / len(pnls), 4) if pnls else 0,
            round(max(pnls), 2) if pnls else 0,
            round(min(pnls), 2) if pnls else 0,
        ))
    logger.info(f"Daily summary saved for {today}")
