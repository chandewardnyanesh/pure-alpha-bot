"""
PureAlpha Bot — Auto Scheduler
================================
Automatically starts the bot at 09:15 IST and kills it at 15:45 IST,
Monday–Friday (Indian market days).

Usage:
  python scheduler.py           # runs in foreground, manages bot lifecycle
  python scheduler.py --paper   # passes --paper to bot.py

The scheduler itself should be set up as a macOS launchd service so it
survives reboots. See setup instructions at the bottom of this file.
"""

import os
import sys
import time
import signal
import logging
import subprocess
import argparse
from datetime import datetime, date

# ─── Config ───────────────────────────────────────────────────────────────────

BOT_START_TIME  = "09:15"   # HH:MM — when to launch bot.py
BOT_KILL_TIME   = "15:45"   # HH:MM — when to terminate it
CHECK_INTERVAL  = 30        # seconds between scheduler ticks

# Directory containing bot.py
BOT_DIR  = os.path.dirname(os.path.abspath(__file__))
BOT_SCRIPT = os.path.join(BOT_DIR, "bot.py")

# Use same Python interpreter as the one running this scheduler
PYTHON   = sys.executable

# ─── Logging ──────────────────────────────────────────────────────────────────

log_path = os.path.join(BOT_DIR, "logs", "scheduler.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [SCHED] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("scheduler")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")

def is_market_day() -> bool:
    """Monday=0 … Friday=4 are market days. Skip weekends."""
    return date.today().weekday() < 5

def is_start_time() -> bool:
    return now_hhmm() >= BOT_START_TIME

def is_kill_time() -> bool:
    return now_hhmm() >= BOT_KILL_TIME


# ─── Bot process management ───────────────────────────────────────────────────

class BotProcess:
    def __init__(self, paper: bool = True):
        self.paper   = paper
        self.proc: subprocess.Popen | None = None

    def start(self):
        if self.proc and self.proc.poll() is None:
            logger.info("Bot already running (pid=%d)", self.proc.pid)
            return

        cmd = [PYTHON, BOT_SCRIPT]
        if self.paper:
            cmd.append("--paper")

        log_stdout = open(os.path.join(BOT_DIR, "logs", "bot_stdout.log"), "a")
        self.proc  = subprocess.Popen(
            cmd,
            cwd    = BOT_DIR,
            stdout = log_stdout,
            stderr = subprocess.STDOUT,
        )
        logger.info("✅ Bot STARTED — pid=%d  cmd=%s", self.proc.pid, " ".join(cmd))

    def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            logger.info("Bot is not running — nothing to stop")
            self.proc = None
            return

        logger.info("🛑 Sending SIGTERM to bot (pid=%d) ...", self.proc.pid)
        self.proc.send_signal(signal.SIGTERM)

        # Give it 30s to shut down gracefully (EOD square-off)
        try:
            self.proc.wait(timeout=30)
            logger.info("Bot exited cleanly.")
        except subprocess.TimeoutExpired:
            logger.warning("Bot did not exit in 30s — sending SIGKILL")
            self.proc.kill()
            self.proc.wait()

        self.proc = None

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


# ─── Main scheduler loop ──────────────────────────────────────────────────────

def run(paper: bool = True):
    bot   = BotProcess(paper=paper)
    today = date.today()

    logger.info("=" * 60)
    logger.info("  PureAlpha Bot Scheduler")
    logger.info(f"  Start: {BOT_START_TIME}  |  Kill: {BOT_KILL_TIME}")
    logger.info(f"  Mode: {'PAPER' if paper else 'LIVE'}")
    logger.info(f"  Today: {today} ({'Market day' if is_market_day() else 'Weekend — idle'})")
    logger.info("=" * 60)

    # Graceful shutdown on SIGTERM/SIGINT
    def _shutdown(sig, frame):
        logger.info("Scheduler shutdown requested — stopping bot...")
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    while True:
        # Reset tracking if a new day started
        if date.today() != today:
            today = date.today()
            logger.info(f"New day: {today} — resetting state")

        # Weekend / holiday skip
        if not is_market_day():
            if bot.is_alive():
                logger.info("Weekend — stopping bot")
                bot.stop()
            time.sleep(CHECK_INTERVAL)
            continue

        # ── Kill window ───────────────────────────────────────────────────────
        if is_kill_time():
            if bot.is_alive():
                logger.info(f"⏰ {BOT_KILL_TIME} — market closing, stopping bot")
                bot.stop()
            time.sleep(CHECK_INTERVAL)
            continue

        # ── Trading window ────────────────────────────────────────────────────
        if is_start_time():
            if not bot.is_alive():
                logger.info(f"⏰ {BOT_START_TIME} — market open, starting bot")
                bot.start()
            else:
                # Watchdog: restart if bot crashed unexpectedly
                rc = bot.proc.poll()
                if rc is not None and rc != 0:
                    logger.warning(f"Bot exited with code {rc} — restarting in 60s ...")
                    time.sleep(60)
                    bot.start()
        else:
            logger.debug(f"Waiting for market open ({BOT_START_TIME}) — now {now_hhmm()}")

        time.sleep(CHECK_INTERVAL)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PureAlpha Bot Scheduler")
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Run bot in paper trade mode (default: True)")
    parser.add_argument("--live",  action="store_true",
                        help="Run bot in LIVE mode (real orders)")
    args = parser.parse_args()

    paper_mode = not args.live
    run(paper=paper_mode)


# ─── macOS launchd setup ───────────────────────────────────────────────────────
# To auto-start this scheduler on login (so bot starts every market morning):
#
# 1. Create plist:
#      python scheduler.py --install
#
# 2. Load it:
#      launchctl load ~/Library/LaunchAgents/com.purealpha.bot.plist
#
# 3. Verify:
#      launchctl list | grep purealpha
#
# The plist is also generated by setup_launchd.py in this directory.
