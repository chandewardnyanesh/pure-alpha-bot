"""
PureAlpha Bot — macOS launchd setup
=====================================
Run this ONCE to install the scheduler as a macOS LaunchAgent.
After installation, the scheduler auto-starts on every login and
manages the bot's 09:15 start / 15:45 kill cycle automatically.

Usage:
    python setup_launchd.py           # install (paper mode, default)
    python setup_launchd.py --live    # install (live mode)
    python setup_launchd.py --remove  # uninstall
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

BOT_DIR     = Path(__file__).parent.resolve()
PYTHON      = sys.executable
PLIST_NAME  = "com.purealpha.bot"
PLIST_PATH  = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_NAME}.plist"
LOG_DIR     = BOT_DIR / "logs"


def install(paper: bool = True):
    LOG_DIR.mkdir(exist_ok=True)
    mode_flag = "" if paper else " --live"
    cmd_args  = f"{PYTHON} {BOT_DIR}/scheduler.py{mode_flag}"

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{PYTHON}</string>
        <string>{BOT_DIR}/scheduler.py</string>
        {"" if paper else "<string>--live</string>"}
    </array>

    <!-- Start at login and keep alive (restart on crash) -->
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>

    <!-- Working directory -->
    <key>WorkingDirectory</key>
    <string>{BOT_DIR}</string>

    <!-- stdout / stderr logs -->
    <key>StandardOutPath</key>
    <string>{LOG_DIR}/launchd_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{LOG_DIR}/launchd_stderr.log</string>

    <!-- Environment: inherit PATH so yfinance / kiteconnect are found -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>

    <!-- Throttle restarts: wait 60s before re-launching if scheduler crashes -->
    <key>ThrottleInterval</key>
    <integer>60</integer>
</dict>
</plist>
"""

    PLIST_PATH.write_text(plist_content)
    print(f"✅ Plist written: {PLIST_PATH}")

    # Unload existing if present, then load
    subprocess.run(["launchctl", "unload", str(PLIST_PATH)],
                   capture_output=True)
    result = subprocess.run(["launchctl", "load", str(PLIST_PATH)],
                            capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ LaunchAgent loaded — scheduler will start on every login.")
        print(f"   Mode: {'PAPER' if paper else 'LIVE'}")
        print(f"   Bot starts: 09:15 IST | Bot stops: 15:45 IST (Mon–Fri)")
        print()
        print("   Useful commands:")
        print(f"   launchctl list | grep purealpha   # check status")
        print(f"   launchctl stop  {PLIST_NAME}      # manual stop")
        print(f"   launchctl start {PLIST_NAME}      # manual start")
        print(f"   tail -f {LOG_DIR}/scheduler.log   # live scheduler log")
        print(f"   tail -f {LOG_DIR}/bot.log         # live bot log")
    else:
        print(f"❌ launchctl load failed: {result.stderr}")
        sys.exit(1)


def remove():
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)],
                       capture_output=True)
        PLIST_PATH.unlink()
        print(f"✅ Removed: {PLIST_PATH}")
        print("   PureAlpha Bot scheduler will no longer auto-start.")
    else:
        print(f"ℹ️  No plist found at {PLIST_PATH} — nothing to remove.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Install/remove PureAlpha Bot macOS LaunchAgent")
    parser.add_argument("--live",   action="store_true",
                        help="Install in LIVE trading mode (default: paper)")
    parser.add_argument("--remove", action="store_true",
                        help="Uninstall the LaunchAgent")
    args = parser.parse_args()

    if args.remove:
        remove()
    else:
        install(paper=not args.live)
