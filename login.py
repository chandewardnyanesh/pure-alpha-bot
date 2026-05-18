"""
One-time login — run once per day before starting the bot.
Usage:
  python3 login.py                          # interactive (opens browser)
  python3 login.py --token REQUEST_TOKEN    # non-interactive
"""
import os, sys, argparse, webbrowser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kiteconnect import KiteConnect

API_KEY    = os.getenv("KITE_API_KEY",    "cjcxfltnoi5guhov")
API_SECRET = os.getenv("KITE_API_SECRET", "rruguucqsl2k6tq9q94qa1c9sxwix3sf")
_BASE      = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(_BASE, "data", "access_token.txt")
CREDS_FILE = os.path.join(_BASE, "data", "credentials.txt")

os.makedirs(os.path.join(_BASE, "data"), exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--token", help="request_token from Zerodha redirect URL")
args = parser.parse_args()

kite = KiteConnect(api_key=API_KEY)

if args.token:
    req_token = args.token.strip()
else:
    url = kite.login_url()
    print("\n" + "="*60)
    print("  ZERODHA KITE LOGIN")
    print("="*60)
    print(f"\n  Open this URL:\n\n  {url}\n")
    print("  After login, copy the request_token from the redirect URL.")
    print("="*60)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    req_token = input("\n  Paste request_token: ").strip()

session = kite.generate_session(req_token, api_secret=API_SECRET)
token   = session["access_token"]

with open(TOKEN_FILE, "w") as f:
    f.write(token)

with open(CREDS_FILE, "w") as f:
    f.write(f"api_key={API_KEY}\napi_secret={API_SECRET}\n")

kite.set_access_token(token)
profile = kite.profile()
margins = kite.margins(segment="equity")

print(f"\n  Logged in  : {profile['user_name']}  ({profile['user_id']})")
print(f"  Capital    : ₹{margins['net']:,.2f}")
print(f"  Token saved: {TOKEN_FILE}")
print("\n  Run: python3 bot.py --paper")
print("="*60 + "\n")
