"""
Upstox daily login script.
Run this every morning before 9:15 AM IST to get a fresh access token.

Usage:
    python scripts/upstox_login.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs

import requests
from dotenv import load_dotenv, set_key

ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.getenv("UPSTOX_API_KEY", "")
API_SECRET = os.getenv("UPSTOX_API_SECRET", "")
REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:3000")

if not API_KEY or not API_SECRET:
    print("ERROR: UPSTOX_API_KEY or UPSTOX_API_SECRET not set in .env")
    sys.exit(1)


def get_auth_url() -> str:
    params = {
        "response_type": "code",
        "client_id": API_KEY,
        "redirect_uri": REDIRECT_URI,
    }
    return "https://api.upstox.com/v2/login/authorization/dialog?" + urlencode(params)


def exchange_code(auth_code: str) -> str:
    resp = requests.post(
        "https://api.upstox.com/v2/login/authorization/token",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data={
            "code": auth_code,
            "client_id": API_KEY,
            "client_secret": API_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        },
    )
    if resp.status_code != 200:
        print(f"ERROR: Token exchange failed ({resp.status_code}): {resp.text}")
        sys.exit(1)
    return resp.json()["access_token"]


def extract_code_from_url(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        return params.get("code", [None])[0]
    except Exception:
        return None


def main() -> None:
    print("\n=== Upstox Daily Login ===\n")
    print("Step 1: Open this URL in your browser:\n")
    print(f"  {get_auth_url()}\n")
    print("Step 2: Log in with your Upstox username, password, and 6-digit TOTP code.")
    print("Step 3: After login, your browser will redirect to a page that may not load.")
    print("        That's fine. Copy the FULL URL from your browser's address bar.\n")

    raw = input("Paste the full redirect URL here: ").strip()

    # Try to extract code from full URL
    code = extract_code_from_url(raw)

    # If they pasted just the code directly
    if not code:
        code = raw.strip()

    if not code:
        print("ERROR: Could not extract auth code. Make sure you pasted the full URL.")
        sys.exit(1)

    print("\nExchanging code for access token...")
    token = exchange_code(code)

    # Write to .env
    set_key(str(ENV_PATH), "UPSTOX_ACCESS_TOKEN", token)
    print(f"\nSuccess! Access token saved to {ENV_PATH}")
    print("\nRestart the backend now:\n")
    print("  tmux kill-session -t backend")
    print("  tmux new-session -d -s backend -c /opt/yegedge 'source .venv/bin/activate && uvicorn server.main:app --host 0.0.0.0 --port 8000'")


if __name__ == "__main__":
    main()
