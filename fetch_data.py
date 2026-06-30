"""
fetch_data.py - Sigenergy API -> Excel updater
=============================================
Pulls daily historical data from the Sigenergy OpenAPI for every date
from the meter start (5 Dec 2025) up to today, then writes/updates the
two sheets (MIL_Usage_Monthly and MIL_Solar_hourly) in the Excel file.

Usage:
    python fetch_data.py              # fetches all missing days up to today
    python fetch_data.py --full       # re-fetches everything from start date
    python fetch_data.py --date 2026-07-15   # fetch a single specific date

Credentials are read from a .env file in the same folder - never hard-code them.
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

# --- CONFIG -------------------------------------------------------------------

# Load credentials from .env file in the same folder as this script
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

# Verify credentials loaded - remove these print lines once working
_u = os.getenv("SIGEN_USERNAME")
_p = os.getenv("SIGEN_PASSWORD")
if not _u or not _p:
    print(f"[ERROR]  .env not loaded from {env_path}")
else:
    print(f"[OK]  Credentials loaded for: {_u}")

API_BASE     = os.getenv("SIGEN_API_BASE", "https://openapi-eu.sigencloud.com")   # update if different
USERNAME     = os.getenv("SIGEN_USERNAME")
PASSWORD     = os.getenv("SIGEN_PASSWORD")
SYSTEM_ID    = os.getenv("SIGEN_SYSTEM_ID", "HUCUDI764140703")

EXCEL_FILE   = Path(__file__).parent / "MIL_Battery_readings_EMS.xlsx"
START_DATE   = date(2025, 12, 5)   # meter start date

# API field -> Excel column name mapping
# Based on Sigenergy API itemList fields from documentation
# API field -> Excel column name mapping
# Using the cumulative ENERGY fields (kWh, reset daily at midnight)
# NOT the instantaneous POWER fields (kW) which end in "Power"
# The energy fields are: powerGeneration, powerUse, powerFromGrid,
# powerToGrid, esCharging, esDischarging
# These are cumulative daily totals — build_hourly_rows converts them to per-interval deltas
HOURLY_COL_MAP = {
    "powerGeneration": "Total Solar Production Energy (kWh)",
    "powerUse":        "Total Load Consumed Energy (kWh)",
    "esCharging":      "Total Battery Charge Energy (kWh)",
    "esDischarging":   "Total Battery Discharge Energy (kWh)",
    "powerFromGrid":   "Total Grid Imported Energy (kWh)",
    "powerToGrid":     "Total Grid Exported Energy (kWh)",
}

DAILY_COL_MAP = {
    "powerGeneration": "Solar Production Energy (kWh)",
    "powerUse":        "Load Consumed Energy (kWh)",
    "esCharging":      "Battery Charge Energy (kWh)",
    "esDischarging":   "Battery Discharge Energy (kWh)",
    "powerFromGrid":   "Grid Imported Energy (kWh)",
    "powerToGrid":     "Grid Exported Energy (kWh)",
}

import json

# --- BILLING RUN CONFIG -------------------------------------------------------

def load_billing_runs() -> list[dict]:
    """
    Load billing run schedule from billing_runs.json.
    Returns list of dicts sorted by end_date ascending.
    """
    config_path = Path(__file__).parent / "billing_runs.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"billing_runs.json not found at {config_path}\n"
            "Create it with your billing run end dates."
        )
    with open(config_path) as f:
        data = json.load(f)
    runs = data["billing_runs"]
    # Sort by end date ascending
    runs = sorted(runs, key=lambda r: r["end_date"])
    return runs

def get_billing_run(d: date) -> str:
    """
    Assign a date to its billing run name using billing_runs.json.
    Each run covers from the day after the previous run ended
    up to and including its own end_date.
    Dates before the first run end date → first run name.
    Dates after the last run end date → 'Pending'.
    """
    runs = load_billing_runs()

    for run in runs:
        end = date.fromisoformat(run["end_date"])
        if d <= end:
            return run["name"]

    return "Pending"

# --- AUTHENTICATION -----------------------------------------------------------

def get_token() -> str:
    """Authenticate with Sigenergy API and return Bearer token."""
    if not USERNAME or not PASSWORD:
        raise ValueError(
            "SIGEN_USERNAME and SIGEN_PASSWORD must be set in your .env file.\n"
            "Create a file called .env in your app folder with:\n"
            "  SIGEN_USERNAME=your@email.com\n"
            "  SIGEN_PASSWORD=yourpassword\n"
            "  SIGEN_SYSTEM_ID=HUCUDI764140703\n"
            "  SIGEN_API_BASE=https://openapi-eu.sigencloud.com"
        )

    url = f"{API_BASE}/openapi/auth/login/password"
    resp = requests.post(url, json={"username": USERNAME, "password": PASSWORD}, timeout=30)
    resp.raise_for_status()
    body = resp.json()

    if body.get("code") != 0:
        raise RuntimeError(f"Authentication failed: {body.get('msg')} (code {body.get('code')})")

    token_data = json.loads(body["data"]) if isinstance(body["data"], str) else body["data"]
    return token_data["accessToken"]

# --- DATA FETCH ---------------------------------------------------------------

def fetch_day(token: str, day: date) -> dict | None:
    """
    Fetch historical data for a single day.
    Returns the full response data dict, or None if no data.
    Rate limit: one request per 5 minutes per station - we add a small delay.
    """
    url = f"{API_BASE}/openapi/systems/{SYSTEM_ID}/history"
    headers = {"Authorization": f"Bearer {token}"}
    params  = {"level": "Day", "date": day.strftime("%Y-%m-%d")}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()

    if body.get("code") != 0:
        print(f"  [WARN]  {day}: API error {body.get('code')} - {body.get('msg')}")
        return None

    data = body.get("data")
    # API returns data as a JSON string - parse it if needed
    if isinstance(data, str):
        import json as _json
        data = _json.loads(data)
    return data

# --- DATA PROCESSING ----------------------------------------------------------

def build_hourly_rows(day: date, data: dict) -> list[dict]:
    """
    Convert API itemList into per-interval usage rows for the hourly sheet.

    The API returns CUMULATIVE daily energy totals (kWh, resetting at midnight)
    in fields like powerGeneration, powerUse, esCharging etc.
    We convert these to per-interval USAGE by subtracting the previous value.

    Note: Do NOT use the instantaneous *Power fields (pvTotalPower, loadPower etc.)
    as those are kW snapshots, not energy values.

    Example:
        API row 1: powerGeneration = 1.5  (cumulative from midnight)
        API row 2: powerGeneration = 3.2
        Stored row 1: 1.5 - 0.0 = 1.5 kWh used in interval 1
        Stored row 2: 3.2 - 1.5 = 1.7 kWh used in interval 2
    """
    items = data.get("itemList", [])
    if not items:
        return []

    rows = []
    # Track previous cumulative values per field — reset at midnight (first item of day)
    prev_values = {api_field: 0.0 for api_field in HOURLY_COL_MAP}

    for item in items:
        dt_str = item.get("dataTime", "")
        try:
            dt = pd.Timestamp(dt_str)
        except Exception:
            continue

        row = {"Date": dt}
        for api_field, col_name in HOURLY_COL_MAP.items():
            current = float(item.get(api_field, 0) or 0)
            prev    = prev_values[api_field]

            # Determine if this is a genuine midnight reset or just noise/rounding.
            # A genuine reset means current is significantly smaller than prev
            # (e.g. daily cumulative restarted at midnight).
            # Small drops (< 1 kWh) are noise — treat delta as 0, keep prev.
            # Large drops (>= 1 kWh below prev) are genuine resets — start fresh.
            drop = prev - current
            if drop > 1.0:
                # Genuine reset — new day started mid-fetch or data anomaly
                delta = current
            elif drop > 0:
                # Tiny noise/rounding drop — treat as zero usage in this interval
                delta = 0.0
            else:
                # Normal increment
                delta = current - prev

            row[col_name]          = round(max(0.0, delta), 6)
            prev_values[api_field] = current

        rows.append(row)
    return rows

def build_daily_row(day: date, hourly_rows: list[dict]) -> dict:
    """
    Build daily summary row by summing the already-converted per-interval rows.
    This ensures the daily sheet always matches the sum of the hourly sheet.
    """
    row = {
        "Billing run": get_billing_run(day),
        "Date":        pd.Timestamp(day),
    }
    # Map hourly column names back to daily column names
    hourly_to_daily = {v: daily_v for (_, v), (_, daily_v)
                       in zip(HOURLY_COL_MAP.items(), DAILY_COL_MAP.items())}
    hourly_to_daily = {
        "Total Solar Production Energy (kWh)": "Solar Production Energy (kWh)",
        "Total Load Consumed Energy (kWh)":    "Load Consumed Energy (kWh)",
        "Total Battery Charge Energy (kWh)":   "Battery Charge Energy (kWh)",
        "Total Battery Discharge Energy (kWh)":"Battery Discharge Energy (kWh)",
        "Total Grid Imported Energy (kWh)":    "Grid Imported Energy (kWh)",
        "Total Grid Exported Energy (kWh)":    "Grid Exported Energy (kWh)",
    }

    # Sum all intervals for the day
    for hourly_col, daily_col in hourly_to_daily.items():
        total = sum(float(r.get(hourly_col, 0) or 0) for r in hourly_rows)
        row[daily_col] = round(total, 3)

    # Generator column not in hourly data — set to 0
    row["From Generator (kWh)"] = 0.0
    row["Revenue(R)"]           = 0.0
    return row

# --- EXCEL READ / WRITE -------------------------------------------------------

def load_existing_excel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load existing Excel sheets, or return empty DataFrames if file doesn't exist."""
    if EXCEL_FILE.exists():
        xl = pd.ExcelFile(EXCEL_FILE)
        daily  = pd.read_excel(xl, sheet_name=xl.sheet_names[0])
        hourly = pd.read_excel(xl, sheet_name=xl.sheet_names[1])
        daily["Date"]  = pd.to_datetime(daily["Date"])
        hourly["Date"] = pd.to_datetime(hourly["Date"])
        return daily, hourly
    else:
        print(f"  [INFO]  No existing Excel file found - will create from scratch.")
        return pd.DataFrame(), pd.DataFrame()

def save_excel(daily: pd.DataFrame, hourly: pd.DataFrame):
    """Write both sheets back to the Excel file."""
    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
        daily.to_excel(writer,  sheet_name="MIL_Usage_Monthly", index=False)
        hourly.to_excel(writer, sheet_name="MIL_Solar_hourly",  index=False)
    print(f"  [OK]  Saved -> {EXCEL_FILE}")

def get_existing_dates(daily: pd.DataFrame) -> set[date]:
    """Return set of dates already in the daily sheet."""
    if daily.empty or "Date" not in daily.columns:
        return set()
    return set(pd.to_datetime(daily["Date"]).dt.date)

# --- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Sigenergy data and update Excel")
    parser.add_argument("--full",  action="store_true", help="Re-fetch all dates from start")
    parser.add_argument("--date",  type=str, default=None, help="Fetch a single date (yyyy-mm-dd)")
    args = parser.parse_args()

    print("[AUTH]  Authenticating with Sigenergy API...")
    token = get_token()
    print("    Token obtained successfully.\n")

    # Determine which dates to fetch
    today = date.today()
    if args.date:
        dates_to_fetch = [date.fromisoformat(args.date)]
    else:
        daily_existing, hourly_existing = load_existing_excel()
        existing_dates = get_existing_dates(daily_existing)

        all_dates = [START_DATE + timedelta(days=i)
                     for i in range((today - START_DATE).days + 1)]

        if args.full:
            dates_to_fetch = all_dates
            print(f"  --full flag set: re-fetching all {len(all_dates)} dates.\n")
        else:
            dates_to_fetch = [d for d in all_dates if d not in existing_dates]
            print(f"  {len(existing_dates)} dates already in Excel. "
                  f"Fetching {len(dates_to_fetch)} new dates.\n")

    if not dates_to_fetch:
        print("[OK]  Excel is already up to date - nothing to fetch.")
        return

    # Load existing data
    daily_df, hourly_df = load_existing_excel()
    new_daily_rows  = []
    new_hourly_rows = []

    # Fetch day by day
    # API rate limit: 1 request per 5 MINUTES per station (error 1201 if exceeded).
    # We wait 310 seconds between requests to stay safely within the limit.
    # Fetching 5 days takes ~25 minutes - let it run in the background.
    for i, day in enumerate(sorted(dates_to_fetch)):
        print(f"  [{i+1}/{len(dates_to_fetch)}] Fetching {day}...", end=" ")
        data = fetch_day(token, day)

        if data is None:
            print("no data")
            continue

        hourly_rows = build_hourly_rows(day, data)
        daily_row   = build_daily_row(day, hourly_rows)

        new_daily_rows.append(daily_row)
        new_hourly_rows.extend(hourly_rows)
        print(f"[OK]  ({len(hourly_rows)} interval records)")

        # Wait 310 seconds (5 min 10 sec) between requests to avoid error 1201
        if i < len(dates_to_fetch) - 1:
            remaining = len(dates_to_fetch) - i - 1
            print(f"      Waiting 310s before next request ({remaining} remaining)...")
            time.sleep(310)

    if not new_daily_rows:
        print("\n[WARN]  No new data returned from API.")
        return

    # Merge with existing data
    new_daily_df  = pd.DataFrame(new_daily_rows)
    new_hourly_df = pd.DataFrame(new_hourly_rows)

    if not daily_df.empty:
        # Remove any existing rows for dates we just re-fetched (handles --full)
        fetched_dates = {r["Date"].date() if hasattr(r["Date"], "date") else r["Date"]
                         for r in new_daily_rows}
        daily_df  = daily_df[~pd.to_datetime(daily_df["Date"]).dt.date.isin(fetched_dates)]
        hourly_df = hourly_df[~pd.to_datetime(hourly_df["Date"]).dt.date.isin(fetched_dates)]

        daily_df  = pd.concat([daily_df,  new_daily_df],  ignore_index=True)
        hourly_df = pd.concat([hourly_df, new_hourly_df], ignore_index=True)
    else:
        daily_df  = new_daily_df
        hourly_df = new_hourly_df

    # Sort by date
    daily_df  = daily_df.sort_values("Date").reset_index(drop=True)
    hourly_df = hourly_df.sort_values("Date").reset_index(drop=True)

    print(f"\n[DATA]  Summary:")
    print(f"    Daily rows:  {len(daily_df)}")
    print(f"    Hourly rows: {len(hourly_df)}")
    print(f"    Date range:  {daily_df['Date'].min().date()} -> {daily_df['Date'].max().date()}")

    save_excel(daily_df, hourly_df)
    print("\n[OK]  Done. Push to GitHub to update the live app:")
    print('    git add MIL_Battery_readings_EMS.xlsx')
    print('    git commit -m "Update EMS data"')
    print('    git push')

if __name__ == "__main__":
    main()