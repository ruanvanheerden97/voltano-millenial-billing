"""
push_readings.py - Virtual meter cumulative readings -> SFTP
============================================================
Reads the Excel file produced by fetch_data.py, calculates cumulative
TOU meter readings for each virtual meter from the start date (5 Dec 2025)
up to the latest available hourly timestamp, then pushes a CSV to the
SFTP server for import into EMS.

CSV format (matches MIL_SFTP_file_Example.xlsx):
  METER_ADDRESS, READING_DATE, READING_VALUE, PEAK, STD, OFFPEAK

Virtual meters:
  Solar      -> Total Solar Production  (cumulative kWh)
  Battery Charge  -> Total Battery Charge    (cumulative kWh)
  Battery Discharge -> Total Battery Discharge (cumulative kWh)

Each meter has 4 rows in the CSV:
  - Total (1.8.0 equivalent)
  - Peak  (1.8.1)
  - Std   (1.8.2)
  - Off-Peak (1.8.3)

Usage:
    python push_readings.py              # push current readings now
    python push_readings.py --dry-run    # preview CSV without uploading
"""

import os
import io
import csv
import sys
import json
import argparse
import paramiko
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# --- CONFIG -------------------------------------------------------------------

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

SFTP_HOST     = os.getenv("SFTP_HOST")
SFTP_PORT     = int(os.getenv("SFTP_PORT", "22"))
SFTP_USERNAME = os.getenv("SFTP_USERNAME")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
SFTP_PATH     = os.getenv("SFTP_REMOTE_PATH", "/")   # folder on SFTP to drop file

EXCEL_FILE    = Path(__file__).parent / "MIL_Battery_readings_EMS.xlsx"
START_DATE    = pd.Timestamp("2025-12-05")
TOU_CONFIG_FILE = Path(__file__).parent / "tou_tariffs.json"

# Virtual meter serial numbers -> (column in hourly sheet, display name)
METERS = {
    "Solar":              "Total Solar Production Energy (kWh)",
    "Battery Charge":     "Total Battery Charge Energy (kWh)",
    "Battery Discharge":  "Total Battery Discharge Energy (kWh)",
}

# SA timezone offset
SA_TZ = timezone(timedelta(hours=2))

# --- TOU CLASSIFICATION (config-driven, vectorised) ---------------------------

def load_tou_config() -> list:
    """Load TOU periods from tou_tariffs.json, sorted by effective_from.
    Same file used by app.py, so the Pi scripts and the Streamlit app
    always agree on the schedule for any given date."""
    if not TOU_CONFIG_FILE.exists():
        print(f"[ERROR]  tou_tariffs.json not found at {TOU_CONFIG_FILE}")
        print("    This file defines the TOU schedule. Cannot continue without it.")
        sys.exit(1)
    with open(TOU_CONFIG_FILE) as f:
        raw = json.load(f)
    return sorted(raw["periods"], key=lambda p: p["effective_from"])

TOU_PERIODS = load_tou_config()
_TOU_PERIOD_STARTS = np.array(
    [pd.Timestamp(p["effective_from"]) for p in TOU_PERIODS], dtype="datetime64[ns]"
)

def assign_tou_vectorised(df: pd.DataFrame) -> pd.DataFrame:
    """Assign TOU slot to each hourly row - fully vectorised, period-aware.
    Rows are grouped by which tariff period they fall under (by effective_from
    date) BEFORE the TOU schedule is applied, so historical rows always use
    the schedule that was actually in effect at the time — even after a new
    tariff period is added to tou_tariffs.json for a later year."""
    df = df.copy()
    dt  = df["Date"]
    h   = dt.dt.hour
    dow = dt.dt.weekday
    mon = dt.dt.month

    dt_arr = dt.values.astype("datetime64[ns]")
    period_idx = np.clip(
        np.searchsorted(_TOU_PERIOD_STARTS, dt_arr, side="right") - 1,
        0, len(TOU_PERIODS) - 1
    )

    is_weekday = dow < 5
    is_sat     = dow == 5
    is_sun     = dow == 6

    slot = pd.Series("1.8.3", index=df.index)

    for p_idx, period in enumerate(TOU_PERIODS):
        in_period = period_idx == p_idx
        if not in_period.any():
            continue

        is_high = mon.isin(period["season_months"]["high"]) & in_period
        is_low  = in_period & ~is_high

        day_type_masks = [
            ("sunday", is_sun & in_period),
            ("saturday", is_sat & in_period),
            ("weekday", is_weekday & in_period),
        ]
        season_masks = [("low", is_low), ("high", is_high)]

        for day_type, day_mask in day_type_masks:
            if not day_mask.any():
                continue
            for season_name, season_mask in season_masks:
                base_mask = day_mask & season_mask
                if not base_mask.any():
                    continue
                slots_cfg = period["schedule"][day_type][season_name]

                if "standard" in slots_cfg:
                    std_mask = pd.Series(False, index=df.index)
                    for start, end in slots_cfg["standard"]:
                        std_mask = std_mask | h.between(start, end - 1)
                    slot = slot.where(~(base_mask & std_mask), "1.8.2")

                if "peak" in slots_cfg:
                    peak_mask = pd.Series(False, index=df.index)
                    for start, end in slots_cfg["peak"]:
                        peak_mask = peak_mask | h.between(start, end - 1)
                    slot = slot.where(~(base_mask & peak_mask), "1.8.1")

    df["tou_slot"] = slot
    return df

# --- CUMULATIVE READING CALCULATION ------------------------------------------

def calc_cumulative_readings(hourly: pd.DataFrame) -> dict:
    """
    For each virtual meter, calculate cumulative kWh from START_DATE
    to the latest available timestamp, split by TOU slot.

    Returns dict keyed by meter serial:
      { total, peak, std, offpeak, reading_dt }
    """
    # Filter from meter start date only
    h = hourly[hourly["Date"] >= START_DATE].copy()
    h = assign_tou_vectorised(h)

    reading_dt = h["Date"].max()   # latest timestamp in the data

    results = {}
    for serial, col in METERS.items():
        if col not in h.columns:
            print(f"  [WARN]  Column '{col}' not found in hourly sheet - skipping {serial}")
            continue

        total   = h[col].sum()
        peak    = h[h["tou_slot"] == "1.8.1"][col].sum()
        std     = h[h["tou_slot"] == "1.8.2"][col].sum()
        offpeak = h[h["tou_slot"] == "1.8.3"][col].sum()

        results[serial] = {
            "total":      round(total, 3),
            "peak":       round(peak, 3),
            "std":        round(std, 3),
            "offpeak":    round(offpeak, 3),
            "reading_dt": reading_dt,
        }

    return results

# --- CSV GENERATION -----------------------------------------------------------

def format_reading_date(dt: pd.Timestamp) -> str:
    """Format timestamp to match EMS expected format: DD/MM/YYYY HH:MM:SS GMT+2"""
    local = dt.tz_localize("Africa/Johannesburg") if dt.tzinfo is None else dt
    return local.strftime("%d/%m/%Y %H:%M:%S GMT+2")

def build_csv(readings: dict) -> str:
    """
    Build CSV string with one row per meter.
    Columns: METER_ADDRESS, READING_DATE, READING_VALUE, PEAK, STD, OFFPEAK
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    # Header
    writer.writerow(["METER_ADDRESS", "READING_DATE", "READING_VALUE", "PEAK", "STD", "OFFPEAK"])

    for serial, data in readings.items():
        reading_date = format_reading_date(data["reading_dt"])
        writer.writerow([
            serial,
            reading_date,
            data["total"],
            data["peak"],
            data["std"],
            data["offpeak"],
        ])

    return output.getvalue()

# --- SFTP UPLOAD --------------------------------------------------------------

def push_to_sftp(csv_content: str, filename: str):
    """Connect to SFTP and upload the CSV file."""
    if not all([SFTP_HOST, SFTP_USERNAME, SFTP_PASSWORD]):
        raise ValueError(
            "SFTP credentials not set. Add to your .env file:\n"
            "  SFTP_HOST=your.sftp.server.com\n"
            "  SFTP_PORT=22\n"
            "  SFTP_USERNAME=yourusername\n"
            "  SFTP_PASSWORD=yourpassword\n"
            "  SFTP_REMOTE_PATH=/path/to/upload/folder"
        )

    remote_path = os.path.join(SFTP_PATH, filename).replace("\\", "/")

    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    try:
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            with sftp.open(remote_path, "w") as f:
                f.write(csv_content)
            print(f"  [OK]  Uploaded -> {SFTP_HOST}:{remote_path}")
        finally:
            sftp.close()
    finally:
        transport.close()

# --- FILENAME -----------------------------------------------------------------

def make_filename() -> str:
    """
    Generate filename: Millenial_Sigenergy_Data_YYYY-MM-DD_HH-MM.csv
    Uses current SA local time.
    """
    now = datetime.now(SA_TZ)
    return f"Millenial_Sigenergy_Data_{now.strftime('%Y-%m-%d_%H-%M')}.csv"

# --- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Push meter readings to SFTP")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview CSV output without uploading to SFTP")
    args = parser.parse_args()

    # Load Excel
    if not EXCEL_FILE.exists():
        print(f"[ERROR]  Excel file not found: {EXCEL_FILE}")
        print("    Run fetch_data.py first to populate the data file.")
        sys.exit(1)

    print(f"[FILE]  Loading hourly data from {EXCEL_FILE.name}...")
    xl     = pd.ExcelFile(EXCEL_FILE)
    hourly = pd.read_excel(xl, sheet_name=xl.sheet_names[1])
    hourly["Date"] = pd.to_datetime(hourly["Date"])
    print(f"    {len(hourly):,} hourly rows loaded. "
          f"Latest: {hourly['Date'].max().strftime('%Y-%m-%d %H:%M')}")

    # Calculate readings
    print("\n[DATA]  Calculating cumulative TOU readings...")
    readings = calc_cumulative_readings(hourly)

    for serial, data in readings.items():
        print(f"    {serial:<22}  Total={data['total']:>10.3f} kWh  "
              f"Peak={data['peak']:>9.3f}  "
              f"Std={data['std']:>9.3f}  "
              f"OffPk={data['offpeak']:>9.3f}")

    # Build CSV
    csv_content = build_csv(readings)
    filename    = make_filename()

    print(f"\n[CSV]  CSV preview ({filename}):")
    print("    " + "\n    ".join(csv_content.strip().splitlines()))

    if args.dry_run:
        print("\n[WARN]   Dry run - not uploading to SFTP.")
        # Save locally for inspection
        local_path = Path(__file__).parent / filename
        local_path.write_text(csv_content)
        print(f"    Saved locally -> {local_path}")
        return

    # Upload to SFTP
    print(f"\n[SFTP]  Uploading to SFTP...")
    push_to_sftp(csv_content, filename)
    print("\n[OK]  Done.")

if __name__ == "__main__":
    main()