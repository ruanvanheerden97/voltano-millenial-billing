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

# Virtual meter serial numbers -> (column in hourly sheet, display name)
METERS = {
    "Solar":              "Total Solar Production Energy (kWh)",
    "Battery Charge":     "Total Battery Charge Energy (kWh)",
    "Battery Discharge":  "Total Battery Discharge Energy (kWh)",
}

# SA timezone offset
SA_TZ = timezone(timedelta(hours=2))

# --- TOU CLASSIFICATION (vectorised) -----------------------------------------

def assign_tou_vectorised(df: pd.DataFrame) -> pd.DataFrame:
    """Assign TOU slot and season to each hourly row - fully vectorised."""
    dt  = df["Date"]
    h   = dt.dt.hour
    dow = dt.dt.weekday
    mon = dt.dt.month

    is_high    = mon.isin([6, 7, 8])
    is_weekday = dow < 5
    is_sat     = dow == 5
    is_sun     = dow == 6

    slot = pd.Series("1.8.3", index=df.index)

    # Sunday
    std_sun = is_sun & (
        (~is_high & h.between(18, 19)) |
        (is_high  & h.between(17, 18))
    )
    slot = slot.where(~std_sun, "1.8.2")

    # Saturday
    std_sat = is_sat & (
        (~is_high & (h.between(7, 11) | h.between(18, 19))) |
        (is_high  & (h.between(7, 11) | h.between(17, 18)))
    )
    slot = slot.where(~std_sat, "1.8.2")

    # Weekday standard
    std_wd = is_weekday & (
        (~is_high & ((h == 6) | h.between(9, 17) | (h == 21))) |
        (is_high  & (h.between(8, 16) | h.between(20, 21)))
    )
    slot = slot.where(~std_wd, "1.8.2")

    # Weekday peak
    peak_wd = is_weekday & (
        (~is_high & (h.between(7, 8)  | h.between(18, 20))) |
        (is_high  & (h.between(6, 7)  | h.between(17, 19)))
    )
    slot = slot.where(~peak_wd, "1.8.1")

    df = df.copy()
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
