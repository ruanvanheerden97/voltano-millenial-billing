"""
export_live_csv.py - Export live_readings.db to lightweight CSV for GitHub sync
================================================================================
Exports two small CSV files from live_readings.db:
  - live_latest.csv   : just the most recent reading (1 row)
  - live_history.csv  : last 7 days of readings (for trend charts)

These are plain text and diff-friendly, unlike the SQLite .db file itself,
so committing them hourly to GitHub doesn't bloat the repo the way committing
the binary .db file every run would. Streamlit Cloud's app.py can fall back
to reading these CSVs when running on Cloud (where it has no access to the
Pi's live_readings.db), while the Pi-hosted version keeps reading the DB
directly for true real-time data.

Run hourly via cron, AFTER live_logger.py's normal 5-minute runs have
populated some data for that hour:
    0 * * * * cd ~/Sigenergy && python3 export_live_csv.py >> export_live_csv.log 2>&1 && git add live_latest.csv live_history.csv && git commit -m "Update live data export" && git push

(See setup_export_cron.sh for installing this safely with proper error handling.)
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

DB_PATH = Path(__file__).parent / "live_readings.db"
LATEST_CSV = Path(__file__).parent / "live_latest.csv"
HISTORY_CSV = Path(__file__).parent / "live_history.csv"


def main():
    if not DB_PATH.exists():
        print(f"[ERROR]  {DB_PATH} not found - nothing to export.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)

    # Export latest single row
    cursor = conn.execute("SELECT * FROM live_readings ORDER BY ts DESC LIMIT 1")
    cols = [d[0] for d in cursor.description]
    latest_row = cursor.fetchone()

    if not latest_row:
        print("[WARN]  live_readings table is empty - nothing to export yet.")
        conn.close()
        return

    with open(LATEST_CSV, "w") as f:
        f.write(",".join(cols) + "\n")
        f.write(",".join(str(v) for v in latest_row) + "\n")
    print(f"[OK]  Exported latest reading -> {LATEST_CSV.name} (ts={latest_row[0]})")

    # Export last 7 days for trend charts
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    cursor = conn.execute(
        "SELECT * FROM live_readings WHERE ts >= ? ORDER BY ts", (cutoff,)
    )
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]

    with open(HISTORY_CSV, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"[OK]  Exported {len(rows)} rows of 7-day history -> {HISTORY_CSV.name}")

    conn.close()


if __name__ == "__main__":
    main()