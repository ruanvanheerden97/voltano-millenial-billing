#!/bin/bash
# hourly_export_push.sh - Export live data to CSV and push to GitHub
# ============================================================================
# Run hourly via cron. Exports the latest live reading + 7-day history to
# small CSV files, then commits and pushes them to GitHub so the Streamlit
# Cloud version of the app (which has no access to the Pi's live_readings.db)
# can show reasonably fresh data, falling back from true real-time.
#
# CRONTAB ENTRY (add via `crontab -e`):
#   0 * * * * /home/arnodt95/Sigenergy/hourly_export_push.sh >> /home/arnodt95/Sigenergy/hourly_export_push.log 2>&1
#
# This is INDEPENDENT of hourly_update.sh (which handles the Excel/SFTP
# billing pipeline) and live_logger.py (which handles the 5-minute live
# data fetch on the Pi). This script only relays a CSV snapshot of what
# live_logger.py has already collected - it makes no API calls itself.
# ============================================================================

cd "$(dirname "$0")" || exit 1

echo "[$(date '+%Y-%m-%d %H:%M:%S')]  hourly_export_push starting..."

# Step 1 - Export CSVs from the SQLite DB
python3 export_live_csv.py
if [ $? -ne 0 ]; then
    echo "  [ERROR]  export_live_csv.py failed - skipping git push."
    exit 1
fi

# Step 2 - Commit and push (only if something actually changed)
git add live_latest.csv live_history.csv

if git diff --cached --quiet; then
    echo "  [INFO]   No changes to live_latest.csv/live_history.csv - nothing to commit."
else
    git commit -m "Update live data export - $(date '+%Y-%m-%d %H:%M')"
    if [ $? -ne 0 ]; then
        echo "  [ERROR]  git commit failed."
        exit 1
    fi

    git push
    if [ $? -ne 0 ]; then
        echo "  [ERROR]  git push failed - changes committed locally but not pushed."
        echo "           Check network/credentials and push manually if needed."
        exit 1
    fi
    echo "  [OK]  Pushed updated live data export to GitHub."
fi

echo "--- Done ---"