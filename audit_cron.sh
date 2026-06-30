#!/bin/bash
# audit_cron.sh - Full health check of all Voltano cron jobs
# ============================================================================
# Run this anytime to get a one-shot summary of:
#   1. What's actually in crontab right now
#   2. Whether each referenced script file exists
#   3. When each job last ran (from its log file)
#   4. Whether the last run looked successful or had errors
#   5. Data freshness in live_readings.db and the Excel file
#
# Usage: bash audit_cron.sh
# ============================================================================

SIGENERGY_DIR="$HOME/Sigenergy"
cd "$SIGENERGY_DIR" || { echo "[ERROR] Cannot cd to $SIGENERGY_DIR"; exit 1; }

echo "============================================================================"
echo "  1. CURRENT CRONTAB"
echo "============================================================================"
crontab -l 2>/dev/null || echo "[WARN]  No crontab found for this user."
echo ""

echo "============================================================================"
echo "  2. SCRIPT FILE CHECKS"
echo "============================================================================"

SCRIPTS=$(crontab -l 2>/dev/null | grep -oE '[a-zA-Z0-9_./]+\.(sh|py)' | sort -u)

if [ -z "$SCRIPTS" ]; then
    echo "[WARN]  No .sh or .py scripts found referenced in crontab."
else
    for script in $SCRIPTS; do
        if [[ "$script" == /* ]]; then
            full_path="$script"
        else
            full_path="$SIGENERGY_DIR/$script"
        fi

        if [ -f "$full_path" ]; then
            mtime=$(stat -c '%y' "$full_path" 2>/dev/null | cut -d'.' -f1)
            echo "[OK]    $script  (exists, last modified: $mtime)"
        else
            echo "[MISSING]  $script  -> NOT FOUND at $full_path"
        fi
    done
fi
echo ""

echo "============================================================================"
echo "  3. LOG FILE HEALTH (last run + last error, if any)"
echo "============================================================================"

check_log() {
    local name="$1"
    local logfile="$2"
    local error_pattern="$3"

    echo "--- $name ---"
    if [ ! -f "$logfile" ]; then
        echo "  [WARN]  Log file not found: $logfile"
        echo "          (either it hasn't run yet, or logs to a different file)"
        echo ""
        return
    fi

    local last_mod=$(stat -c '%y' "$logfile" 2>/dev/null | cut -d'.' -f1)
    echo "  Log last written: $last_mod"

    local last_lines=$(tail -5 "$logfile")
    echo "  Last 5 lines:"
    echo "$last_lines" | sed 's/^/    /'

    if echo "$last_lines" | grep -qiE "$error_pattern"; then
        echo "  [WARN]  Recent error-looking output detected in this log."
    else
        echo "  [OK]    No obvious errors in the last 5 lines."
    fi
    echo ""
}

check_log "hourly_update.sh (Excel/SFTP pipeline)" "$SIGENERGY_DIR/hourly_log.txt" "error|traceback|failed"
check_log "live_logger.py (5-min live data)"       "$SIGENERGY_DIR/live_logger.log" "error|traceback"
check_log "hourly_export_push.sh (CSV->GitHub)"    "$SIGENERGY_DIR/hourly_export_push.log" "error|traceback|failed"

echo "============================================================================"
echo "  4. DATA FRESHNESS CHECK"
echo "============================================================================"

if [ -f "$SIGENERGY_DIR/live_readings.db" ]; then
    echo "--- live_readings.db ---"
    python3 -c "
import sqlite3
from datetime import datetime, timezone
conn = sqlite3.connect('$SIGENERGY_DIR/live_readings.db')
row = conn.execute('SELECT ts FROM live_readings ORDER BY ts DESC LIMIT 1').fetchone()
conn.close()
if row:
    ts = datetime.fromisoformat(row[0])
    age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
    print(f'  Latest reading: {row[0]}')
    print(f'  Age: {age_min:.1f} minutes ago', '[OK - fresh]' if age_min < 10 else '[WARN - stale, expected <10 min]')
else:
    print('  [WARN]  No rows in live_readings table.')
"
else
    echo "  [WARN]  live_readings.db not found."
fi
echo ""

if [ -f "$SIGENERGY_DIR/MIL_Battery_readings_EMS.xlsx" ]; then
    echo "--- MIL_Battery_readings_EMS.xlsx ---"
    mtime=$(stat -c '%y' "$SIGENERGY_DIR/MIL_Battery_readings_EMS.xlsx" | cut -d'.' -f1)
    echo "  Last modified: $mtime"
    now_epoch=$(date +%s)
    file_epoch=$(stat -c '%Y' "$SIGENERGY_DIR/MIL_Battery_readings_EMS.xlsx")
    age_min=$(( (now_epoch - file_epoch) / 60 ))
    if [ "$age_min" -lt 90 ]; then
        echo "  Age: $age_min minutes ago [OK - fresh]"
    else
        echo "  Age: $age_min minutes ago [WARN - stale, expected <90 min for hourly job]"
    fi
else
    echo "  [WARN]  MIL_Battery_readings_EMS.xlsx not found."
fi
echo ""

echo "============================================================================"
echo "  AUDIT COMPLETE"
echo "============================================================================"