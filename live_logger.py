"""
live_logger.py - Background live data + weather logger for Voltano MIL Dashboard
==================================================================================
Runs on the Pi via cron every 5 minutes. Fetches:
  - Live Sigenergy data (energyFlow, summary, device realtimeInfo)
  - Current weather + irradiance (Open-Meteo, free, no key)

Writes one row per run to live_readings.db. The Streamlit app's Live
Dashboard tab reads from this DB instead of calling the APIs directly,
so the dashboard can refresh as often as you like (e.g. every 30s during
a demo) without ever touching the rate-limited Sigenergy endpoints.

IMPORTANT — Sigenergy rate limit (confirmed from official API docs):
  "One account can only access one station/device once every five minutes"
  applies identically to /summary, /energyFlow, and /devices/{sn}/realtimeInfo.
  This script must NOT run more often than every 5 minutes, and nothing
  else should call these same endpoints in between (e.g. don't also run
  this manually while the cron job might fire, and don't re-enable the
  Streamlit app's direct API calls alongside this).

Cron (every 5 minutes):
    */5 * * * * cd ~/Sigenergy && python3 live_logger.py >> live_logger.log 2>&1

Old rows (> 7 days) are pruned automatically on each run to keep the DB lean.
This is independent of fetch_data.py / push_readings.py / hourly_update.sh,
which handle the hourly Excel + SFTP pipeline using the separate /history
endpoint (also 5-min limited, but on its own schedule via cron).
"""

import os
import json
import sqlite3
import time
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

API_BASE    = os.getenv("SIGEN_API_BASE", "https://openapi-eu.sigencloud.com")
USERNAME    = os.getenv("SIGEN_USERNAME")
PASSWORD    = os.getenv("SIGEN_PASSWORD")
SYSTEM_ID   = os.getenv("SIGEN_SYSTEM_ID", "HUCUD1764140703")

# This site has 3 inverters. Confirmed via the /openapi/system/{systemId}/devices
# endpoint on 2026-06-30 - these are the real serial numbers (the old single
# SIGEN_INVERTER_SN in .env was stale/wrong, which is why all device fields
# were reading back as 0). Querying all 3 in one run is safe: the "one access
# per device every 5 minutes" rate limit is per-device, and this script also
# runs once every 5 minutes, so each inverter gets touched once per window.
INVERTER_SNS = [
    os.getenv("SIGEN_INVERTER_SN_1", "110A133M0196"),
    os.getenv("SIGEN_INVERTER_SN_2", "110A133M0203"),
    os.getenv("SIGEN_INVERTER_SN_3", "110A133M0198"),
]

DB_PATH     = Path(__file__).parent / "live_readings.db"
LAT, LON    = -29.7215, 31.0498    # Umhlanga
SA_TZ       = timezone(timedelta(hours=2))

# ── DB helpers ────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_readings (
            ts           TEXT PRIMARY KEY,
            pv_kw        REAL, grid_kw      REAL,
            load_kw      REAL, battery_kw   REAL,
            battery_soc  REAL, cloud_cover  REAL,
            irradiance   REAL, temperature  REAL,
            wind_speed   REAL, precipitation REAL,
            pv_daily_kwh REAL, pv_month_kwh REAL,
            pv_year_kwh  REAL, pv_life_kwh  REAL,
            inv1_temp REAL, inv2_temp REAL, inv3_temp REAL,
            inv1_pv_kw REAL, inv2_pv_kw REAL, inv3_pv_kw REAL,
            bat_dischd   REAL, co2_saved    REAL,
            coal_saved   REAL, trees        REAL,
            inv1_pv1_v REAL, inv1_pv1_a REAL, inv1_pv2_v REAL, inv1_pv2_a REAL,
            inv1_pv3_v REAL, inv1_pv3_a REAL, inv1_pv4_v REAL, inv1_pv4_a REAL,
            inv2_pv1_v REAL, inv2_pv1_a REAL, inv2_pv2_v REAL, inv2_pv2_a REAL,
            inv2_pv3_v REAL, inv2_pv3_a REAL, inv2_pv4_v REAL, inv2_pv4_a REAL,
            inv3_pv1_v REAL, inv3_pv1_a REAL, inv3_pv2_v REAL, inv3_pv2_a REAL,
            inv3_pv3_v REAL, inv3_pv3_a REAL, inv3_pv4_v REAL, inv3_pv4_a REAL,
            phase_a_v REAL, phase_a_a REAL,
            phase_b_v REAL, phase_b_a REAL,
            phase_c_v REAL, phase_c_a REAL,
            power_factor REAL, grid_freq REAL
        )
    """)
    # Separate weather table that is NEVER pruned - this builds an indefinite
    # historical weather record for the site (Umhlanga), independent of the
    # 7-day rolling power data above. Same 5-minute cadence as the Sigenergy
    # fetch, since Open-Meteo has no meaningful rate limit for this volume.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_history (
            ts           TEXT PRIMARY KEY,
            temperature  REAL,
            cloud_cover  REAL,
            wind_speed   REAL,
            precipitation REAL,
            irradiance   REAL
        )
    """)
    conn.commit()
    conn.close()

def save_row(ts: str, data: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO live_readings VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
         ?,?,?,
         ?,?,?,
         ?,?,?,?,
         ?,?,?,?,?,?,?,?,
         ?,?,?,?,?,?,?,?,
         ?,?,?,?,?,?,?,?,
         ?,?,?,?,?,?,
         ?,?)
    """, (
        ts,
        data.get("pv_kw", 0),         data.get("grid_kw", 0),
        data.get("load_kw", 0),       data.get("battery_kw", 0),
        data.get("battery_soc", 0),   data.get("cloud_cover", 0),
        data.get("irradiance", 0),    data.get("temperature", 0),
        data.get("wind_speed", 0),    data.get("precipitation", 0),
        data.get("pv_daily_kwh", 0),  data.get("pv_month_kwh", 0),
        data.get("pv_year_kwh", 0),   data.get("pv_life_kwh", 0),
        data.get("inv1_temp", 0), data.get("inv2_temp", 0), data.get("inv3_temp", 0),
        data.get("inv1_pv_kw", 0), data.get("inv2_pv_kw", 0), data.get("inv3_pv_kw", 0),
        data.get("bat_dischd", 0),    data.get("co2_saved", 0),
        data.get("coal_saved", 0),    data.get("trees", 0),
        data.get("inv1_pv1_v", 0), data.get("inv1_pv1_a", 0),
        data.get("inv1_pv2_v", 0), data.get("inv1_pv2_a", 0),
        data.get("inv1_pv3_v", 0), data.get("inv1_pv3_a", 0),
        data.get("inv1_pv4_v", 0), data.get("inv1_pv4_a", 0),
        data.get("inv2_pv1_v", 0), data.get("inv2_pv1_a", 0),
        data.get("inv2_pv2_v", 0), data.get("inv2_pv2_a", 0),
        data.get("inv2_pv3_v", 0), data.get("inv2_pv3_a", 0),
        data.get("inv2_pv4_v", 0), data.get("inv2_pv4_a", 0),
        data.get("inv3_pv1_v", 0), data.get("inv3_pv1_a", 0),
        data.get("inv3_pv2_v", 0), data.get("inv3_pv2_a", 0),
        data.get("inv3_pv3_v", 0), data.get("inv3_pv3_a", 0),
        data.get("inv3_pv4_v", 0), data.get("inv3_pv4_a", 0),
        data.get("phase_a_v", 0), data.get("phase_a_a", 0),
        data.get("phase_b_v", 0), data.get("phase_b_a", 0),
        data.get("phase_c_v", 0), data.get("phase_c_a", 0),
        data.get("power_factor", 0), data.get("grid_freq", 0),
    ))
    conn.commit()
    conn.close()

def save_weather_row(ts: str, weather: dict):
    """Save a row to the permanent weather_history table. Never pruned -
    this is the standing historical weather record for the site."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO weather_history VALUES (?,?,?,?,?,?)
    """, (
        ts,
        weather.get("temperature", 0),
        weather.get("cloud_cover", 0),
        weather.get("wind_speed", 0),
        weather.get("precipitation", 0),
        weather.get("irradiance", 0),
    ))
    conn.commit()
    conn.close()

def prune_old_rows(days=7):
    """Delete rows older than `days` from live_readings ONLY (the rolling
    power-data table that feeds the 7-day trend charts). weather_history is
    deliberately never pruned here - it's the standing historical weather
    record for the site and is meant to grow indefinitely."""
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    deleted = conn.execute(
        "DELETE FROM live_readings WHERE ts < ?", (cutoff,)
    ).rowcount
    conn.commit()
    conn.close()
    return deleted

# ── Sigenergy API helpers ───────────────────────────────────────────────────────

def get_token(retry_on_failure: bool = True) -> str:
    try:
        r = requests.post(
            f"{API_BASE}/openapi/auth/login/password",
            json={"username": USERNAME, "password": PASSWORD},
            timeout=15
        )
    except requests.exceptions.Timeout:
        if retry_on_failure:
            print("    [get_token]  timeout, retrying once...")
            time.sleep(3)
            return get_token(retry_on_failure=False)
        raise RuntimeError("Auth request timed out twice in a row.")

    try:
        body = r.json()
    except json.JSONDecodeError:
        raw_preview = (r.text or "")[:200]
        if retry_on_failure:
            print(f"    [get_token]  status={r.status_code} body_len={len(r.text or '')} "
                  f"raw={raw_preview!r} - retrying once...")
            time.sleep(3)
            return get_token(retry_on_failure=False)
        raise RuntimeError(
            f"Auth response not valid JSON after retry: status={r.status_code} "
            f"raw={raw_preview!r}"
        )

    if body.get("code") != 0:
        raise RuntimeError(f"Auth failed: {body.get('msg')} (code {body.get('code')})")
    d = body["data"]
    if isinstance(d, str):
        d = json.loads(d)
    return d["accessToken"]

def api_get(token: str, path: str, retry_on_failure: bool = True) -> dict | None:
    """GET a Sigenergy endpoint. Retries once after a short pause if the
    request times out OR if the response body is empty/malformed (observed
    in practice: occasionally a device returns an empty body that fails to
    parse as JSON, distinct from a proper error response with a code field).
    """
    try:
        r = requests.get(
            f"{API_BASE}{path}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15
        )
    except requests.exceptions.Timeout:
        if retry_on_failure:
            print(f"    [api_get]  {path} -> timeout, retrying once...")
            time.sleep(3)
            return api_get(token, path, retry_on_failure=False)
        print(f"    [api_get]  {path} -> timed out again on retry, giving up.")
        return None

    try:
        body = r.json()
    except json.JSONDecodeError:
        # Log the raw response details so we can tell apart an actually-empty
        # body, a non-JSON error page, a 5xx, etc. - "Expecting value" alone
        # doesn't distinguish these.
        raw_preview = (r.text or "")[:200]
        if retry_on_failure:
            print(f"    [api_get]  {path} -> status={r.status_code} "
                  f"body_len={len(r.text or '')} raw={raw_preview!r} "
                  f"- retrying once...")
            time.sleep(3)
            return api_get(token, path, retry_on_failure=False)
        print(f"    [api_get]  {path} -> status={r.status_code} "
              f"body_len={len(r.text or '')} raw={raw_preview!r} "
              f"- failed again on retry, giving up.")
        return None

    if body.get("code") != 0:
        print(f"    [api_get]  {path} -> code={body.get('code')} msg={body.get('msg')}")
        return None
    d = body.get("data", {})
    return json.loads(d) if isinstance(d, str) else d

# ── Weather helper ───────────────────────────────────────────────────────────────

def fetch_weather() -> dict:
    """Fetch current weather + irradiance from Open-Meteo. No API key, no rate limit
    that matters here - this is independent of the Sigenergy 5-minute restriction."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":  LAT,
                "longitude": LON,
                "current":   "temperature_2m,cloud_cover,wind_speed_10m,precipitation",
                "hourly":    "shortwave_radiation",
                "forecast_days": 1,
                "timezone":  "Africa/Johannesburg",
            },
            timeout=10
        )
        d = r.json()
        curr = d.get("current", {})
        now_hour = datetime.now().strftime("%Y-%m-%dT%H:00")
        hourly_times = d.get("hourly", {}).get("time", [])
        irr = 0.0
        if now_hour in hourly_times:
            irr = d["hourly"]["shortwave_radiation"][hourly_times.index(now_hour)] or 0.0
        return {
            "temperature":   curr.get("temperature_2m", 0),
            "cloud_cover":   curr.get("cloud_cover", 0),
            "wind_speed":    curr.get("wind_speed_10m", 0),
            "precipitation": curr.get("precipitation", 0),
            "irradiance":    irr,
        }
    except Exception as e:
        print(f"  [WARN]  Weather fetch failed: {e}")
        return {"temperature": 0, "cloud_cover": 0, "wind_speed": 0, "precipitation": 0, "irradiance": 0}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now_sa  = datetime.now(SA_TZ).strftime("%Y-%m-%d %H:%M:%S SAST")
    now_utc = datetime.now(timezone.utc).isoformat()

    print(f"[{now_sa}]  live_logger starting...")

    init_db()

    # Fetch Sigenergy live data
    try:
        token   = get_token()
        flow    = api_get(token, f"/openapi/systems/{SYSTEM_ID}/energyFlow")
        summary = api_get(token, f"/openapi/systems/{SYSTEM_ID}/summary")

        if not flow:
            print("  [WARN]  No energy flow data returned - skipping this run.")
            return

        data = {
            "pv_kw":        float(flow.get("pvPower", 0) or 0),
            "grid_kw":      float(flow.get("gridPower", 0) or 0),
            "load_kw":      float(flow.get("loadPower", 0) or 0),
            "battery_kw":   float(flow.get("batteryPower", 0) or 0),
            "battery_soc":  float(flow.get("batterySoc", 0) or 0),
            "pv_daily_kwh": float(summary.get("dailyPowerGeneration", 0) or 0) if summary else 0,
            "pv_month_kwh": float(summary.get("monthlyPowerGeneration", 0) or 0) if summary else 0,
            "pv_year_kwh":  float(summary.get("annualPowerGeneration", 0) or 0) if summary else 0,
            "pv_life_kwh":  float(summary.get("lifetimePowerGeneration", 0) or 0) if summary else 0,
            "co2_saved":    float(summary.get("lifetimeCo2", 0) or 0) if summary else 0,
            "coal_saved":   float(summary.get("lifetimeCoal", 0) or 0) if summary else 0,
            "trees":        float(summary.get("lifetimeTreeEquivalent", 0) or 0) if summary else 0,
            "bat_dischd":   0.0,  # filled in below from inverter 1's device data
        }

        # This site has 3 inverters - query each one's realtimeInfo in this
        # same run. Each inverter is touched once per 5-minute window, which
        # respects the documented "one access per device every 5 minutes"
        # rate limit (it's per-device, not per-account-wide for this endpoint).
        any_device_ok = False
        for i, sn in enumerate(INVERTER_SNS, start=1):
            try:
                dev_raw = api_get(token, f"/openapi/systems/{SYSTEM_ID}/devices/{sn}/realtimeInfo")
                dev = dev_raw.get("realTimeInfo", dev_raw) if dev_raw else {}
                if not dev:
                    if dev_raw is None:
                        print(f"  [WARN]  Inverter {i} ({sn}): api_get returned None "
                              f"(see [api_get] line above for the actual error code/msg).")
                    else:
                        print(f"  [WARN]  Inverter {i} ({sn}): api_get succeeded but "
                              f"realTimeInfo was empty. dev_raw keys: {list(dev_raw.keys())}")
                    continue
                any_device_ok = True

                data[f"inv{i}_temp"]  = float(dev.get("internalTemperature", 0) or 0)
                data[f"inv{i}_pv_kw"] = float(dev.get("pvTotalPower", 0) or 0)
                for s in range(1, 5):
                    data[f"inv{i}_pv{s}_v"] = float(dev.get(f"pV{s}Voltage", 0) or 0)
                    data[f"inv{i}_pv{s}_a"] = float(dev.get(f"pV{s}Current", 0) or 0)

                # Battery discharge today, phase voltages/currents, power
                # factor, and grid frequency are read from the shared grid
                # connection point - inverter 1 is used as the representative
                # reading since all 3 inverters tie into the same site grid.
                if i == 1:
                    data["bat_dischd"]    = float(dev.get("esDischargingDay", 0) or 0)
                    data["phase_a_v"]     = float(dev.get("aPhaseVoltage", 0) or 0)
                    data["phase_a_a"]     = float(dev.get("aPhaseCurrent", 0) or 0)
                    data["phase_b_v"]     = float(dev.get("bPhaseVoltage", 0) or 0)
                    data["phase_b_a"]     = float(dev.get("bPhaseCurrent", 0) or 0)
                    data["phase_c_v"]     = float(dev.get("cPhaseVoltage", 0) or 0)
                    data["phase_c_a"]     = float(dev.get("cPhaseCurrent", 0) or 0)
                    data["power_factor"]  = float(dev.get("powerFactor", 0) or 0)
                    data["grid_freq"]     = float(dev.get("gridFrequency", 0) or 0)
            except Exception as e:
                print(f"  [WARN]  Inverter {i} ({sn}) fetch failed: {e}")

        if not any_device_ok:
            print("  [WARN]  No inverter device data returned this run - "
                  "energyFlow/summary still saved, but per-inverter fields will be 0.")

    except Exception as e:
        print(f"  [ERROR]  Sigenergy API failed: {e}")
        return

    # Fetch weather (same run, same 5-min cadence per your preference)
    weather = fetch_weather()
    data.update(weather)

    # Write to DB - power data to the rolling 7-day table, weather to both
    # the rolling table (for the dashboard's combined power+weather charts)
    # AND the permanent weather_history table (standing site weather record)
    save_row(now_utc, data)
    save_weather_row(now_utc, weather)

    # Prune old rows (live_readings only - weather_history is permanent)
    pruned = prune_old_rows(days=7)

    print(
        f"  [OK]  Saved: PV={data['pv_kw']:.2f}kW  "
        f"Load={data['load_kw']:.2f}kW  "
        f"Batt={data['battery_soc']:.0f}%SoC  "
        f"Inv1={data.get('inv1_temp',0):.1f}C "
        f"Inv2={data.get('inv2_temp',0):.1f}C "
        f"Inv3={data.get('inv3_temp',0):.1f}C  "
        f"WxTemp={data['temperature']:.1f}C  "
        f"Cloud={data['cloud_cover']:.0f}%  "
        f"Pruned={pruned} old rows"
    )

if __name__ == "__main__":
    main()