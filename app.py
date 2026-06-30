import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import time
import numpy as np
import os

st.set_page_config(
    page_title="Voltano Billing Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── TOU CLASSIFICATION (config-driven, multi-period) ─────────────────────────
# Schedule and tariff rates are loaded from tou_tariffs.json as a list of
# PERIODS, each with its own effective_from date. This means when tariffs
# change each year, a new period is appended — historical data keeps using
# whichever schedule/rates were in effect at the time, instead of being
# silently recalculated with the newest rates. See load_tou_config() below.

import json as _json_tou
import bisect as _bisect_tou

@st.cache_data
def load_tou_config() -> dict:
    """Load TOU periods from tou_tariffs.json, sorted by effective_from."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tou_tariffs.json")
    if not os.path.exists(config_path):
        st.error(
            "tou_tariffs.json not found. This file defines the TOU time slots "
            "and tariff rates — the app cannot calculate billing without it."
        )
        st.stop()
    with open(config_path) as f:
        raw = _json_tou.load(f)
    periods = sorted(raw["periods"], key=lambda p: p["effective_from"])
    return periods

TOU_PERIODS = load_tou_config()
_TOU_PERIOD_STARTS = [pd.Timestamp(p["effective_from"]) for p in TOU_PERIODS]

# Most recent period's rates are used for "current" displays (e.g. live tab).
# Kept for backward compatibility with code that references TARIFF/SELL_RATE directly.
TARIFF    = TOU_PERIODS[-1]["tariffs_rkwh"]
SELL_RATE = TOU_PERIODS[-1]["sell_rate_rkwh"]

def get_period_for_date(dt) -> dict:
    """Return the tariff period dict that was in effect on the given date.
    Strips timezone info before comparing, since _TOU_PERIOD_STARTS are
    timezone-naive (parsed from plain date strings in tou_tariffs.json),
    while callers like the live dashboard may pass timezone-aware datetimes."""
    d = pd.Timestamp(dt)
    if d.tzinfo is not None:
        d = d.tz_localize(None)
    idx = _bisect_tou.bisect_right(_TOU_PERIOD_STARTS, d) - 1
    idx = max(0, min(idx, len(TOU_PERIODS) - 1))
    return TOU_PERIODS[idx]

def get_season(dt):
    """Return 'high' or 'low' demand season, using the period in effect on dt."""
    period = get_period_for_date(dt)
    return "high" if dt.month in period["season_months"]["high"] else "low"

def _hour_in_ranges(h, ranges):
    """Check if hour h falls in any [start, end) range from the config."""
    return any(start <= h < end for start, end in ranges)

def get_tou_slot(dt):
    """
    Classify a datetime into Peak (1.8.1), Standard (1.8.2), or Off-Peak (1.8.3),
    using whichever tariff period (schedule) was in effect on that date.
    """
    period = get_period_for_date(dt)
    season = "high" if dt.month in period["season_months"]["high"] else "low"
    dow = dt.weekday()   # 0=Mon, 5=Sat, 6=Sun
    h = dt.hour

    if dow == 6:
        day_type = "sunday"
    elif dow == 5:
        day_type = "saturday"
    else:
        day_type = "weekday"

    slots = period["schedule"][day_type][season]
    if "peak" in slots and _hour_in_ranges(h, slots["peak"]):
        return "1.8.1"
    if "standard" in slots and _hour_in_ranges(h, slots["standard"]):
        return "1.8.2"
    return "1.8.3"

def get_tariff_for_date(dt, tou_slot: str) -> float:
    """Return the R/kWh rate for a given datetime + TOU slot, using the
    period that was in effect on that date. Use this instead of TARIFF[s][t]
    directly anywhere a specific date is involved."""
    period = get_period_for_date(dt)
    season = "high" if dt.month in period["season_months"]["high"] else "low"
    return period["tariffs_rkwh"][season][tou_slot]

def get_sell_rate_for_date(dt) -> float:
    """Return the flat sell rate (R/kWh) in effect on the given date."""
    return get_period_for_date(dt)["sell_rate_rkwh"]


def assign_tou_vectorised(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Fully vectorised TOU slot, season, and tariff assignment, period-aware.
    Adds columns: tou_slot, season, tariff_rkwh, sell_rate_rkwh.
    Rows are grouped by which tariff period they fall under (by effective_from
    date) BEFORE the TOU schedule is applied, so historical rows always use
    the schedule/rates that were actually in effect at the time — even after
    a new tariff period is added to tou_tariffs.json for a later year.
    """
    hourly = hourly.copy()
    dt  = hourly["datetime"]
    h   = dt.dt.hour
    dow = dt.dt.weekday          # 0=Mon ... 6=Sun
    mon = dt.dt.month

    period_starts_arr = np.array(_TOU_PERIOD_STARTS, dtype="datetime64[ns]")
    dt_arr = dt.values.astype("datetime64[ns]")
    period_idx = np.clip(
        np.searchsorted(period_starts_arr, dt_arr, side="right") - 1,
        0, len(TOU_PERIODS) - 1
    )

    hourly["tou_slot"]      = "1.8.3"
    hourly["season"]        = ""
    hourly["tariff_rkwh"]   = 0.0
    hourly["sell_rate_rkwh"] = 0.0

    is_weekday = dow < 5
    is_sat     = dow == 5
    is_sun     = dow == 6

    for p_idx, period in enumerate(TOU_PERIODS):
        in_period = period_idx == p_idx
        if not in_period.any():
            continue

        is_high = mon.isin(period["season_months"]["high"]) & in_period
        is_low  = in_period & ~is_high
        hourly.loc[is_high, "season"] = "high"
        hourly.loc[is_low, "season"]  = "low"
        hourly.loc[in_period, "sell_rate_rkwh"] = period["sell_rate_rkwh"]

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
                    std_mask = pd.Series(False, index=hourly.index)
                    for start, end in slots_cfg["standard"]:
                        std_mask = std_mask | h.between(start, end - 1)
                    hourly.loc[base_mask & std_mask, "tou_slot"] = "1.8.2"

                if "peak" in slots_cfg:
                    peak_mask = pd.Series(False, index=hourly.index)
                    for start, end in slots_cfg["peak"]:
                        peak_mask = peak_mask | h.between(start, end - 1)
                    hourly.loc[base_mask & peak_mask, "tou_slot"] = "1.8.1"

        # Assign tariff rate per row based on season + tou_slot, within this period only
        for season_name in ["low", "high"]:
            for slot_code in ["1.8.1", "1.8.2", "1.8.3"]:
                mask = in_period & (hourly["season"] == season_name) & (hourly["tou_slot"] == slot_code)
                if mask.any():
                    rate = period["tariffs_rkwh"][season_name][slot_code]
                    hourly.loc[mask, "tariff_rkwh"] = rate

    return hourly

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data(filepath):
    """Load and process both sheets from the Excel file."""
    xl = pd.ExcelFile(filepath)

    # ── Monthly / daily sheet ──
    daily = pd.read_excel(xl, sheet_name=xl.sheet_names[0])
    daily.columns = daily.columns.str.strip()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.rename(columns={
        "Billing run":                       "billing_run",
        "Date":                              "date",
        "Solar Production Energy (kWh)":     "solar_kwh",
        "Load Consumed Energy (kWh)":        "load_kwh",
        "Battery Charge Energy (kWh)":       "batt_charge_kwh",
        "Battery Discharge Energy (kWh)":    "batt_discharge_kwh",
        "From Generator (kWh)":              "generator_kwh",
        "Grid Imported Energy (kWh)":        "grid_import_kwh",
        "Grid Exported Energy (kWh)":        "grid_export_kwh",
        "Revenue(R)":                        "revenue_r",
    })

    # ── Hourly sheet ──
    hourly = pd.read_excel(xl, sheet_name=xl.sheet_names[1])
    hourly.columns = hourly.columns.str.strip()
    hourly["Date"] = pd.to_datetime(hourly["Date"])
    hourly = hourly.rename(columns={
        "Date":                                     "datetime",
        "Total Solar Production Energy (kWh)":      "solar_kwh",
        "Total Load Consumed Energy (kWh)":         "load_kwh",
        "Total Battery Charge Energy (kWh)":        "batt_charge_kwh",
        "Total Battery Discharge Energy (kWh)":     "batt_discharge_kwh",
        "Total Grid Imported Energy (kWh)":         "grid_import_kwh",
        "Total Grid Exported Energy (kWh)":         "grid_export_kwh",
    })

    # Assign TOU slot and season — vectorised (fast)
    hourly = assign_tou_vectorised(hourly)
    hourly["date"] = hourly["datetime"].dt.date

    return daily, hourly

# ─── METER READINGS ───────────────────────────────────────────────────────────

# All energy columns available in the hourly sheet, with display labels
METER_COLUMNS = {
    "Grid Import (kWh)":       "grid_import_kwh",
    "Grid Export (kWh)":       "grid_export_kwh",
    "Solar Production (kWh)":  "solar_kwh",
    "Load Consumed (kWh)":     "load_kwh",
    "Battery Charge (kWh)":    "batt_charge_kwh",
    "Battery Discharge (kWh)": "batt_discharge_kwh",
}

def compute_meter_readings(hourly: pd.DataFrame, billing_run_dates: dict, site_name: str, col: str):
    """
    For each billing run, compute cumulative meter readings (kWh) from
    the start date (5 Dec 2025, reading = 0) up to the end of each period,
    for the specified energy column.

    Returns a DataFrame with one row per billing run showing:
      x.0  total
      x.1  peak
      x.2  standard
      x.3  off-peak
    """
    start_date = pd.Timestamp("2025-12-05")
    rows = []

    for billing_run, end_date in sorted(billing_run_dates.items(), key=lambda x: x[1]):
        mask = (hourly["datetime"] >= start_date) & (hourly["datetime"] < end_date + pd.Timedelta(days=1))
        period = hourly[mask]

        total   = period[col].sum()
        peak    = period[period["tou_slot"] == "1.8.1"][col].sum()
        std     = period[period["tou_slot"] == "1.8.2"][col].sum()
        offpeak = period[period["tou_slot"] == "1.8.3"][col].sum()

        rows.append({
            "Site":           site_name,
            "Billing Run":    billing_run,
            "Period End":     end_date.date(),
            "x.0 Total":      round(total, 3),
            "x.1 Peak":       round(peak, 3),
            "x.2 Standard":   round(std, 3),
            "x.3 Off-Peak":   round(offpeak, 3),
        })

    return pd.DataFrame(rows)


def compute_usage(hourly: pd.DataFrame, billing_run_dates: dict, site_name: str, col: str):
    """
    For each billing run, compute the usage (kWh) consumed WITHIN that specific
    billing period only (not cumulative). This is the delta between consecutive
    meter readings — i.e. what the database should bill for each period.

    Returns a DataFrame with one row per billing run showing:
      x.0  total usage for the period
      x.1  peak usage for the period
      x.2  standard usage for the period
      x.3  off-peak usage for the period
    """
    start_date = pd.Timestamp("2025-12-05")
    sorted_runs = sorted(billing_run_dates.items(), key=lambda x: x[1])
    rows = []

    for i, (billing_run, end_date) in enumerate(sorted_runs):
        # Period start: day after the previous billing run ended (or meter start for first run)
        if i == 0:
            period_start = start_date
        else:
            prev_end = sorted_runs[i - 1][1]
            period_start = prev_end + pd.Timedelta(days=1)

        mask = (
            (hourly["datetime"] >= period_start) &
            (hourly["datetime"] < end_date + pd.Timedelta(days=1))
        )
        period = hourly[mask]

        total   = period[col].sum()
        peak    = period[period["tou_slot"] == "1.8.1"][col].sum()
        std     = period[period["tou_slot"] == "1.8.2"][col].sum()
        offpeak = period[period["tou_slot"] == "1.8.3"][col].sum()

        rows.append({
            "Site":           site_name,
            "Billing Run":    billing_run,
            "Period Start":   period_start.date(),
            "Period End":     end_date.date(),
            "x.0 Total":      round(total, 3),
            "x.1 Peak":       round(peak, 3),
            "x.2 Standard":   round(std, 3),
            "x.3 Off-Peak":   round(offpeak, 3),
        })

    return pd.DataFrame(rows)


# ─── UI ───────────────────────────────────────────────────────────────────────

st.title("⚡ Voltano Billing & Performance Tool")
st.caption("Internal use · MIL Estate · Virtual Meter TOU Tracker")

# Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Data")
    site_name = st.text_input("Site name", value="MIL Estate")

    # ── Auto-load from fixed path ─────────────────────────────────────────
    # Place your Excel file in the same folder as app.py and update the
    # filename below if it ever changes.
    DATA_FILE = os.path.join(os.path.dirname(__file__), "MIL_Battery_readings_EMS.xlsx")

    if os.path.exists(DATA_FILE):
        file_mtime = os.path.getmtime(DATA_FILE)
        last_updated = pd.Timestamp(file_mtime, unit="s").strftime("%d %b %Y %H:%M")
        st.success(f"✅ Data file loaded")
        st.caption(f"File: `{os.path.basename(DATA_FILE)}`  \nLast modified: {last_updated}")
        if st.button("🔄 Reload data", help="Click after updating the Excel file to refresh the app"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.error(
            f"❌ Data file not found.\n\n"
            f"Place **{os.path.basename(DATA_FILE)}** in the same folder as `app.py` and restart."
        )
        st.stop()

    st.divider()
    st.header("⚙️ Settings")
    show_raw = st.checkbox("Show raw hourly data", value=False)

# ── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    daily, hourly = load_data(DATA_FILE)

# ── Billing runs from JSON config ──────────────────────────────────────────────
import json as _json

def load_billing_runs_config() -> dict:
    """
    Load billing run schedule from billing_runs.json.
    Returns dict of {run_name: end_date as pd.Timestamp} sorted chronologically.
    Falls back to grouping by billing_run column if JSON not found.
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "billing_runs.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            data = _json.load(f)
        runs = sorted(data["billing_runs"], key=lambda r: r["end_date"])
        return {r["name"]: pd.Timestamp(r["end_date"]) for r in runs}
    else:
        # Fallback: derive from data
        st.warning("billing_runs.json not found — using dates from data. Add billing_runs.json for accurate billing periods.")
        runs = daily.groupby("billing_run")["date"].max().to_dict()
        return {k: pd.Timestamp(v) for k, v in runs.items()}

billing_runs = load_billing_runs_config()

# Re-assign billing_run column in daily data using the JSON config
# so sorting and grouping is always correct
def compute_grid_import_tou_share(hourly: pd.DataFrame, billing_run_dates: dict) -> pd.DataFrame:
    """
    For each billing run, compute what % of TOTAL grid import (kWh) fell into
    each TOU slot (Peak / Standard / Off-Peak). This is %-based rather than
    raw kWh because total load and total solar generation both change over
    time (new appliances, seasonal variation, panel degradation, etc.) - a
    rising kWh of peak import could just mean "the site uses more power now",
    whereas a rising % of peak import specifically means "the battery
    strategy is shifting less of the load away from peak than it used to".

    This is the right metric to judge whether the battery dispatch strategy
    (charge off-peak/solar, discharge at peak on weekdays and into the
    weekend standard windows) is actually working, independent of how much
    the site's overall consumption has grown or shrunk.

    Returns a DataFrame with one row per billing run:
      Period, Period Start, Period End, Total Grid Import (kWh),
      Peak %, Standard %, Off-Peak %,
      Peak pp Change, Standard pp Change, Off-Peak pp Change
    (pp = percentage points vs. the immediately preceding billing run -
    NOT vs. the same row's own kWh, since these are share-of-total deltas)
    """
    start_date = pd.Timestamp("2025-12-05")
    sorted_runs = sorted(billing_run_dates.items(), key=lambda x: x[1])
    rows = []

    for i, (billing_run, end_date) in enumerate(sorted_runs):
        if i == 0:
            period_start = start_date
        else:
            prev_end = sorted_runs[i - 1][1]
            period_start = prev_end + pd.Timedelta(days=1)

        mask = (
            (hourly["datetime"] >= period_start) &
            (hourly["datetime"] < end_date + pd.Timedelta(days=1))
        )
        period = hourly[mask]

        total_import = period["grid_import_kwh"].sum()
        peak_import    = period[period["tou_slot"] == "1.8.1"]["grid_import_kwh"].sum()
        std_import     = period[period["tou_slot"] == "1.8.2"]["grid_import_kwh"].sum()
        offpeak_import = period[period["tou_slot"] == "1.8.3"]["grid_import_kwh"].sum()

        if total_import > 0:
            peak_pct    = round(100 * peak_import / total_import, 2)
            std_pct     = round(100 * std_import / total_import, 2)
            offpeak_pct = round(100 * offpeak_import / total_import, 2)
        else:
            peak_pct = std_pct = offpeak_pct = 0.0

        rows.append({
            "Billing Run":              billing_run,
            "Period Start":             period_start.date(),
            "Period End":               end_date.date(),
            "Total Grid Import (kWh)":  round(total_import, 1),
            "Peak %":                   peak_pct,
            "Standard %":               std_pct,
            "Off-Peak %":               offpeak_pct,
        })

    df = pd.DataFrame(rows)

    # Month-over-month change in percentage points (this row's % minus the
    # previous row's %) - this is the "are we shaving more or less than last
    # period" signal. Positive Peak pp Change = a BIGGER share of grid import
    # is now happening at peak than last period (worse shaving). Negative =
    # smaller share at peak (better shaving).
    for col in ["Peak %", "Standard %", "Off-Peak %"]:
        change_col = col.replace(" %", " pp Change")
        df[change_col] = df[col].diff().round(2)

    return df

def assign_billing_run(d: pd.Timestamp, runs: dict) -> str:
    """Assign a date to the correct billing run based on end dates."""
    d_date = d.date() if hasattr(d, "date") else d
    for run_name, end_date in sorted(runs.items(), key=lambda x: x[1]):
        if pd.Timestamp(d_date) <= end_date:
            return run_name
    return "Pending"

daily["billing_run"] = daily["date"].apply(lambda d: assign_billing_run(d, billing_runs))
daily = daily.sort_values("date").reset_index(drop=True)

# ─── TARIFF CONSTANTS ─────────────────────────────────────────────────────────
# TARIFF and SELL_RATE are loaded from tou_tariffs.json near the top of this
# file (see load_tou_config()). Edit that JSON file to update rates each year.

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔢 Meter Readings",
    "📊 System Performance",
    "💰 Profitability",
    "🔋 Battery Health",
    "🌤️ Seasonal Patterns",
    "⚡ Live Dashboard",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — METER READINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Virtual Meter — TOU Breakdown")

    # ── Controls row ──────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([2, 1])

    with ctrl_col1:
        selected_label = st.selectbox(
            "📟 Energy channel",
            options=list(METER_COLUMNS.keys()),
            index=0,
            help="Choose which energy flow to compute TOU readings for.",
        )
    with ctrl_col2:
        view_mode = st.radio(
            "View mode",
            options=["Meter Reading", "Period Usage"],
            horizontal=True,
            help=(
                "Meter Reading — cumulative total from meter start (5 Dec 2025 = 0). "
                "Period Usage — energy consumed within each billing period only."
            ),
        )

    selected_col = METER_COLUMNS[selected_label]

    # Register prefix for column headers
    register_prefix = {
        "grid_import_kwh":    "1.8",
        "grid_export_kwh":    "2.8",
        "solar_kwh":          "S.0",
        "load_kwh":           "L.0",
        "batt_charge_kwh":    "B.C",
        "batt_discharge_kwh": "B.D",
    }.get(selected_col, "x")

    # ── Compute the right table ───────────────────────────────────────────────
    if view_mode == "Meter Reading":
        result_df = compute_meter_readings(hourly, billing_runs, site_name, selected_col)
        mode_caption = (
            "Cumulative reading from meter start (5 Dec 2025 = 0.000 kWh). "
            "Each row shows the meter value AT THE END of that billing run."
        )
        filename_suffix = "meter_readings"
        # No Period Start column for readings
        display_df = result_df.drop(columns=["Site"])
    else:
        result_df = compute_usage(hourly, billing_runs, site_name, selected_col)
        mode_caption = (
            "Energy consumed WITHIN each billing period only. "
            "Each row shows usage between Period Start and Period End."
        )
        filename_suffix = "period_usage"
        display_df = result_df.drop(columns=["Site"])

    # Rename x.N columns to use the correct register prefix
    display_df = display_df.rename(columns={
        "x.0 Total":    f"{register_prefix}.0  Total",
        "x.1 Peak":     f"{register_prefix}.1  Peak",
        "x.2 Standard": f"{register_prefix}.2  Standard",
        "x.3 Off-Peak": f"{register_prefix}.3  Off-Peak",
    })

    st.caption(mode_caption)

    fmt_cols = {c: "{:.3f}" for c in display_df.columns if any(x in c for x in [".0 ", ".1 ", ".2 ", ".3 "])}
    st.dataframe(
        display_df.style.format(fmt_cols),
        use_container_width=True,
        hide_index=True,
    )

    # Download
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download {view_mode.lower()} CSV",
        csv,
        file_name=f"{site_name}_{selected_col}_{filename_suffix}.csv",
        mime="text/csv",
    )

    st.divider()

    # ── Stacked bar chart ─────────────────────────────────────────────────────
    chart_title = f"{'Cumulative Meter Reading' if view_mode == 'Meter Reading' else 'Period Usage'} — {selected_label}"
    st.subheader(chart_title)

    fig_tou = go.Figure()
    fig_tou.add_bar(
        name=f"Peak ({register_prefix}.1)",
        x=result_df["Billing Run"], y=result_df["x.1 Peak"],
        marker_color="#E24B4A",
    )
    fig_tou.add_bar(
        name=f"Standard ({register_prefix}.2)",
        x=result_df["Billing Run"], y=result_df["x.2 Standard"],
        marker_color="#EF9F27",
    )
    fig_tou.add_bar(
        name=f"Off-Peak ({register_prefix}.3)",
        x=result_df["Billing Run"], y=result_df["x.3 Off-Peak"],
        marker_color="#1D9E75",
    )
    fig_tou.update_layout(
        barmode="stack",
        xaxis_title="Billing Run",
        yaxis_title=f"{'Cumulative' if view_mode == 'Meter Reading' else 'Period'} {selected_label}",
        legend_title="TOU Slot",
        height=400,
    )
    st.plotly_chart(fig_tou, use_container_width=True)

    # ── Usage trend line (only meaningful for Period Usage) ───────────────────
    if view_mode == "Period Usage":
        st.subheader("Period Usage Trend")
        fig_trend = go.Figure()
        fig_trend.add_scatter(
            x=result_df["Billing Run"], y=result_df["x.0 Total"],
            mode="lines+markers", name="Total",
            line=dict(color="#7F77DD", width=2), marker=dict(size=7),
        )
        fig_trend.add_scatter(
            x=result_df["Billing Run"], y=result_df["x.1 Peak"],
            mode="lines+markers", name="Peak",
            line=dict(color="#E24B4A", dash="dot"), marker=dict(size=5),
        )
        fig_trend.add_scatter(
            x=result_df["Billing Run"], y=result_df["x.2 Standard"],
            mode="lines+markers", name="Standard",
            line=dict(color="#EF9F27", dash="dot"), marker=dict(size=5),
        )
        fig_trend.add_scatter(
            x=result_df["Billing Run"], y=result_df["x.3 Off-Peak"],
            mode="lines+markers", name="Off-Peak",
            line=dict(color="#1D9E75", dash="dot"), marker=dict(size=5),
        )
        fig_trend.update_layout(
            xaxis_title="Billing Run",
            yaxis_title=selected_label,
            legend_title="Register",
            height=380,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Peak/Standard Shaving Effectiveness ────────────────────────────────────
    st.divider()
    st.subheader("⚖️ Peak/Standard Shaving Effectiveness")
    st.caption(
        "What % of total GRID IMPORT falls into each TOU slot, per billing period. "
        "This is %-based rather than raw kWh because total load and solar generation "
        "both change over time — a falling **% share at Peak/Standard** means the battery "
        "strategy is genuinely shifting more consumption away from expensive hours, "
        "independent of whether the site is using more or less power overall. "
        "The 'pp Change' columns show the percentage-point shift vs. the immediately "
        "preceding billing run. For **Peak** and **Standard**, negative is good (shrinking "
        "share of expensive import) and positive is bad. For **Off-Peak**, it's the mirror "
        "image — positive is good (more import successfully shifted to the cheapest hours) "
        "and negative is bad."
    )

    shaving_df = compute_grid_import_tou_share(hourly, billing_runs)

    if len(shaving_df) > 0:
        # Color logic differs by slot: for Peak/Standard, a SHRINKING share is
        # good (green) and a GROWING share is bad (red) - we want less import
        # at expensive hours. For Off-Peak, it's the mirror image: a GROWING
        # share is good (green) since that import shifted away from
        # Peak/Standard, and a SHRINKING share is bad (red).
        def _pp_change_color_lower_is_better(val):
            if pd.isna(val):
                return ""
            if val < 0:
                return "color: #1D9E75"   # improving - smaller share at this slot than last period
            elif val > 0:
                return "color: #E24B4A"   # worsening - bigger share at this slot than last period
            return ""

        def _pp_change_color_higher_is_better(val):
            if pd.isna(val):
                return ""
            if val > 0:
                return "color: #1D9E75"   # improving - more import shifted to off-peak than last period
            elif val < 0:
                return "color: #E24B4A"   # worsening - less import at off-peak than last period
            return ""

        pct_cols = ["Peak %", "Standard %", "Off-Peak %"]

        styled_shaving = shaving_df.style.format(
            {**{c: "{:.2f}%" for c in pct_cols},
             **{c: "{:+.2f} pp" for c in ["Peak pp Change", "Standard pp Change", "Off-Peak pp Change"]},
             "Total Grid Import (kWh)": "{:.1f}"}
        ).map(_pp_change_color_lower_is_better, subset=["Peak pp Change", "Standard pp Change"]
        ).map(_pp_change_color_higher_is_better, subset=["Off-Peak pp Change"])

        st.dataframe(styled_shaving, use_container_width=True, hide_index=True)

        # Download
        shaving_csv = shaving_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download shaving effectiveness CSV",
            shaving_csv,
            file_name=f"{site_name}_grid_import_tou_share.csv",
            mime="text/csv",
        )

        # ── Trend chart: % share by TOU slot over billing periods ─────────────
        fig_shaving = go.Figure()
        fig_shaving.add_scatter(
            x=shaving_df["Billing Run"], y=shaving_df["Peak %"],
            mode="lines+markers", name="Peak %",
            line=dict(color="#E24B4A", width=2), marker=dict(size=7),
        )
        fig_shaving.add_scatter(
            x=shaving_df["Billing Run"], y=shaving_df["Standard %"],
            mode="lines+markers", name="Standard %",
            line=dict(color="#EF9F27", width=2), marker=dict(size=7),
        )
        fig_shaving.add_scatter(
            x=shaving_df["Billing Run"], y=shaving_df["Off-Peak %"],
            mode="lines+markers", name="Off-Peak %",
            line=dict(color="#1D9E75", width=2), marker=dict(size=7),
        )
        fig_shaving.update_layout(
            xaxis_title="Billing Run",
            yaxis_title="% of Total Grid Import",
            yaxis=dict(range=[0, 100]),
            legend_title="TOU Slot",
            height=380,
            title="Grid Import Share by TOU Slot — Trend Across Billing Periods",
        )
        st.plotly_chart(fig_shaving, use_container_width=True)

        # ── Latest period vs. previous period summary callout ─────────────────
        if len(shaving_df) >= 2:
            latest = shaving_df.iloc[-1]
            prev   = shaving_df.iloc[-2]
            peak_change = latest["Peak pp Change"]
            std_change  = latest["Standard pp Change"]

            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                if pd.notna(peak_change):
                    direction = "improved" if peak_change < 0 else "worsened" if peak_change > 0 else "held steady"
                    st.metric(
                        f"Peak Share — {latest['Billing Run']} vs {prev['Billing Run']}",
                        f"{latest['Peak %']:.2f}%",
                        delta=f"{peak_change:+.2f} pp ({direction})",
                        delta_color="inverse",  # negative (smaller peak share) shown as "good" green
                    )
            with summary_col2:
                if pd.notna(std_change):
                    direction = "improved" if std_change < 0 else "worsened" if std_change > 0 else "held steady"
                    st.metric(
                        f"Standard Share — {latest['Billing Run']} vs {prev['Billing Run']}",
                        f"{latest['Standard %']:.2f}%",
                        delta=f"{std_change:+.2f} pp ({direction})",
                        delta_color="inverse",
                    )
    else:
        st.info("No billing run data available yet to calculate shaving effectiveness.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SYSTEM PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("System Performance Overview")

    # ── Date range slider ─────────────────────────────────────────────────────
    perf_min_date = daily["date"].min().to_pydatetime()
    perf_max_date = daily["date"].max().to_pydatetime()

    perf_date_range = st.slider(
        "Date range",
        min_value=perf_min_date,
        max_value=perf_max_date,
        value=(perf_min_date, perf_max_date),
        format="DD MMM YYYY",
        key="perf_slider",
    )

    perf_daily  = daily[(daily["date"] >= pd.Timestamp(perf_date_range[0])) &
                        (daily["date"] <= pd.Timestamp(perf_date_range[1]))]
    perf_hourly = hourly[(hourly["datetime"].dt.date >= perf_date_range[0].date()) &
                         (hourly["datetime"].dt.date <= perf_date_range[1].date())]

    # KPIs (reflect slider range)
    total_solar      = perf_daily["solar_kwh"].sum()
    total_load       = perf_daily["load_kwh"].sum()
    total_import     = perf_daily["grid_import_kwh"].sum()
    total_export     = perf_daily["grid_export_kwh"].sum()
    self_sufficiency = (1 - total_import / total_load) * 100 if total_load > 0 else 0
    solar_fraction   = (total_solar / total_load) * 100 if total_load > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Solar Produced", f"{total_solar:,.1f} kWh")
    c2.metric("Total Load Consumed",  f"{total_load:,.1f} kWh")
    c3.metric("Self-Sufficiency",     f"{self_sufficiency:.1f}%", help="% of load NOT sourced from grid")
    c4.metric("Solar Fraction",       f"{solar_fraction:.1f}%",   help="Solar as % of total load")

    st.divider()

    # Daily energy flows
    fig_flows = go.Figure()
    fig_flows.add_scatter(x=perf_daily["date"], y=perf_daily["solar_kwh"],       name="Solar",       line=dict(color="#EF9F27"))
    fig_flows.add_scatter(x=perf_daily["date"], y=perf_daily["load_kwh"],        name="Load",        line=dict(color="#E24B4A"))
    fig_flows.add_scatter(x=perf_daily["date"], y=perf_daily["grid_import_kwh"], name="Grid Import", line=dict(color="#7F77DD"))
    fig_flows.add_scatter(x=perf_daily["date"], y=perf_daily["grid_export_kwh"], name="Grid Export", line=dict(color="#1D9E75"))
    fig_flows.update_layout(
        title="Daily Energy Flows",
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        legend_title="Source",
        height=400,
    )
    st.plotly_chart(fig_flows, use_container_width=True)

    # Average hourly profile (within selected range)
    st.subheader("Average Daily Load & Solar Profile")
    hourly_avg = perf_hourly.groupby(perf_hourly["datetime"].dt.hour).agg(
        solar=("solar_kwh", "mean"),
        load=("load_kwh", "mean"),
        grid_import=("grid_import_kwh", "mean"),
    ).reset_index().rename(columns={"datetime": "hour"})

    fig_profile = go.Figure()
    fig_profile.add_bar(x=hourly_avg["hour"], y=hourly_avg["solar"],           name="Solar",       marker_color="#EF9F27")
    fig_profile.add_bar(x=hourly_avg["hour"], y=hourly_avg["load"],            name="Load",        marker_color="#E24B4A")
    fig_profile.add_scatter(x=hourly_avg["hour"], y=hourly_avg["grid_import"], name="Grid Import", line=dict(color="#7F77DD", dash="dash"))
    fig_profile.update_layout(
        barmode="overlay",
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1),
        yaxis_title="Avg Energy (kWh)",
        height=380,
    )
    st.plotly_chart(fig_profile, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROFITABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Revenue & Profitability")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROFITABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("💰 Profitability Analysis")
    st.caption("All calculations derived from actual metered data and your input parameters — not from the inverter revenue column.")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION P0 — INPUT PARAMETERS
    # ════════════════════════════════════════════════════════════════════════
    with st.expander("⚙️ System Cost & Rate Parameters", expanded=True):

        # ── 1. SOLAR PANELS ───────────────────────────────────────────────
        st.markdown("#### ☀️ Solar Panels")
        sol_c1, sol_c2, sol_c3 = st.columns(3)
        with sol_c1:
            solar_capex = st.number_input(
                "Total solar panel cost (R)",
                min_value=0.0, value=500_000.0, step=10_000.0, format="%.0f",
                help="Total installed cost of solar panels only (excluding battery & inverter)",
            )
        with sol_c2:
            solar_kwp = st.number_input(
                "Solar panel peak output (kWp)",
                min_value=0.1, value=100.0, step=1.0, format="%.1f",
                help="Total installed peak capacity in kilowatt-peak",
            )
        with sol_c3:
            # Standard: panels rated 20 years to 80% output
            # Lifetime kWh = kWp × 20 yr × 365 days × avg daily peak sun hours
            solar_peak_sun_hours = st.number_input(
                "Avg daily peak sun hours",
                min_value=1.0, max_value=12.0, value=5.5, step=0.1, format="%.1f",
                help="Average peak sun hours per day for your location (typically 4.5–6.5 for KZN)",
            )
            solar_lifetime_yr = 20
            solar_lifetime_kwh = solar_kwp * solar_peak_sun_hours * 365 * solar_lifetime_yr
            st.metric(
                "Calculated lifetime production",
                f"{solar_lifetime_kwh:,.0f} kWh",
                help=f"{solar_kwp:.1f} kWp × {solar_peak_sun_hours:.1f} h/day × 365 × {solar_lifetime_yr} yr",
            )

        # Solar cost per kWh — choose method
        sol_m1, sol_m2, sol_m3 = st.columns(3)
        with sol_m1:
            cost_method = st.radio(
                "Solar cost basis",
                options=["Calculate from capex", "Enter manually"],
                help="Auto-calculate R/kWh from capex ÷ lifetime kWh, or enter a known figure",
                key="solar_cost_method",
            )
        with sol_m2:
            if cost_method == "Calculate from capex":
                solar_kwh_cost = solar_capex / solar_lifetime_kwh if solar_lifetime_kwh > 0 else 0.0
                st.metric(
                    "Solar cost (auto-calculated)",
                    f"R {solar_kwh_cost:.4f} / kWh",
                    help="Solar capex ÷ lifetime production",
                )
            else:
                solar_kwh_cost = st.number_input(
                    "Solar cost (manual R/kWh)",
                    min_value=0.0, value=0.50, step=0.01, format="%.4f",
                    key="solar_manual_cost",
                )
        with sol_m3:
            bc_levy_pct = st.number_input(
                "Body corporate levy (%)",
                min_value=0.0, max_value=100.0, value=8.0, step=0.1, format="%.1f",
                help="% of total solar produced credited to body corporate at flat sell rate",
            )

        st.divider()

        # ── 2. BATTERY ────────────────────────────────────────────────────
        st.markdown("#### 🔋 Battery")
        bat_c1, bat_c2, bat_c3, bat_c4, bat_c5 = st.columns(5)
        with bat_c1:
            battery_capex = st.number_input(
                "Total battery cost (R)",
                min_value=0.0, value=400_000.0, step=10_000.0, format="%.0f",
                help="Total installed cost of battery bank only",
            )
        with bat_c2:
            battery_replacement_cost = st.number_input(
                "Battery replacement cost (R)",
                min_value=0.0, value=350_000.0, step=10_000.0, format="%.0f",
                help="Cost to replace battery bank at end of rated cycle life",
            )
        with bat_c3:
            p0_batt_kwh = st.number_input(
                "Battery capacity (kWh)",
                min_value=1.0, value=80.0, step=1.0, format="%.1f",
                help="Usable capacity of the battery bank in kWh",
            )
        with bat_c4:
            p0_batt_cycles = st.number_input(
                "Rated cycle life",
                min_value=100, value=10000, step=100,
                help="Manufacturer rated number of full cycles at the specified DoD",
            )
        with bat_c5:
            p0_batt_dod = st.slider(
                "Depth of Discharge (%)",
                min_value=50, max_value=100, value=80, step=5,
                key="p0_dod",
                help="DoD at which the cycle rating applies (e.g. 10 000 cycles @ 80% DoD)",
            )

        # Derived battery metrics
        bat_m1, bat_m2, bat_m3 = st.columns(3)
        p0_dod_f            = p0_batt_dod / 100.0
        total_lifetime_kwh_b= p0_batt_cycles * p0_batt_kwh * p0_dod_f
        repl_per_kwh_p0     = battery_replacement_cost / total_lifetime_kwh_b if total_lifetime_kwh_b > 0 else 0
        bat_m1.metric("Total lifetime discharge (kWh)",
                      f"{total_lifetime_kwh_b:,.0f} kWh",
                      help=f"{p0_batt_cycles:,} cycles × {p0_batt_kwh:.0f} kWh × {p0_dod_f:.0%} DoD")
        bat_m2.metric("Replacement cost / kWh discharged",
                      f"R {repl_per_kwh_p0:.4f}",
                      help="Battery replacement cost ÷ total lifetime discharge kWh")
        bat_m3.metric("Battery cost / kWh capacity",
                      f"R {battery_capex / p0_batt_kwh:,.0f} / kWh" if p0_batt_kwh > 0 else "—")

        st.divider()

        # ── 3. TOTAL SYSTEM ───────────────────────────────────────────────
        st.markdown("#### 🏗️ Total System")
        sys_c1, sys_c2 = st.columns(2)
        with sys_c1:
            total_system_cost = st.number_input(
                "Total system cost (R)",
                min_value=0.0, value=1_000_000.0, step=10_000.0, format="%.0f",
                help="All-in installed cost: panels + battery + inverter + installation + commissioning",
            )
        with sys_c2:
            sys_cost_auto = solar_capex + battery_capex
            st.metric(
                "Solar + Battery sub-total",
                f"R {sys_cost_auto:,.0f}",
                delta=f"R {total_system_cost - sys_cost_auto:,.0f} other costs" if total_system_cost != sys_cost_auto else "Matches total",
                help="Sum of solar panel cost + battery cost — difference is inverter, installation, etc.",
            )

    bc_rate     = bc_levy_pct / 100.0
    batt_margin = 0.15   # you keep 15% of the tariff; pay 85% to battery/solar company

    # ════════════════════════════════════════════════════════════════════════
    # SECTION P1 — PER-HOUR REVENUE & COST CALCULATION
    # Requires hourly_batt from Tab 4 — we need to ensure it exists.
    # Since Tab 4 inputs (battery_capacity, solar_cost_per_kwh) are defined
    # inside Tab 4's with-block and Tab 3 runs first, we use the global
    # hourly frame directly and replicate the split logic here.
    # ════════════════════════════════════════════════════════════════════════

    # --- Replicate charge split (same as Tab 4 but self-contained) ----------
    h = hourly.copy()
    h["solar_to_batt"] = np.minimum(h["batt_charge_kwh"], h["solar_kwh"])
    h["grid_to_batt"]  = np.maximum(0.0, h["batt_charge_kwh"] - h["solar_kwh"])

    # Per-row tariff already assigned by assign_tou_vectorised() using the
    # correct period for each row's date — no date-blind lookup needed.
    h["slot_tariff"] = h["tariff_rkwh"]

    is_weekday_h = h["datetime"].dt.weekday < 5

    # ── Revenue Stream 1: Grid-charged battery → discharge at Peak/Standard ──
    # Discharge slot tariff × 15% − grid charge cost (off-peak tariff already paid)
    # Attribution: proportion of total discharge = grid_to_batt / batt_charge (where charge > 0)
    charge_total = h["batt_charge_kwh"].replace(0, np.nan)
    h["grid_frac"]  = (h["grid_to_batt"]  / charge_total).fillna(0).clip(0, 1)
    h["solar_frac"] = (h["solar_to_batt"] / charge_total).fillna(0).clip(0, 1)

    # Discharge hours split by source (proportional)
    h["disc_from_grid"]  = h["batt_discharge_kwh"] * h["grid_frac"]
    h["disc_from_solar"] = h["batt_discharge_kwh"] * h["solar_frac"]

    # Stream 1 revenue: 15% of slot tariff for grid-discharge kWh
    # Cost recovery: off-peak tariff paid to municipality
    # Per-row off-peak/standard tariffs, date-aware (vectorised via period+season groups)
    offpeak_tariff_h = pd.Series(0.0, index=h.index)
    std_tariff_h     = pd.Series(0.0, index=h.index)
    for period in TOU_PERIODS:
        p_start = pd.Timestamp(period["effective_from"])
        p_mask = h["datetime"] >= p_start
        # Narrow to just this period's rows (later periods will overwrite earlier ones below)
        for season_name in ["low", "high"]:
            s_mask = p_mask & (h["season"] == season_name)
            offpeak_tariff_h = offpeak_tariff_h.where(~s_mask, period["tariffs_rkwh"][season_name]["1.8.3"])
            std_tariff_h     = std_tariff_h.where(~s_mask, period["tariffs_rkwh"][season_name]["1.8.2"])

    h["rev_grid_disc"]  = h["disc_from_grid"]  * h["slot_tariff"] * batt_margin
    h["cost_grid_charge"]= h["grid_to_batt"]   * offpeak_tariff_h   # what you paid municipality

    # Stream 2: Solar-charged battery → discharge
    # Revenue: 15% of Standard tariff (you keep margin on peak mitigation via solar)
    # Cost: solar capex cost per kWh
    h["rev_solar_disc"] = h["disc_from_solar"] * std_tariff_h * batt_margin
    h["cost_solar_batt"]= h["solar_to_batt"]   * solar_kwh_cost

    # Stream 3: Solar → direct load or grid export (not going into battery)
    # Solar available for export/direct = total solar − solar_to_batt
    h["solar_direct"]   = np.maximum(0.0, h["solar_kwh"] - h["solar_to_batt"])
    h["rev_solar_direct"]= h["solar_direct"] * h["sell_rate_rkwh"]
    h["cost_solar_direct"]= h["solar_direct"] * solar_kwh_cost

    # Body corporate levy — 8% of ALL solar at sell rate (deducted from total)
    # Calculated at billing-run level below

    # Grid import cost (non-battery grid usage = grid import − grid_to_batt)
    h["grid_load_import"]  = np.maximum(0.0, h["grid_import_kwh"] - h["grid_to_batt"])
    h["cost_grid_load"]    = h["grid_load_import"] * h["slot_tariff"]

    # Battery replacement cost — pro-rated per kWh discharged
    # Uses: battery replacement cost ÷ (rated cycles × capacity kWh × DoD)
    _lifetime_disc_kwh   = p0_batt_cycles * p0_batt_kwh * p0_dod_f
    battery_replacement_cost_per_kwh = (
        battery_replacement_cost / _lifetime_disc_kwh if _lifetime_disc_kwh > 0 else 0
    )
    h["cost_batt_replacement"] = h["batt_discharge_kwh"] * battery_replacement_cost_per_kwh

    # ── Aggregate by billing run ─────────────────────────────────────────────
    daily2 = daily.copy()
    billing_summary = daily2.groupby("billing_run").agg(
        solar_kwh=("solar_kwh", "sum"),
        grid_import=("grid_import_kwh", "sum"),
        grid_export=("grid_export_kwh", "sum"),
        days=("date", "count"),
        period_end=("date", "max"),
    ).reset_index().sort_values("period_end").reset_index(drop=True)

    _sorted_runs_p3 = sorted(
        billing_summary.set_index("billing_run")["period_end"].to_dict().items(),
        key=lambda x: x[1]
    )
    _start_p3 = pd.Timestamp("2025-12-05")

    prof_rows = []
    for i, (run, end_date) in enumerate(_sorted_runs_p3):
        ps = _start_p3 if i == 0 else _sorted_runs_p3[i-1][1] + pd.Timedelta(days=1)
        seg = h[(h["datetime"] >= ps) & (h["datetime"] < end_date + pd.Timedelta(days=1))]

        solar_total     = seg["solar_kwh"].sum()
        bc_kwh          = solar_total * bc_rate
        # Body corporate cost uses each row's own sell rate, weighted by its
        # share of solar — correct even if this billing run spans a tariff
        # period boundary (e.g. an April rate change mid-run).
        bc_cost_r       = (seg["solar_kwh"] * bc_rate * seg["sell_rate_rkwh"]).sum()

        rev_grid        = seg["rev_grid_disc"].sum()
        rev_solar_disc  = seg["rev_solar_disc"].sum()
        rev_solar_dir   = seg["rev_solar_direct"].sum()
        total_revenue_p3= rev_grid + rev_solar_disc + rev_solar_dir

        cost_grid_ch    = seg["cost_grid_charge"].sum()
        cost_solar_b    = seg["cost_solar_batt"].sum()
        cost_solar_d    = seg["cost_solar_direct"].sum()
        cost_grid_load  = seg["cost_grid_load"].sum()
        cost_batt_rep   = seg["cost_batt_replacement"].sum()
        total_costs     = cost_grid_ch + cost_solar_b + cost_solar_d + bc_cost_r + cost_batt_rep

        gross_profit    = total_revenue_p3 - total_costs
        grid_only_cost  = seg["slot_tariff"].mul(seg["grid_import_kwh"] + seg["load_kwh"]).sum() / 2
        # Savings = what grid-only scenario would have cost vs actual grid import cost
        actual_grid_cost= seg["cost_grid_load"].sum() + seg["cost_grid_charge"].sum()
        hypothetical    = (seg["load_kwh"] * seg["slot_tariff"]).sum()
        peak_saving     = hypothetical - actual_grid_cost

        disc_peak       = seg[(seg["tou_slot"]=="1.8.1") & (is_weekday_h)]["batt_discharge_kwh"].sum()
        disc_std_wknd   = seg[(seg["tou_slot"]=="1.8.2") & (~is_weekday_h)]["batt_discharge_kwh"].sum()

        prof_rows.append({
            "Billing Run":          run,
            "Period Start":         ps.date(),
            "Period End":           end_date.date(),
            "Solar Produced (kWh)": round(solar_total, 1),
            "BC Levy (kWh)":        round(bc_kwh, 1),
            "BC Levy Cost (R)":     round(bc_cost_r, 2),
            "Rev: Grid Discharge (R)":   round(rev_grid, 2),
            "Rev: Solar Discharge (R)":  round(rev_solar_disc, 2),
            "Rev: Solar Direct (R)":     round(rev_solar_dir, 2),
            "Total Revenue (R)":         round(total_revenue_p3, 2),
            "Cost: Grid Charge (R)":     round(cost_grid_ch, 2),
            "Cost: Solar (Battery) (R)": round(cost_solar_b, 2),
            "Cost: Solar (Direct) (R)":  round(cost_solar_d, 2),
            "Cost: BC Levy (R)":         round(bc_cost_r, 2),
            "Cost: Batt Replacement (R)":round(cost_batt_rep, 2),
            "Total Costs (R)":           round(total_costs, 2),
            "Gross Profit (R)":          round(gross_profit, 2),
            "Peak Mitigation Saving (R)":round(peak_saving, 2),
            "Peak Discharged (kWh)":     round(disc_peak, 1),
            "Std Weekend Disc (kWh)":    round(disc_std_wknd, 1),
        })

    prof_df = pd.DataFrame(prof_rows)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION P2 — KPI METRICS
    # ════════════════════════════════════════════════════════════════════════
    st.divider()
    tot_rev   = prof_df["Total Revenue (R)"].sum()
    tot_cost  = prof_df["Total Costs (R)"].sum()
    tot_profit= prof_df["Gross Profit (R)"].sum()
    tot_save  = prof_df["Peak Mitigation Saving (R)"].sum()
    tot_bc    = prof_df["BC Levy Cost (R)"].sum()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Revenue",         f"R {tot_rev:,.0f}")
    k2.metric("Total Costs",           f"R {tot_cost:,.0f}")
    k3.metric("Gross Profit",          f"R {tot_profit:,.0f}",
              delta=f"{tot_profit/tot_rev*100:.1f}% margin" if tot_rev else "")
    k4.metric("Peak Mitigation Saving",f"R {tot_save:,.0f}",
              help="Hypothetical all-grid cost minus actual grid cost")
    k5.metric("BC Levy Paid",          f"R {tot_bc:,.0f}")

    # ── Revenue waterfall chart ──────────────────────────────────────────────
    st.subheader("Revenue & Cost Breakdown by Billing Run")
    fig_prof = go.Figure()
    fig_prof.add_bar(name="Grid Discharge Rev",  x=prof_df["Billing Run"], y=prof_df["Rev: Grid Discharge (R)"],  marker_color="#1D9E75")
    fig_prof.add_bar(name="Solar Discharge Rev", x=prof_df["Billing Run"], y=prof_df["Rev: Solar Discharge (R)"], marker_color="#EF9F27")
    fig_prof.add_bar(name="Solar Direct Rev",    x=prof_df["Billing Run"], y=prof_df["Rev: Solar Direct (R)"],    marker_color="#3B8BD4")
    fig_prof.add_bar(name="Grid Charge Cost",    x=prof_df["Billing Run"], y=-prof_df["Cost: Grid Charge (R)"],   marker_color="#E24B4A")
    fig_prof.add_bar(name="Solar Cost (Batt)",   x=prof_df["Billing Run"], y=-prof_df["Cost: Solar (Battery) (R)"], marker_color="#c0392b")
    fig_prof.add_bar(name="BC Levy",             x=prof_df["Billing Run"], y=-prof_df["Cost: BC Levy (R)"],       marker_color="#7F77DD")
    fig_prof.add_bar(name="Batt Replacement",    x=prof_df["Billing Run"], y=-prof_df["Cost: Batt Replacement (R)"], marker_color="#AAAAAA")
    fig_prof.add_scatter(name="Net Profit",      x=prof_df["Billing Run"], y=prof_df["Gross Profit (R)"],
                         mode="lines+markers", line=dict(color="white", width=2), marker=dict(size=8))
    fig_prof.update_layout(
        barmode="relative",
        xaxis=dict(title="Billing Run", categoryorder="array",
                   categoryarray=prof_df["Billing Run"].tolist()),
        yaxis_title="R",
        legend_title="", height=440,
    )
    st.plotly_chart(fig_prof, use_container_width=True)

    # ── Peak mitigation saving ───────────────────────────────────────────────
    st.subheader("Peak Mitigation — Grid Cost Avoided")
    fig_save = go.Figure()
    fig_save.add_bar(name="Peak Mitigation Saving", x=prof_df["Billing Run"], y=prof_df["Peak Mitigation Saving (R)"],
                     marker_color="#1D9E75",
                     text=prof_df["Peak Mitigation Saving (R)"].apply(lambda v: f"R{v:,.0f}"),
                     textposition="outside")
    fig_save.update_layout(
        xaxis=dict(categoryorder="array", categoryarray=prof_df["Billing Run"].tolist()),
        yaxis_title="R Saved", height=340,
    )
    st.plotly_chart(fig_save, use_container_width=True)

    # ── Detailed table ───────────────────────────────────────────────────────
    st.subheader("Detailed Profitability Table")
    table_cols = [
        "Billing Run", "Period Start", "Period End",
        "Solar Produced (kWh)", "BC Levy (kWh)", "BC Levy Cost (R)",
        "Rev: Grid Discharge (R)", "Rev: Solar Discharge (R)", "Rev: Solar Direct (R)", "Total Revenue (R)",
        "Cost: Grid Charge (R)", "Cost: Solar (Battery) (R)", "Cost: Solar (Direct) (R)",
        "Cost: BC Levy (R)", "Cost: Batt Replacement (R)", "Total Costs (R)",
        "Gross Profit (R)", "Peak Mitigation Saving (R)",
    ]
    money_cols = [c for c in table_cols if "(R)" in c]
    kwh_cols   = [c for c in table_cols if "kWh" in c]
    fmt_p = {c: "R {:,.2f}" for c in money_cols}
    fmt_p.update({c: "{:,.1f}" for c in kwh_cols})
    st.dataframe(prof_df[table_cols].style.format(fmt_p), use_container_width=True, hide_index=True)

    csv_p = prof_df[table_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Profitability CSV", csv_p,
                       file_name=f"{site_name}_profitability_detailed.csv", mime="text/csv")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION P3 — PAYBACK PERIOD
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📈 Payback Period Analysis")
    st.caption("Based on gross profit per billing period extrapolated forward at current run rate.")

    avg_monthly_profit = tot_profit / len(prof_df) if len(prof_df) > 0 else 1

    payback_total   = total_system_cost  / avg_monthly_profit if avg_monthly_profit > 0 else np.inf
    payback_solar   = solar_capex        / avg_monthly_profit if avg_monthly_profit > 0 else np.inf
    payback_battery = battery_capex      / avg_monthly_profit if avg_monthly_profit > 0 else np.inf

    pb1, pb2, pb3, pb4 = st.columns(4)
    pb1.metric("Avg Monthly Profit",   f"R {avg_monthly_profit:,.0f}")
    pb2.metric("Total System Payback", f"{payback_total:.1f} months  ({payback_total/12:.1f} yrs)" if payback_total < 600 else "N/A")
    pb3.metric("Solar Only Payback",   f"{payback_solar:.1f} months  ({payback_solar/12:.1f} yrs)"   if payback_solar  < 600 else "N/A")
    pb4.metric("Battery Only Payback", f"{payback_battery:.1f} months  ({payback_battery/12:.1f} yrs)" if payback_battery < 600 else "N/A")

    # Cumulative profit curve vs capex lines
    prof_df["Cumulative Profit (R)"] = prof_df["Gross Profit (R)"].cumsum()
    n_periods = max(int(payback_total * 1.3), len(prof_df) + 6)
    future_months = list(range(len(prof_df) + 1, min(n_periods + 1, 121)))
    future_cum    = [prof_df["Cumulative Profit (R)"].iloc[-1] + avg_monthly_profit * i
                     for i in range(1, len(future_months) + 1)]

    all_labels = prof_df["Billing Run"].tolist() + [f"M+{i}" for i in range(1, len(future_months) + 1)]
    all_cum    = prof_df["Cumulative Profit (R)"].tolist() + future_cum

    fig_pb = go.Figure()
    fig_pb.add_scatter(x=prof_df["Billing Run"], y=prof_df["Cumulative Profit (R)"],
                       name="Actual cumulative profit", mode="lines+markers",
                       line=dict(color="#1D9E75", width=2))
    if future_months:
        fig_pb.add_scatter(x=[f"M+{i}" for i in range(1, len(future_months)+1)], y=future_cum,
                           name="Projected (current rate)", mode="lines",
                           line=dict(color="#1D9E75", dash="dash", width=1))
    for label, cost, color in [
        ("Total system cost",  total_system_cost,  "#E24B4A"),
        ("Solar capex",        solar_capex,         "#EF9F27"),
        ("Battery capex",      battery_capex,       "#7F77DD"),
    ]:
        fig_pb.add_hline(y=cost, line_dash="dot", line_color=color,
                         annotation_text=label, annotation_position="right")
    fig_pb.update_layout(
        title="Cumulative Profit vs Capital Cost",
        xaxis_title="Billing Run / Month", yaxis_title="R",
        height=420, legend_title="",
    )
    st.plotly_chart(fig_pb, use_container_width=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION P4 — BATTERY CYCLE LIFE & REPLACEMENT
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("🔋 Battery Cycle Life & Replacement Timeline")
    st.caption(
        f"Using battery parameters from the Battery section above: "
        f"**{p0_batt_kwh:.0f} kWh** capacity · **{p0_batt_cycles:,}** rated cycles · "
        f"**{p0_batt_dod}% DoD** · replacement cost **R {battery_replacement_cost:,.0f}**"
    )

    # Compute cycles per billing period using p0 inputs
    cycle_rows = []
    for i, (run, end_date) in enumerate(_sorted_runs_p3):
        ps = _start_p3 if i == 0 else _sorted_runs_p3[i-1][1] + pd.Timedelta(days=1)
        seg_disc = h[(h["datetime"] >= ps) & (h["datetime"] < end_date + pd.Timedelta(days=1))]["batt_discharge_kwh"].sum()
        cycle_rows.append({
            "Billing Run":    run,
            "Discharge (kWh)": round(seg_disc, 2),
            "Cycles Used":    seg_disc / (p0_batt_kwh * p0_dod_f) if p0_batt_kwh * p0_dod_f > 0 else 0,
        })
    cycle_df = pd.DataFrame(cycle_rows)
    cycle_df["Cumulative Cycles"] = cycle_df["Cycles Used"].cumsum()
    cycle_df["Health (%)"]        = ((1 - cycle_df["Cumulative Cycles"] / p0_batt_cycles) * 100).clip(0, 100)

    total_cycles_used = cycle_df["Cumulative Cycles"].iloc[-1]
    cycles_remaining  = max(0, p0_batt_cycles - total_cycles_used)
    avg_cycles_period = cycle_df["Cycles Used"].mean()
    periods_to_eol    = cycles_remaining / avg_cycles_period if avg_cycles_period > 0 else np.inf
    current_health    = cycle_df["Health (%)"].iloc[-1]

    cy1, cy2, cy3, cy4 = st.columns(4)
    cy1.metric("Cycles Used",      f"{total_cycles_used:.2f}",
               help=f"Cumulative equivalent full cycles at {p0_batt_dod}% DoD")
    cy2.metric("Cycles Remaining", f"{cycles_remaining:,.0f}",
               help=f"Out of {p0_batt_cycles:,} rated cycles")
    cy3.metric("Current Health",   f"{current_health:.2f}%")
    cy4.metric("Est. Months to EOL",
               f"{periods_to_eol:.0f} months ({periods_to_eol/12:.1f} yrs)"
               if periods_to_eol < 600 else ">50 yrs")

    st.info(
        f"ℹ️ Replacement cost per kWh discharged: **R {battery_replacement_cost_per_kwh:.4f}/kWh** "
        f"(R{battery_replacement_cost:,.0f} ÷ {p0_batt_cycles:,} cycles "
        f"× {p0_batt_kwh:.0f} kWh × {p0_dod_f:.0%} DoD = {_lifetime_disc_kwh:,.0f} kWh lifetime)"
    )

    fig_cyc = go.Figure()
    fig_cyc.add_scatter(
        x=cycle_df["Billing Run"], y=cycle_df["Health (%)"],
        mode="lines+markers", name="Battery Health",
        line=dict(color="#EF9F27", width=2), marker=dict(size=8),
    )
    fig_cyc.add_hline(y=80, line_dash="dash", line_color="#E24B4A",
                      annotation_text="80% — typical end-of-warranty threshold")
    fig_cyc.update_layout(
        xaxis_title="Billing Run", yaxis_title="Health (%)",
        yaxis_range=[0, 105], height=300,
    )
    st.plotly_chart(fig_cyc, use_container_width=True)

    st.dataframe(
        cycle_df.style.format({
            "Discharge (kWh)":   "{:,.2f}",
            "Cycles Used":       "{:.4f}",
            "Cumulative Cycles": "{:.4f}",
            "Health (%)":        "{:.2f}%",
        }),
        use_container_width=True, hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BATTERY HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Battery Health & Cycle Tracking")

    # ── Battery parameters ────────────────────────────────────────────────────
    bp_col1, bp_col2, bp_col3 = st.columns(3)
    with bp_col1:
        battery_capacity = st.number_input(
            "Battery nominal capacity (kWh)",
            min_value=1.0, max_value=5000.0, value=80.0, step=1.0,
            help="Total usable capacity of the battery bank in kWh",
        )
    with bp_col2:
        warranty_cycles = st.number_input(
            "Warranty cycle life (at 80% DoD)",
            min_value=100, max_value=50000, value=10000, step=100,
            help="Manufacturer rated cycle count before battery reaches end-of-life (default: 10 000 cycles at 80% DoD)",
        )
    with bp_col3:
        solar_cost_per_kwh = st.number_input(
            "Solar generation cost (R/kWh)",
            min_value=0.0, max_value=10.0, value=0.50, step=0.01,
            format="%.4f",
            help=(
                "Your blended cost to generate 1 kWh from solar (e.g. system capex + opex ÷ lifetime kWh). "
                "Used as the charge cost when the battery is charged from solar during Standard hours. "
                "Grid Off-Peak charging still uses the municipality tariff."
            ),
        )

    st.divider()

    # ── Charging source & discharging destination logic ───────────────────────
    # Charge source is derived purely from the recorded data:
    #   solar_to_battery = min(batt_charge_kwh, solar_kwh)  — solar fills battery first
    #   grid_to_battery  = max(0, batt_charge_kwh - solar_kwh) — remainder from grid
    # Discharge TOU slot is used to show when discharging occurred and what tariff it displaced.

    @st.cache_data
    def classify_hourly_battery(h_df: pd.DataFrame, solar_cost: float) -> pd.DataFrame:
        """
        Derive solar vs grid battery charge split from recorded hourly values.
        Fully vectorised — no row-by-row apply(). Cached so it only re-runs when
        solar_cost changes (not on every slider interaction).
        """
        df = h_df.copy()

        # ── Data-driven charge source split ──────────────────────────────────
        df["solar_to_battery"] = np.minimum(df["batt_charge_kwh"], df["solar_kwh"])
        df["grid_to_battery"]  = np.maximum(0.0, df["batt_charge_kwh"] - df["solar_kwh"])

        # Dominant source label
        df["charge_source"] = np.where(
            df["solar_to_battery"] >= df["grid_to_battery"], "Solar", "Grid"
        )

        # Discharge destination
        is_weekday = df["datetime"].dt.weekday < 5
        df["discharge_dest"] = np.where(
            (is_weekday  & (df["tou_slot"] == "1.8.1")) |
            (~is_weekday & (df["tou_slot"] == "1.8.2")),
            "Scheduled", "Unscheduled"
        )

        # ── Per-row tariff for grid charge cost (already date-aware) ─────────
        # tariff_rkwh and sell_rate_rkwh are assigned by assign_tou_vectorised()
        # using the correct period for each row's date.
        df["tou_tariff"] = df["tariff_rkwh"]

        df["grid_charge_cost_r"]  = df["grid_to_battery"]  * df["tou_tariff"]
        df["solar_charge_cost_r"] = df["solar_to_battery"] * solar_cost
        df["charge_cost_r"]       = df["grid_charge_cost_r"] + df["solar_charge_cost_r"]

        df["discharge_value_r"] = df["batt_discharge_kwh"] * df["sell_rate_rkwh"]
        df["net_benefit_r"]     = df["discharge_value_r"] - df["charge_cost_r"]

        return df

    hourly_batt = classify_hourly_battery(hourly, solar_cost_per_kwh)

    # ── Health calculation using DISCHARGE data ───────────────────────────────
    dod_factor          = 0.80
    total_discharge_kwh = daily["batt_discharge_kwh"].sum()
    total_charge_kwh    = daily["batt_charge_kwh"].sum()
    total_cycles_equiv  = total_discharge_kwh / (battery_capacity * dod_factor)
    health_pct          = max(0.0, (1 - total_cycles_equiv / warranty_cycles) * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Charge Energy",      f"{total_charge_kwh:,.1f} kWh")
    c2.metric("Total Discharge Energy",   f"{total_discharge_kwh:,.1f} kWh")
    c3.metric("Equivalent Full Cycles",   f"{total_cycles_equiv:.2f}",
              help=f"Discharge energy ÷ ({battery_capacity} kWh × {dod_factor:.0%} DoD)")
    c4.metric("Estimated Battery Health", f"{health_pct:.2f}%",
              help=f"Based on {int(warranty_cycles):,} cycle warranty at 80% DoD")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # BILLING-RUN TABLES — charge and discharge shown together
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📋 Billing Run Summary — Charge & Discharge")
    st.caption(
        "Meter Reading = cumulative from 5 Dec 2025 = 0. "
        "Period Usage = energy within that billing period only. "
        "TOU columns show when the energy flowed, split by Grid and Solar source for charge."
    )

    batt_table_mode = st.radio(
        "View mode", options=["Meter Reading", "Period Usage"],
        horizontal=True, key="batt_table_mode",
    )

    # Build per-billing-run rows from hourly_batt
    _start_date  = pd.Timestamp("2025-12-05")
    _sorted_runs = sorted(billing_runs.items(), key=lambda x: x[1])

    batt_meter_rows, batt_usage_rows = [], []

    for i, (run, end_date) in enumerate(_sorted_runs):
        period_start = _start_date if i == 0 else _sorted_runs[i - 1][1] + pd.Timedelta(days=1)

        cum_mask = (hourly_batt["datetime"] >= _start_date) & \
                   (hourly_batt["datetime"] <  end_date + pd.Timedelta(days=1))
        per_mask = (hourly_batt["datetime"] >= period_start) & \
                   (hourly_batt["datetime"] <  end_date + pd.Timedelta(days=1))

        for mask, row_list in [(cum_mask, batt_meter_rows), (per_mask, batt_usage_rows)]:
            seg = hourly_batt[mask]

            # ── Charge TOU split — grid and solar separately ─────────────────
            c_total       = seg["batt_charge_kwh"].sum()
            c_grid_total  = seg["grid_to_battery"].sum()
            c_solar_total = seg["solar_to_battery"].sum()

            # Grid → battery by TOU slot
            cg_peak    = seg[seg["tou_slot"] == "1.8.1"]["grid_to_battery"].sum()
            cg_std     = seg[seg["tou_slot"] == "1.8.2"]["grid_to_battery"].sum()
            cg_offpeak = seg[seg["tou_slot"] == "1.8.3"]["grid_to_battery"].sum()

            # Solar → battery by TOU slot
            cs_peak    = seg[seg["tou_slot"] == "1.8.1"]["solar_to_battery"].sum()
            cs_std     = seg[seg["tou_slot"] == "1.8.2"]["solar_to_battery"].sum()
            cs_offpeak = seg[seg["tou_slot"] == "1.8.3"]["solar_to_battery"].sum()

            c_cost = seg["charge_cost_r"].sum()

            # ── Discharge TOU split ──────────────────────────────────────────
            d_total   = seg["batt_discharge_kwh"].sum()
            d_peak    = seg[seg["tou_slot"] == "1.8.1"]["batt_discharge_kwh"].sum()
            d_std     = seg[seg["tou_slot"] == "1.8.2"]["batt_discharge_kwh"].sum()
            d_offpeak = seg[seg["tou_slot"] == "1.8.3"]["batt_discharge_kwh"].sum()
            d_value   = seg["discharge_value_r"].sum()

            row_list.append({
                "Billing Run":            run,
                "Period Start":           period_start.date(),
                "Period End":             end_date.date(),
                # Charge totals
                "C Total (kWh)":          round(c_total, 3),
                "C Grid Total (kWh)":     round(c_grid_total, 3),
                "C Solar Total (kWh)":    round(c_solar_total, 3),
                # Grid→battery by TOU
                "CG Peak (kWh)":          round(cg_peak, 3),
                "CG Std (kWh)":           round(cg_std, 3),
                "CG Off-Peak (kWh)":      round(cg_offpeak, 3),
                # Solar→battery by TOU
                "CS Peak (kWh)":          round(cs_peak, 3),
                "CS Std (kWh)":           round(cs_std, 3),
                "CS Off-Peak (kWh)":      round(cs_offpeak, 3),
                # Charge cost
                "C Cost (R)":             round(c_cost, 2),
                # Discharge totals + TOU
                "D Total (kWh)":          round(d_total, 3),
                "D Peak (kWh)":           round(d_peak, 3),
                "D Std (kWh)":            round(d_std, 3),
                "D Off-Peak (kWh)":       round(d_offpeak, 3),
                "D Value (R)":            round(d_value, 2),
                "Net Benefit (R)":        round(d_value - c_cost, 2),
            })

    batt_meter_df = pd.DataFrame(batt_meter_rows)
    batt_usage_df = pd.DataFrame(batt_usage_rows)
    src_df = batt_meter_df if batt_table_mode == "Meter Reading" else batt_usage_df

    base_cols = ["Billing Run", "Period End"]
    if batt_table_mode == "Period Usage":
        base_cols = ["Billing Run", "Period Start", "Period End"]

    # ── CHARGE TABLE ──────────────────────────────────────────────────────────
    st.markdown("**🔋 Charge — Grid & Solar split by TOU slot**")
    st.caption(
        "Grid→Battery and Solar→Battery totals, then broken down by TOU slot. "
        "Grid portion uses actual TOU tariff; Solar portion uses your generation cost."
    )
    charge_cols = base_cols + [
        "C Total (kWh)",
        "C Grid Total (kWh)", "CG Peak (kWh)", "CG Std (kWh)", "CG Off-Peak (kWh)",
        "C Solar Total (kWh)", "CS Peak (kWh)", "CS Std (kWh)", "CS Off-Peak (kWh)",
        "C Cost (R)",
    ]
    charge_rename = {
        "C Total (kWh)":       "Total Charge",
        "C Grid Total (kWh)":  "Grid Total",
        "CG Peak (kWh)":       "Grid — Peak",
        "CG Std (kWh)":        "Grid — Std",
        "CG Off-Peak (kWh)":   "Grid — Off-Pk",
        "C Solar Total (kWh)": "Solar Total",
        "CS Peak (kWh)":       "Solar — Peak",
        "CS Std (kWh)":        "Solar — Std",
        "CS Off-Peak (kWh)":   "Solar — Off-Pk",
        "C Cost (R)":          "Charge Cost (R)",
    }
    charge_fmt = {v: "{:.3f}" for v in charge_rename.values() if "(R)" not in v}
    charge_fmt["Charge Cost (R)"] = "R {:,.2f}"

    charge_display = src_df[charge_cols].rename(columns=charge_rename)
    st.dataframe(charge_display.style.format(charge_fmt), use_container_width=True, hide_index=True)

    csv_charge = charge_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download Charge {batt_table_mode} CSV", csv_charge,
        file_name=f"{site_name}_battery_charge_{batt_table_mode.lower().replace(' ','_')}.csv",
        mime="text/csv", key="dl_charge",
    )

    st.divider()

    # ── DISCHARGE TABLE ───────────────────────────────────────────────────────
    st.markdown("**⚡ Discharge — TOU slot breakdown**")
    st.caption(
        "Shows when the battery discharged across Peak, Standard and Off-Peak slots. "
        "Value = discharge kWh × R3.2795 sell rate. Net Benefit = Discharge Value − Charge Cost."
    )
    discharge_cols = base_cols + [
        "D Total (kWh)", "D Peak (kWh)", "D Std (kWh)", "D Off-Peak (kWh)",
        "D Value (R)", "Net Benefit (R)",
    ]
    discharge_rename = {
        "D Total (kWh)":    "Total Discharge",
        "D Peak (kWh)":     "Peak Discharged",
        "D Std (kWh)":      "Std Discharged",
        "D Off-Peak (kWh)": "Off-Pk Discharged",
        "D Value (R)":      "Discharge Value (R)",
        "Net Benefit (R)":  "Net Benefit (R)",
    }
    discharge_fmt = {v: "{:.3f}" for v in discharge_rename.values() if "(R)" not in v}
    discharge_fmt["Discharge Value (R)"] = "R {:,.2f}"
    discharge_fmt["Net Benefit (R)"]     = "R {:,.2f}"

    discharge_display = src_df[discharge_cols].rename(columns=discharge_rename)
    st.dataframe(discharge_display.style.format(discharge_fmt), use_container_width=True, hide_index=True)

    csv_discharge = discharge_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download Discharge {batt_table_mode} CSV", csv_discharge,
        file_name=f"{site_name}_battery_discharge_{batt_table_mode.lower().replace(' ','_')}.csv",
        mime="text/csv", key="dl_discharge",
    )

    st.divider()

    # ── Date range controls ───────────────────────────────────────────────────
    batt_min = daily["date"].min().to_pydatetime()
    batt_max = daily["date"].max().to_pydatetime()

    batt_date_input_col, batt_slider_col = st.columns([1, 2])
    with batt_date_input_col:
        st.date_input("From", value=batt_min, min_value=batt_min, max_value=batt_max, key="batt_start")
        st.date_input("To",   value=batt_max, min_value=batt_min, max_value=batt_max, key="batt_end")
    with batt_slider_col:
        batt_range = st.slider(
            "Or drag to adjust range",
            min_value=batt_min, max_value=batt_max,
            value=(batt_min, batt_max),
            format="DD MMM YYYY",
            key="batt_slider",
        )
    effective_start = pd.Timestamp(batt_range[0])
    effective_end   = pd.Timestamp(batt_range[1])

    batt_daily  = daily[(daily["date"] >= effective_start) & (daily["date"] <= effective_end)]
    batt_hourly = hourly_batt[
        (hourly_batt["datetime"].dt.date >= effective_start.date()) &
        (hourly_batt["datetime"].dt.date <= effective_end.date())
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A — Charge source breakdown (Grid vs Solar)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("⚡ Charge Source — Grid vs Solar")
    st.caption(
        f"Solar-to-battery = min(battery charge, solar production) each hour. "
        f"Grid-to-battery = remainder beyond what solar could supply. "
        f"Solar cost: **R{solar_cost_per_kwh:.4f}/kWh**. Grid cost: actual TOU tariff at time of charge."
    )

    grid_charge_kwh      = batt_hourly["grid_to_battery"].sum()
    solar_charge_kwh     = batt_hourly["solar_to_battery"].sum()
    total_charge_kwh_sel = batt_hourly["batt_charge_kwh"].sum()
    grid_charge_cost     = batt_hourly["grid_charge_cost_r"].sum()
    solar_charge_cost    = batt_hourly["solar_charge_cost_r"].sum()
    solar_pct = (solar_charge_kwh / total_charge_kwh_sel * 100) if total_charge_kwh_sel > 0 else 0
    grid_pct  = 100 - solar_pct

    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.metric("Grid → Battery (kWh)",  f"{grid_charge_kwh:,.1f} kWh",  help=f"{grid_pct:.1f}% of total charge")
    cs2.metric("Solar → Battery (kWh)", f"{solar_charge_kwh:,.1f} kWh", help=f"{solar_pct:.1f}% of total charge")
    cs3.metric("Total Charge Cost",     f"R {grid_charge_cost + solar_charge_cost:,.2f}",
               help=f"Grid: actual TOU tariff. Solar: R{solar_cost_per_kwh:.4f}/kWh.")
    cs4.metric("Solar Fraction",        f"{solar_pct:.1f}%")

    batt_hourly_sorted = batt_hourly.sort_values("datetime")

    # ── TOU charge breakdown — Grid and Solar separately with % ───────────────
    tou_labels = {"1.8.1": "Peak", "1.8.2": "Standard", "1.8.3": "Off-Peak"}
    tou_charge = batt_hourly.groupby("tou_slot").agg(
        grid=("grid_to_battery", "sum"),
        solar=("solar_to_battery", "sum"),
    ).reindex(["1.8.1", "1.8.2", "1.8.3"]).fillna(0).reset_index()
    tou_charge["label"] = tou_charge["tou_slot"].map(tou_labels)
    tou_charge["total"] = tou_charge["grid"] + tou_charge["solar"]
    tou_charge["grid_pct"]  = (tou_charge["grid"]  / total_charge_kwh_sel * 100).round(1)
    tou_charge["solar_pct"] = (tou_charge["solar"] / total_charge_kwh_sel * 100).round(1)

    ca_col, cb_col = st.columns(2)

    with ca_col:
        # Pie: overall Grid vs Solar
        charge_pie_df = pd.DataFrame({
            "source": ["Grid", "Solar"],
            "kwh": [grid_charge_kwh, solar_charge_kwh],
            "pct": [grid_pct, solar_pct],
        })
        fig_c_pie = px.pie(
            charge_pie_df, names="source", values="kwh",
            color="source",
            color_discrete_map={"Grid": "#7F77DD", "Solar": "#EF9F27"},
            title="Charge Source Split",
            hole=0.45,
        )
        fig_c_pie.update_traces(
            texttemplate="%{label}<br>%{value:.1f} kWh<br>(%{percent})"
        )
        fig_c_pie.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_c_pie, use_container_width=True)

    with cb_col:
        # Grouped bar: Grid and Solar per TOU slot with % labels
        fig_c_tou_bar = go.Figure()
        fig_c_tou_bar.add_bar(
            name="Grid", x=tou_charge["label"], y=tou_charge["grid"],
            marker_color="#7F77DD",
            text=[f"{v:.1f} kWh<br>({p:.1f}%)" for v, p in zip(tou_charge["grid"], tou_charge["grid_pct"])],
            textposition="outside",
        )
        fig_c_tou_bar.add_bar(
            name="Solar", x=tou_charge["label"], y=tou_charge["solar"],
            marker_color="#EF9F27",
            text=[f"{v:.1f} kWh<br>({p:.1f}%)" for v, p in zip(tou_charge["solar"], tou_charge["solar_pct"])],
            textposition="outside",
        )
        fig_c_tou_bar.update_layout(
            barmode="group", title="Charge by TOU Slot — Grid vs Solar",
            xaxis_title="TOU Slot", yaxis_title="kWh",
            legend_title="Source", height=320,
        )
        st.plotly_chart(fig_c_tou_bar, use_container_width=True)

    # Stacked % bar: proportion of each source per TOU slot
    fig_c_pct = go.Figure()
    tou_totals_c = tou_charge["total"].replace(0, np.nan)
    fig_c_pct.add_bar(
        name="Grid %", x=tou_charge["label"],
        y=(tou_charge["grid"] / tou_totals_c * 100).fillna(0).round(1),
        marker_color="#7F77DD",
        text=(tou_charge["grid"] / tou_totals_c * 100).fillna(0).round(1).astype(str) + "%",
        textposition="inside",
    )
    fig_c_pct.add_bar(
        name="Solar %", x=tou_charge["label"],
        y=(tou_charge["solar"] / tou_totals_c * 100).fillna(0).round(1),
        marker_color="#EF9F27",
        text=(tou_charge["solar"] / tou_totals_c * 100).fillna(0).round(1).astype(str) + "%",
        textposition="inside",
    )
    fig_c_pct.update_layout(
        barmode="stack", title="% Source Mix within each TOU Slot (Charge)",
        xaxis_title="TOU Slot", yaxis=dict(title="%", range=[0, 105]),
        legend_title="Source", height=300,
    )
    st.plotly_chart(fig_c_pct, use_container_width=True)

    # Daily stacked bar + hourly TOU chart (using rangebreaks instead of vrect loops)
    daily_split = batt_hourly.groupby(batt_hourly["datetime"].dt.date).agg(
        grid=("grid_to_battery", "sum"),
        solar=("solar_to_battery", "sum"),
    ).reset_index()
    daily_split.columns = ["date", "grid", "solar"]
    daily_split["total"] = daily_split["grid"] + daily_split["solar"]
    daily_split["solar_pct_d"] = (daily_split["solar"] / daily_split["total"].replace(0, np.nan) * 100).round(1)

    fig_c_daily = go.Figure()
    fig_c_daily.add_bar(x=daily_split["date"], y=daily_split["grid"],  name="Grid",  marker_color="#7F77DD")
    fig_c_daily.add_bar(x=daily_split["date"], y=daily_split["solar"], name="Solar", marker_color="#EF9F27")
    fig_c_daily.add_scatter(
        x=daily_split["date"], y=daily_split["solar_pct_d"],
        name="Solar %", yaxis="y2", mode="lines+markers",
        line=dict(color="#E24B4A", dash="dot"), marker=dict(size=5),
    )
    fig_c_daily.update_layout(
        barmode="stack", title="Daily Charge by Source with Solar %",
        xaxis_title="Date",
        yaxis=dict(title="Charge (kWh)"),
        yaxis2=dict(title="Solar %", overlaying="y", side="right", range=[0, 115],
                    showgrid=False, ticksuffix="%"),
        legend_title="", height=360,
    )
    st.plotly_chart(fig_c_daily, use_container_width=True)

    st.markdown("**Hourly Charge Profile with TOU Zones**")
    st.caption("Colour-coded bars: purple = Grid, orange = Solar. Background tint shows TOU slot.")
    fig_charge_tou = go.Figure()
    fig_charge_tou.add_bar(
        x=batt_hourly_sorted["datetime"], y=batt_hourly_sorted["grid_to_battery"],
        name="Grid Charge", marker_color="#7F77DD",
    )
    fig_charge_tou.add_bar(
        x=batt_hourly_sorted["datetime"], y=batt_hourly_sorted["solar_to_battery"],
        name="Solar Charge", marker_color="#EF9F27",
    )
    # TOU colour legend (dummy scatter entries)
    for label, color in [("🟢 Off-Peak zone", "#1D9E75"), ("🟠 Standard zone", "#EF9F27"), ("🔴 Peak zone", "#E24B4A")]:
        fig_charge_tou.add_scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, symbol="square"), name=label)
    fig_charge_tou.update_layout(
        barmode="stack", xaxis_title="Date / Hour", yaxis_title="Charge (kWh)",
        legend_title="Source / TOU", height=400,
    )
    st.plotly_chart(fig_charge_tou, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B — Discharge value & tariff savings
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("💡 Discharge Value — Tariff Savings")
    st.caption(
        "Discharge value = energy discharged × sell rate (R3.2795/kWh). "
        "TOU breakdown shows in which slots the battery discharged."
    )

    total_charge_cost_r = batt_hourly["charge_cost_r"].sum()
    total_disc_value_r  = batt_hourly["discharge_value_r"].sum()
    net_benefit_r       = total_disc_value_r - total_charge_cost_r
    total_disc_kwh      = batt_hourly["batt_discharge_kwh"].sum()

    db1, db2, db3, db4 = st.columns(4)
    db1.metric("Total Discharge (kWh)",  f"{total_disc_kwh:,.1f} kWh")
    db2.metric("Discharge Value",        f"R {total_disc_value_r:,.2f}",
               help=f"Calculated using each row's own period sell rate (currently R{SELL_RATE}/kWh)")
    db3.metric("Total Charge Cost",      f"R {total_charge_cost_r:,.2f}",
               help=f"Grid TOU tariff + Solar R{solar_cost_per_kwh:.4f}/kWh")
    db4.metric("Net Battery Benefit",    f"R {net_benefit_r:,.2f}",
               delta=f"{'Profit' if net_benefit_r >= 0 else 'Loss'}",
               help="Discharge value minus total charge cost")

    st.info(
        f"ℹ️ **Grid charge cost** = grid-to-battery kWh × actual TOU tariff.  "
        f"**Solar charge cost** = solar-to-battery kWh × R{solar_cost_per_kwh:.4f}/kWh.  "
        f"**Discharge value** = discharge kWh × R{SELL_RATE}/kWh.  "
        f"**Net Benefit** = Discharge Value − Total Charge Cost."
    )

    # ── TOU discharge breakdown with % ───────────────────────────────────────
    tou_disc = batt_hourly.groupby("tou_slot").agg(
        discharge=("batt_discharge_kwh", "sum"),
    ).reindex(["1.8.1", "1.8.2", "1.8.3"]).fillna(0).reset_index()
    tou_disc["label"] = tou_disc["tou_slot"].map(tou_labels)
    tou_disc["pct"]   = (tou_disc["discharge"] / total_disc_kwh * 100).round(1) if total_disc_kwh > 0 else 0

    da_col, db_col2 = st.columns(2)

    with da_col:
        # Pie: discharge by TOU slot
        fig_d_pie = px.pie(
            tou_disc, names="label", values="discharge",
            color="label",
            color_discrete_map={"Peak": "#E24B4A", "Standard": "#EF9F27", "Off-Peak": "#1D9E75"},
            title="Discharge by TOU Slot",
            hole=0.45,
        )
        fig_d_pie.update_traces(
            texttemplate="%{label}<br>%{value:.1f} kWh<br>(%{percent})"
        )
        fig_d_pie.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_d_pie, use_container_width=True)

    with db_col2:
        # Bar: discharge per TOU slot with % labels
        tou_colors_bar = {"Peak": "#E24B4A", "Standard": "#EF9F27", "Off-Peak": "#1D9E75"}
        fig_d_bar = go.Figure()
        for _, row in tou_disc.iterrows():
            fig_d_bar.add_bar(
                x=[row["label"]], y=[row["discharge"]],
                name=row["label"],
                marker_color=tou_colors_bar.get(row["label"], "#AAAAAA"),
                text=f"{row['discharge']:.1f} kWh<br>({row['pct']:.1f}%)",
                textposition="outside",
            )
        fig_d_bar.update_layout(
            title="Discharge by TOU Slot",
            xaxis_title="TOU Slot", yaxis_title="kWh",
            showlegend=False, height=320,
        )
        st.plotly_chart(fig_d_bar, use_container_width=True)

    # Daily discharge with TOU stack + % line
    daily_disc = batt_hourly.groupby(
        [batt_hourly["datetime"].dt.date, "tou_slot"]
    )["batt_discharge_kwh"].sum().unstack("tou_slot").fillna(0).reset_index()
    daily_disc.columns.name = None
    for col in ["1.8.1", "1.8.2", "1.8.3"]:
        if col not in daily_disc.columns:
            daily_disc[col] = 0.0
    daily_disc["total"] = daily_disc[["1.8.1", "1.8.2", "1.8.3"]].sum(axis=1)
    daily_disc["peak_pct"] = (daily_disc["1.8.1"] / daily_disc["total"].replace(0, np.nan) * 100).round(1)

    fig_d_daily = go.Figure()
    fig_d_daily.add_bar(x=daily_disc["datetime"], y=daily_disc["1.8.1"], name="Peak",     marker_color="#E24B4A")
    fig_d_daily.add_bar(x=daily_disc["datetime"], y=daily_disc["1.8.2"], name="Standard", marker_color="#EF9F27")
    fig_d_daily.add_bar(x=daily_disc["datetime"], y=daily_disc["1.8.3"], name="Off-Peak", marker_color="#1D9E75")
    fig_d_daily.add_scatter(
        x=daily_disc["datetime"], y=daily_disc["peak_pct"],
        name="Peak %", yaxis="y2", mode="lines+markers",
        line=dict(color="#7F77DD", dash="dot"), marker=dict(size=5),
    )
    fig_d_daily.update_layout(
        barmode="stack", title="Daily Discharge by TOU Slot with Peak %",
        xaxis_title="Date",
        yaxis=dict(title="Discharge (kWh)"),
        yaxis2=dict(title="Peak %", overlaying="y", side="right", range=[0, 115],
                    showgrid=False, ticksuffix="%"),
        legend_title="TOU Slot", height=380,
    )
    st.plotly_chart(fig_d_daily, use_container_width=True)

    st.markdown("**Hourly Discharge Profile with TOU Zones**")
    st.caption("Colour-coded bars show discharge by TOU slot: red = Peak, orange = Standard, green = Off-Peak.")
    fig_disc_tou = go.Figure()
    for slot, color, label in [("1.8.1", "#E24B4A", "Peak"), ("1.8.2", "#EF9F27", "Standard"), ("1.8.3", "#1D9E75", "Off-Peak")]:
        slot_h = batt_hourly_sorted[batt_hourly_sorted["tou_slot"] == slot]
        fig_disc_tou.add_bar(
            x=slot_h["datetime"], y=slot_h["batt_discharge_kwh"],
            name=label, marker_color=color,
        )
    fig_disc_tou.update_layout(
        barmode="stack", xaxis_title="Date / Hour", yaxis_title="Discharge (kWh)",
        legend_title="TOU Slot", height=400,
    )
    st.plotly_chart(fig_disc_tou, use_container_width=True)

    # ── Daily net benefit chart ───────────────────────────────────────────────
    st.markdown("**Daily Net Battery Benefit (R)**")
    daily_benefit = batt_hourly.groupby(batt_hourly["datetime"].dt.date).agg(
        disc_value=("discharge_value_r", "sum"),
        charge_cost=("charge_cost_r", "sum"),
    ).reset_index()
    daily_benefit.columns = ["date", "disc_value", "charge_cost"]
    daily_benefit["net_r"] = daily_benefit["disc_value"] - daily_benefit["charge_cost"]

    fig_net = go.Figure()
    fig_net.add_bar(
        x=daily_benefit["date"],
        y=daily_benefit["net_r"],
        marker_color=np.where(daily_benefit["net_r"] >= 0, "#1D9E75", "#E24B4A"),
        name="Net Benefit (R)",
    )
    fig_net.add_hline(y=0, line_color="gray", line_width=1)
    fig_net.update_layout(xaxis_title="Date", yaxis_title="Net Benefit (R)", height=340)
    st.plotly_chart(fig_net, use_container_width=True)

    # ── Summary table by TOU slot ─────────────────────────────────────────────
    st.markdown("**TOU Slot Summary Table**")
    tou_summary = batt_hourly.groupby("tou_slot").agg(
        charge_kwh=("batt_charge_kwh", "sum"),
        grid_kwh=("grid_to_battery", "sum"),
        solar_kwh=("solar_to_battery", "sum"),
        discharge_kwh=("batt_discharge_kwh", "sum"),
        charge_cost_r=("charge_cost_r", "sum"),
        discharge_value_r=("discharge_value_r", "sum"),
    ).reindex(["1.8.1", "1.8.2", "1.8.3"]).reset_index()
    tou_summary["tou_label"]    = tou_summary["tou_slot"].map({"1.8.1": "Peak", "1.8.2": "Standard", "1.8.3": "Off-Peak"})
    tou_summary["charge_pct"]   = (tou_summary["charge_kwh"]    / total_charge_kwh_sel * 100).round(1)
    tou_summary["disc_pct"]     = (tou_summary["discharge_kwh"] / total_disc_kwh       * 100).round(1)
    tou_summary["net_r"]        = tou_summary["discharge_value_r"] - tou_summary["charge_cost_r"]
    cols_order = ["tou_label", "charge_kwh", "charge_pct", "grid_kwh", "solar_kwh",
                  "charge_cost_r", "discharge_kwh", "disc_pct", "discharge_value_r", "net_r"]
    tou_summary = tou_summary[cols_order]
    tou_summary.columns = ["TOU Slot", "Charge (kWh)", "Charge %", "Grid Charged (kWh)", "Solar Charged (kWh)",
                           "Charge Cost (R)", "Discharge (kWh)", "Discharge %", "Discharge Value (R)", "Net (R)"]
    st.dataframe(
        tou_summary.style.format({
            "Charge (kWh)":          "{:.2f}",  "Charge %":             "{:.1f}%",
            "Grid Charged (kWh)":    "{:.2f}",  "Solar Charged (kWh)":  "{:.2f}",
            "Charge Cost (R)":       "R {:,.2f}",
            "Discharge (kWh)":       "{:.2f}",  "Discharge %":          "{:.1f}%",
            "Discharge Value (R)":   "R {:,.2f}", "Net (R)":             "R {:,.2f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION C — Battery health projection
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🔋 Battery Health Projection")
    daily_work = daily.copy()
    daily_work["cycles_equiv"]      = daily_work["batt_discharge_kwh"] / (battery_capacity * dod_factor)
    daily_work["cumulative_cycles"] = daily_work["cycles_equiv"].cumsum()
    daily_work["projected_health"]  = (1 - daily_work["cumulative_cycles"] / warranty_cycles).clip(0, 1) * 100

    fig_deg = go.Figure()
    fig_deg.add_scatter(
        x=daily_work["date"], y=daily_work["projected_health"],
        name="Estimated Health (%)",
        fill="tozeroy", line=dict(color="#EF9F27"), fillcolor="rgba(239,159,39,0.15)",
    )
    fig_deg.add_hline(
        y=80, line_dash="dash", line_color="#E24B4A",
        annotation_text="80% threshold (typical end-of-warranty)",
    )
    fig_deg.update_layout(
        xaxis_title="Date", yaxis_title="Estimated Health (%)",
        yaxis_range=[0, 105], height=360,
    )
    st.plotly_chart(fig_deg, use_container_width=True)

    st.info(
        f"ℹ️ Health = 1 − (cumulative discharge kWh ÷ ({battery_capacity:.0f} kWh × 80% DoD × {int(warranty_cycles):,} cycles)). "
        "Uses actual discharge energy from data. Actual degradation also depends on temperature, cell chemistry, and charge rate."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SEASONAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Seasonal Patterns & Weather Influence")

    daily["month"]        = daily["date"].dt.month
    daily["month_name"]   = daily["date"].dt.strftime("%b %Y")
    daily["season_label"] = daily["date"].apply(
        lambda d: "High Demand (Jun–Aug)" if d.month in [6, 7, 8] else "Low Demand (Sep–May)"
    )

    # Solar vs load by billing run
    fig_season = go.Figure()
    for run in daily["billing_run"].unique():
        sub = daily[daily["billing_run"] == run]
        fig_season.add_scatter(x=sub["date"], y=sub["solar_kwh"], name=f"{run} — Solar", mode="lines")
    fig_season.update_layout(
        title="Daily Solar Production by Billing Run",
        xaxis_title="Date", yaxis_title="Solar (kWh)", height=380,
    )
    st.plotly_chart(fig_season, use_container_width=True)

    # Hourly solar profile by season
    hourly["season_label"] = hourly["datetime"].apply(
        lambda d: "High Demand (Jun–Aug)" if d.month in [6, 7, 8] else "Low Demand (Sep–May)"
    )
    hourly["hour"] = hourly["datetime"].dt.hour

    hourly_season = hourly.groupby(["season_label", "hour"]).agg(
        solar=("solar_kwh", "mean"),
        load=("load_kwh", "mean"),
        grid_import=("grid_import_kwh", "mean"),
    ).reset_index()

    fig_hourly_season = px.line(
        hourly_season,
        x="hour", y="solar",
        color="season_label",
        labels={"hour": "Hour of Day", "solar": "Avg Solar (kWh)", "season_label": "Season"},
        title="Average Hourly Solar Profile by Season",
        height=380,
    )
    fig_hourly_season.update_xaxes(tickmode="linear", dtick=1)
    st.plotly_chart(fig_hourly_season, use_container_width=True)

    # ── Seasonal % change chart ───────────────────────────────────────────────
    st.subheader("Seasonal % Change — Billing Run vs Billing Run")
    st.caption(
        "Percentage increase or decrease in key metrics from one billing run to the next. "
        "Positive = improvement / increase, negative = decline / decrease."
    )

    # Build per-billing-run aggregates sorted chronologically
    seasonal_agg = daily.groupby("billing_run").agg(
        solar=("solar_kwh", "sum"),
        load=("load_kwh", "sum"),
        grid_import=("grid_import_kwh", "sum"),
        grid_export=("grid_export_kwh", "sum"),
        batt_discharge=("batt_discharge_kwh", "sum"),
        period_end=("date", "max"),
    ).reset_index().sort_values("period_end").reset_index(drop=True)

    # % change from previous billing run
    pct_cols = ["solar", "load", "grid_import", "grid_export", "batt_discharge"]
    pct_change = seasonal_agg[["billing_run"] + pct_cols].copy()
    for col in pct_cols:
        pct_change[f"{col}_pct"] = pct_change[col].pct_change() * 100

    pct_change = pct_change.dropna(subset=[f"{pct_cols[0]}_pct"])

    # Metric selector for % change chart
    pct_metric_map = {
        "Solar Production":    "solar_pct",
        "Load Consumed":       "load_pct",
        "Grid Import":         "grid_import_pct",
        "Grid Export":         "grid_export_pct",
        "Battery Discharge":   "batt_discharge_pct",
    }
    selected_pct_metrics = st.multiselect(
        "Select metrics to compare",
        options=list(pct_metric_map.keys()),
        default=["Solar Production", "Load Consumed", "Grid Import"],
    )

    if selected_pct_metrics:
        fig_pct = go.Figure()
        colors = ["#EF9F27", "#E24B4A", "#7F77DD", "#1D9E75", "#3B8BD4"]
        for i, metric in enumerate(selected_pct_metrics):
            col_name = pct_metric_map[metric]
            fig_pct.add_scatter(
                x=pct_change["billing_run"],
                y=pct_change[col_name].round(1),
                name=metric,
                mode="lines+markers",
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=8),
            )
        fig_pct.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig_pct.update_layout(
            title="% Change in Energy Metrics Between Billing Runs",
            xaxis_title="Billing Run",
            yaxis_title="% Change vs Previous Run",
            legend_title="Metric",
            height=400,
        )
        st.plotly_chart(fig_pct, use_container_width=True)

        # Table of % changes
        display_pct = pct_change[["billing_run"] + [pct_metric_map[m] for m in selected_pct_metrics]].copy()
        display_pct.columns = ["Billing Run"] + selected_pct_metrics
        fmt = {m: "{:+.1f}%" for m in selected_pct_metrics}
        st.dataframe(display_pct.style.format(fmt), use_container_width=True, hide_index=True)
    else:
        st.info("Select at least one metric above to see the % change chart.")

    # Load heatmap — day of week vs hour
    st.subheader("Load Pattern — Day of Week vs Hour")
    hourly["dow_name"] = hourly["datetime"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    load_pivot = hourly.groupby(["dow_name", "hour"])["load_kwh"].mean().unstack("hour")
    load_pivot = load_pivot.reindex(dow_order)

    fig_load_heat = px.imshow(
        load_pivot,
        labels={"x": "Hour", "y": "Day", "color": "Avg Load (kWh)"},
        color_continuous_scale="YlOrRd",
        title="Average Load (kWh) by Day of Week and Hour",
        aspect="auto",
    )
    fig_load_heat.update_layout(height=380)
    st.plotly_chart(fig_load_heat, use_container_width=True)

    # Solar self-consumption ratio over time
    st.subheader("Solar Self-Consumption")
    daily["self_consumption_pct"] = (
        (daily["solar_kwh"] - daily["grid_export_kwh"]) / daily["solar_kwh"].replace(0, np.nan) * 100
    ).clip(0, 100)

    fig_sc = px.line(
        daily.dropna(subset=["self_consumption_pct"]),
        x="date", y="self_consumption_pct",
        labels={"date": "Date", "self_consumption_pct": "Self-Consumption (%)"},
        title="Daily Solar Self-Consumption Rate",
        height=360,
    )
    fig_sc.add_hline(
        y=daily["self_consumption_pct"].mean(), line_dash="dash",
        annotation_text=f"Average: {daily['self_consumption_pct'].mean():.1f}%",
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    import requests as _requests
    import sqlite3
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    # Umhlanga coordinates - used only for the free Open-Meteo forecast call
    # below (Section 4). This is NOT the Sigenergy API and has no meaningful
    # rate limit, so it's fine to call directly on every tab render.
    LAT, LON = -29.7215, 31.0498

    # ── DB path (shared with live_logger.py on the Pi) ────────────────────────
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_readings.db")

    # ── CSV fallback paths ─────────────────────────────────────────────────────
    # When this app runs on Streamlit Cloud, it has no access to the Pi's
    # local live_readings.db at all - that file only exists on the Pi's disk.
    # hourly_export_push.sh (cron, on the Pi) exports the DB to these two CSVs
    # and pushes them to GitHub every hour, so Streamlit Cloud can show
    # reasonably fresh (up to ~1hr old) data instead of nothing. The Pi-hosted
    # version of this app always prefers the live DB (true 5-min freshness)
    # and only falls back to CSV if the DB is genuinely missing.
    LATEST_CSV_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_latest.csv")
    HISTORY_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_history.csv")

    # ── Read-only DB helpers ───────────────────────────────────────────────────
    # NOTE: This tab does NOT call the Sigenergy API directly. live_logger.py
    # runs on the Pi via cron every 5 minutes (matching the documented API
    # rate limit of "one access per station/device every 5 minutes" which
    # applies identically to /energyFlow, /summary, and /devices/.../realtimeInfo).
    # This tab only ever reads from live_readings.db (or its CSV export as a
    # fallback), so the dashboard can be auto-refreshed as often as you like
    # (e.g. every 30s during a demo) without ever risking error 1201.

    def load_latest() -> tuple[dict, str]:
        """Return (reading_dict, source) where source is 'db', 'csv', or 'none'.
        Tries live_readings.db first (true real-time, available on the Pi);
        falls back to live_latest.csv (up to ~1hr old, available wherever
        GitHub synced it, e.g. Streamlit Cloud) if the DB doesn't exist.
        The source matters because the two have very different freshness
        cadences - 5 minutes for the DB, up to an hour for the CSV - so the
        "is this stale" check downstream needs to use a different threshold
        depending on which one we actually read from."""
        if os.path.exists(DB_PATH):
            try:
                conn = sqlite3.connect(DB_PATH)
                row = conn.execute(
                    "SELECT * FROM live_readings ORDER BY ts DESC LIMIT 1"
                ).fetchone()
                cols = [d[0] for d in conn.execute("SELECT * FROM live_readings LIMIT 0").description] if row else []
                conn.close()
                if row:
                    return dict(zip(cols, row)), "db"
            except Exception:
                pass

        # Fallback: CSV export (Streamlit Cloud scenario)
        if os.path.exists(LATEST_CSV_PATH):
            try:
                df = pd.read_csv(LATEST_CSV_PATH)
                if len(df) > 0:
                    return df.iloc[0].to_dict(), "csv"
            except Exception:
                pass

        return {}, "none"

    def load_history(days=7) -> pd.DataFrame:
        """Return readings from the last `days` days, oldest first. Same
        DB-first, CSV-fallback pattern as load_latest()."""
        if os.path.exists(DB_PATH):
            try:
                conn = sqlite3.connect(DB_PATH)
                cutoff = (_dt.now(_tz.utc) - _td(days=days)).isoformat()
                df = pd.read_sql(
                    "SELECT * FROM live_readings WHERE ts >= ? ORDER BY ts",
                    conn, params=(cutoff,)
                )
                conn.close()
                return df
            except Exception:
                pass

        # Fallback: CSV export (Streamlit Cloud scenario). Already filtered
        # to 7 days by export_live_csv.py, but re-filter here in case `days`
        # differs from that script's default.
        if os.path.exists(HISTORY_CSV_PATH):
            try:
                df = pd.read_csv(HISTORY_CSV_PATH)
                if "ts" in df.columns and len(df) > 0:
                    cutoff = (_dt.now(_tz.utc) - _td(days=days)).isoformat()
                    df = df[df["ts"] >= cutoff].sort_values("ts")
                return df
            except Exception:
                pass

        return pd.DataFrame()

    # ── Load latest reading from DB (or CSV fallback) ──────────────────────────
    d, data_source = load_latest()

    data_fresh = False
    reading_age_seconds = None
    if d and d.get("ts"):
        try:
            reading_ts = pd.Timestamp(d["ts"])
            if reading_ts.tzinfo is None:
                reading_ts = reading_ts.tz_localize("UTC")
            reading_age_seconds = (pd.Timestamp.now(tz="UTC") - reading_ts).total_seconds()
            # Freshness threshold depends on the data source:
            # - DB: live_logger.py runs every 5 min, so 10 min allows one
            #   missed cycle plus slack before flagging as stale.
            # - CSV: hourly_export_push.sh runs once per hour, so a much
            #   longer threshold is needed - otherwise the CSV fallback would
            #   ALWAYS show as stale, defeating the point of having it. 90
            #   min allows one full hourly cycle plus slack.
            freshness_threshold = 600 if data_source == "db" else 5400
            data_fresh = reading_age_seconds < freshness_threshold
        except Exception:
            data_fresh = False

    # Reconstruct nested structures the rest of this tab expects, from the
    # flat columns live_logger.py writes.
    live_data = None
    if d:
        live_data = {
            "pv_strings": {
                f"{inv}-{s}": {"v": d.get(f"inv{inv}_pv{s}_v", 0), "a": d.get(f"inv{inv}_pv{s}_a", 0)}
                for inv in range(1, 4) for s in range(1, 5)
            },
            "phases": {ph: {"v": d.get(f"phase_{ph}_v", 0), "a": d.get(f"phase_{ph}_a", 0)} for ph in ["a", "b", "c"]},
            "power_factor": d.get("power_factor", 0),
            "grid_freq": d.get("grid_freq", 0),
            "inv_temps": [d.get(f"inv{i}_temp", 0) for i in range(1, 4)],
        }

    weather = {
        "temperature":   d.get("temperature", 0),
        "cloud_cover":   d.get("cloud_cover", 0),
        "wind_speed":    d.get("wind_speed", 0),
        "precipitation": d.get("precipitation", 0),
        "irradiance":    d.get("irradiance", 0),
    } if d else {}

    # ── Header ────────────────────────────────────────────────────────────────
    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        st.subheader("Live System Dashboard — The Millennial")
        _source_label = "updates every 5 min" if data_source == "db" else "synced hourly from Pi via GitHub"
        if data_fresh:
            reading_ts_local = pd.Timestamp(d["ts"])
            if reading_ts_local.tzinfo is None:
                reading_ts_local = reading_ts_local.tz_localize("UTC")
            reading_ts_local = reading_ts_local.tz_convert("Africa/Johannesburg")
            st.caption(f"Last reading: {reading_ts_local.strftime('%d %b %Y %H:%M:%S')} SAST "
                       f"({_source_label})")
        elif d:
            reading_ts_local = pd.Timestamp(d["ts"])
            if reading_ts_local.tzinfo is None:
                reading_ts_local = reading_ts_local.tz_localize("UTC")
            reading_ts_local = reading_ts_local.tz_convert("Africa/Johannesburg")
            age_min = int(reading_age_seconds // 60) if reading_age_seconds else 0
            st.caption(f"Last reading: {reading_ts_local.strftime('%d %b %Y %H:%M:%S')} SAST "
                       f"({age_min} min ago)")
            if data_source == "db":
                st.warning(
                    "The background logger (live_logger.py) hasn't reported in over "
                    "10 minutes — check the Pi's cron job and live_logger.log."
                )
            else:
                st.warning(
                    "The hourly GitHub sync (hourly_export_push.sh on the Pi) hasn't "
                    "updated in over 90 minutes — check that script's cron job and log "
                    "on the Pi."
                )
        else:
            st.info(
                "No live data yet. Make sure live_logger.py is running on the Pi "
                "(cron job every 5 minutes) to populate live_readings.db, and that "
                "hourly_export_push.sh is syncing it to GitHub if you're viewing this "
                "on Streamlit Cloud."
            )
    with hdr_right:
        auto_refresh = st.toggle("Auto-refresh 30s", value=False)
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()

    if not d:
        st.stop()

    # ── Extract values ────────────────────────────────────────────────────────
    pv_kw        = d.get("pv_kw", 0)
    grid_kw      = d.get("grid_kw", 0)
    load_kw      = d.get("load_kw", 0)
    batt_kw      = d.get("battery_kw", 0)
    batt_soc     = d.get("battery_soc", 0)
    grid_import  = max(0, -grid_kw)
    grid_export  = max(0,  grid_kw)
    batt_charge  = max(0,  batt_kw)
    batt_disc    = max(0, -batt_kw)
    # Average across the 3 inverters for the summary card; individual
    # per-inverter temps are shown in the Inverter & Grid Details expander.
    _inv_temps_raw = [d.get(f"inv{i}_temp", 0) for i in range(1, 4)]
    inv_temp     = sum(_inv_temps_raw) / len(_inv_temps_raw) if _inv_temps_raw else 0
    cloud        = d.get("cloud_cover", 0)
    irradiance   = d.get("irradiance", 0)
    temperature  = d.get("temperature", 0)

    # ── SECTION 1 — Power Flow Cards ──────────────────────────────────────────
    st.markdown("---")

    # Custom styled power flow cards using HTML
    def power_card(title, value, unit, subtitle, color):
        return f"""
        <div style='background:{color}18;border:1px solid {color}44;border-radius:12px;
                    padding:16px 20px;text-align:center;height:130px;'>
            <div style='color:#aaa;font-size:13px;margin-bottom:4px'>{title}</div>
            <div style='color:{color};font-size:32px;font-weight:700;line-height:1.1'>
                {value}<span style='font-size:16px;font-weight:400'> {unit}</span>
            </div>
            <div style='color:#888;font-size:12px;margin-top:6px'>{subtitle}</div>
        </div>"""

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(power_card(
            "Solar Generation", f"{pv_kw:.1f}", "kW",
            f"Irradiance: {irradiance:.0f} W/m2",
            "#EF9F27"), unsafe_allow_html=True)
    with c2:
        st.markdown(power_card(
            "Site Load", f"{load_kw:.1f}", "kW",
            f"Total consumption",
            "#E24B4A"), unsafe_allow_html=True)
    with c3:
        if grid_import > 0:
            st.markdown(power_card(
                "Grid Import", f"{grid_import:.1f}", "kW",
                "Buying from grid", "#7F77DD"), unsafe_allow_html=True)
        else:
            st.markdown(power_card(
                "Grid Export", f"{grid_export:.1f}", "kW",
                "Selling to grid", "#1D9E75"), unsafe_allow_html=True)
    with c4:
        if batt_charge > 0:
            st.markdown(power_card(
                "Battery Charging", f"{batt_charge:.1f}", "kW",
                f"SoC: {batt_soc:.0f}%", "#1D9E75"), unsafe_allow_html=True)
        elif batt_disc > 0:
            st.markdown(power_card(
                "Battery Discharging", f"{batt_disc:.1f}", "kW",
                f"SoC: {batt_soc:.0f}%", "#EF9F27"), unsafe_allow_html=True)
        else:
            st.markdown(power_card(
                "Battery", "Idle", "",
                f"SoC: {batt_soc:.0f}%", "#888888"), unsafe_allow_html=True)
    with c5:
        soc_color = "#E24B4A" if batt_soc < 20 else "#EF9F27" if batt_soc < 50 else "#1D9E75"
        _inv_temp_label = f"{inv_temp:.1f}degC" if inv_temp > 0 else "N/A"
        st.markdown(power_card(
            "Battery SoC", f"{batt_soc:.0f}", "%",
            f"Inverter: {_inv_temp_label}", soc_color), unsafe_allow_html=True)

    # ── SECTION 1B — Animated Sankey Energy Flow ──────────────────────────────
    st.markdown("---")
    st.markdown("**Live Energy Flow**")

    # Calculate energy flows for Sankey
    solar_to_load    = min(pv_kw, max(0, load_kw - batt_disc - max(0, -grid_kw)))
    solar_to_batt    = min(pv_kw - solar_to_load, batt_charge) if batt_charge > 0 else 0
    solar_to_grid    = max(0, pv_kw - solar_to_load - solar_to_batt)
    grid_to_load     = max(0, grid_import - 0)
    grid_to_batt     = max(0, batt_charge - solar_to_batt)
    batt_to_load     = max(0, batt_disc)

    # 3-column layout matching the Sigenergy portal's own energy flow diagram:
    #   Left column   (sources):      Solar, Grid
    #   Middle column  (always Battery, whether charging or discharging):
    #   Right column  (destinations): Load, Grid Export
    # Battery sits in a TRUE middle column regardless of charge/discharge
    # state - it always both receives (from Solar/Grid when charging) and
    # sends (to Load when discharging) so it never collapses into a thin
    # pass-through or gets pinned to an edge it doesn't belong on.
    sankey_labels = ["Solar", "Grid", "Battery", "Load", "Grid Export"]
    sankey_source = []
    sankey_target = []
    sankey_value  = []
    sankey_colors = []

    flow_map = [
        (0, 3, solar_to_load,  "rgba(239,159,39,0.6)"),   # Solar -> Load (direct, bypasses battery)
        (0, 2, solar_to_batt,  "rgba(239,159,39,0.4)"),   # Solar -> Battery (charging)
        (0, 4, solar_to_grid,  "rgba(239,159,39,0.5)"),   # Solar -> Grid Export
        (1, 3, grid_to_load,   "rgba(127,119,221,0.6)"),  # Grid -> Load (direct, bypasses battery)
        (1, 2, grid_to_batt,   "rgba(127,119,221,0.4)"),  # Grid -> Battery (charging from grid)
        (2, 3, batt_to_load,   "rgba(29,158,117,0.6)"),   # Battery -> Load (discharging)
    ]
    for src, tgt, val, col in flow_map:
        if val > 0.05:
            sankey_source.append(src)
            sankey_target.append(tgt)
            sankey_value.append(round(val, 2))
            sankey_colors.append(col)

    if sankey_source:
        fig_sankey = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20, thickness=25,
                line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                label=sankey_labels,
                color=["#EF9F27","#7F77DD","#1D9E75","#E24B4A","#3B8BD4"],
                # True 3-column layout: x=0 (sources), x=0.5 (battery, always
                # middle), x=1.0 (destinations) - matches the Sigenergy
                # portal's own Energy Statistics diagram structure.
                x=[0.0, 0.0, 0.5, 1.0, 1.0],
                y=[0.1, 0.7, 0.4, 0.3, 0.7],
            ),
            link=dict(
                source=sankey_source,
                target=sankey_target,
                value=sankey_value,
                color=sankey_colors,
                label=[f"{v:.2f} kW" for v in sankey_value],
            )
        ))
        fig_sankey.update_layout(
            height=320,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=13),
        )
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("No significant power flows to display — system may be idle.")

    # ── SECTION 2 — Gauges + Donut ────────────────────────────────────────────
    st.markdown("---")
    g1, g2, g3 = st.columns(3)

    with g1:
        soc_color = "#E24B4A" if batt_soc < 20 else "#EF9F27" if batt_soc < 50 else "#1D9E75"
        fig_soc_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=batt_soc,
            title={"text": "Battery SoC (%)"},
            delta={"reference": 50, "valueformat": ".0f"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": soc_color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 20],  "color": "rgba(226,75,74,0.15)"},
                    {"range": [20, 50], "color": "rgba(239,159,39,0.15)"},
                    {"range": [50, 100],"color": "rgba(29,158,117,0.15)"},
                ],
            }
        ))
        fig_soc_g.update_layout(height=250, margin=dict(t=50,b=10,l=20,r=20),
                                 paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_soc_g, use_container_width=True)

    with g2:
        max_solar = 78.0
        theoretical_kw = (irradiance / 1000) * max_solar if irradiance > 0 else max_solar
        performance_ratio = min(100, (pv_kw / theoretical_kw * 100)) if theoretical_kw > 0 else 0
        fig_perf = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(performance_ratio, 1),
            title={"text": "Solar Performance (%)"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#EF9F27", "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 40],  "color": "rgba(226,75,74,0.15)"},
                    {"range": [40, 70], "color": "rgba(239,159,39,0.15)"},
                    {"range": [70, 100],"color": "rgba(29,158,117,0.15)"},
                ],
            }
        ))
        fig_perf.update_layout(height=250, margin=dict(t=50,b=10,l=20,r=20),
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_perf, use_container_width=True)
        st.caption(f"Actual: {pv_kw:.1f} kW / Theoretical: {theoretical_kw:.1f} kW at {irradiance:.0f} W/m2")

    with g3:
        solar_to_load_d  = min(pv_kw, load_kw)
        batt_to_load_d   = min(batt_disc, max(0, load_kw - solar_to_load_d))
        grid_to_load_d   = max(0, load_kw - solar_to_load_d - batt_to_load_d)
        donut_vals   = [max(0,solar_to_load_d), max(0,batt_to_load_d), max(0,grid_to_load_d)]
        donut_labels = ["Solar", "Battery", "Grid"]
        donut_colors = ["#EF9F27", "#1D9E75", "#7F77DD"]
        fig_donut = go.Figure(go.Pie(
            values=donut_vals, labels=donut_labels,
            marker_colors=donut_colors,
            hole=0.55, textinfo="percent+label", textfont_size=12,
        ))
        fig_donut.update_layout(
            title="Current Load Source Mix",
            height=250, margin=dict(t=50,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── SECTION 3 — Peak Shaving + Live Rand Savings ──────────────────────────
    st.markdown("---")
    st.markdown("**Peak Shaving Effectiveness & Live Savings**")

    # Current TOU slot
    now_sa = _dt.now(_tz(_td(hours=2)))
    _tou_now = get_tou_slot(now_sa) if 'get_tou_slot' in dir() else "1.8.2"
    _season_now = get_season(now_sa)
    _tariff_now = get_tariff_for_date(now_sa, _tou_now)
    _tou_name = {"1.8.1":"PEAK","1.8.2":"Standard","1.8.3":"Off-Peak"}[_tou_now]
    _tou_color = {"1.8.1":"#E24B4A","1.8.2":"#EF9F27","1.8.3":"#1D9E75"}[_tou_now]

    ps1, ps2, ps3, ps4 = st.columns(4)
    ps1.markdown(f"""
    <div style='background:{_tou_color}18;border:1px solid {_tou_color}44;
                border-radius:12px;padding:16px;text-align:center'>
        <div style='color:#aaa;font-size:12px'>Current TOU Slot</div>
        <div style='color:{_tou_color};font-size:28px;font-weight:700'>{_tou_name}</div>
        <div style='color:#888;font-size:12px'>R{_tariff_now:.4f}/kWh</div>
    </div>""", unsafe_allow_html=True)

    # Grid avoided by battery discharge right now
    grid_avoided_kw = batt_disc
    rand_saved_per_hr = grid_avoided_kw * _tariff_now
    ps2.metric("Grid Avoided Now",   f"{grid_avoided_kw:.1f} kW",
               help="Battery discharge currently displacing grid import")
    ps3.metric("Saving Rate",        f"R {rand_saved_per_hr:.2f}/hr",
               help=f"At {_tou_name} tariff R{_tariff_now:.4f}/kWh")

    # Estimated today's saving from battery discharge
    daily_disc = d.get("bat_dischd", 0)
    est_daily_saving = daily_disc * _tariff_now
    ps4.metric("Est. Today's Saving", f"R {est_daily_saving:.2f}",
               help="Battery discharge today × current tariff rate")

    # ── SECTION 4 — Daily Forecast ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Today's Production Forecast vs Actual**")

    try:
        wx_r = _requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": LAT, "longitude": LON,
            "hourly": "shortwave_radiation",
            "forecast_days": 1,
            "timezone": "Africa/Johannesburg",
        }, timeout=10)
        wx_d = wx_r.json()
        hours_today = wx_d["hourly"]["time"]
        irr_today   = wx_d["hourly"]["shortwave_radiation"]
        # Theoretical production per hour = kWp × irradiance/1000
        theoretical_hourly = [round((i or 0) / 1000 * 78.0, 2) for i in irr_today]
        forecast_total = sum(theoretical_hourly)
        actual_today   = d.get("pv_daily_kwh", 0)

        fig_forecast = go.Figure()
        fig_forecast.add_bar(
            x=hours_today, y=theoretical_hourly,
            name="Forecast (kWh/hr)", marker_color="rgba(239,159,39,0.3)",
        )
        # Mark current hour
        curr_hr_str = now_sa.strftime("%Y-%m-%dT%H:00")
        if curr_hr_str in hours_today:
            curr_idx = hours_today.index(curr_hr_str)
            actual_by_hour = [0] * len(hours_today)
            # Distribute actual evenly up to current hour as rough approximation
            if curr_idx > 0:
                per_hr = actual_today / curr_idx
                for i in range(curr_idx):
                    actual_by_hour[i] = round(per_hr, 2)
            fig_forecast.add_bar(
                x=hours_today[:curr_idx+1], y=actual_by_hour[:curr_idx+1],
                name="Actual (kWh/hr)", marker_color="rgba(239,159,39,0.9)",
            )
        fig_forecast.add_vline(
            x=curr_hr_str, line_dash="dash", line_color="white",
            annotation_text="Now", annotation_position="top right"
        )
        fig_forecast.update_layout(
            barmode="overlay",
            title=f"Forecast: {forecast_total:.0f} kWh | Actual so far: {actual_today:.1f} kWh",
            xaxis_title="Hour", yaxis_title="kWh",
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title="",
        )
        fig_forecast.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig_forecast.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig_forecast, use_container_width=True)

        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Forecast Total Today", f"{forecast_total:.0f} kWh")
        fc2.metric("Actual So Far",        f"{actual_today:.1f} kWh")
        fc3.metric("Remaining Expected",   f"{max(0, forecast_total - actual_today):.0f} kWh")

    except Exception:
        st.caption("Weather forecast unavailable.")

    # ── SECTION 5 — PV String Health ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("**PV String Health Monitor**")

    # Detect inverters where the ENTIRE device fetch likely failed (all 4
    # strings AND pv_kw are exactly 0) vs. genuinely idle/disconnected
    # strings on an inverter that otherwise reported real data. A failed
    # fetch defaults every field to 0 via .get(..., 0), which is a different
    # situation from a string that's truly producing nothing.
    _missing_inverters = []
    for inv in range(1, 4):
        _inv_pv_kw = d.get(f"inv{inv}_pv_kw", 0)
        _inv_strings_all_zero = all(
            d.get(f"inv{inv}_pv{s}_v", 0) == 0 and d.get(f"inv{inv}_pv{s}_a", 0) == 0
            for s in range(1, 5)
        )
        if _inv_pv_kw == 0 and _inv_strings_all_zero:
            _missing_inverters.append(inv)

    if _missing_inverters:
        _names = ", ".join(f"Inverter {i}" for i in _missing_inverters)
        st.warning(
            f"{_names} reported no data on the last logger run (likely a "
            f"transient API timeout/error - live_logger.py retries automatically, "
            f"so this should resolve on the next 5-minute cycle). Strings for "
            f"{'this inverter' if len(_missing_inverters)==1 else 'these inverters'} "
            f"are hidden below rather than shown as disconnected."
        )

    if live_data and live_data.get("pv_strings"):
        strings = {i: v for i, v in live_data["pv_strings"].items()
                   if (abs(v["v"]) > 0.1 or abs(v["a"]) > 0.01)
                   and int(i.split("-")[0]) not in _missing_inverters}
        if strings:
            str_rows = []
            powers = []
            for i, sv in strings.items():
                p = round(sv["v"] * sv["a"] / 1000, 3)
                powers.append(p)
                inv_num, string_num = i.split("-")
                str_rows.append({
                    "String": f"Inv{inv_num} PV{string_num}",
                    "Voltage (V)": round(sv["v"], 2),
                    "Current (A)": round(sv["a"], 3),
                    "Power (kW)":  p,
                })
            avg_p = sum(powers) / len(powers) if powers else 0
            # Flag strings more than 20% below average
            for row in str_rows:
                if avg_p > 0.05 and row["Power (kW)"] < avg_p * 0.80:
                    row["Status"] = "FAULT"
                elif avg_p > 0.05:
                    row["Status"] = "OK"
                else:
                    row["Status"] = "-"

            str_df = pd.DataFrame(str_rows)

            # Bar chart with fault highlighting
            bar_colors = []
            for row in str_rows:
                if row.get("Status") == "FAULT":
                    bar_colors.append("#E24B4A")
                else:
                    bar_colors.append("#EF9F27")

            fig_strings = go.Figure()
            fig_strings.add_bar(
                x=[r["String"] for r in str_rows],
                y=[r["Power (kW)"] for r in str_rows],
                marker_color=bar_colors,
                text=[f"{r['Power (kW)']:.3f} kW" for r in str_rows],
                textposition="outside",
            )
            if avg_p > 0.05:
                fig_strings.add_hline(y=avg_p, line_dash="dash", line_color="white",
                                      annotation_text=f"Avg: {avg_p:.3f} kW")
                fig_strings.add_hline(y=avg_p * 0.80, line_dash="dot", line_color="#E24B4A",
                                      annotation_text="Fault threshold (80%)")
            fig_strings.update_layout(
                title="PV String Power Output",
                xaxis_title="String", yaxis_title="Power (kW)",
                height=300, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_strings.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            fig_strings.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_strings, use_container_width=True)

            faults = [r for r in str_rows if r.get("Status") == "FAULT"]
            if faults:
                st.error(f"Potential fault detected on: {', '.join(r['String'] for r in faults)} "
                         f"— producing >20% below average output.")
            elif avg_p > 0.05:
                st.success("All PV strings operating within normal range.")

            st.dataframe(str_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No PV string data available — panels may not be generating.")
    else:
        st.caption("PV string data not available in current reading.")

    # ── SECTION 6 — CO2 & Environmental Impact ───────────────────────────────
    st.markdown("---")
    st.markdown("**Environmental Impact**")

    env1, env2, env3, env4 = st.columns(4)
    co2_life  = d.get("co2_saved", 0)
    coal_life = d.get("coal_saved", 0)
    trees     = d.get("trees", 0)
    pv_life   = d.get("pv_life_kwh", 0)

    # Equivalent cars off road (avg car = 4.6 tons CO2/year)
    cars_equiv = co2_life / 4.6 if co2_life > 0 else 0

    env1.markdown(f"""
    <div style='background:rgba(29,158,117,0.1);border:1px solid rgba(29,158,117,0.3);
                border-radius:12px;padding:16px;text-align:center'>
        <div style='font-size:32px'>🌱</div>
        <div style='color:#1D9E75;font-size:22px;font-weight:700'>{co2_life:.1f} tons</div>
        <div style='color:#aaa;font-size:12px'>CO2 Avoided (lifetime)</div>
    </div>""", unsafe_allow_html=True)
    env2.markdown(f"""
    <div style='background:rgba(59,139,212,0.1);border:1px solid rgba(59,139,212,0.3);
                border-radius:12px;padding:16px;text-align:center'>
        <div style='font-size:32px'>🌳</div>
        <div style='color:#3B8BD4;font-size:22px;font-weight:700'>{trees:.0f} trees</div>
        <div style='color:#aaa;font-size:12px'>Equivalent Trees Planted</div>
    </div>""", unsafe_allow_html=True)
    env3.markdown(f"""
    <div style='background:rgba(239,159,39,0.1);border:1px solid rgba(239,159,39,0.3);
                border-radius:12px;padding:16px;text-align:center'>
        <div style='font-size:32px'>🚗</div>
        <div style='color:#EF9F27;font-size:22px;font-weight:700'>{cars_equiv:.1f}</div>
        <div style='color:#aaa;font-size:12px'>Cars off road (equiv. year)</div>
    </div>""", unsafe_allow_html=True)
    env4.markdown(f"""
    <div style='background:rgba(127,119,221,0.1);border:1px solid rgba(127,119,221,0.3);
                border-radius:12px;padding:16px;text-align:center'>
        <div style='font-size:32px'>⚡</div>
        <div style='color:#7F77DD;font-size:22px;font-weight:700'>{pv_life:,.0f}</div>
        <div style='color:#aaa;font-size:12px'>kWh Generated (lifetime)</div>
    </div>""", unsafe_allow_html=True)

    # ── SECTION 7 — Today's Energy + Weather ─────────────────────────────────
    st.markdown("---")
    en_col, wx_col = st.columns(2)

    with en_col:
        st.markdown("**Today's Energy Summary**")
        e1, e2 = st.columns(2)
        e1.metric("Solar Today",      f"{d.get('pv_daily_kwh',0):.1f} kWh")
        e2.metric("Solar This Month",  f"{d.get('pv_month_kwh',0):.1f} kWh")
        e3, e4 = st.columns(2)
        e3.metric("Solar This Year",   f"{d.get('pv_year_kwh',0):.1f} kWh")
        e4.metric("Battery Discharged",f"{d.get('bat_dischd',0):.1f} kWh")

    with wx_col:
        st.markdown("**Current Weather — Umhlanga**")
        w1, w2 = st.columns(2)
        w1.metric("Temperature",    f"{weather.get('temperature',0):.1f} degC")
        w2.metric("Cloud Cover",    f"{weather.get('cloud_cover',0):.0f}%")
        w3, w4 = st.columns(2)
        w3.metric("Wind Speed",     f"{weather.get('wind_speed',0):.1f} km/h")
        w4.metric("Irradiance",     f"{weather.get('irradiance',0):.0f} W/m2")
        cloud_pct = weather.get("cloud_cover", 0)
        cloud_color = "#1D9E75" if cloud_pct < 30 else "#EF9F27" if cloud_pct < 70 else "#7F77DD"
        st.markdown(f"""
        <div style='margin-top:8px'>
            <div style='font-size:12px;color:#aaa;margin-bottom:4px'>Cloud cover solar impact</div>
            <div style='background:#222;border-radius:6px;height:12px;width:100%'>
                <div style='background:{cloud_color};height:12px;width:{cloud_pct}%;
                            border-radius:6px'></div>
            </div>
            <div style='font-size:11px;color:#888;margin-top:2px'>
                Est. solar reduction: {cloud_pct:.0f}%
            </div>
        </div>""", unsafe_allow_html=True)

    # ── SECTION 8 — Inverter Details (collapsed) ──────────────────────────────
    st.markdown("---")
    with st.expander("Inverter & Grid Details", expanded=False):
        _total_inv_pv_kw = sum(d.get(f"inv{i}_pv_kw", 0) for i in range(1, 4))
        inv1, inv2, inv3, inv4 = st.columns(4)
        inv1.metric("Total PV Power (3 inverters)", f"{_total_inv_pv_kw:.2f} kW")
        inv2.metric("Avg Inverter Temp",
                    f"{inv_temp:.1f} degC" if inv_temp > 0 else "N/A",
                    delta=("High" if inv_temp > 60 else "Normal") if inv_temp > 0 else None,
                    delta_color="inverse" if inv_temp > 60 else "off")
        inv3.metric("Power Factor",   f"{live_data.get('power_factor',0):.3f}" if live_data else "—")
        inv4.metric("Grid Frequency", f"{live_data.get('grid_freq',0):.2f} Hz" if live_data else "—",
                    delta="OK" if live_data and 49.5 <= live_data.get('grid_freq',50) <= 50.5 else "Out of range",
                    delta_color="off" if live_data and 49.5 <= live_data.get('grid_freq',50) <= 50.5 else "inverse")

        st.markdown("**Per-Inverter Temperature & PV Output**")
        st.caption(
            "Note: internal temperature isn't reported by this inverter model/firmware "
            "(also absent from the Sigenergy portal's own Real Time Info panel) - "
            "shown as '—' when unavailable rather than a misleading 0 degC."
        )
        inv_rows = [{
            "Inverter": f"Inverter {i}",
            "Temp (degC)": (f"{d.get(f'inv{i}_temp', 0):.1f}" if d.get(f"inv{i}_temp", 0) > 0 else "—"),
            "PV Power (kW)": round(d.get(f"inv{i}_pv_kw", 0), 2),
            "Status": ("High" if d.get(f"inv{i}_temp", 0) > 60
                        else "Normal" if d.get(f"inv{i}_temp", 0) > 0
                        else "—"),
        } for i in range(1, 4)]
        st.dataframe(pd.DataFrame(inv_rows), use_container_width=True, hide_index=True)

        if live_data and live_data.get("phases"):
            st.markdown("**Grid Phases**")
            ph_rows = [{"Phase": ph.upper(),
                        "Voltage (V)": round(live_data["phases"][ph]["v"],2),
                        "Current (A)": round(live_data["phases"][ph]["a"],3),
                        "Power (kW)":  round(live_data["phases"][ph]["v"]*live_data["phases"][ph]["a"]/1000,3)}
                       for ph in ["a","b","c"]]
            st.dataframe(pd.DataFrame(ph_rows), use_container_width=True, hide_index=True)

    # ── SECTION 9 — 7-day trends & PV string diagnostics ──────────────────────
    st.markdown("---")
    st.header("📈 Trends & Panel Diagnostics — Last 7 Days")

    hist = load_history(days=7)
    if not hist.empty:
        hist["ts"] = pd.to_datetime(hist["ts"]).dt.tz_convert("Africa/Johannesburg")
        hist = hist.sort_values("ts").reset_index(drop=True)

        # ── 9A. Power Flow — its own full-width chart, one y-axis (kW) ────────
        st.subheader("⚡ Power Flow")
        fig_power = go.Figure()
        fig_power.add_scatter(
            x=hist["ts"], y=hist["pv_kw"], name="Solar",
            line=dict(color="#EF9F27", width=2),
            fill="tozeroy", fillcolor="rgba(239,159,39,0.12)",
        )
        fig_power.add_scatter(
            x=hist["ts"], y=hist["load_kw"], name="Load",
            line=dict(color="#E24B4A", width=2, dash="dot"),
        )
        fig_power.add_scatter(
            x=hist["ts"], y=hist["grid_kw"].apply(lambda v: max(0, -v)),
            name="Grid import", line=dict(color="#7F77DD", width=2),
        )
        fig_power.update_layout(
            height=360,
            yaxis_title="kW",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
        )
        fig_power.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        fig_power.update_yaxes(gridcolor="rgba(255,255,255,0.06)", rangemode="tozero")
        st.plotly_chart(fig_power, use_container_width=True)

        # ── 9B. Battery SoC — separate chart, own 0-100% axis ──────────────────
        st.subheader("🔋 Battery state of charge")
        fig_bsoc = go.Figure()
        fig_bsoc.add_scatter(
            x=hist["ts"], y=hist["battery_soc"], name="Battery SoC",
            fill="tozeroy", line=dict(color="#1D9E75", width=2),
            fillcolor="rgba(29,158,117,0.12)",
        )
        fig_bsoc.add_hline(
            y=20, line_dash="dash", line_color="#E24B4A",
            annotation_text="Low SoC threshold (20%)", annotation_font_color="#E24B4A",
        )
        fig_bsoc.update_layout(
            height=260, yaxis=dict(title="SoC (%)", range=[0, 105]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
        )
        fig_bsoc.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        fig_bsoc.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig_bsoc, use_container_width=True)

        # ── 9C. Weather + Solar Production — stacked, synced x-axis ───────────
        # Three genuinely different units (kW, W/m², %) so each gets its own
        # y-axis - but stacking them with a SHARED x-axis (rather than side
        # by side) lets you trace a vertical line straight down through all
        # three and see exactly how a cloud cover dip lines up with an
        # irradiance drop and the resulting dip in actual solar output.
        st.subheader("🌤️ Weather vs. solar production")
        st.caption(
            "Stacked with a shared time axis so you can trace cause and effect "
            "vertically: a cloud cover spike should line up with an irradiance "
            "dip, which should line up with a dip in solar production. Each "
            "panel keeps its own scale since the three are different units."
        )

        fig_weather_combo = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.34, 0.33, 0.33],
            subplot_titles=("Solar production (kW)", "Irradiance (W/m²)", "Cloud cover (%)"),
            vertical_spacing=0.06,
        )
        fig_weather_combo.add_trace(
            go.Scatter(
                x=hist["ts"], y=hist["pv_kw"], name="Solar production",
                line=dict(color="#EF9F27", width=2),
                fill="tozeroy", fillcolor="rgba(239,159,39,0.15)",
            ), row=1, col=1
        )
        fig_weather_combo.add_trace(
            go.Scatter(
                x=hist["ts"], y=hist["irradiance"], name="Irradiance",
                line=dict(color="#D85A30", width=2),
                fill="tozeroy", fillcolor="rgba(216,90,48,0.15)",
            ), row=2, col=1
        )
        fig_weather_combo.add_trace(
            go.Scatter(
                x=hist["ts"], y=hist["cloud_cover"], name="Cloud cover",
                line=dict(color="#7F77DD", width=2),
                fill="tozeroy", fillcolor="rgba(127,119,221,0.15)",
            ), row=3, col=1
        )
        fig_weather_combo.update_layout(
            height=620, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=10, l=10, r=10),
        )
        fig_weather_combo.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        fig_weather_combo.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        fig_weather_combo.update_yaxes(rangemode="tozero", row=1, col=1)
        fig_weather_combo.update_yaxes(rangemode="tozero", row=2, col=1)
        fig_weather_combo.update_yaxes(range=[0, 100], row=3, col=1)
        st.plotly_chart(fig_weather_combo, use_container_width=True)

        # ── 9D. Per-inverter PV power comparison ───────────────────────────────
        st.subheader("☀️ Per-inverter solar output")
        st.caption(
            "Each inverter's own reported PV power output, plus the SITE TOTAL "
            "(pv_kw from the live energy flow reading, measured independently "
            "of the 3 inverters' own self-reports) and the sum of the 3 "
            "inverters' outputs. If Sum of inverters tracks the Site total "
            "closely, the inverters' own readings are trustworthy. A gap "
            "between them points to a whole inverter dropping out or "
            "misreporting — distinct from the per-string mismatch covered "
            "below, which only catches faults INSIDE one inverter's own data."
        )
        fig_inv_compare = go.Figure()
        inv_colors = {"inv1_pv_kw": "#EF9F27", "inv2_pv_kw": "#7F77DD", "inv3_pv_kw": "#1D9E75"}
        inv_cols_present = [c for c in ["inv1_pv_kw", "inv2_pv_kw", "inv3_pv_kw"] if c in hist.columns]

        for i, col in enumerate(["inv1_pv_kw", "inv2_pv_kw", "inv3_pv_kw"], start=1):
            if col in hist.columns:
                fig_inv_compare.add_scatter(
                    x=hist["ts"], y=hist[col], name=f"Inverter {i}",
                    line=dict(color=inv_colors[col], width=2),
                )

        if inv_cols_present:
            inv_sum = hist[inv_cols_present].sum(axis=1)
            fig_inv_compare.add_scatter(
                x=hist["ts"], y=inv_sum, name="Sum of inverters",
                line=dict(color="#D85A30", width=2, dash="dot"),
            )

        if "pv_kw" in hist.columns:
            fig_inv_compare.add_scatter(
                x=hist["ts"], y=hist["pv_kw"], name="Site total (independent reading)",
                line=dict(color="#FFFFFF", width=2.5),
            )

        fig_inv_compare.update_layout(
            height=340, yaxis=dict(title="kW", rangemode="tozero"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
        )
        fig_inv_compare.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        fig_inv_compare.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig_inv_compare, use_container_width=True)

        # ── Quantify the sum-vs-total gap over the daylight window ────────────
        if inv_cols_present and "pv_kw" in hist.columns:
            daylight_compare = hist[hist["irradiance"] > 50] if "irradiance" in hist.columns else hist[hist["pv_kw"] > 0.5]
            if len(daylight_compare) > 0:
                avg_sum = daylight_compare[inv_cols_present].sum(axis=1).mean()
                avg_total = daylight_compare["pv_kw"].mean()
                if avg_total > 0.05:
                    site_gap_pct = 100 * (avg_total - avg_sum) / avg_total
                else:
                    site_gap_pct = 0.0

                gap_col1, gap_col2, gap_col3 = st.columns(3)
                gap_col1.metric("Avg sum of inverters", f"{avg_sum:.2f} kW")
                gap_col2.metric("Avg site total", f"{avg_total:.2f} kW")
                gap_col3.metric(
                    "Gap (site total vs. sum)",
                    f"{site_gap_pct:+.1f}%",
                    delta_color="off",
                )
                if abs(site_gap_pct) > 10:
                    st.warning(
                        f"The 3 inverters' own readings sum to {site_gap_pct:+.1f}% "
                        f"away from the independently-measured site total. This "
                        f"suggests at least one inverter is under- or "
                        f"over-reporting its own output, separate from any "
                        f"single bad string inside it — check the per-inverter "
                        f"diagnostics table below first to rule out a string-level "
                        f"cause, then consider whether the energyFlow reading "
                        f"itself or an inverter's communication link needs checking."
                    )

        # ── 9E. PV string fault detection ───────────────────────────────────────
        st.subheader("🔍 PV string diagnostics — fault & mismatch detection")
        st.caption(
            "Compares the SUM of each inverter's 4 individual string readings "
            "(voltage x current) against that inverter's OWN reported total PV "
            "power. A meaningful gap between the two points to a specific "
            "string with a problem - the per-string table below shows exactly "
            "which one. A string reading near-zero voltage AND near-zero "
            "current while its siblings are healthy usually means a "
            "disconnected, faulty, or severely shaded/soiled panel string. "
            "If the inverter's own total is ALSO near-zero, that's more "
            "consistent with no strings being connected to that inverter at "
            "all (or it being genuinely idle at night) rather than a fault."
        )

        # Restrict the fault analysis to meaningful daylight hours (irradiance
        # above a low threshold) so we don't flag every inverter as "faulty"
        # simply because it's night and everything reads near zero.
        daylight = hist[hist["irradiance"] > 50] if "irradiance" in hist.columns else hist[hist["pv_kw"] > 0.5]

        if len(daylight) > 0:
            diag_rows = []
            for inv in [1, 2, 3]:
                pv_kw_col = f"inv{inv}_pv_kw"
                if pv_kw_col not in daylight.columns:
                    continue
                string_sum_kw = sum(
                    daylight[f"inv{inv}_pv{s}_v"] * daylight[f"inv{inv}_pv{s}_a"] / 1000
                    for s in range(1, 5)
                    if f"inv{inv}_pv{s}_v" in daylight.columns
                )
                reported_kw = daylight[pv_kw_col]
                avg_string_sum = string_sum_kw.mean()
                avg_reported = reported_kw.mean()
                if avg_reported > 0.05:
                    discrepancy_pct = 100 * (avg_reported - avg_string_sum) / avg_reported
                else:
                    discrepancy_pct = 0.0

                if avg_reported < 0.1 and avg_string_sum < 0.1:
                    status = "Idle / no strings connected"
                elif abs(discrepancy_pct) > 15:
                    status = "String data mismatch — check wiring/comms"
                elif discrepancy_pct > 5:
                    status = "Minor mismatch — monitor"
                else:
                    status = "OK"

                diag_rows.append({
                    "Inverter": f"Inverter {inv}",
                    "Avg string-sum (kW)": round(avg_string_sum, 2),
                    "Avg reported (kW)": round(avg_reported, 2),
                    "Discrepancy": f"{discrepancy_pct:+.1f}%",
                    "Status": status,
                })

            diag_df = pd.DataFrame(diag_rows)

            def _status_color(val):
                if "OK" in str(val):
                    return "color: #1D9E75"
                elif "Idle" in str(val):
                    return "color: #888888"
                elif "mismatch" in str(val).lower():
                    return "color: #E24B4A"
                return ""

            st.dataframe(
                diag_df.style.map(_status_color, subset=["Status"]),
                use_container_width=True, hide_index=True,
            )

            # ── Per-string breakdown — pinpoint exactly which string is bad ───
            st.markdown("**Per-string average power (daylight hours only)**")
            string_rows = []
            for inv in [1, 2, 3]:
                for s in range(1, 5):
                    v_col, a_col = f"inv{inv}_pv{s}_v", f"inv{inv}_pv{s}_a"
                    if v_col not in daylight.columns:
                        continue
                    p = daylight[v_col] * daylight[a_col] / 1000
                    avg_p = p.mean()
                    avg_v = daylight[v_col].mean()
                    string_rows.append({
                        "String": f"Inv{inv} PV{s}",
                        "Avg voltage (V)": round(avg_v, 1),
                        "Avg power (kW)": round(avg_p, 3),
                    })
            string_df = pd.DataFrame(string_rows)

            if len(string_df) > 0:
                # Flag strings producing well below their siblings' average
                # within the SAME inverter (a fairer comparison than the
                # site-wide average, since inverters can legitimately have
                # different numbers of panels per string).
                string_df["_inv"] = string_df["String"].str.extract(r"Inv(\d)")[0]
                inv_avg = string_df.groupby("_inv")["Avg power (kW)"].transform("mean")
                string_df["% of inverter avg"] = (
                    100 * string_df["Avg power (kW)"] / inv_avg.replace(0, np.nan)
                ).fillna(0).round(0)
                string_df = string_df.drop(columns=["_inv"])

                def _string_pct_color(val):
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        return ""
                    if v < 50:
                        return "color: #E24B4A"
                    elif v < 80:
                        return "color: #EF9F27"
                    return "color: #1D9E75"

                fig_strings = go.Figure()
                bar_colors = [
                    "#E24B4A" if p < 50 else "#EF9F27" if p < 80 else "#1D9E75"
                    for p in string_df["% of inverter avg"]
                ]
                fig_strings.add_bar(
                    x=string_df["String"], y=string_df["Avg power (kW)"],
                    marker_color=bar_colors,
                    text=[f"{p:.0f}%" for p in string_df["% of inverter avg"]],
                    textposition="outside",
                )
                fig_strings.update_layout(
                    height=320,
                    yaxis_title="Avg power (kW)",
                    xaxis_title="",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                )
                fig_strings.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
                fig_strings.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
                st.plotly_chart(fig_strings, use_container_width=True)
                st.caption(
                    "Bar labels show each string's output as a % of its own "
                    "inverter's average string output. Below 50% (red) strongly "
                    "suggests a fault, disconnection, or heavy "
                    "soiling/shading specific to that string."
                )

                st.dataframe(
                    string_df.style.format({"% of inverter avg": "{:.0f}%"})
                                    .map(_string_pct_color, subset=["% of inverter avg"]),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info(
                "No daylight-hours data in the last 7 days to run string "
                "diagnostics on yet."
            )

    else:
        st.info(
            "No 7-day trend data yet — history builds up as live_logger.py "
            "runs on the Pi every 5 minutes. Check back in a few hours."
        )

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if auto_refresh:
        import time as _time
        _time.sleep(30)
        st.cache_data.clear()
        st.rerun()


# ── Raw data (optional) ───────────────────────────────────────────────────────
if show_raw:
    st.divider()
    st.subheader("Raw Hourly Data")
    st.dataframe(hourly, use_container_width=True)