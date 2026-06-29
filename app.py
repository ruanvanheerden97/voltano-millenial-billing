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

# ─── TOU CLASSIFICATION ───────────────────────────────────────────────────────

def get_season(dt):
    """Return 'high' (Jun-Aug) or 'low' (Sep-May) demand season."""
    return "high" if dt.month in [6, 7, 8] else "low"

def get_tou_slot(dt):
    """
    Classify a datetime into Peak (1.8.1), Standard (1.8.2), or Off-Peak (1.8.3).
    Returns the register string: '1.8.1', '1.8.2', or '1.8.3'
    Based on Eskom TOU tariff schedule.
    """
    season = get_season(dt)
    dow = dt.weekday()   # 0=Mon, 5=Sat, 6=Sun
    h = dt.hour          # hour of day (0-23), represents the hour STARTING at that time

    # ── SUNDAY (all seasons) ──────────────────────────────────────
    if dow == 6:
        if season == "low":
            if 18 <= h < 20:
                return "1.8.2"
            else:
                return "1.8.3"
        else:  # high
            if 17 <= h < 19:
                return "1.8.2"
            else:
                return "1.8.3"

    # ── SATURDAY ─────────────────────────────────────────────────
    elif dow == 5:
        if season == "low":
            if h < 7 or h >= 20:
                return "1.8.3"
            elif 7 <= h < 12 or 18 <= h < 20:
                return "1.8.2"
            else:  # 12:00-18:00
                return "1.8.3"
        else:  # high
            if h < 7 or h >= 19:
                return "1.8.3"
            elif 7 <= h < 12 or 17 <= h < 19:
                return "1.8.2"
            else:  # 12:00-17:00
                return "1.8.3"

    # ── WEEKDAY (Mon-Fri) ────────────────────────────────────────
    else:
        if season == "low":
            if h < 6 or h >= 22:
                return "1.8.3"
            elif h == 6 or (9 <= h < 18) or (21 <= h < 22):
                return "1.8.2"
            elif (7 <= h < 9) or (18 <= h < 21):
                return "1.8.1"
            else:
                return "1.8.2"
        else:  # high
            if h < 6 or h >= 22:
                return "1.8.3"
            elif (8 <= h < 17) or (20 <= h < 22):
                return "1.8.2"
            elif (6 <= h < 8) or (17 <= h < 20):
                return "1.8.1"
            else:
                return "1.8.2"


def assign_tou_vectorised(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Fully vectorised TOU slot and season assignment — no row-by-row apply().
    ~50× faster than apply(get_tou_slot) on large datasets.
    """
    dt  = hourly["datetime"]
    h   = dt.dt.hour
    dow = dt.dt.weekday          # 0=Mon … 6=Sun
    mon = dt.dt.month

    season = pd.Series(
        np.where(mon.isin([6, 7, 8]), "high", "low"),
        index=hourly.index
    )

    is_high    = season == "high"
    is_weekday = dow < 5
    is_sat     = dow == 5
    is_sun     = dow == 6

    # Start everything as Off-Peak then overwrite in priority order
    slot = pd.Series("1.8.3", index=hourly.index)

    # ── SUNDAY ────────────────────────────────────────────────────
    std_sun = is_sun & (
        (~is_high & h.between(18, 19)) |
        (is_high  & h.between(17, 18))
    )
    slot = slot.where(~std_sun, "1.8.2")

    # ── SATURDAY ──────────────────────────────────────────────────
    std_sat = is_sat & (
        (~is_high & (h.between(7, 11)  | h.between(18, 19))) |
        (is_high  & (h.between(7, 11)  | h.between(17, 18)))
    )
    slot = slot.where(~std_sat, "1.8.2")

    # ── WEEKDAY ───────────────────────────────────────────────────
    # Standard
    std_wd = is_weekday & (
        (~is_high & ((h == 6) | h.between(9, 17) | (h == 21))) |
        (is_high  & (h.between(8, 16) | h.between(20, 21)))
    )
    slot = slot.where(~std_wd, "1.8.2")

    # Peak (overwrites standard where applicable)
    peak_wd = is_weekday & (
        (~is_high & (h.between(7, 8)  | h.between(18, 20))) |
        (is_high  & (h.between(6, 7)  | h.between(17, 19)))
    )
    slot = slot.where(~peak_wd, "1.8.1")

    hourly = hourly.copy()
    hourly["tou_slot"] = slot
    hourly["season"]   = season
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
TARIFF = {
    "low":  {"1.8.1": 3.1682, "1.8.2": 2.5487, "1.8.3": 1.4826},
    "high": {"1.8.1": 5.9163, "1.8.2": 1.9068, "1.8.3": 1.1100},
}
SELL_RATE = 3.2795  # R/kWh — flat sell rate regardless of season/TOU

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

    # Vectorised tariff lookup
    _tmap = {(s, t): TARIFF[s][t]
             for s in ["low", "high"] for t in ["1.8.1", "1.8.2", "1.8.3"]}
    h["slot_tariff"] = pd.Series(
        [_tmap[(s, t)] for s, t in zip(h["season"], h["tou_slot"])],
        index=h.index
    )

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
    offpeak_tariff_h = h["season"].map({"low": TARIFF["low"]["1.8.3"], "high": TARIFF["high"]["1.8.3"]})
    std_tariff_h     = h["season"].map({"low": TARIFF["low"]["1.8.2"], "high": TARIFF["high"]["1.8.2"]})

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
    h["rev_solar_direct"]= h["solar_direct"] * SELL_RATE
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
        bc_cost_r       = bc_kwh * SELL_RATE

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

        # ── Vectorised tariff lookup for grid charge cost ─────────────────────
        # Build a tariff series from season + tou_slot without apply()
        tariff_map = {
            ("low",  "1.8.1"): TARIFF["low"]["1.8.1"],
            ("low",  "1.8.2"): TARIFF["low"]["1.8.2"],
            ("low",  "1.8.3"): TARIFF["low"]["1.8.3"],
            ("high", "1.8.1"): TARIFF["high"]["1.8.1"],
            ("high", "1.8.2"): TARIFF["high"]["1.8.2"],
            ("high", "1.8.3"): TARIFF["high"]["1.8.3"],
        }
        tariff_key           = list(zip(df["season"], df["tou_slot"]))
        df["tou_tariff"]     = pd.Series([tariff_map[k] for k in tariff_key], index=df.index)

        df["grid_charge_cost_r"]  = df["grid_to_battery"]  * df["tou_tariff"]
        df["solar_charge_cost_r"] = df["solar_to_battery"] * solar_cost
        df["charge_cost_r"]       = df["grid_charge_cost_r"] + df["solar_charge_cost_r"]

        df["discharge_value_r"] = df["batt_discharge_kwh"] * SELL_RATE
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
               help=f"At flat sell rate of R{SELL_RATE}/kWh")
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
    import json as _json_live
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    st.subheader("Live System Dashboard")
    st.caption("Real-time power flow from Sigenergy API, overlaid with weather data from Open-Meteo.")

    # ── Config ────────────────────────────────────────────────────────────────
    # Load credentials from st.secrets (Streamlit Cloud) or os.getenv (local .env)
    def _get_secret(key, default=None):
        try:
            return st.secrets[key]
        except Exception:
            return os.getenv(key, default)

    LIVE_API_BASE    = _get_secret("SIGEN_API_BASE", "https://openapi-eu.sigencloud.com")
    LIVE_SYSTEM_ID   = _get_secret("SIGEN_SYSTEM_ID", "HUCUD1764140703")
    LIVE_USERNAME    = _get_secret("SIGEN_USERNAME")
    LIVE_PASSWORD    = _get_secret("SIGEN_PASSWORD")
    LIVE_INVERTER_SN = _get_secret("SIGEN_INVERTER_SN", "110B1K500388")

    # Umhlanga coordinates for weather
    LAT, LON = -29.7215, 31.0498
    SA_TZ_OFFSET = _td(hours=2)

    # SQLite DB path — stores live readings for trend charts
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_readings.db")

    # ── SQLite setup ──────────────────────────────────────────────────────────
    def init_db():
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_readings (
                ts          TEXT PRIMARY KEY,
                pv_kw       REAL,
                grid_kw     REAL,
                load_kw     REAL,
                battery_kw  REAL,
                battery_soc REAL,
                cloud_cover REAL,
                irradiance  REAL,
                temperature REAL
            )
        """)
        conn.commit()
        conn.close()

    def save_reading(ts, pv, grid, load, batt, soc, cloud, irr, temp):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR REPLACE INTO live_readings
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (ts, pv, grid, load, batt, soc, cloud, irr, temp))
        conn.commit()
        conn.close()

    def load_history(days=7):
        conn = sqlite3.connect(DB_PATH)
        cutoff = (_dt.now(_tz.utc) - _td(days=days)).isoformat()
        df = pd.read_sql(
            "SELECT * FROM live_readings WHERE ts >= ? ORDER BY ts",
            conn, params=(cutoff,)
        )
        conn.close()
        return df

    init_db()

    # ── API helpers ───────────────────────────────────────────────────────────
    @st.cache_data(ttl=300)  # cache token for 5 minutes
    def get_live_token():
        r = _requests.post(
            f"{LIVE_API_BASE}/openapi/auth/login/password",
            json={"username": LIVE_USERNAME, "password": LIVE_PASSWORD},
            timeout=10
        )
        body = r.json()
        if body.get("code") != 0:
            return None
        data = _json_live.loads(body["data"]) if isinstance(body["data"], str) else body["data"]
        return data["accessToken"]

    def fetch_energy_flow(token):
        """GET /openapi/systems/{systemId}/energyFlow — live power flow."""
        headers = {"Authorization": f"Bearer {token}"}
        r = _requests.get(
            f"{LIVE_API_BASE}/openapi/systems/{LIVE_SYSTEM_ID}/energyFlow",
            headers=headers, timeout=10
        )
        body = r.json()
        if body.get("code") != 0:
            return None
        data = body.get("data", {})
        if isinstance(data, str):
            data = _json_live.loads(data)
        return data

    def fetch_system_summary(token):
        """GET /openapi/systems/{systemId}/summary — daily/monthly totals."""
        headers = {"Authorization": f"Bearer {token}"}
        r = _requests.get(
            f"{LIVE_API_BASE}/openapi/systems/{LIVE_SYSTEM_ID}/summary",
            headers=headers, timeout=10
        )
        body = r.json()
        if body.get("code") != 0:
            return None
        data = body.get("data", {})
        if isinstance(data, str):
            data = _json_live.loads(data)
        return data

    def fetch_device_realtime(token, serial_number):
        """GET /openapi/systems/{systemId}/devices/{serialNumber}/realtimeInfo"""
        headers = {"Authorization": f"Bearer {token}"}
        r = _requests.get(
            f"{LIVE_API_BASE}/openapi/systems/{LIVE_SYSTEM_ID}/devices/{serial_number}/realtimeInfo",
            headers=headers, timeout=10
        )
        body = r.json()
        if body.get("code") != 0:
            return None
        data = body.get("data", {})
        if isinstance(data, str):
            data = _json_live.loads(data)
        return data.get("realTimeInfo", data)

    def fetch_weather():
        """Fetch current weather and irradiance from Open-Meteo (free, no key)."""
        try:
            r = _requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":  LAT,
                    "longitude": LON,
                    "current":   "temperature_2m,cloud_cover,wind_speed_10m,precipitation",
                    "hourly":    "shortwave_radiation,cloud_cover,temperature_2m",
                    "forecast_days": 1,
                    "timezone":  "Africa/Johannesburg",
                },
                timeout=10
            )
            d = r.json()
            curr = d.get("current", {})
            # Get current hour's irradiance
            now_hour = _dt.now().strftime("%Y-%m-%dT%H:00")
            hourly_times = d.get("hourly", {}).get("time", [])
            irr = 0.0
            if now_hour in hourly_times:
                idx = hourly_times.index(now_hour)
                irr = d["hourly"]["shortwave_radiation"][idx] or 0.0
            return {
                "temperature": curr.get("temperature_2m", 0),
                "cloud_cover": curr.get("cloud_cover", 0),
                "wind_speed":  curr.get("wind_speed_10m", 0),
                "precipitation": curr.get("precipitation", 0),
                "irradiance":  irr,
            }
        except Exception:
            return {"temperature": 0, "cloud_cover": 0, "wind_speed": 0,
                    "precipitation": 0, "irradiance": 0}

    # ── Refresh controls ──────────────────────────────────────────────────────
    col_refresh, col_auto = st.columns([1, 2])
    with col_refresh:
        manual_refresh = st.button("Refresh now")
    with col_auto:
        auto_refresh = st.toggle("Auto-refresh every 30s", value=True)

    if auto_refresh:
        st.caption("Auto-refreshing every 30 seconds...")

    # ── Fetch live data ───────────────────────────────────────────────────────
    live_error = None
    flow = None
    summary = None
    device_rt = None
    weather = {}

    try:
        token = get_live_token()
        if token:
            flow      = fetch_energy_flow(token)
            summary   = fetch_system_summary(token)
            device_rt = fetch_device_realtime(token, LIVE_INVERTER_SN)
        else:
            live_error = "Authentication failed — check SIGEN credentials in .env"
    except Exception as e:
        live_error = str(e)

    weather = fetch_weather()

    if live_error:
        st.error(f"API error: {live_error}")
    elif flow is None:
        st.warning("No live data returned from API.")
    else:
        # Extract values
        pv_kw   = float(flow.get("pvPower", 0) or 0)
        grid_kw = float(flow.get("gridPower", 0) or 0)  # +ve = export, -ve = import
        load_kw = float(flow.get("loadPower", 0) or 0)
        batt_kw = float(flow.get("batteryPower", 0) or 0)  # +ve = charging, -ve = discharging
        batt_soc= float(flow.get("batterySoc", 0) or 0)

        grid_import_kw = max(0, -grid_kw)
        grid_export_kw = max(0, grid_kw)
        batt_charge_kw = max(0, batt_kw)
        batt_disc_kw   = max(0, -batt_kw)

        now_ts = _dt.now(_tz.utc).isoformat()
        save_reading(now_ts, pv_kw, grid_kw, load_kw, batt_kw, batt_soc,
                     weather.get("cloud_cover", 0),
                     weather.get("irradiance", 0),
                     weather.get("temperature", 0))

        # ── SECTION 1 — Live KPIs ─────────────────────────────────────────────
        st.subheader("Live Power Flow")
        now_str = _dt.now(_tz(_td(hours=2))).strftime("%d %b %Y %H:%M:%S SAST")
        st.caption(f"Last updated: {now_str}")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Solar", f"{pv_kw:.1f} kW",
                  help="Current PV generation")
        k2.metric("Load", f"{load_kw:.1f} kW",
                  help="Current site load")
        k3.metric("Grid",
                  f"{grid_import_kw:.1f} kW import" if grid_import_kw > 0 else f"{grid_export_kw:.1f} kW export",
                  delta=f"{'Importing' if grid_import_kw > 0 else 'Exporting'}",
                  delta_color="inverse" if grid_import_kw > 0 else "normal")
        k4.metric("Battery",
                  f"{batt_charge_kw:.1f} kW" if batt_charge_kw > 0 else f"{batt_disc_kw:.1f} kW",
                  delta="Charging" if batt_charge_kw > 0 else "Discharging" if batt_disc_kw > 0 else "Idle")
        k5.metric("Battery SoC", f"{batt_soc:.0f}%",
                  help="Battery state of charge")

        # Battery SoC gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=batt_soc,
            title={"text": "Battery State of Charge (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#1D9E75"},
                "steps": [
                    {"range": [0, 20],  "color": "#E24B4A"},
                    {"range": [20, 50], "color": "#EF9F27"},
                    {"range": [50, 100],"color": "#1D9E75"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": batt_soc,
                }
            }
        ))
        fig_gauge.update_layout(height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Inverter realtime details ──────────────────────────────────────────
        if device_rt:
            st.divider()
            st.subheader("Inverter Details")
            inv_c1, inv_c2, inv_c3, inv_c4, inv_c5, inv_c6 = st.columns(6)
            inv_c1.metric("PV Power",       f"{float(device_rt.get('pvTotalPower', 0) or 0):.2f} kW")
            inv_c2.metric("Inverter Temp",  f"{float(device_rt.get('internalTemperature', 0) or 0):.1f} degC")
            inv_c3.metric("Battery Power",  f"{float(device_rt.get('batPower', 0) or 0):.2f} kW",
                          help="Positive = discharging, Negative = charging")
            inv_c4.metric("Daily PV",       f"{float(device_rt.get('pvEnergyDaily', 0) or 0):.2f} kWh")
            inv_c5.metric("Lifetime PV",    f"{float(device_rt.get('pvEnergyTotal', 0) or 0):.1f} kWh")
            inv_c6.metric("Batt Discharged Today", f"{float(device_rt.get('esDischargingDay', 0) or 0):.2f} kWh")

            # PV Strings — show individual string voltages and currents
            st.markdown("**PV String Details**")
            strings_data = []
            for i in range(1, 5):
                v = float(device_rt.get(f"pv{i}Voltage", 0) or 0)
                c = float(device_rt.get(f"pv{i}Current", 0) or 0)
                p = round(v * c / 1000, 3)  # kW
                if v != 0 or c != 0:
                    strings_data.append({
                        "String": f"PV String {i}",
                        "Voltage (V)": round(v, 2),
                        "Current (A)": round(c, 3),
                        "Power (kW)":  p,
                    })
            if strings_data:
                st.dataframe(pd.DataFrame(strings_data), use_container_width=True, hide_index=True)

            # Phase voltages and currents
            st.markdown("**Grid Phase Details**")
            phase_data = []
            for ph in ["a", "b", "c"]:
                v = float(device_rt.get(f"{ph}PhaseVoltage", 0) or 0)
                c = float(device_rt.get(f"{ph}PhaseCurrent", 0) or 0)
                phase_data.append({
                    "Phase": ph.upper(),
                    "Voltage (V)": round(v, 2),
                    "Current (A)": round(c, 3),
                    "Power (kW)":  round(v * c / 1000, 3),
                })
            st.dataframe(pd.DataFrame(phase_data), use_container_width=True, hide_index=True)

            pf = device_rt.get("powerFactor")
            freq = device_rt.get("gridFrequency")
            if pf or freq:
                pf_col, freq_col = st.columns(2)
                if pf:
                    pf_col.metric("Power Factor", f"{float(pf):.3f}")
                if freq:
                    freq_col.metric("Grid Frequency", f"{float(freq):.2f} Hz")

        # Summary metrics from system summary
        if summary:
            st.divider()
            st.subheader("Today's Energy Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Solar Today",      f"{summary.get('dailyPowerGeneration', 0):.1f} kWh")
            s2.metric("Solar This Month",  f"{summary.get('monthlyPowerGeneration', 0):.1f} kWh")
            s3.metric("Solar This Year",   f"{summary.get('annualPowerGeneration', 0):.1f} kWh")
            s4.metric("Lifetime Solar",    f"{summary.get('lifetimePowerGeneration', 0):.1f} kWh")

            env1, env2, env3 = st.columns(3)
            env1.metric("CO2 Saved (lifetime)", f"{summary.get('lifetimeCo2', 0):.2f} tons")
            env2.metric("Coal Saved (lifetime)", f"{summary.get('lifetimeCoal', 0):.2f} tons")
            env3.metric("Trees Equivalent",      f"{summary.get('lifetimeTreeEquivalent', 0):.0f} trees")

    # ── SECTION 2 — Weather ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Current Weather — Umhlanga")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Temperature",   f"{weather.get('temperature', 0):.1f} degC")
    w2.metric("Cloud Cover",   f"{weather.get('cloud_cover', 0):.0f}%")
    w3.metric("Wind Speed",    f"{weather.get('wind_speed', 0):.1f} km/h")
    w4.metric("Solar Irradiance", f"{weather.get('irradiance', 0):.0f} W/m2")

    # ── SECTION 3 — 7-day history chart ──────────────────────────────────────
    st.divider()
    st.subheader("Last 7 Days — Solar Power vs Weather")

    hist = load_history(days=7)
    if not hist.empty:
        hist["ts"] = pd.to_datetime(hist["ts"]).dt.tz_convert("Africa/Johannesburg")

        fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hist.add_trace(
            go.Scatter(x=hist["ts"], y=hist["pv_kw"],
                       name="Solar (kW)", line=dict(color="#EF9F27", width=2)),
            secondary_y=False
        )
        fig_hist.add_trace(
            go.Scatter(x=hist["ts"], y=hist["load_kw"],
                       name="Load (kW)", line=dict(color="#E24B4A", dash="dot")),
            secondary_y=False
        )
        fig_hist.add_trace(
            go.Scatter(x=hist["ts"], y=hist["cloud_cover"],
                       name="Cloud Cover (%)", line=dict(color="#7F77DD", dash="dash"),
                       opacity=0.7),
            secondary_y=True
        )
        fig_hist.add_trace(
            go.Scatter(x=hist["ts"], y=hist["irradiance"],
                       name="Irradiance (W/m2)", line=dict(color="#3B8BD4", dash="dot"),
                       opacity=0.7),
            secondary_y=True
        )
        fig_hist.update_layout(
            title="Solar Power vs Cloud Cover & Irradiance",
            xaxis_title="Date / Time",
            legend_title="Metric",
            height=420,
        )
        fig_hist.update_yaxes(title_text="Power (kW)", secondary_y=False)
        fig_hist.update_yaxes(title_text="Cloud % / Irradiance W/m2", secondary_y=True)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Battery SoC trend
        fig_soc = go.Figure()
        fig_soc.add_scatter(x=hist["ts"], y=hist["battery_soc"],
                            name="Battery SoC (%)",
                            fill="tozeroy", line=dict(color="#1D9E75"),
                            fillcolor="rgba(29,158,117,0.2)")
        fig_soc.update_layout(
            title="Battery State of Charge — Last 7 Days",
            xaxis_title="Date / Time",
            yaxis=dict(title="SoC (%)", range=[0, 105]),
            height=300,
        )
        st.plotly_chart(fig_soc, use_container_width=True)
    else:
        st.info("No historical live data yet — readings are stored each time this tab is refreshed. Check back in a few minutes.")

    # Auto-refresh using Streamlit's rerun
    if auto_refresh:
        import time as _time
        _time.sleep(30)
        st.rerun()

# ── Raw data (optional) ───────────────────────────────────────────────────────
if show_raw:
    st.divider()
    st.subheader("Raw Hourly Data")
    st.dataframe(hourly, use_container_width=True)