import os
import glob

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="NSE Valuation Dashboard", layout="wide", page_icon="📊")
st.title("📊 NSE Valuation Dashboard: P/E & P/B")
st.markdown("Comparative analysis of **Nifty 50, Midcap 150, Smallcap 250, and Total Market**.")

# Only these indices are shown (ignore anything else found in the folder)
ALLOWED_INDICES = {
    "NIFTY 50",
    "NIFTY MIDCAP 150",
    "NIFTY SMALLCAP 250",
    "NIFTY TOTAL MARKET",
}


# ==========================================================================
#  DATA LOADING
#  Auto-discovers every PE/PB/DIV CSV in the folder, parses each one, then
#  MERGES them incrementally: rows are deduplicated on (Index Name, Date)
#  keeping the LAST occurrence. Because newer files are read last, they win
#  on any overlapping ("common") months, so the combined dataset only ever
#  grows by the genuinely-newer months from each additional file.
# ==========================================================================
@st.cache_data
def parse_single_csv(filepath: str) -> pd.DataFrame:
    """Parse one PE/PB/DIV export into a clean long-format frame."""
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return pd.DataFrame()

    # Clean column names (strip stray quotes / whitespace)
    df.columns = df.columns.str.strip().str.replace('"', '', regex=False)

    # This dashboard needs valuation ratios. PR / OHLC files (Open/High/Low/
    # Close) are silently skipped so dropping a price file in the folder by
    # accident can't corrupt the valuation data.
    if not {"P/E", "P/B"}.issubset(df.columns):
        return pd.DataFrame()

    # Standardise the index-name column
    if "IndexName" in df.columns:
        df["Index Name"] = df["IndexName"].astype(str).str.strip().str.replace('"', "", regex=False)
    elif "Index Name" in df.columns:
        df["Index Name"] = df["Index Name"].astype(str).str.strip().str.replace('"', "", regex=False)
    else:
        # Fall back to the filename, e.g. NIFTY_50_Historical... -> NIFTY 50
        base = os.path.basename(filepath)
        name = base.split("_Historical")[0] if "_Historical" in base else base.split(".")[0]
        df["Index Name"] = name.replace("_", " ").strip()

    if "Date" not in df.columns:
        return pd.DataFrame()

    # Robust date parsing (handles "01 Jul 2026", "01-Jul-2026", etc.)
    df["Date"] = df["Date"].astype(str).str.strip().str.replace('"', "", regex=False)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")

    # Numeric cleaning
    for col in ["P/E", "P/B", "Div Yield %"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().str.replace('"', "", regex=False),
                errors="coerce",
            )
    if "Div Yield %" not in df.columns:
        df["Div Yield %"] = np.nan

    out = df[["Date", "Index Name", "P/E", "P/B", "Div Yield %"]].dropna(
        subset=["Date", "P/E", "P/B"]
    )
    return out


@st.cache_data
def load_and_merge_data() -> pd.DataFrame:
    # Discover every valuation export in the working directory
    patterns = [
        "*Historical_PE_PB_DIV_Data*.csv",
        "NIFTY*_PE_PB_DIV*.csv",
    ]
    files = sorted(set(sum((glob.glob(p) for p in patterns), [])))

    if not files:
        return pd.DataFrame()

    st.sidebar.markdown("**📂 Files merged**")
    frames = []
    for f in files:
        part = parse_single_csv(f)
        if part.empty:
            st.sidebar.caption(f"⚠️ {os.path.basename(f)} — skipped (no P/E–P/B or unreadable)")
            continue
        rng = f"{part['Date'].min().date()} → {part['Date'].max().date()}"
        st.sidebar.caption(f"✅ {os.path.basename(f)}  ({len(part)} rows, {rng})")
        frames.append(part)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # --- CRITICAL MERGE STEP -------------------------------------------------
    # Sort so that, on any overlapping (Index, Date), the LAST row survives.
    # keep="last" => common months are collapsed to a single point and only
    # the further months from newer files actually extend the series.
    combined.sort_values(by=["Index Name", "Date"], ascending=[True, True], inplace=True)
    combined.drop_duplicates(subset=["Index Name", "Date"], keep="last", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    # ------------------------------------------------------------------------

    # Keep only the indices we care about
    combined = combined[combined["Index Name"].isin(ALLOWED_INDICES)]

    return combined


# Load Data
df = load_and_merge_data()

if not df.empty:
    st.sidebar.markdown(
        f"**Coverage:** {df['Date'].min().date()} → {df['Date'].max().date()}  "
        f"({df['Index Name'].nunique()} indices, {len(df):,} rows)"
    )

    # --- 2. METRIC SELECTION ---
    st.sidebar.header("⚙️ Configuration")
    metric_choice = st.sidebar.radio("Select Metric:", ["P/E Ratio", "P/B Ratio", "Div Yield %"])
    col_map = {"P/E Ratio": "P/E", "P/B Ratio": "P/B", "Div Yield %": "Div Yield %"}
    selected_col = col_map[metric_choice]

    # --- 3. SUMMARY METRICS ---
    st.subheader(f"📋 Valuation Summary: {metric_choice}")

    # Because the frame is globally sorted by Date, .last() correctly grabs the
    # most recent point (now extended by whatever the newest file added).
    summary = (
        df.groupby("Index Name")[selected_col]
        .agg(["last", "mean", "min", "max", "std"])
        .reset_index()
    )
    summary.columns = ["Index", "Current", "Average", "Min", "Max", "Volatility"]

    def get_status(row):
        curr, avg = row["Current"], row["Average"]
        if curr > avg * 1.05:
            return "Expensive 🔴"
        elif curr < avg * 0.95:
            return "Cheap 🟢"
        return "Fair 🟡"

    summary["Status"] = summary.apply(get_status, axis=1)

    st.dataframe(
        summary.style.format(
            {
                "Current": "{:.2f}", "Average": "{:.2f}",
                "Min": "{:.2f}", "Max": "{:.2f}", "Volatility": "{:.2f}",
            }
        ).background_gradient(subset=["Current"], cmap="Reds"),
        use_container_width=True,
    )

    # --- 4. VISUALIZATIONS ---
    st.divider()
    tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "📊 Relative Value", "📉 Matrix"])

    with tab1:
        st.subheader("Historical Trend")
        fig = px.line(df, x="Date", y=selected_col, color="Index Name",
                      title=f"{metric_choice} Trend", height=500)
        fig.update_layout(hovermode="x unified", yaxis_title=metric_choice)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Premium / Discount vs Period Average")
        bar_data = summary.copy()
        bar_data["% Diff"] = ((bar_data["Current"] - bar_data["Average"]) / bar_data["Average"]) * 100
        fig_bar = px.bar(bar_data, x="Index", y="% Diff", color="% Diff",
                         color_continuous_scale="RdYlGn_r")
        fig_bar.add_hline(y=0, line_color="black")
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Valuation Matrix (Current P/E vs P/B)")
        latest_snapshot = df.groupby("Index Name").tail(1)
        fig_scatter = px.scatter(latest_snapshot, x="P/E", y="P/B", color="Index Name",
                                 size="P/E", text="Index Name", title="Risk vs Value Matrix")
        fig_scatter.add_vline(x=latest_snapshot["P/E"].mean(), line_dash="dash", line_color="grey")
        fig_scatter.add_hline(y=latest_snapshot["P/B"].mean(), line_dash="dash", line_color="grey")
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.warning(
        "No valuation CSVs found. Place your `*_Historical_PE_PB_DIV_Data_*.csv` files "
        "in the same folder as this script. (Price / `_PR_` OHLC files are ignored here — "
        "they have no P/E–P/B data.)"
    )
