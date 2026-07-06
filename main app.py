import os
import glob

import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="NSE Monthly Valuation Table", layout="wide", page_icon="📊")
st.title("📊 NSE Monthly Valuation Table")
st.markdown(
    "Month-by-month valuation across **Nifty 50, Midcap 150, Smallcap 250, and Total Market**."
)

# Only these indices are shown (the stray "NIFTY SMALLCAP 100" file is ignored).
ALLOWED_INDICES = {
    "NIFTY 50",
    "NIFTY MIDCAP 150",
    "NIFTY SMALLCAP 250",
    "NIFTY TOTAL MARKET",
}
# Fixed left-to-right column order (large -> broad)
INDEX_ORDER = ["NIFTY 50", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250", "NIFTY TOTAL MARKET"]


# ==========================================================================
#  DATA LOADING
#  Auto-discovers every PE/PB/DIV CSV in the folder and MERGES them by
#  deduplicating on (Index Name, Date) keeping the LAST occurrence. This
#  absorbs every overlap in your folder automatically:
#    - 01012025to01012026  vs  01052025to23042026  vs  01012026to02072026
#    - "... - Copy" duplicate files
#  Overlapping ("common") dates collapse to one row; only the further dates
#  from each additional file actually extend the history.
# ==========================================================================
@st.cache_data
def parse_single_csv(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return pd.DataFrame()

    df.columns = df.columns.str.strip().str.replace('"', "", regex=False)

    # Must be a valuation file (skip any PR / OHLC price files).
    if not {"P/E", "P/B"}.issubset(df.columns):
        return pd.DataFrame()

    if "IndexName" in df.columns:
        df["Index Name"] = df["IndexName"].astype(str).str.strip().str.replace('"', "", regex=False)
    elif "Index Name" in df.columns:
        df["Index Name"] = df["Index Name"].astype(str).str.strip().str.replace('"', "", regex=False)
    else:
        base = os.path.basename(filepath)
        name = base.split("_Historical")[0] if "_Historical" in base else base.split(".")[0]
        df["Index Name"] = name.replace("_", " ").strip()

    if "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = df["Date"].astype(str).str.strip().str.replace('"', "", regex=False)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")

    for col in ["P/E", "P/B", "Div Yield %"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().str.replace('"', "", regex=False),
                errors="coerce",
            )
    if "Div Yield %" not in df.columns:
        df["Div Yield %"] = np.nan

    return df[["Date", "Index Name", "P/E", "P/B", "Div Yield %"]].dropna(
        subset=["Date", "P/E", "P/B"]
    )


@st.cache_data
def load_and_merge_data() -> pd.DataFrame:
    patterns = ["*Historical_PE_PB_DIV_Data*.csv", "NIFTY*_PE_PB_DIV*.csv"]
    files = sorted(set(sum((glob.glob(p) for p in patterns), [])))
    if not files:
        return pd.DataFrame()

    st.sidebar.markdown("**📂 Files merged**")
    frames = []
    for f in files:
        part = parse_single_csv(f)
        if part.empty:
            st.sidebar.caption(f"⚠️ {os.path.basename(f)} — skipped")
            continue
        rng = f"{part['Date'].min().date()} → {part['Date'].max().date()}"
        st.sidebar.caption(f"✅ {os.path.basename(f)}  ({len(part)}r, {rng})")
        frames.append(part)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(by=["Index Name", "Date"], ascending=[True, True], inplace=True)
    combined.drop_duplicates(subset=["Index Name", "Date"], keep="last", inplace=True)
    combined = combined[combined["Index Name"].isin(ALLOWED_INDICES)]
    combined.reset_index(drop=True, inplace=True)
    return combined


# --------------------------------------------------------------------------
df = load_and_merge_data()

if df.empty:
    st.warning(
        "No valuation CSVs found. Place your `*_Historical_PE_PB_DIV_Data_*.csv` files "
        "in the same folder as this script."
    )
    st.stop()

st.sidebar.markdown(
    f"**Coverage:** {df['Date'].min().date()} → {df['Date'].max().date()}  "
    f"({df['Index Name'].nunique()} indices)"
)

# --- CONFIG ---
st.sidebar.header("⚙️ Configuration")
metric_choice = st.sidebar.radio("Metric:", ["P/E Ratio", "P/B Ratio", "Div Yield %"], index=0)
agg_choice = st.sidebar.radio(
    "Monthly value:", ["Month-end (last trading day)", "Monthly average"], index=0
)
col_map = {"P/E Ratio": "P/E", "P/B Ratio": "P/B", "Div Yield %": "Div Yield %"}
selected_col = col_map[metric_choice]

# --- BUILD YEAR/MONTH TABLE ---
d = df.copy()
d["Period"] = d["Date"].dt.to_period("M")

if agg_choice.startswith("Month-end"):
    monthly = (
        d.sort_values("Date")
        .groupby(["Period", "Index Name"])[selected_col]
        .last()
        .reset_index()
    )
else:
    monthly = (
        d.groupby(["Period", "Index Name"])[selected_col].mean().reset_index()
    )

pivot = monthly.pivot(index="Period", columns="Index Name", values=selected_col)
pivot = pivot.reindex(columns=[c for c in INDEX_ORDER if c in pivot.columns])
pivot = pivot.sort_index(ascending=False)  # newest month on top

# Split Period into Year / Month display columns
table = pivot.copy()
table.insert(0, "Month", table.index.strftime("%b"))
table.insert(0, "Year", table.index.year)
table = table.reset_index(drop=True)

# --- CURRENT-LEVEL SNAPSHOT: average / median + premium/discount -----------
# Computed on the full DAILY history (Jan 2024 -> latest) for the selected
# metric. "% vs Avg" / "% vs Median" show where the current reading sits
# relative to its own history (positive = richer than usual).
snap_rows = []
for idx in INDEX_ORDER:
    s = df.loc[df["Index Name"] == idx, selected_col].dropna()
    if s.empty:
        continue
    cur = s.iloc[-1]            # df is date-sorted within index, so this is latest
    avg = s.mean()
    med = s.median()
    snap_rows.append(
        {
            "Index": idx,
            "Current": cur,
            "Average": avg,
            "Median": med,
            "% vs Avg": (cur / avg - 1) * 100,
            "% vs Median": (cur / med - 1) * 100,
        }
    )
snapshot = pd.DataFrame(snap_rows)

# ONE TAB — snapshot on top, month-wise table below
tab, = st.tabs([f"📅 {metric_choice} — Levels & Month-wise Table"])
with tab:
    st.subheader(f"📌 Current {metric_choice}: Average & Median")
    st.caption(
        f"Full-history basis ({df['Date'].min().strftime('%b %Y')} → "
        f"{df['Date'].max().strftime('%b %Y')}). Current = latest daily value; "
        "'% vs' shows premium (+) or discount (–) to its own average / median."
    )
    snap_styled = (
        snapshot.style
        .format(
            {
                "Current": "{:.2f}", "Average": "{:.2f}", "Median": "{:.2f}",
                "% vs Avg": "{:+.1f}%", "% vs Median": "{:+.1f}%",
            }
        )
        .background_gradient(subset=["% vs Avg", "% vs Median"], cmap="RdYlGn_r")
    )
    st.dataframe(snap_styled, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader(f"📅 Month-wise {metric_choice}")
    basis = "month-end" if agg_choice.startswith("Month-end") else "monthly-average"
    st.caption(
        f"Each row is one calendar month ({basis} {metric_choice}). "
        f"Newest month first. {len(table)} months, "
        f"{df['Date'].min().strftime('%b %Y')} → {df['Date'].max().strftime('%b %Y')}."
    )

    idx_cols = [c for c in INDEX_ORDER if c in table.columns]
    styled = (
        table.style
        .format({c: "{:.2f}" for c in idx_cols})
        .background_gradient(subset=idx_cols, cmap="RdYlGn_r")
        .set_properties(subset=["Year", "Month"], **{"font-weight": "600"})
    )
    st.dataframe(styled, use_container_width=True, height=640, hide_index=True)

    st.download_button(
        f"⬇️ Download month-wise {selected_col} table (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"nifty_monthly_{selected_col.replace('/', '')}.csv",
        mime="text/csv",
    )
