import os
import glob

import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="NSE Valuation Tables", layout="wide", page_icon="📊")
st.title("📊 NSE Valuation Tables")
st.markdown(
    "Month-by-month valuation for the **broad-market** indices and the "
    "**sector** indices — same analysis, separate tabs."
)

# Broad-market indices (fixed large -> broad order).
BROAD_ORDER = ["NIFTY 50", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250", "NIFTY TOTAL MARKET"]
BROAD_INDICES = set(BROAD_ORDER)

# Broad-market analysis starts here (snapshot avg/median AND month-wise table).
# Rows before this date are dropped for the broad tab only; sectors are unaffected.
BROAD_START = pd.Timestamp("2024-03-01")

# Stray valuation files we never want to show (e.g. a duplicate SMALLCAP 100 export).
IGNORE_INDICES = {"NIFTY SMALLCAP 100"}

# Two readings for the same (index, date) count as a real "conflict" only if
# they differ by more than this. Below it, it's just rounding noise.
CONFLICT_TOL = 0.01


# ==========================================================================
#  DATA LOADING
#  Auto-discovers every PE/PB/DIV CSV, MERGES them, and partitions into
#  BROAD vs SECTOR by index name.
#
#  OVERLAP HANDLING: where two files carry the SAME (index, date):
#    - identical value  -> collapses to one row (just extends history)
#    - DIFFERENT value  -> the row from the file whose own coverage ends
#                          LATEST wins (the newer export's restated number),
#                          deterministically. Every shared date and every
#                          genuine conflict is reported in the sidebar.
#  (The abutting sector windows ...to01092025 / 02092025to... don't overlap,
#   so they simply stitch; the older broad-market files may overlap.)
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

    df["__src"] = os.path.basename(filepath)  # provenance for overlap resolution
    return df[["Date", "Index Name", "P/E", "P/B", "Div Yield %", "__src"]].dropna(
        subset=["Date", "P/E", "P/B"]
    )


@st.cache_data
def load_and_merge_data():
    """Returns (combined_df_all_indices, overlap_report_dict)."""
    patterns = ["*Historical_PE_PB_DIV_Data*.csv", "NIFTY*_PE_PB_DIV*.csv"]
    files = sorted(set(sum((glob.glob(p) for p in patterns), [])))
    if not files:
        return pd.DataFrame(), {}

    file_captions = []
    frames = []
    for f in files:
        part = parse_single_csv(f)
        if part.empty:
            file_captions.append(f"WARN {os.path.basename(f)} - skipped")
            continue
        rng = f"{part['Date'].min().date()} -> {part['Date'].max().date()}"
        file_captions.append(f"OK {os.path.basename(f)}  ({len(part)}r, {rng})")
        part["__rank"] = part["Date"].max()  # file recency = its own max date
        frames.append(part)

    if not frames:
        return pd.DataFrame(), {"captions": file_captions}

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[~combined["Index Name"].isin(IGNORE_INDICES)]

    # ---- OVERLAP / CONFLICT DIAGNOSTICS (before dedup) --------------------
    grp = combined.groupby(["Index Name", "Date"])
    src_counts = grp["__src"].nunique()
    pe_spread = grp["P/E"].agg(lambda s: s.max() - s.min())
    pb_spread = grp["P/B"].agg(lambda s: s.max() - s.min())
    conflict_mask = src_counts.gt(1) & ((pe_spread > CONFLICT_TOL) | (pb_spread > CONFLICT_TOL))
    conflicts = (
        pd.DataFrame(
            {
                "Index Name": [k[0] for k in conflict_mask[conflict_mask].index],
                "Date": [k[1] for k in conflict_mask[conflict_mask].index],
                "dP/E": pe_spread[conflict_mask].values,
                "dP/B": pb_spread[conflict_mask].values,
            }
        )
        .sort_values("dP/E", ascending=False)
        .reset_index(drop=True)
    )

    report = {
        "captions": file_captions,
        "shared_dates": int((src_counts > 1).sum()),
        "conflicts": conflicts,
    }

    # ---- DETERMINISTIC DEDUP: newest file wins ---------------------------
    combined.sort_values(by=["Index Name", "Date", "__rank"], kind="mergesort", inplace=True)
    combined.drop_duplicates(subset=["Index Name", "Date"], keep="last", inplace=True)
    combined.drop(columns=["__rank", "__src"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined, report


# ==========================================================================
#  ONE SHARED RENDERER - snapshot + month-wise table for any index group.
# ==========================================================================
def monthly_long(dfg: pd.DataFrame, selected_col: str, month_end: bool) -> pd.DataFrame:
    """Collapse daily -> ONE value per (month, index): month-end last value, or
    monthly mean. Both the snapshot and the month-wise table are built from this,
    so their numbers are identical and neither is exposed to daily-tick outliers."""
    d = dfg.copy()
    d["Period"] = d["Date"].dt.to_period("M")
    if month_end:
        m = (
            d.sort_values("Date")
            .groupby(["Period", "Index Name"])[selected_col].last().reset_index()
        )
    else:
        m = d.groupby(["Period", "Index Name"])[selected_col].mean().reset_index()
    return m


def build_snapshot(monthly: pd.DataFrame, order, selected_col: str) -> pd.DataFrame:
    """Average / Median / Min / Max computed on the MONTHLY series (one point per
    month), NOT on daily ticks. Current = latest month's value (= top row of table)."""
    rows = []
    for idx in order:
        sub = monthly.loc[monthly["Index Name"] == idx].sort_values("Period")
        sub = sub[sub[selected_col].notna()]
        if sub.empty:
            continue
        s = sub[selected_col]
        cur, avg, med = s.iloc[-1], s.mean(), s.median()
        rows.append(
            {
                "Index": idx,
                "From": sub["Period"].iloc[0].strftime("%b %Y"),  # first month used
                "Months": int(len(sub)),                          # # monthly obs -> uneven coverage
                "Current": cur,
                "Average": avg,
                "Median": med,
                "Min": s.min(),                                   # month-level outlier check
                "Max": s.max(),
                "% vs Avg": (cur / avg - 1) * 100,
                "% vs Median": (cur / med - 1) * 100,
            }
        )
    return pd.DataFrame(rows)


def build_month_table(monthly: pd.DataFrame, order, selected_col: str) -> pd.DataFrame:
    pivot = monthly.pivot(index="Period", columns="Index Name", values=selected_col)
    pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns])
    pivot = pivot.sort_index(ascending=False)  # newest month on top

    table = pivot.copy()
    table.insert(0, "Month", table.index.strftime("%b"))
    table.insert(0, "Year", table.index.year)
    return table.reset_index(drop=True)


def render_group(dfg: pd.DataFrame, order, selected_col: str, metric_choice: str,
                 agg_choice: str, key_prefix: str):
    if dfg.empty:
        st.info("No data for this group.")
        return

    order = [c for c in order if c in set(dfg["Index Name"].unique())]
    month_end = agg_choice.startswith("Month-end")
    basis = "month-end" if month_end else "monthly-average"

    # Build the monthly series ONCE; snapshot and table both read from it.
    monthly = monthly_long(dfg, selected_col, month_end)

    # --- snapshot (computed on the monthly series) ---
    st.subheader(f"Current {metric_choice}: Average & Median")
    st.caption(
        f"Computed on the {basis} MONTHLY series (one point per month), per index over "
        f"its own months within {dfg['Date'].min().strftime('%b %Y')} -> "
        f"{dfg['Date'].max().strftime('%b %Y')}. Current = latest month's value (= top row "
        "below). 'From'/'Months' show start + count per index; Min/Max flag month-level "
        "outliers. '% vs' = premium (+) / discount (-) vs own avg / median."
    )
    snapshot = build_snapshot(monthly, order, selected_col)
    st.dataframe(
        snapshot.style.format(
            {
                "Months": "{:d}",
                "Current": "{:.2f}", "Average": "{:.2f}", "Median": "{:.2f}",
                "Min": "{:.2f}", "Max": "{:.2f}",
                "% vs Avg": "{:+.1f}%", "% vs Median": "{:+.1f}%",
            }
        ).background_gradient(subset=["% vs Avg", "% vs Median"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # --- month-wise table (same monthly series) ---
    st.subheader(f"Month-wise {metric_choice}")
    table = build_month_table(monthly, order, selected_col)
    st.caption(
        f"Each row is one calendar month ({basis} {metric_choice}). Newest first. "
        f"{len(table)} months, {dfg['Date'].min().strftime('%b %Y')} -> "
        f"{dfg['Date'].max().strftime('%b %Y')}."
    )
    idx_cols = [c for c in order if c in table.columns]
    st.dataframe(
        table.style.format({c: "{:.2f}" for c in idx_cols})
        .background_gradient(subset=idx_cols, cmap="RdYlGn_r")
        .set_properties(subset=["Year", "Month"], **{"font-weight": "600"}),
        use_container_width=True, height=640, hide_index=True,
    )
    st.download_button(
        f"Download {key_prefix} month-wise {selected_col} table (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"nifty_{key_prefix}_monthly_{selected_col.replace('/', '')}.csv",
        mime="text/csv",
        key=f"dl_{key_prefix}",
    )


# --------------------------------------------------------------------------
df, report = load_and_merge_data()

# --- SIDEBAR: files + overlap diagnostics ---------------------------------
st.sidebar.markdown("**Files merged**")
for cap in report.get("captions", []):
    st.sidebar.caption(cap)

if df.empty:
    st.warning(
        "No valuation CSVs found. Place your `*_Historical_PE_PB_DIV_Data_*.csv` files "
        "in the same folder as this script."
    )
    st.stop()

# Partition into broad vs sector.
df_broad = df[df["Index Name"].isin(BROAD_INDICES)].copy()
df_broad = df_broad[df_broad["Date"] >= BROAD_START]  # floor broad tab at Mar 2024
df_sector = df[~df["Index Name"].isin(BROAD_INDICES)].copy()
SECTOR_ORDER = sorted(df_sector["Index Name"].unique())

st.sidebar.markdown(
    f"**Coverage:** {df['Date'].min().date()} -> {df['Date'].max().date()}  "
    f"({len(BROAD_INDICES & set(df['Index Name']))} broad, {len(SECTOR_ORDER)} sectors)"
)

# Overlap panel.
st.sidebar.markdown("**Overlap**")
conflicts = report.get("conflicts", pd.DataFrame())
st.sidebar.caption(f"Shared (index, date) pairs across files: **{report.get('shared_dates', 0)}**")
if conflicts is not None and not conflicts.empty:
    st.sidebar.caption(f"**{len(conflicts)}** had differing values - newest file kept. Top:")
    st.sidebar.dataframe(
        conflicts.assign(Date=lambda x: x["Date"].dt.date).head(10),
        use_container_width=True, hide_index=True,
    )
else:
    st.sidebar.caption("All overlapping dates agreed (clean merge).")

# --- CONFIG (shared across both tabs) -------------------------------------
st.sidebar.header("Configuration")
metric_choice = st.sidebar.radio("Metric:", ["P/E Ratio", "P/B Ratio", "Div Yield %"], index=0)
agg_choice = st.sidebar.radio(
    "Monthly value:", ["Month-end (last trading day)", "Monthly average"], index=0
)
selected_col = {"P/E Ratio": "P/E", "P/B Ratio": "P/B", "Div Yield %": "Div Yield %"}[metric_choice]

# --- TABS -----------------------------------------------------------------
tab_broad, tab_sector = st.tabs(["Broad Market", "Sectors"])
with tab_broad:
    render_group(df_broad, BROAD_ORDER, selected_col, metric_choice, agg_choice, "broad")
with tab_sector:
    render_group(df_sector, SECTOR_ORDER, selected_col, metric_choice, agg_choice, "sector")
