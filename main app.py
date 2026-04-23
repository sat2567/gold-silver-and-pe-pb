import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NSE Valuation Dashboard",
    layout="wide",
    page_icon="📊"
)

# --- INDICES TO INCLUDE (no Smallcap 100) ---
ALLOWED_INDICES = {
    "NIFTY 50",
    "NIFTY MIDCAP 150",
    "NIFTY SMALLCAP 250",
    "NIFTY TOTAL MARKET",
}

# ==========================================
#  DATA LOADING (Auto-discover & merge CSVs)
# ==========================================

def discover_csv_files(search_dirs=None):
    if search_dirs is None:
        search_dirs = ["."]
    found_files = []
    for directory in search_dirs:
        for pattern in [
            "*Historical_PE_PB_DIV_Data*.csv",
            "*_Historical_PE_PB_DIV_Data_*.csv",
            "NIFTY*_PE_PB_DIV*.csv",
        ]:
            found_files.extend(glob.glob(os.path.join(directory, pattern)))
    return sorted(set(found_files))


def parse_single_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.replace('"', '', regex=False)

        if not {'P/E', 'P/B'}.issubset(set(df.columns)):
            return pd.DataFrame()

        if 'IndexName' in df.columns:
            df['Index Name'] = df['IndexName'].str.strip().str.replace('"', '', regex=False)
            df.drop(columns=['IndexName'], inplace=True, errors='ignore')
        elif 'Index Name' in df.columns:
            df['Index Name'] = df['Index Name'].str.strip().str.replace('"', '', regex=False)
        else:
            basename = os.path.basename(filepath)
            name_part = basename.split("_Historical")[0] if "_Historical" in basename else basename.split(".")[0]
            df['Index Name'] = name_part.replace("_", " ")

        if 'Date' not in df.columns:
            return pd.DataFrame()
        df['Date'] = df['Date'].astype(str).str.strip().str.replace('"', '', regex=False)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)

        for col in ['P/E', 'P/B', 'Div Yield %']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.strip().str.replace('"', '', regex=False),
                    errors='coerce'
                )

        if 'Div Yield %' not in df.columns:
            df['Div Yield %'] = np.nan

        result = df[['Date', 'Index Name', 'P/E', 'P/B', 'Div Yield %']].dropna(subset=['Date', 'P/E', 'P/B'])

        # Filter out unwanted indices (e.g. Smallcap 100)
        result = result[result['Index Name'].isin(ALLOWED_INDICES)]
        return result

    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_valuation_data():
    all_files = discover_csv_files(["."])
    if not all_files:
        return pd.DataFrame()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📂 Loaded Files:**")

    all_dfs = []
    for f in all_files:
        df = parse_single_csv(f)
        if not df.empty:
            basename = os.path.basename(f)
            date_min = df['Date'].min().strftime('%d-%b-%Y')
            date_max = df['Date'].max().strftime('%d-%b-%Y')
            idx_names = df['Index Name'].unique()
            st.sidebar.caption(
                f"✅ `{basename}`  \n{', '.join(idx_names)} — {len(df)} rows  \n{date_min} → {date_max}"
            )
            all_dfs.append(df)
        else:
            st.sidebar.caption(f"⚠️ `{os.path.basename(f)}` — skipped")

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.drop_duplicates(subset=['Index Name', 'Date'], keep='last', inplace=True)
    combined_df.sort_values(by=['Index Name', 'Date'], ascending=[True, True], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    indices = combined_df['Index Name'].unique()
    total_from = combined_df['Date'].min().strftime('%d-%b-%Y')
    total_to = combined_df['Date'].max().strftime('%d-%b-%Y')
    st.sidebar.success(
        f"**Merged:** {len(indices)} indices, {len(combined_df)} rows  \n"
        f"**Range:** {total_from} → {total_to}"
    )

    return combined_df


# ==========================================
#  DASHBOARD
# ==========================================

def main():
    st.title("📊 NSE Valuation Dashboard: P/E & P/B")
    st.markdown("Comparative analysis of **Nifty 50, Midcap 150, Smallcap 250 & Total Market**.")

    df = load_valuation_data()

    if df.empty:
        st.error(
            "No CSV files found. Place your Nifty valuation CSVs "
            "(e.g. `NIFTY_50_Historical_PE_PB_DIV_Data_*.csv`) in the same folder as this app.\n\n"
            "The app auto-discovers all matching files, merges them, and deduplicates by date."
        )
        return

    # --- METRIC SELECTION ---
    col_ctrl1, _ = st.columns([1, 3])
    with col_ctrl1:
        metric_choice = st.radio("Select Metric:", ["P/E Ratio", "P/B Ratio", "Div Yield %"])

    col_map = {"P/E Ratio": "P/E", "P/B Ratio": "P/B", "Div Yield %": "Div Yield %"}
    selected_col = col_map[metric_choice]

    # --- SUMMARY TABLE ---
    st.markdown(f"### 📋 Valuation Summary: {metric_choice}")

    summary = df.groupby('Index Name')[selected_col].agg(
        ['last', 'mean', 'median', 'min', 'max', 'std']
    ).reset_index()
    summary.columns = ['Index', 'Current', 'Average', 'Median', 'Min', 'Max', 'Volatility']
    summary['% vs Avg'] = ((summary['Current'] - summary['Average']) / summary['Average']) * 100

    def get_status(row):
        diff = row['% vs Avg']
        if selected_col == "Div Yield %":
            if diff > 5:
                return "Undervalued (High Yield) 🟢"
            elif diff < -5:
                return "Overvalued (Low Yield) 🔴"
            else:
                return "Fair Value 🟡"
        else:
            if diff > 5:
                return "Expensive 🔴"
            elif diff < -5:
                return "Cheap 🟢"
            else:
                return "Fair 🟡"

    summary['Status'] = summary.apply(get_status, axis=1)

    date_ranges = df.groupby('Index Name')['Date'].agg(['min', 'max', 'count']).reset_index()
    date_ranges.columns = ['Index', 'From', 'To', 'Data Points']
    date_ranges['From'] = date_ranges['From'].dt.strftime('%d-%b-%Y')
    date_ranges['To'] = date_ranges['To'].dt.strftime('%d-%b-%Y')
    summary = summary.merge(date_ranges, on='Index', how='left')

    def highlight_status(val):
        color = 'white'
        if isinstance(val, str):
            if 'Expensive' in val or 'Overvalued' in val:
                color = '#ffcccc'
            elif 'Cheap' in val or 'Undervalued' in val:
                color = '#ccffcc'
            elif 'Fair' in val:
                color = '#fff5cc'
        return f'background-color: {color}; color: black'

    display_summary = summary.set_index('Index')
    st.dataframe(
        display_summary.style.format({
            'Current': '{:.2f}', 'Average': '{:.2f}', 'Median': '{:.2f}',
            'Min': '{:.2f}', 'Max': '{:.2f}',
            'Volatility': '{:.2f}', '% vs Avg': '{:+.2f}%'
        }).map(highlight_status, subset=['Status']
        ).background_gradient(subset=['Current'], cmap="Reds"),
        use_container_width=True
    )

    st.divider()
    indices = df['Index Name'].unique()

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Monthly Historical Data", "📈 Trend Analysis",
        "📊 Relative Value", "📉 Matrix"
    ])

    with tab1:
        st.subheader(f"📅 Monthly {metric_choice} Data Table")
        df_tab = df.copy()
        df_tab['Year'] = df_tab['Date'].dt.year
        df_tab['Month'] = df_tab['Date'].dt.month_name().str[:3]
        selected_index = st.selectbox("Select Index for Tabular View:", indices)
        subset = df_tab[df_tab['Index Name'] == selected_index]
        pivot_table = subset.groupby(['Year', 'Month'])[selected_col].mean().reset_index()
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot_table['Month'] = pd.Categorical(pivot_table['Month'], categories=months_order, ordered=True)
        final_table = pivot_table.pivot(index='Year', columns='Month', values=selected_col)
        st.write(f"**Average Monthly {metric_choice} for {selected_index}**")
        st.dataframe(
            final_table.style.format("{:.2f}").background_gradient(cmap="YlOrRd"),
            use_container_width=True
        )

    with tab2:
        st.subheader("Historical Trend")
        fig = px.line(df, x='Date', y=selected_col, color='Index Name',
                      title=f"{metric_choice} Trend", height=500)
        fig.update_layout(hovermode="x unified", yaxis_title=metric_choice)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Premium/Discount vs Historical Average")
        bar_data = summary.copy()
        fig_bar = px.bar(bar_data, x='Index', y='% vs Avg', color='% vs Avg',
                         color_continuous_scale="RdYlGn_r",
                         title="Premium/Discount vs Historical Average (%)")
        fig_bar.add_hline(y=0, line_color="black")
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        st.subheader("Valuation Matrix (Current P/E vs P/B)")
        latest = df.groupby('Index Name').tail(1)
        fig_scat = px.scatter(latest, x='P/E', y='P/B', color='Index Name',
                              size='P/E', text='Index Name',
                              title="Risk vs Value Matrix")
        avg_pe = latest['P/E'].mean()
        avg_pb = latest['P/B'].mean()
        fig_scat.add_vline(x=avg_pe, line_dash="dash", line_color="grey")
        fig_scat.add_hline(y=avg_pb, line_dash="dash", line_color="grey")
        fig_scat.update_traces(textposition='top center')
        st.plotly_chart(fig_scat, use_container_width=True)


if __name__ == "__main__":
    main()
