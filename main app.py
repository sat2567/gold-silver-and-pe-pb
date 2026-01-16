import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. GLOBAL PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro Market Intelligence Super-App", 
    layout="wide", 
    page_icon="🚀"
)

# ==========================================
#  MODULE 1: PRECIOUS METALS (Gold vs Silver)
# ==========================================

def get_gold_tax_multiplier(date):
    """Returns (1 + Tax Rate) for 'Landed Cost' calculation."""
    dt = pd.Timestamp(date)
    if dt < pd.Timestamp("2012-01-17"): return 1.02
    elif dt < pd.Timestamp("2013-08-13"): return 1.06
    elif dt < pd.Timestamp("2017-07-01"): return 1.11
    elif dt < pd.Timestamp("2019-07-05"): return 1.13
    elif dt < pd.Timestamp("2021-02-01"): return 1.155
    elif dt < pd.Timestamp("2022-07-01"): return 1.14
    elif dt < pd.Timestamp("2024-07-23"): return 1.18
    else: return 1.09

def clean_outliers(df, col_name, threshold=0.10):
    # 1. Remove Zeros
    df = df[df[col_name] > 10].copy()
    # 2. Remove Massive Spikes (Glitches > 10%)
    daily_ret = df[col_name].pct_change()
    mask = daily_ret.abs().fillna(0) < threshold
    return df[mask]

@st.cache_data
def fetch_metals_data():
    tickers = ['GOLDBEES.NS', 'SI=F', 'INR=X']
    # Use 'Close' instead of 'Adj Close' for raw price data if preferred, but usually 'Close' is fine
    data = yf.download(tickers, period="20y", interval="1d")['Close']
    data.ffill(inplace=True)
    data.dropna(inplace=True)
    
    df = pd.DataFrame(index=data.index)
    
    # Gold (Cleaned)
    df['Gold (India)'] = data['GOLDBEES.NS']
    df = clean_outliers(df, 'Gold (India)', threshold=0.10)
    
    # Silver (Synthetic)
    common_index = df.index
    si_price = data.loc[common_index, 'SI=F']
    inr_price = data.loc[common_index, 'INR=X']
    tax_series = common_index.to_series().apply(get_gold_tax_multiplier)
    df['Silver (India)'] = si_price * inr_price * 32.15 * tax_series
    
    df.dropna(inplace=True)
    return df

def calculate_rolling_metrics(df):
    metrics = df.copy()
    # 1-Year Rolling Return (Window = 252 trading days)
    metrics['Gold_1Y'] = metrics['Gold (India)'].pct_change(periods=252) * 100
    metrics['Silver_1Y'] = metrics['Silver (India)'].pct_change(periods=252) * 100
    
    # 3-Year Rolling CAGR (Window = 756 trading days)
    metrics['Gold_3Y_CAGR'] = ((metrics['Gold (India)'] / metrics['Gold (India)'].shift(756))**(1/3) - 1) * 100
    metrics['Silver_3Y_CAGR'] = ((metrics['Silver (India)'] / metrics['Silver (India)'].shift(756))**(1/3) - 1) * 100
    
    return metrics

def get_stats_table(series, name):
    return {
        "Metric": name,
        "Current": series.iloc[-1],
        "Average (Mean)": series.mean(),
        "Median": series.median(),
        "Best Case (Max)": series.max(),
        "Worst Case (Min)": series.min(),
        "% Positive Periods": (series > 0).mean() * 100
    }

def show_metals_dashboard():
    st.title("✨ Pro Quant Dashboard: Gold vs. Silver")
    st.markdown("Analysis of **Rolling Return Distributions** (Mean of all 1Y/3Y periods).")

    with st.spinner('Fetching & Cleaning market data...'):
        raw_df = fetch_metals_data()
        df_analysis = calculate_rolling_metrics(raw_df)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Rolling Statistics", "📈 Charts & Distribution", "📉 Drawdowns", "📅 Seasonality"])

    with tab1:
        st.subheader("Rolling Return Statistics")
        stats_data = []
        stats_data.append(get_stats_table(df_analysis['Gold_1Y'].dropna(), "Gold: 1-Year Rolling Returns"))
        stats_data.append(get_stats_table(df_analysis['Silver_1Y'].dropna(), "Silver: 1-Year Rolling Returns"))
        stats_data.append(get_stats_table(df_analysis['Gold_3Y_CAGR'].dropna(), "Gold: 3-Year Rolling CAGR"))
        stats_data.append(get_stats_table(df_analysis['Silver_3Y_CAGR'].dropna(), "Silver: 3-Year Rolling CAGR"))
        
        stats_df = pd.DataFrame(stats_data).set_index("Metric")
        format_dict = {"Current": "{:.2f}%", "Average (Mean)": "{:.2f}%", "Median": "{:.2f}%", "Best Case (Max)": "{:.2f}%", "Worst Case (Min)": "{:.2f}%", "% Positive Periods": "{:.1f}%"}
        st.dataframe(stats_df.style.format(format_dict).background_gradient(subset=["Average (Mean)"], cmap="Greens"), use_container_width=True)

    with tab2:
        st.subheader("Rolling Returns Visualization")
        period_select = st.radio("Select Period:", ["1-Year Rolling Return", "3-Year Rolling CAGR"], horizontal=True)
        cols = ['Gold_1Y', 'Silver_1Y'] if period_select == "1-Year Rolling Return" else ['Gold_3Y_CAGR', 'Silver_3Y_CAGR']
        
        fig_roll = go.Figure()
        colors = {'Gold': '#FFD700', 'Silver': '#C0C0C0'}
        for c in cols:
            name = c.split('_')[0]
            fig_roll.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis[c], name=name, line=dict(color=colors.get(name, 'black'), width=1.5)))
        for c in cols:
             name = c.split('_')[0]
             mean_val = df_analysis[c].mean()
             fig_roll.add_hline(y=mean_val, line_dash="dot", line_color=colors.get(name, 'black'), annotation_text=f"{name} Avg: {mean_val:.1f}%")
        fig_roll.add_hline(y=0, line_dash="solid", line_color="white")
        st.plotly_chart(fig_roll, use_container_width=True)

        fig_hist = go.Figure()
        for c in cols:
            name = c.split('_')[0]
            fig_hist.add_trace(go.Histogram(x=df_analysis[c], name=name, opacity=0.75, marker_color=colors.get(name, 'black')))
        fig_hist.update_layout(barmode='overlay', title="Distribution of Returns", xaxis_title="Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Drawdown Analysis")
        dd_data = raw_df.copy()
        dd_summary = []
        for c in dd_data.columns:
            peak = dd_data[c].cummax()
            dd_series = (dd_data[c] / peak - 1) * 100
            dd_data[c] = dd_series
            dd_summary.append({"Asset": c, "Max Drawdown": dd_series.min(), "Current Drawdown": dd_series.iloc[-1], "Average Drawdown": dd_series.mean()})
            
        dd_df = pd.DataFrame(dd_summary).set_index("Asset")
        st.dataframe(dd_df.style.format("{:.2f}%").background_gradient(cmap="Reds_r", subset=["Max Drawdown", "Current Drawdown"]), use_container_width=True)
        fig_dd = px.area(dd_data, x=dd_data.index, y=dd_data.columns, color_discrete_map={"Gold (India)": "#FFD700", "Silver (India)": "#C0C0C0"})
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab4:
        st.subheader("Seasonality Heatmap")
        col_a, col_b = st.columns(2)
        def plot_heatmap(asset, ax):
            m_ret = raw_df[asset].pct_change().dropna()
            grp = m_ret.groupby(m_ret.index.month).mean() * 100
            heatmap_data = pd.DataFrame(grp).T
            sns.heatmap(heatmap_data, cmap="RdYlGn", center=0, annot=True, fmt=".2f", ax=ax, cbar=False)
            ax.set_title(asset)
            ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
            ax.set_yticklabels([])
        
        with col_a:
            fig_g, ax_g = plt.subplots(figsize=(6, 2))
            plot_heatmap('Gold (India)', ax_g)
            st.pyplot(fig_g)
        with col_b:
            fig_s, ax_s = plt.subplots(figsize=(6, 2))
            plot_heatmap('Silver (India)', ax_s)
            st.pyplot(fig_s)

# ==========================================
#  MODULE 2: NSE VALUATION (Logic from pe.py)
# ==========================================

@st.cache_data
def load_valuation_data():
    files = {
        "Nifty 50 (2025)": "NIFTY 50_Historical_PE_PB_DIV_Data_01012025to01012026.csv",
        "Nifty Midcap 150 (2025)": "NIFTY MIDCAP 150_Historical_PE_PB_DIV_Data_01012025to01012026.csv",
        "Nifty Smallcap 250 (2025)": "NIFTY SMALLCAP 250_Historical_PE_PB_DIV_Data_01012025to01012026.csv",
        "Nifty Total Market (2025)": "NIFTY TOTAL MARKET_Historical_PE_PB_DIV_Data_01012025to01012026.csv",
        "Nifty 50 (2024)": "NIFTY 50_Historical_PE_PB_DIV_Data_01012024to31122024.csv",
        "Nifty Midcap 150 (2024)": "NIFTY MIDCAP 150_Historical_PE_PB_DIV_Data_01012024to31122024.csv",
        "Nifty Smallcap 250 (2024)": "NIFTY SMALLCAP 250_Historical_PE_PB_DIV_Data_01012024to31122024.csv",
        "Nifty Total Market (2024)": "NIFTY TOTAL MARKET_Historical_PE_PB_DIV_Data_01012024to31122024.csv"
    }
    
    combined_df = pd.DataFrame()
    for label, filename in files.items():
        try:
            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip().str.replace('"', '')
            if 'P/B' not in df.columns: continue
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                if 'IndexName' in df.columns: df['Index Name'] = df['IndexName']
                elif 'Index Name' not in df.columns: df['Index Name'] = label
                combined_df = pd.concat([combined_df, df])
        except FileNotFoundError:
            pass # Skip missing files silently
    
    # Sort Globally to ensure correct Timeline
    if not combined_df.empty:
        combined_df.sort_values(by=['Index Name', 'Date'], ascending=[True, True], inplace=True)
    return combined_df

def show_valuation_dashboard():
    st.title("📊 NSE Valuation Dashboard: P/E & P/B")
    
    df = load_valuation_data()
    
    if df.empty:
        st.error("No CSV files found. Please ensure the Nifty CSVs are in the folder.")
        return

    # Configuration
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        metric_choice = st.radio("Select Metric:", ["P/E Ratio", "P/B Ratio", "Div Yield %"])
    
    col_map = {"P/E Ratio": "P/E", "P/B Ratio": "P/B", "Div Yield %": "Div Yield %"}
    selected_col = col_map[metric_choice]

    # --- NEW SECTION: Current vs Average Status Table ---
    st.markdown(f"### 🚦 Valuation Status: {metric_choice}")
    
    # Calculate Stats per Index
    summary_data = []
    indices = df['Index Name'].unique()
    
    for idx in indices:
        subset = df[df['Index Name'] == idx]
        current_val = subset[selected_col].iloc[-1]
        avg_val = subset[selected_col].mean()
        diff_pct = ((current_val - avg_val) / avg_val) * 100
        
        # Determine Status
        if selected_col == "Div Yield %":
            # For Dividend Yield, Higher is Cheaper/Better
            if diff_pct > 5: status = "Undervalued (High Yield) 🟢"
            elif diff_pct < -5: status = "Overvalued (Low Yield) 🔴"
            else: status = "Fair Value 🟡"
        else:
            # For P/E and P/B, Higher is Expensive
            if diff_pct > 5: status = "Overvalued 🔴"
            elif diff_pct < -5: status = "Undervalued 🟢"
            else: status = "Fair Value 🟡"
            
        summary_data.append({
            "Index Name": idx,
            "Current": current_val,
            "Historical Average": avg_val,
            "Diff (%)": diff_pct,
            "Valuation Status": status
        })
        
    summary_df = pd.DataFrame(summary_data).set_index("Index Name")
    
    # Display the comparison table with highlighting
    def highlight_status(val):
        color = 'white'
        if 'Overvalued' in val: color = '#ffcccc' # Light Red
        elif 'Undervalued' in val: color = '#ccffcc' # Light Green
        elif 'Fair' in val: color = '#fff5cc' # Light Yellow
        return f'background-color: {color}; color: black'

    st.dataframe(
        summary_df.style.format({
            "Current": "{:.2f}", 
            "Historical Average": "{:.2f}", 
            "Diff (%)": "{:+.2f}%"
        }).applymap(highlight_status, subset=['Valuation Status']),
        use_container_width=True
    )
    
    st.divider()

    # Visuals Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Monthly Historical Data", "📈 Trend Analysis", "📊 Relative Value", "📉 Matrix"])
    
    with tab1:
        st.subheader(f"📅 Monthly {metric_choice} Data Table")
        
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month_name().str[:3]
        
        selected_index = st.selectbox("Select Index for Tabular View:", indices)
        subset = df[df['Index Name'] == selected_index]
        
        pivot_table = subset.groupby(['Year', 'Month'])[selected_col].mean().reset_index()
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot_table['Month'] = pd.Categorical(pivot_table['Month'], categories=months_order, ordered=True)
        final_table = pivot_table.pivot(index='Year', columns='Month', values=selected_col)
        
        st.write(f"**Average Monthly {metric_choice} for {selected_index}**")
        st.dataframe(final_table.style.format("{:.2f}").background_gradient(cmap="YlOrRd"), use_container_width=True)

    with tab2:
        fig = px.line(df, x='Date', y=selected_col, color='Index Name', title=f"{metric_choice} Trend")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        fig_bar = px.bar(summary_df.reset_index(), x='Index Name', y='Diff (%)', color='Diff (%)', 
                         color_continuous_scale="RdYlGn_r", title="Premium/Discount vs Historical Average (%)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with tab4:
        latest = df.groupby('Index Name').tail(1)
        fig_scat = px.scatter(latest, x='P/E', y='P/B', color='Index Name', size='P/E', text='Index Name', title="Risk vs Value Matrix")
        fig_scat.add_vline(x=latest['P/E'].mean(), line_dash="dash")
        fig_scat.add_hline(y=latest['P/B'].mean(), line_dash="dash")
        st.plotly_chart(fig_scat, use_container_width=True)

# ==========================================
#  MAIN APP CONTROLLER
# ==========================================

def main():
    st.sidebar.title("🚀 Navigation")
    app_mode = st.sidebar.radio("Go To:", ["Precious Metals (Gold/Silver)", "NSE Valuations (P/E & P/B)"])
    
    st.sidebar.divider()
    st.sidebar.info("**About:**\n\n1. **Metals:** Rolling Returns Mean/Max/Min.\n2. **Valuations:** Overvalued/Undervalued Status.")
    
    if app_mode == "Precious Metals (Gold/Silver)":
        show_metals_dashboard()
    elif app_mode == "NSE Valuations (P/E & P/B)":
        show_valuation_dashboard()

if __name__ == "__main__":
    main()
