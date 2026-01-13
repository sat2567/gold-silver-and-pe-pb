import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. GLOBAL PAGE CONFIGURATION (Must be at the very top) ---
st.set_page_config(
    page_title="Pro Market Intelligence Super-App", 
    layout="wide", 
    page_icon="🚀"
)

# ==========================================
#  MODULE 1: PRECIOUS METALS (Logic from data.py)
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
    tickers = ['GOLDBEES.NS', 'SI=F', 'INR=X', '^NSEI']
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
    
    # Nifty
    df['Nifty 50'] = data.loc[common_index, '^NSEI']
    df.dropna(inplace=True)
    return df

def calculate_rolling_metrics(df):
    metrics = df.copy()
    # 1-Year
    metrics['Gold_1Y'] = metrics['Gold (India)'].pct_change(periods=252) * 100
    metrics['Silver_1Y'] = metrics['Silver (India)'].pct_change(periods=252) * 100
    metrics['Nifty_1Y'] = metrics['Nifty 50'].pct_change(periods=252) * 100
    # 3-Year
    metrics['Gold_3Y_CAGR'] = ((metrics['Gold (India)'] / metrics['Gold (India)'].shift(756))**(1/3) - 1) * 100
    metrics['Silver_3Y_CAGR'] = ((metrics['Silver (India)'] / metrics['Silver (India)'].shift(756))**(1/3) - 1) * 100
    metrics['Nifty_3Y_CAGR'] = ((metrics['Nifty 50'] / metrics['Nifty 50'].shift(756))**(1/3) - 1) * 100
    return metrics

def show_metals_dashboard():
    st.title("✨ Pro Quant Dashboard: Gold vs. Silver")
    st.markdown("Analysis with **Outlier Removal** and **Synthetic Tax Calculations**.")

    with st.spinner('Fetching & Cleaning market data...'):
        raw_df = fetch_metals_data()
        df_analysis = calculate_rolling_metrics(raw_df)

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    years = (raw_df.index[-1] - raw_df.index[0]).days / 365.25
    gold_cagr = (raw_df['Gold (India)'].iloc[-1] / raw_df['Gold (India)'].iloc[0])**(1/years) - 1
    silver_cagr = (raw_df['Silver (India)'].iloc[-1] / raw_df['Silver (India)'].iloc[0])**(1/years) - 1

    with col1: st.metric("Gold CAGR", f"{gold_cagr*100:.2f}%")
    with col2: st.metric("Silver CAGR", f"{silver_cagr*100:.2f}%")
    with col3: st.metric("Gold 3Y Return (Curr)", f"{df_analysis['Gold_3Y_CAGR'].iloc[-1]:.2f}%")
    with col4: st.metric("Silver 3Y Return (Curr)", f"{df_analysis['Silver_3Y_CAGR'].iloc[-1]:.2f}%")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Wealth Index", "🔄 Rolling Returns", "📉 Drawdowns", "📅 Seasonality"])

    with tab1:
        st.subheader("Growth of ₹100")
        normalized = (raw_df / raw_df.iloc[0]) * 100
        fig_norm = px.line(normalized, x=normalized.index, y=normalized.columns)
        st.plotly_chart(fig_norm, use_container_width=True)

    with tab2:
        st.subheader("Rolling Returns Analysis")
        period_select = st.radio("Select Period:", ["1-Year Rolling Return", "3-Year Rolling CAGR"], horizontal=True)
        cols = ['Gold_1Y', 'Silver_1Y', 'Nifty_1Y'] if period_select == "1-Year Rolling Return" else ['Gold_3Y_CAGR', 'Silver_3Y_CAGR', 'Nifty_3Y_CAGR']
        fig_roll = go.Figure()
        colors = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Nifty': '#0000FF'}
        for c in cols:
            name = c.split('_')[0]
            fig_roll.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis[c], name=name, line=dict(color=colors.get(name, 'black'), width=1.5)))
        fig_roll.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_roll, use_container_width=True)

    with tab3:
        st.subheader("Underwater Plot (Risk)")
        dd_data = raw_df.copy()
        for c in dd_data.columns:
            dd_data[c] = (dd_data[c] / dd_data[c].cummax() - 1) * 100
        fig_dd = px.area(dd_data, x=dd_data.index, y=dd_data.columns)
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab4:
        st.subheader("Seasonality Heatmap")
        col_a, col_b = st.columns(2)
        def plot_heatmap(asset, ax):
            m_ret = raw_df[asset].pct_change().dropna()
            grp = m_ret.groupby(m_ret.index.month).mean() * 100
            sns.heatmap(pd.DataFrame(grp).T, cmap="RdYlGn", center=0, annot=True, fmt=".2f", ax=ax, cbar=False)
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
        "Nifty Smallcap 100 (2025)": "NIFTY SMALLCAP 100_Historical_PE_PB_DIV_Data_01012025to01012026.csv",
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
    st.markdown("Comparative analysis of **Nifty 50, Midcap, Smallcap, and Total Market**.")
    
    df = load_valuation_data()
    
    if df.empty:
        st.error("No CSV files found. Please ensure the Nifty CSVs are in the folder.")
        return

    # Configuration
    metric_choice = st.radio("Select Metric:", ["P/E Ratio", "P/B Ratio", "Div Yield %"], horizontal=True)
    col_map = {"P/E Ratio": "P/E", "P/B Ratio": "P/B", "Div Yield %": "Div Yield %"}
    selected_col = col_map[metric_choice]

    # Summary
    st.subheader(f"📋 Valuation Summary: {metric_choice}")
    summary = df.groupby('Index Name')[selected_col].agg(['last', 'mean', 'min', 'max', 'std']).reset_index()
    summary.columns = ['Index', 'Current', 'Average', 'Min', 'Max', 'Volatility']
    
    def get_status(row):
        if row['Current'] > row['Average'] * 1.05: return "Expensive 🔴"
        elif row['Current'] < row['Average'] * 0.95: return "Cheap 🟢"
        else: return "Fair 🟡"
    summary['Status'] = summary.apply(get_status, axis=1)
    
    st.dataframe(summary.style.format({'Current':'{:.2f}', 'Average':'{:.2f}', 'Min':'{:.2f}', 'Max':'{:.2f}', 'Volatility':'{:.2f}'}).background_gradient(subset=['Current'], cmap="Reds"), use_container_width=True)

    # Visuals
    tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "📊 Relative Value", "📉 Matrix"])
    
    with tab1:
        fig = px.line(df, x='Date', y=selected_col, color='Index Name', title=f"{metric_choice} Trend")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        bar_data = summary.copy()
        bar_data['% Diff'] = ((bar_data['Current'] - bar_data['Average']) / bar_data['Average']) * 100
        fig_bar = px.bar(bar_data, x='Index', y='% Diff', color='% Diff', color_continuous_scale="RdYlGn_r", title="Premium/Discount vs Avg")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with tab3:
        latest = df.groupby('Index Name').tail(1)
        fig_scat = px.scatter(latest, x='P/E', y='P/B', color='Index Name', size='P/E', text='Index Name', title="Risk vs Value Matrix")
        fig_scat.add_vline(x=latest['P/E'].mean(), line_dash="dash")
        fig_scat.add_hline(y=latest['P/B'].mean(), line_dash="dash")
        st.plotly_chart(fig_scat, use_container_width=True)


# ==========================================
#  MAIN APP CONTROLLER (The Switcher)
# ==========================================

def main():
    st.sidebar.title("🚀 Navigation")
    app_mode = st.sidebar.radio("Go To:", ["Precious Metals (Gold/Silver)", "NSE Valuations (P/E & P/B)"])
    
    st.sidebar.divider()
    st.sidebar.info("**About:**\n\n1. **Metals:** Real-time + Synthetic data via Yahoo Finance.\n2. **Valuations:** Historical data via NSE CSV dumps.")
    
    if app_mode == "Precious Metals (Gold/Silver)":
        show_metals_dashboard()
    elif app_mode == "NSE Valuations (P/E & P/B)":
        show_valuation_dashboard()

if __name__ == "__main__":
    main()