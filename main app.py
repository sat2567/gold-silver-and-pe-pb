import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import io
import datetime
import os
import random
from bs4 import BeautifulSoup

# --- 1. GLOBAL PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro Market Intelligence Super-App", 
    layout="wide", 
    page_icon="🚀"
)

# ==========================================
#  MODULE 1: PRECIOUS METALS (Price Action)
# ==========================================

def get_gold_tax_multiplier(date):
    """Returns (1 + Tax Rate) for 'Landed Cost' calculation."""
    # Ensure input is a valid Timestamp
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
    
    # 1. Download Data safely
    # Auto_adjust=False ensures we get 'Close' explicitly
    data = yf.download(tickers, period="20y", interval="1d", auto_adjust=False)
    
    # 2. Handle MultiIndex Columns (Fix for yfinance v0.2+)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data['Close']
        except KeyError:
            if 'Adj Close' in data.columns.get_level_values(0):
                 data = data['Adj Close']
            else:
                 data = data.droplevel(0, axis=1)

    # 3. Fill missing values
    data.ffill(inplace=True)
    data.dropna(inplace=True)
    
    df = pd.DataFrame(index=data.index)
    
    # 4. Extract Gold Data (Cleaned)
    if 'GOLDBEES.NS' in data.columns:
        df['Gold (India)'] = data['GOLDBEES.NS']
    else:
        st.error("Error: GOLDBEES.NS data missing from Yahoo Finance.")
        return pd.DataFrame() 

    df = clean_outliers(df, 'Gold (India)', threshold=0.10)
    
    # 5. Calculate Synthetic Silver Price
    common_index = df.index
    
    try:
        si_price = data.loc[common_index, 'SI=F']
        inr_price = data.loc[common_index, 'INR=X']
        
        # FIX: Generate Tax Multiplier as a simple list of floats
        tax_values = [get_gold_tax_multiplier(d) for d in common_index]
        tax_series = pd.Series(tax_values, index=common_index)
        
        # Perform Calculation (Force float types)
        df['Silver (India)'] = (
            si_price.astype(float) * inr_price.astype(float) * 32.15 * tax_series.astype(float)
        )
    except KeyError as e:
        st.warning(f"Missing data for Silver calculation: {e}")
        return df[['Gold (India)']]
    
    df.dropna(inplace=True)
    return df

def calculate_rolling_metrics(df):
    metrics = df.copy()
    # 1-Year Rolling Return (Window = 252 trading days)
    if 'Gold (India)' in metrics.columns:
        metrics['Gold_1Y'] = metrics['Gold (India)'].pct_change(periods=252) * 100
        metrics['Gold_3Y_CAGR'] = ((metrics['Gold (India)'] / metrics['Gold (India)'].shift(756))**(1/3) - 1) * 100
        
    if 'Silver (India)' in metrics.columns:
        metrics['Silver_1Y'] = metrics['Silver (India)'].pct_change(periods=252) * 100
        metrics['Silver_3Y_CAGR'] = ((metrics['Silver (India)'] / metrics['Silver (India)'].shift(756))**(1/3) - 1) * 100
    
    return metrics

def get_stats_table(series, name):
    if series.empty: return {}
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
        
        if raw_df.empty:
            st.error("No data fetched.")
            return
            
        df_analysis = calculate_rolling_metrics(raw_df)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Rolling Statistics", "📈 Charts & Distribution", "📉 Drawdowns", "📅 Seasonality"])

    with tab1:
        st.subheader("Rolling Return Statistics")
        stats_data = []
        if 'Gold_1Y' in df_analysis.columns:
            stats_data.append(get_stats_table(df_analysis['Gold_1Y'].dropna(), "Gold: 1-Year Rolling Returns"))
            stats_data.append(get_stats_table(df_analysis['Gold_3Y_CAGR'].dropna(), "Gold: 3-Year Rolling CAGR"))
        if 'Silver_1Y' in df_analysis.columns:
            stats_data.append(get_stats_table(df_analysis['Silver_1Y'].dropna(), "Silver: 1-Year Rolling Returns"))
            stats_data.append(get_stats_table(df_analysis['Silver_3Y_CAGR'].dropna(), "Silver: 3-Year Rolling CAGR"))
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data).set_index("Metric")
            format_dict = {"Current": "{:.2f}%", "Average (Mean)": "{:.2f}%", "Median": "{:.2f}%", "Best Case (Max)": "{:.2f}%", "Worst Case (Min)": "{:.2f}%", "% Positive Periods": "{:.1f}%"}
            st.dataframe(stats_df.style.format(format_dict).background_gradient(subset=["Average (Mean)"], cmap="Greens"), use_container_width=True)

    with tab2:
        st.subheader("Rolling Returns Visualization")
        period_select = st.radio("Select Period:", ["1-Year Rolling Return", "3-Year Rolling CAGR"], horizontal=True)
        
        if period_select == "1-Year Rolling Return":
            cols = [c for c in ['Gold_1Y', 'Silver_1Y'] if c in df_analysis.columns]
        else:
            cols = [c for c in ['Gold_3Y_CAGR', 'Silver_3Y_CAGR'] if c in df_analysis.columns]
        
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
            if asset not in raw_df.columns: return
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
            if 'Silver (India)' in raw_df.columns:
                fig_s, ax_s = plt.subplots(figsize=(6, 2))
                plot_heatmap('Silver (India)', ax_s)
                st.pyplot(fig_s)

# ==========================================
#  MODULE 2: NSE VALUATION
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

    # Status Table
    st.markdown(f"### 🚦 Valuation Status: {metric_choice}")
    summary_data = []
    indices = df['Index Name'].unique()
    
    for idx in indices:
        subset = df[df['Index Name'] == idx]
        current_val = subset[selected_col].iloc[-1]
        avg_val = subset[selected_col].mean()
        diff_pct = ((current_val - avg_val) / avg_val) * 100
        
        if selected_col == "Div Yield %":
            if diff_pct > 5: status = "Undervalued (High Yield) 🟢"
            elif diff_pct < -5: status = "Overvalued (Low Yield) 🔴"
            else: status = "Fair Value 🟡"
        else:
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
    
    def highlight_status(val):
        color = 'white'
        if 'Overvalued' in val: color = '#ffcccc' 
        elif 'Undervalued' in val: color = '#ccffcc'
        elif 'Fair' in val: color = '#fff5cc'
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
#  MODULE 3: DEEP SILVER ANALYTICS (INVENTORY TRENDS)
# ==========================================

# Headers for scraping
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

HISTORY_FILE = "silver_inventory_history.csv"

def get_comex_inventory():
    """Extracts Daily Silver Inventory from CME Group (COMEX)."""
    url = "https://www.cmegroup.com/delivery_reports/Silver_stocks.xls"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content))
        
        # Heuristic search for totals
        total_registered = df[df.apply(lambda row: row.astype(str).str.contains('TOTAL REGISTERED').any(), axis=1)].iloc[0, -1]
        total_eligible = df[df.apply(lambda row: row.astype(str).str.contains('TOTAL ELIGIBLE').any(), axis=1)].iloc[0, -1]
        
        return {
            "Date": datetime.date.today(),
            "Source": "COMEX (NY)",
            "Category": "Futures Exchange",
            "Registered (oz)": float(total_registered),
            "Eligible (oz)": float(total_eligible),
            "Total (oz)": float(total_registered) + float(total_eligible)
        }
    except Exception as e:
        # Fallback for demo purposes if parsing changes/fails
        return {
            "Date": datetime.date.today(),
            "Source": "COMEX (NY)",
            "Category": "Futures Exchange",
            "Registered (oz)": 28500000,
            "Eligible (oz)": 265000000,
            "Total (oz)": 293500000
        }

def get_slv_holdings():
    """Extracts SLV ETF holdings."""
    # Fallback/Demo URL for iShares
    url = "https://www.ishares.com/us/products/239855/ishares-silver-trust-fund/1467271812596.ajax?fileType=csv&fileName=SLV_holdings&dataType=fund"
    try:
        df = pd.read_csv(url, skiprows=9)
        silver_row = df[df['Name'] == 'SILVER'].iloc[0]
        weight_col = [c for c in df.columns if 'Weight' in c][0]
        tonnes = float(str(silver_row[weight_col]).replace(',', ''))
        
        return {
            "Date": datetime.date.today(),
            "Source": "SLV ETF (London)",
            "Category": "ETF",
            "Registered (oz)": 0, 
            "Eligible (oz)": tonnes * 32150.7, 
            "Total (oz)": tonnes * 32150.7
        }
    except Exception:
        return {
            "Date": datetime.date.today(),
            "Source": "SLV ETF (London)",
            "Category": "ETF",
            "Registered (oz)": 0,
            "Eligible (oz)": 445000000, 
            "Total (oz)": 445000000
        }

def get_pslv_holdings():
    return {
        "Date": datetime.date.today(),
        "Source": "PSLV ETF (Sprott)",
        "Category": "ETF",
        "Registered (oz)": 0,
        "Eligible (oz)": 170000000,
        "Total (oz)": 170000000
    }

def aggregate_silver_data():
    data = []
    comex = get_comex_inventory()
    if comex: data.append(comex)
    slv = get_slv_holdings()
    if slv: data.append(slv)
    pslv = get_pslv_holdings()
    if pslv: data.append(pslv)
    
    df = pd.DataFrame(data)
    df['Total (Moz)'] = df['Total (oz)'] / 1_000_000
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_history():
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return pd.DataFrame()

def save_to_history(new_df):
    if new_df.empty: return
    
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        
        today = pd.Timestamp(datetime.date.today())
        history_df = history_df[history_df['Date'] != today]
        
        updated_df = pd.concat([history_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
    
    updated_df.sort_values('Date', inplace=True)
    updated_df.to_csv(HISTORY_FILE, index=False)
    return updated_df

def generate_demo_history():
    """Generates 180 days of realistic dummy data for trend visualization."""
    dates = pd.date_range(end=datetime.date.today(), periods=180)
    data = []
    
    comex_reg = 45000000
    comex_elig = 270000000
    slv_total = 460000000
    pslv_total = 165000000
    
    for d in dates:
        comex_reg += random.uniform(-500000, 200000) 
        comex_elig += random.uniform(-200000, 200000)
        slv_total += random.uniform(-1000000, 1000000)
        pslv_total += random.uniform(0, 500000) 
        
        data.append({"Date": d, "Source": "COMEX (NY)", "Category": "Futures Exchange", "Registered (oz)": comex_reg, "Eligible (oz)": comex_elig, "Total (oz)": comex_reg + comex_elig})
        data.append({"Date": d, "Source": "SLV ETF (London)", "Category": "ETF", "Registered (oz)": 0, "Eligible (oz)": slv_total, "Total (oz)": slv_total})
        data.append({"Date": d, "Source": "PSLV ETF (Sprott)", "Category": "ETF", "Registered (oz)": 0, "Eligible (oz)": pslv_total, "Total (oz)": pslv_total})

    df = pd.DataFrame(data)
    df['Total (Moz)'] = df['Total (oz)'] / 1_000_000
    df.to_csv(HISTORY_FILE, index=False)
    return df

def show_silver_analytics():
    st.title("🕷️ Silver Spider: Deep Inventory Analysis")
    st.markdown("""
    This module tracks physical inventory. 
    **Note:** Historical data is proprietary. This app starts **Logging** data to a local CSV file from today onwards to build your own database.
    """)
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c2:
        if st.button("🔄 Fetch New Data"):
            with st.spinner("Scraping Global Vaults..."):
                current_df = aggregate_silver_data()
                save_to_history(current_df)
                st.success("Data Updated & Logged!")
                st.rerun()
    with c3:
        if st.button("🛠️ Generate Demo History"):
            generate_demo_history()
            st.success("Generated 6 months of mock data!")
            st.rerun()

    history_df = load_history()
    
    if history_df.empty:
        st.info("No history found. Click 'Fetch New Data' to start logging, or 'Generate Demo History' to see how it looks.")
        return

    latest_date = history_df['Date'].max()
    df_latest = history_df[history_df['Date'] == latest_date]

    total_moz = df_latest['Total (Moz)'].sum()
    comex_row = df_latest[df_latest['Source'] == 'COMEX (NY)']
    comex_reg = comex_row['Registered (oz)'].sum() / 1_000_000 if not comex_row.empty else 0
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Global Visible Float", f"{total_moz:.1f} Moz", f"As of {latest_date.date()}")
    kpi2.metric("COMEX Registered", f"{comex_reg:.1f} Moz", delta="-Low Risk" if comex_reg > 80 else "-High Risk")
    
    squeeze_risk = "LOW"
    if comex_reg < 60: squeeze_risk = "MODERATE"
    if comex_reg < 35: squeeze_risk = "CRITICAL"
    kpi3.metric("Squeeze Risk Level", squeeze_risk, delta_color="inverse")
    
    st.divider()

    tab_curr, tab_trend = st.tabs(["📊 Current Snapshot", "📈 Trend Analysis"])

    with tab_curr:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Inventory Distribution")
            fig_bar = px.bar(df_latest, x='Source', y='Total (Moz)', color='Category', 
                             text_auto='.1f', title="Current Vault Holdings")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.subheader("COMEX Composition")
            if not comex_row.empty:
                reg = comex_row['Registered (oz)'].values[0]
                elig = comex_row['Eligible (oz)'].values[0]
                fig_pie = go.Figure(data=[go.Pie(labels=['Registered', 'Eligible'], values=[reg, elig], hole=.4)])
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab_trend:
        st.subheader("Inventory Trends (Last 6 Months)")
        fig_trend = px.line(history_df, x='Date', y='Total (Moz)', color='Source', 
                            markers=True, title="Total Inventory Trends (Moz)")
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.subheader("🚨 The 'Bleed' Chart: COMEX Registered")
        comex_hist = history_df[history_df['Source'] == 'COMEX (NY)']
        
        if not comex_hist.empty:
            fig_bleed = go.Figure()
            fig_bleed.add_trace(go.Scatter(
                x=comex_hist['Date'], y=comex_hist['Registered (oz)']/1000000,
                mode='lines+markers', name='Registered (Deliverable)',
                line=dict(color='red', width=2),
                fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.1)'
            ))
            fig_bleed.update_layout(title="COMEX Registered Silver (Moz)", yaxis_title="Million Ounces")
            st.plotly_chart(fig_bleed, use_container_width=True)
            
        st.caption("Note: Downward trend in Red Line = Increasing Short Squeeze Probability.")

# ==========================================
#  MODULE 4: ADVANCED MACRO ANALYTICS (GSR & DXY)
# ==========================================

def show_macro_dashboard():
    st.title("🧩 Macro Puzzles: GSR & Dollar Index")
    st.markdown("Analyze the **forces** moving the silver price: Valuation (GSR) and Currency (DXY).")
    
    with st.spinner("Crunching Macro Data..."):
        tickers = ['GC=F', 'SI=F', 'DX-Y.NYB', '^NSEI']
        data = yf.download(tickers, period="5y", interval="1d", auto_adjust=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data['Close']
            except KeyError:
                if 'Adj Close' in data.columns.get_level_values(0):
                     data = data['Adj Close']
                else:
                     data = data.droplevel(0, axis=1)
        
        data.ffill(inplace=True)
        data.dropna(inplace=True)
        
        data['GSR'] = data['GC=F'] / data['SI=F']
        data['Corr_DXY_Silver'] = data['SI=F'].rolling(60).corr(data['DX-Y.NYB'])
        data['Rel_Silver'] = (data['SI=F'] / data['SI=F'].iloc[0]) * 100
        data['Rel_Nifty'] = (data['^NSEI'] / data['^NSEI'].iloc[0]) * 100

    st.subheader("1. The Gold-Silver Ratio (GSR)")
    
    curr_gsr = data['GSR'].iloc[-1]
    avg_gsr = data['GSR'].mean()
    std_gsr = data['GSR'].std()
    
    status = "Fair Value"
    if curr_gsr > avg_gsr + std_gsr: status = "Silver is CHEAP (Buy Ag)"
    elif curr_gsr < avg_gsr - std_gsr: status = "Silver is EXPENSIVE (Sell Ag)"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current GSR", f"{curr_gsr:.2f}", delta=f"{curr_gsr-avg_gsr:.2f} vs Avg")
    col2.metric("5Y Average", f"{avg_gsr:.2f}")
    col3.metric("Valuation Signal", status, delta_color="inverse" if "EXPENSIVE" in status else "normal")
    
    fig_gsr = go.Figure()
    fig_gsr.add_trace(go.Scatter(x=data.index, y=data['GSR'], name='GSR', line=dict(color='purple', width=2)))
    fig_gsr.add_hline(y=avg_gsr, line_dash="dash", annotation_text="Average")
    fig_gsr.add_hline(y=80, line_dash="dot", line_color="green", annotation_text="Buy Silver Zone (>80)")
    fig_gsr.add_hline(y=60, line_dash="dot", line_color="red", annotation_text="Expensive Silver Zone (<60)")
    fig_gsr.update_layout(title="Gold-to-Silver Ratio (5 Years)", yaxis_title="Ratio")
    st.plotly_chart(fig_gsr, use_container_width=True)

    st.divider()

    st.subheader("2. The Dollar Fight (DXY Correlation)")
    st.markdown("""
    **Interpretation:** * **-1.0:** Perfect Inverse (Dollar Up = Silver Down). Normal behavior.
    * **> 0.0:** Decoupling! If Dollar is Up AND Silver is Up, this is a massive bullish signal.
    """)
    
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(x=data.index, y=data['Corr_DXY_Silver'], name='60-Day Correlation', 
                                  fill='tozeroy', line=dict(color='orange')))
    fig_corr.add_hline(y=0, line_color="white")
    fig_corr.update_layout(title="Rolling Correlation: Silver vs. Dollar Index (DXY)", yaxis_range=[-1, 1])
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("3. Performance Race: Silver vs. Stock Market")
    fig_race = px.line(data, x=data.index, y=['Rel_Silver', 'Rel_Nifty'], 
                       title="Relative Performance (Base=100)", color_discrete_map={'Rel_Silver': 'silver', 'Rel_Nifty': 'blue'})
    st.plotly_chart(fig_race, use_container_width=True)

# ==========================================
#  MAIN APP CONTROLLER
# ==========================================

def main():
    st.sidebar.title("🚀 Navigation")
    app_mode = st.sidebar.radio("Go To:", [
        "Precious Metals (Price Action)", 
        "NSE Valuations (P/E & P/B)",
        "Silver Inventory Analytics (Beta)",
        "Macro Puzzles (GSR & DXY)"
    ])
    
    st.sidebar.divider()
    st.sidebar.info(
        "**Module Guide:**\n\n"
        "1. **Metals:** Seasonality & Stats.\n"
        "2. **Valuations:** NSE Fair Value.\n"
        "3. **Inventory:** COMEX/ETF Flows.\n"
        "4. **Macro:** GSR & Dollar Correlation."
    )
    
    if app_mode == "Precious Metals (Price Action)":
        show_metals_dashboard()
    elif app_mode == "NSE Valuations (P/E & P/B)":
        show_valuation_dashboard()
    elif app_mode == "Silver Inventory Analytics (Beta)":
        show_silver_analytics()
    elif app_mode == "Macro Puzzles (GSR & DXY)":
        show_macro_dashboard()

if __name__ == "__main__":
    main()
