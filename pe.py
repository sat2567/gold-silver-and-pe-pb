import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="NSE Valuation Dashboard (Fixed)", layout="wide", page_icon="📊")
st.title("📊 NSE Valuation Dashboard: P/E & P/B (Corrected)")
st.markdown("Comparative analysis of **Nifty 50, Midcap 150, Smallcap 100/250, and Total Market**.")

# --- 1. DATA LOADING ---
@st.cache_data
def load_and_merge_data():
    # List of all files
    files = {
        # 2025-2026 Files
        "Nifty 50 (2025)": "NIFTY 50_Historical_PE_PB_DIV_Data_01012025to01012026.csv",
        "Nifty Midcap 150 (2025)": "NIFTY MIDCAP 150_Historical_PE_PB_DIV_Data_01012025to01012026.csv",
        "Nifty Smallcap 100 (2025)": "NIFTY SMALLCAP 250_Historical_PE_PB_DIV_Data_01012025to31122025.csv",
        "Nifty Total Market (2025)": "NIFTY TOTAL MARKET_Historical_PE_PB_DIV_Data_01012025to01012026.csv",
        
        # 2024 Files
        "Nifty 50 (2024)": "NIFTY 50_Historical_PE_PB_DIV_Data_01012024to31122024.csv",
        "Nifty Midcap 150 (2024)": "NIFTY MIDCAP 150_Historical_PE_PB_DIV_Data_01012024to31122024.csv",
        "Nifty Smallcap 250 (2024)": "NIFTY SMALLCAP 250_Historical_PE_PB_DIV_Data_01012024to31122024.csv",
        "Nifty Total Market (2024)": "NIFTY TOTAL MARKET_Historical_PE_PB_DIV_Data_01012024to31122024.csv"
    }
    
    combined_df = pd.DataFrame()
    
    for label, filename in files.items():
        try:
            df = pd.read_csv(filename)
            # Clean Columns
            df.columns = df.columns.str.strip().str.replace('"', '')
            
            # Ensure P/B exists
            if 'P/B' not in df.columns:
                continue

            # Parse Dates
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Standardize Index Name
                if 'IndexName' in df.columns:
                    df['Index Name'] = df['IndexName']
                elif 'Index Name' not in df.columns:
                    df['Index Name'] = label 

                combined_df = pd.concat([combined_df, df])
                
        except FileNotFoundError:
            st.error(f"❌ File not found: {filename}")
    
    # --- CRITICAL FIX: GLOBAL SORT ---
    # Sort the ENTIRE dataset by date so the timeline is 2024 -> 2025 -> 2026
    if not combined_df.empty:
        combined_df.sort_values(by=['Index Name', 'Date'], ascending=[True, True], inplace=True)
            
    return combined_df

# Load Data
df = load_and_merge_data()

if not df.empty:
    # --- 2. METRIC SELECTION ---
    st.sidebar.header("⚙️ Configuration")
    metric_choice = st.sidebar.radio("Select Metric:", ["P/E Ratio", "P/B Ratio", "Div Yield %"])
    col_map = {"P/E Ratio": "P/E", "P/B Ratio": "P/B", "Div Yield %": "Div Yield %"}
    selected_col = col_map[metric_choice]

    # --- 3. SUMMARY METRICS ---
    st.subheader(f"📋 Valuation Summary: {metric_choice}")
    
    # Group by Index and get stats
    # Since we globally sorted by Date above, .last() will now correctly grab Jan 2026
    summary = df.groupby('Index Name')[selected_col].agg(['last', 'mean', 'min', 'max', 'std']).reset_index()
    summary.columns = ['Index', 'Current', 'Average', 'Min', 'Max', 'Volatility']
    
    # Valuation Logic
    def get_status(row):
        curr = row['Current']
        avg = row['Average']
        if curr > avg * 1.05: return "Expensive 🔴"
        elif curr < avg * 0.95: return "Cheap 🟢"
        else: return "Fair 🟡"
    
    summary['Status'] = summary.apply(get_status, axis=1)
    
    # Display Table
    st.dataframe(
        summary.style.format({
            'Current': '{:.2f}', 'Average': '{:.2f}', 
            'Min': '{:.2f}', 'Max': '{:.2f}', 'Volatility': '{:.2f}'
        }).background_gradient(subset=['Current'], cmap="Reds"),
        use_container_width=True
    )
    
    # --- DEBUG CHECK (Optional) ---
    # Un-comment this if you want to verify the dates onscreen
    # st.write("Latest Data Points Used for 'Current' Value:")
    # st.write(df.groupby('Index Name').tail(1)[['Index Name', 'Date', selected_col]])

    # --- 4. VISUALIZATIONS ---
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "📊 Relative Value", "📉 Matrix"])
    
    with tab1:
        st.subheader(f"Historical Trend (2024-2026)")
        fig = px.line(df, x='Date', y=selected_col, color='Index Name', 
                      title=f"{metric_choice} Trend", height=500)
        fig.update_layout(hovermode="x unified", yaxis_title=metric_choice)
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Premium/Discount vs 2-Year Average")
        bar_data = summary.copy()
        bar_data['% Diff'] = ((bar_data['Current'] - bar_data['Average']) / bar_data['Average']) * 100
        fig_bar = px.bar(bar_data, x='Index', y='% Diff', color='% Diff',
                         color_continuous_scale="RdYlGn_r")
        fig_bar.add_hline(y=0, line_color="black")
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Valuation Matrix (Current P/E vs P/B)")
        latest_snapshot = df.groupby('Index Name').tail(1)
        fig_scatter = px.scatter(latest_snapshot, x='P/E', y='P/B', color='Index Name', size='P/E',
                                 text='Index Name', title="Risk vs Value Matrix")
        avg_pe = latest_snapshot['P/E'].mean()
        avg_pb = latest_snapshot['P/B'].mean()
        fig_scatter.add_vline(x=avg_pe, line_dash="dash", line_color="grey")
        fig_scatter.add_hline(y=avg_pb, line_dash="dash", line_color="grey")
        fig_scatter.update_traces(textposition='top center')
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.warning("No data found.")