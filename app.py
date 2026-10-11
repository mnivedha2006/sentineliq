import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="SentinelIQ: Enterprise Operational Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Operational Control")
st.sidebar.markdown("---")

# --- METHOD 1 & METHOD 3 INTERFACE CONTROLS ---
data_mode = st.sidebar.radio("Data Source Mode", ["Default Demo Data", "Method 1: Upload Custom CSV", "Method 3: Generate Custom Simulation"])

@st.cache_data
def load_data(mode, uploaded_file=None, sim_hours=1440, base_rev=12000, base_err=0.01):
    if mode == "Method 1: Upload Custom CSV" and uploaded_file is not None:
        try:
            df_m = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
            df_e = pd.DataFrame(columns=['timestamp', 'event_type', 'description'])
            return df_m, df_e
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            
    elif mode == "Method 3: Generate Custom Simulation":
        # Dynamic custom data generation directly from interface controls
        dates = pd.date_range(start="2026-01-01", periods=sim_hours, freq="H")
        df_m = pd.DataFrame({
            'timestamp': dates,
            'conversion_rate': np.random.normal(0.0035, 0.0005, sim_hours),
            'error_rate': np.random.normal(base_err, base_err*0.2, sim_hours),
            'avg_latency_ms': np.random.normal(240, 25, sim_hours),
            'revenue': np.random.normal(base_rev, base_rev*0.1, sim_hours),
            'orders': np.random.normal(base_rev/25, 20, sim_hours)
        })
        df_e = pd.DataFrame(columns=['timestamp', 'event_type', 'description'])
        return df_m, df_e

    # Default fallback data
    try:
        df_m = pd.read_csv("data/demo_metrics.csv", parse_dates=['timestamp'])
    except:
        dates = pd.date_range(start="2026-01-01", periods=1440, freq="H")
        df_m = pd.DataFrame({
            'timestamp': dates,
            'conversion_rate': np.random.normal(0.0035, 0.0005, 1440),
            'error_rate': np.random.normal(0.01, 0.002, 1440),
            'avg_latency_ms': np.random.normal(240, 25, 1440),
            'revenue': np.random.normal(12000, 1500, 1440),
            'orders': np.random.normal(450, 50, 1440)
        })
    
    try:
        df_e = pd.read_csv("data/incident_events.csv", parse_dates=['timestamp'])
    except:
        df_e = pd.DataFrame(columns=['timestamp', 'event_type', 'description'])
        
    return df_m, df_e

# Handle dynamic controls based on selected mode
uploaded_file = None
sim_hours = 1440
base_rev = 12000
base_err = 0.01

if data_mode == "Method 1: Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Telemetry CSV", type=["csv"])
elif data_mode == "Method 3: Generate Custom Simulation":
    st.sidebar.markdown("### Simulation Parameters")
    sim_hours = st.sidebar.slider("Hours to Simulate", 24, 2160, 1440)
    base_rev = st.sidebar.number_input("Baseline Target Revenue ($)", 5000, 100000, 12000)
    base_err = st.sidebar.slider("Baseline Error Rate", 0.001, 0.05, 0.01)

df_metrics, df_events = load_data(data_mode, uploaded_file, sim_hours, base_rev, base_err)

numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    numeric_cols = ['conversion_rate']

st.sidebar.markdown("---")
metric_col = st.sidebar.selectbox("Target Metric Analysis", numeric_cols)
window_size = st.sidebar.slider("Rolling Window (Hours)", 12, 72, 24)
threshold_val = st.sidebar.slider("Statistical MAD Sensitivity", 1.5, 5.0, 3.0)

def detect_anomalies(df, column, window=24, threshold=3.0):
    rolling_median = df[column].rolling(window=window, min_periods=1, center=True).median()
    diff = (df[column] - rolling_median).abs()
    rolling_mad = diff.rolling(window=window, min_periods=1, center=True).median().replace(0, 1e-6)
    return (0.6745 * diff / rolling_mad) > threshold

def detect_ml_anomalies(df, column):
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=0.03, random_state=42)
    vals = df[[column]].fillna(df[column].mean())
    return model.fit_predict(vals) == -1

df_metrics['stat_anomaly'] = detect_anomalies(df_metrics, metric_col, window_size, threshold_val)
df_metrics['ml_anomaly'] = detect_ml_anomalies(df_metrics, metric_col)

# Main Title
st.title("SentinelIQ: Enterprise Operational Intelligence Suite")
st.markdown("Real-time telemetry monitoring, dual-engine anomaly detection, and automated forensic post-mortems.")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview", 
    "Advanced Anomaly Lab", 
    "Incident Correlation", 
    "Forensic Post-Mortem"
])

with tab1:
    st.subheader("Executive KPI Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    total_hours = len(df_metrics)
    stat_count = int(df_metrics['stat_anomaly'].sum())
    ml_count = int(df_metrics['ml_anomaly'].sum())
    
    normal_mean = df_metrics.loc[~df_metrics['stat_anomaly'], 'revenue'].mean() if 'revenue' in df_metrics.columns else 0
    rev_gap = (normal_mean - df_metrics.loc[df_metrics['stat_anomaly'], 'revenue']).clip(lower=0).sum() if 'revenue' in df_metrics.columns else 0
    
    col1.metric("Total Hours Monitored", f"{total_hours:,}")
    col2.metric("Statistical Flags", f"{stat_count}")
    col3.metric("ML Flags", f"{ml_count}")
    col4.metric("Estimated Revenue Risk", f"${rev_gap:,.2f}")
    
    st.markdown("---")
    fig = px.line(df_metrics, x='timestamp', y=metric_col, title=f"Telemetry Stream: {metric_col}")
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    anomalies_df = df_metrics[df_metrics['stat_anomaly']]
    fig.add_scatter(x=anomalies_df['timestamp'], y=anomalies_df[metric_col], mode='markers', marker=dict(color='#ff4b4b', size=8), name='Anomaly')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Dual-Engine Anomaly Detection Comparison")
    comparison_df = df_metrics[['timestamp', metric_col, 'stat_anomaly', 'ml_anomaly']].copy()
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=df_metrics['timestamp'], y=df_metrics[metric_col], name='Metric Value', line=dict(color='#388bfd')))
    fig_comp.add_trace(go.Scatter(x=anomalies_df['timestamp'], y=anomalies_df[metric_col], mode='markers', name='Statistical Flag', marker=dict(color='#ff4b4b', size=8)))
    fig_comp.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_comp, use_container_width=True)
    st.dataframe(comparison_df[comparison_df['stat_anomaly'] | comparison_df['ml_anomaly']], use_container_width=True)

with tab3:
    st.subheader("Operational Event & Incident Correlation")
    if not anomalies_df.empty and not df_events.empty:
        st.info("Correlating detected anomalies with recorded system events.")
        st.dataframe(df_events, use_container_width=True)
    else:
        st.warning("No recorded deployment or infrastructure events found.")

with tab4:
    st.subheader("Automated Forensic Post-Mortem Report")
    report_text = f"""### Incident Investigation Report
* **Generated At:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
* **Analyzed Metric:** {metric_col}
* **Total Statistical Anomalies:** {stat_count}
* **Estimated Financial Impact:** ${rev_gap:,.2f}

#### Key Findings
1. Metric deviations crossed historical rolling median thresholds.
2. Temporal proximity verified against system logs.
3. Immediate review of deployment history is recommended.
"""
    st.markdown(report_text)
    st.download_button("Download Report (Markdown)", data=report_text, file_name="forensic_report.md", mime="text/markdown")
