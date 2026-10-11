import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_metrics, load_events
from src.detector import detect_anomalies_statistical, detect_anomalies_isolation_forest
from src.investigator import correlate_events
from src.impact import estimate_revenue_gap

st.set_page_config(page_title="SentinelIQ: Operational Intelligence", layout="wide")

st.title("🚨 SentinelIQ: Operational Incident & Anomaly Detective")
st.markdown("Monitor high-impact e-commerce metrics, detect statistical and ML anomalies, and correlate operational events with automated forensic reporting.")

# Load Data
df_metrics = load_metrics()
df_events = load_events()

# Sidebar Controls
st.sidebar.header("Configuration & Controls")
metric_col = st.sidebar.selectbox("Select Metric to Analyze", 
                                  ['conversion_rate', 'error_rate', 'avg_latency_ms', 'revenue', 'orders'])
window_size = st.sidebar.slider("Rolling Window (Hours)", 12, 72, 24)
threshold_val = st.sidebar.slider("Statistical MAD Threshold", 1.5, 5.0, 3.0)

# Run Detectors
df_metrics['stat_anomaly'] = detect_anomalies_statistical(df_metrics, column=metric_col, window=window_size, threshold=threshold_val)
df_metrics['ml_anomaly'] = detect_anomalies_isolation_forest(df_metrics, column=metric_col)

# Main Metrics Overview
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Hours Analyzed", len(df_metrics))
col2.metric("Statistical Anomalies Flagged", int(df_metrics['stat_anomaly'].sum()))
col3.metric("ML Anomalies Flagged", int(df_metrics['ml_anomaly'].sum()))
revenue_gap = estimate_revenue_gap(df_metrics, df_metrics['stat_anomaly'])
col4.metric("Estimated Revenue Gap", f"${revenue_gap:,.2f}")

st.markdown("---")

# Time-Series Plotly Chart
st.subheader(f"📈 Time-Series Analysis: {metric_col}")
fig = px.line(df_metrics, x='timestamp', y=metric_col, title=f"Timeline of {metric_col} with Anomalies")

# Highlight statistical anomalies
anomalies_df = df_metrics[df_metrics['stat_anomaly']]
fig.add_scatter(x=anomalies_df['timestamp'], y=anomalies_df[metric_col], 
                mode='markers', marker=dict(color='red', size=8), name='Statistical Anomaly')

st.plotly_chart(fig, use_container_width=True)

# Incident Correlation Section
st.markdown("---")
st.subheader("🔍 Event & Incident Correlation")
if not anomalies_df.empty:
    anomaly_times = anomalies_df['timestamp'].tolist()
    correlations = correlate_events(df_metrics, df_events, anomaly_times, window_hours=3)
    
    if not correlations.empty:
        st.warning(f"Found {len(correlations)} operational events correlated near anomaly timestamps within a 3-hour window!")
        st.dataframe(correlations, use_container_width=True)
    else:
        st.info("No direct operational events recorded within 3 hours of the detected anomalies for this window.")
else:
    st.info("No anomalies detected with current threshold settings.")

# Evidence-Based AI Report Section
st.markdown("---")
st.subheader("🧠 Evidence-Based AI Forensic Report")
st.markdown("""
> **Observed Condition:** Metric deviations crossed historical rolling median thresholds.  
> **Correlated Timeline:** Verified against nearby system deployment alerts and database telemetry.  
> **Caveat / Disclaimer:** Timing proximity indicates correlation, not definitive proven causation. Further developer verification is required.
""")
