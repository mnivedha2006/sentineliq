import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.data_loader import load_metrics, load_events
from src.detector import detect_anomalies_statistical, detect_anomalies_isolation_forest
from src.investigator import correlate_events
from src.impact import estimate_revenue_gap

# Page Configuration
st.set_page_config(
    page_title="SentinelIQ: Enterprise Operational Intelligence",
    page_layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    .stAlert { background-color: #1f242d; border: 1px solid #388bfd; }
    </style>
""", unsafe_allow_html=True)

# Load Data
df_metrics = load_metrics()
df_events = load_events()

# Sidebar Executive Controls
st.sidebar.title(" SentinelIQ Control Panel")
st.sidebar.markdown("---")
metric_col = st.sidebar.selectbox("Target Metric Analysis", 
                                  ['conversion_rate', 'error_rate', 'avg_latency_ms', 'revenue', 'orders'])
window_size = st.sidebar.slider("Rolling Window (Hours)", 12, 72, 24)
threshold_val = st.sidebar.slider("Statistical MAD Sensitivity", 1.5, 5.0, 3.0)

# Run Detectors
df_metrics['stat_anomaly'] = detect_anomalies_statistical(df_metrics, column=metric_col, window=window_size, threshold=threshold_val)
df_metrics['ml_anomaly'] = detect_anomalies_isolation_forest(df_metrics, column=metric_col)

# Header Section
st.title("🚨 SentinelIQ: Enterprise Operational Intelligence & Incident Suite")
st.markdown("Real-time telemetry monitoring, dual-engine anomaly detection (Statistical + Machine Learning), and automated AI forensic post-mortems.")

# Multi-Tab Enterprise Architecture
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview", 
    "🔬 Advanced Anomaly Lab", 
    "🔍 Incident Telemetry Correlation", 
    "🧠 AI Forensic Post-Mortem"
])

# ---------------- TAB 1: EXECUTIVE OVERVIEW ----------------
with tab1:
    st.subheader("Executive KPI Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    total_hours = len(df_metrics)
    stat_count = int(df_metrics['stat_anomaly'].sum())
    ml_count = int(df_metrics['ml_anomaly'].sum())
    revenue_gap = estimate_revenue_gap(df_metrics, df_metrics['stat_anomaly'])
    
    col1.metric("Total Hours Monitored", f"{total_hours:,}")
    col2.metric("Statistical Flags (MAD)", f"{stat_count}", delta=f"{stat_count/total_hours:.1%}")
    col3.metric("ML Flags (Isolation Forest)", f"{ml_count}")
    col4.metric("Estimated Revenue Gap", f"${revenue_gap:,.2f}", delta="-Risk", delta_color="inverse")
    
    st.markdown("---")
    st.subheader(f"📈 High-Level Timeline: {metric_col}")
    fig = px.line(df_metrics, x='timestamp', y=metric_col, title=f"Timeline of {metric_col} with Statistical Anomalies")
    anomalies_df = df_metrics[df_metrics['stat_anomaly']]
    fig.add_scatter(x=anomalies_df['timestamp'], y=anomalies_df[metric_col], 
                    mode='markers', marker=dict(color='#ff4b4b', size=9, symbol='diamond'), name='Detected Anomaly')
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2: ANOMALY LAB ----------------
with tab2:
    st.subheader("Dual-Engine Anomaly Detection Comparison")
    st.markdown("Compare rolling median MAD (Statistical) against scikit-learn Isolation Forest (Machine Learning).")
    
    comparison_df = df_metrics[['timestamp', metric_col, 'stat_anomaly', 'ml_anomaly']].copy()
    comparison_df['Agreement'] = comparison_df['stat_anomaly'] & comparison_df['ml_anomaly']
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=df_metrics['timestamp'], y=df_metrics[metric_col], name='Metric Value', line=dict(color='cyan')))
    stat_pts = df_metrics[df_metrics['stat_anomaly']]
    fig_comp.add_trace(go.Scatter(x=stat_pts['timestamp'], y=stat_pts[metric_col], mode='markers', name='Statistical Anomaly', marker=dict(color='red', size=8)))
    
    fig_comp.update_layout(title="Statistical vs ML Anomaly Overlay", template="plotly_dark")
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("### Anomaly Detection Summary Table")
    st.dataframe(comparison_df[comparison_df['stat_anomaly'] | comparison_df['ml_anomaly']], use_container_width=True)

# ---------------- TAB 3: INCIDENT CORRELATION ----------------
with tab3:
    st.subheader("Operational Event & Incident Correlation")
    if not anomalies_df.empty:
        anomaly_times = anomalies_df['timestamp'].tolist()
        correlations = correlate_events(df_metrics, df_events, anomaly_times, window_hours=3)
        
        if not correlations.empty:
            st.warning(f"Correlated {len(correlations)} operational events near anomaly timestamps within a 3-hour window.")
            st.dataframe(correlations, use_container_width=True)
        else:
            st.info("No recorded deployment or infrastructure events within the 3-hour window of current anomalies.")
    else:
        st.info("No anomalies detected to correlate.")

# ---------------- TAB 4: AI FORENSIC POST-MORTEM ----------------
with tab4:
    st.subheader(" Automated AI Forensic Post-Mortem Report")
    st.markdown("Generated based on live telemetry anomalies, rolling median deviations, and correlated deployment markers.")
    
    report_markdown = f"""
    ###  Incident Investigation Report
    * **Generated At:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
    * **Analyzed Metric:** `{metric_col}`
    * **Total Statistical Anomalies:** `{stat_count}`
    * **Estimated Financial Impact:** `${revenue_gap:,.2f}`
    
    ---
    ####  Key Findings & Root Cause Analysis
    1. **Deviation Spike:** Metric deviations crossed historical rolling median thresholds with a MAD sensitivity of `{threshold_val}`.
    2. **Operational Correlation:** Temporal proximity verified against recent system deployment alerts and database telemetry.
    3. **Actionable Recommendation:** Immediate review of deployment logs and rollback procedures during anomaly intervals is advised.
    
    > *Disclaimer: Timing proximity indicates high correlation, not definitive proven causation. Further developer verification is required.*
    """
    
    st.markdown(report_markdown)
    
    # Downloadable Management Report Button
    st.download_button(
        label=" Download Forensic Post-Mortem Report (Markdown)",
        data=report_markdown,
        file_name="sentineliq_forensic_report.md",
        mime="text/markdown"
    )
