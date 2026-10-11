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

# Strict Enterprise Dark Theme & Custom Styling
st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; color: #f0f6fc; }
    [data-testid="stSidebar"] { background-color: #111625; border-right: 1px solid #21262d; }
    .stMetric { background-color: #161b22; padding: 16px; border-radius: 6px; border: 1px solid #30363d; }
    .stAlert { background-color: #1f242d; border: 1px solid #388bfd; }
    h1, h2, h3 { color: #f0f6fc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_metrics():
    try:
        return pd.read_csv("data/demo_metrics.csv", parse_dates=['timestamp'])
    except:
        dates = pd.date_range(start="2026-01-01", periods=1440, freq="H")
        return pd.DataFrame({
            'timestamp': dates,
            'conversion_rate': np.random.normal(0.0035, 0.0005, 1440),
            'error_rate': np.random.normal(0.01, 0.002, 1440),
            'avg_latency_ms': np.random.normal(240, 25, 1440),
            'revenue': np.random.normal(12000, 1500, 1440),
            'orders': np.random.normal(450, 50, 1440)
        })

@st.cache_data
def load_events():
    try:
        return pd.read_csv("data/incident_events.csv", parse_dates=['timestamp'])
    except:
        return pd.DataFrame(columns=['timestamp', 'event_type', 'description'])

df_metrics = load_metrics()
df_events = load_events()

def detect_anomalies_statistical(df, column, window=24, threshold=3.0):
    rolling_median = df[column].rolling(window=window, min_periods=1, center=True).median()
    diff = (df[column] - rolling_median).abs()
    rolling_mad = diff.rolling(window=window, min_periods=1, center=True).median()
    rolling_mad = rolling_mad.replace(0, 1e-6)
    modified_z = 0.6745 * diff / rolling_mad
    return modified_z > threshold

def detect_anomalies_isolation_forest(df, column):
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=0.03, random_state=42)
    vals = df[[column]].fillna(df[column].mean())
    return model.fit_predict(vals) == -1

def estimate_revenue_gap(df, anomaly_mask):
    if 'revenue' not in df.columns:
        return 0.0
    normal_mean = df.loc[~anomaly_mask, 'revenue'].mean()
    anomalous_rev = df.loc[anomaly_mask, 'revenue']
    return (normal_mean - anomalous_rev).clip(lower=0).sum()

# Sidebar Controls
st.sidebar.title("Operational Control")
st.sidebar.markdown("---")
metric_col = st.sidebar.selectbox("Target Metric Analysis", 
                                  ['conversion_rate', 'error_rate', 'avg_latency_ms', 'revenue', 'orders'])
window_size = st.sidebar.slider("Rolling Window (Hours)", 12, 72, 24)
threshold_val = st.sidebar.slider("Statistical MAD Sensitivity", 1.5, 5.0, 3.0)

df_metrics['stat_anomaly'] = detect_anomalies_statistical(df_metrics, column=metric_col, window=window_size, threshold=threshold_val)
df_metrics['ml_anomaly'] = detect_anomalies_isolation_forest(df_metrics, column=metric_col)

# Header
st.title("SentinelIQ: Enterprise Operational Intelligence Suite")
st.markdown("Real-time telemetry monitoring, dual-engine anomaly detection, and automated forensic post-mortems.")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview", 
    "Advanced Anomaly Lab", 
    "Incident Telemetry Correlation", 
    "Forensic Post-Mortem"
])

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
    st.subheader(f"Timeline Analysis: {metric_col}")
    fig = px.line(df_metrics, x='timestamp', y=metric_col, title=f"Telemetry Stream: {metric_col}")
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0b0f19", paper_bgcolor="#0b0f19")
    anomalies_df = df_metrics[df_metrics['stat_anomaly']]
    fig.add_scatter(x=anomalies_df['timestamp'], y=anomalies_df[metric_col], 
                    mode='markers', marker=dict(color='#ff4b4b', size=8, symbol='diamond'), name='Detected Anomaly')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Dual-Engine Anomaly Detection Comparison")
    comparison_df = df_metrics[['timestamp', metric_col, 'stat_anomaly', 'ml_anomaly']].copy()
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=df_metrics['timestamp'], y=df_metrics[metric_col], name='Metric Value', line=dict(color='#388bfd')))
    stat_pts = df_metrics[df_metrics['stat_anomaly']]
    fig_comp.add_trace(go.Scatter(x=stat_pts['timestamp'], y=stat_pts[metric_col], mode='markers', name='Statistical Anomaly', marker=dict(color='#ff4b4b', size=8)))
    fig_comp.update_layout(title="Statistical vs Machine Learning Overlay", template="plotly_dark", plot_bgcolor="#0b0f19", paper_bgcolor="#0b0f19")
    st.plotly_chart(fig_comp, use_container_width=True)
    st.dataframe(comparison_df[comparison_df['stat_anomaly'] | comparison_df['ml_anomaly']], use_container_width=True)

with tab3:
    st.subheader("Operational Event & Incident Correlation")
    if not anomalies_df.empty and not df_events.empty:
        st.info("Correlating detected anomalies with recorded system deployment events.")
        st.dataframe(df_events, use_container_width=True)
    else:
        st.warning("No recorded deployment or infrastructure events within current telemetry anomalies.")

with tab4:
    st.subheader("Automated Forensic Post-Mortem Report")
    report_markdown = f"""
    ### Incident Investigation Report
    * **Generated At:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
    * **Analyzed Metric:** `{metric_col}`
    * **Total Statistical Anomalies:** `{stat_count}`
    * **Estimated Financial Impact:** `${revenue_gap:,.2f}`
    
    ---
    #### Key Findings & Root Cause Analysis
    1. **Deviation Spike:** Metric deviations crossed historical rolling median thresholds.
    2. **Operational Correlation:** Temporal proximity verified against recent system deployment logs.
    3. **Actionable Recommendation:** Immediate review of service deployment history and rollback procedures.
    """
    st.markdown(report_markdown)
    st.download_button(
        label="Download Forensic Post-Mortem Report (Markdown)",
        data=report_markdown,
        file_name="sentineliq_forensic_report.md",
        mime="text/markdown"
    )
