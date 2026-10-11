import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="SentinelIQ: Enterprise Operational Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Operational Control")
st.sidebar.markdown("---")

data_mode = st.sidebar.radio("Data Source Mode", ["Default Demo Data", "Upload Custom CSV", "Generate Custom Simulation"])

@st.cache_data
def load_and_validate_data(mode, uploaded_file=None, sim_hours=1440, base_rev=12000, base_err=0.01):
    df_m = pd.DataFrame()
    df_e = pd.DataFrame(columns=['timestamp', 'event_type', 'description'])
    
    if mode == "Upload Custom CSV" and uploaded_file is not None:
        try:
            df_m = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None, None
            
        if 'timestamp' not in df_m.columns:
            st.error("Validation Error: Missing required 'timestamp' column in uploaded CSV.")
            return None, None
            
        try:
            df_m['timestamp'] = pd.to_datetime(df_m['timestamp'])
        except Exception:
            st.error("Validation Error: Failed to parse 'timestamp' column into valid dates.")
            return None, None
            
        if df_m.empty:
            st.error("Validation Error: The uploaded CSV file is empty.")
            return None, None
            
        numeric_cols = df_m.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("Validation Error: No numeric metric columns found in the uploaded CSV.")
            return None, None
            
    elif mode == "Generate Custom Simulation":
        dates = pd.date_range(start="2026-01-01", periods=sim_hours, freq="H")
        df_m = pd.DataFrame({
            'timestamp': dates,
            'conversion_rate': np.random.normal(0.0035, 0.0005, sim_hours),
            'error_rate': np.random.normal(base_err, base_err*0.2, sim_hours),
            'avg_latency_ms': np.random.normal(240, 25, sim_hours),
            'revenue': np.random.normal(base_rev, base_rev*0.1, sim_hours),
            'orders': np.random.normal(base_rev/25, 20, sim_hours)
        })
    else:
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
        df_e = pd.DataFrame({
            'timestamp': [df_m['timestamp'].min() + timedelta(hours=120), df_m['timestamp'].min() + timedelta(hours=400)],
            'event_type': ['Deployment', 'Infrastructure Update'],
            'description': ['V2.4.1 Microservice Rollout', 'Database Migration & Indexing']
        })
        
    return df_m, df_e

uploaded_file = None
sim_hours = 1440
base_rev = 12000
base_err = 0.01

if data_mode == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Telemetry CSV", type=["csv"])
elif data_mode == "Generate Custom Simulation":
    st.sidebar.markdown("### Simulation Parameters")
    sim_hours = st.sidebar.slider("Hours to Simulate", 24, 2160, 1440)
    base_rev = st.sidebar.number_input("Baseline Target Revenue ($)", 5000, 100000, 12000)
    base_err = st.sidebar.slider("Baseline Error Rate", 0.001, 0.05, 0.01)

df_metrics, df_events = load_and_validate_data(data_mode, uploaded_file, sim_hours, base_rev, base_err)

if df_metrics is None:
    st.stop()

numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns available for analysis.")
    st.stop()

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

def estimate_revenue_impact(df, anomaly_mask):
    if 'revenue' not in df.columns:
        return 0.0
    normal_mean = df.loc[~anomaly_mask, 'revenue'].mean()
    anomalous_rev = df.loc[anomaly_mask, 'revenue']
    shortfall = (normal_mean - anomalous_rev).clip(lower=0).sum()
    return shortfall

revenue_risk = estimate_revenue_impact(df_metrics, df_metrics['stat_anomaly'])

def correlate_incidents(anomalies_df, events_df, window_hours=3):
    if anomalies_df.empty or events_df.empty:
        return pd.DataFrame(columns=['anomaly_time', 'event_type', 'description', 'time_delta_hrs'])
    
    correlated = []
    for _, anom in anomalies_df.iterrows():
        anom_time = anom['timestamp']
        for _, ev in events_df.iterrows():
            ev_time = ev['timestamp']
            delta_hrs = (anom_time - ev_time).total_seconds() / 3600.0
            if abs(delta_hrs) <= window_hours:
                correlated.append({
                    'anomaly_time': anom_time,
                    'event_type': ev['event_type'],
                    'description': ev['description'],
                    'time_delta_hrs': round(delta_hrs, 2)
                })
    return pd.DataFrame(correlated)

anomalies_df = df_metrics[df_metrics['stat_anomaly']]
correlated_events_df = correlate_incidents(anomalies_df, df_events, window_hours=3)

st.title("SentinelIQ: Enterprise Operational Intelligence Suite")
st.markdown("Real-time telemetry monitoring, dual-engine anomaly detection, and automated forensic post-mortems.")
st.markdown("---")

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
    
    col1.metric("Total Hours Monitored", f"{total_hours:,}")
    col2.metric("Statistical Flags", f"{stat_count}")
    col3.metric("ML Flags", f"{ml_count}")
    col4.metric("Estimated Revenue Risk", f"${revenue_risk:,.2f}")
    
    st.markdown("---")
    fig = px.line(df_metrics, x='timestamp', y=metric_col, title=f"Telemetry Stream: {metric_col}")
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
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
    st.subheader("Operational Event & Incident Time-Window Correlation (±3 Hours)")
    if not correlated_events_df.empty:
        st.success(f"Successfully correlated {len(correlated_events_df)} anomalies with infrastructure deployment events.")
        st.dataframe(correlated_events_df, use_container_width=True)
    else:
        st.warning("No operational events matched within the ±3 hour window of detected anomalies.")
        if not df_events.empty:
            st.markdown("**All Recorded System Events:**")
            st.dataframe(df_events, use_container_width=True)

with tab4:
    st.subheader("Automated Forensic Post-Mortem Report")
    
    findings_bullets = ""
    if not anomalies_df.empty:
        first_anom = anomalies_df.iloc[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        last_anom = anomalies_df.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        findings_bullets += f"1. **Detection Window:** Primary deviations for `{metric_col}` were recorded between `{first_anom}` and `{last_anom}`.\n"
    else:
        findings_bullets += "1. **Detection Window:** No critical statistical anomalies detected under current sensitivity parameters.\n"
        
    if not correlated_events_df.empty:
        ev_name = correlated_events_df.iloc[0]['event_type']
        findings_bullets += f"2. **Operational Correlation (Hypothesis):** Temporal proximity detected relative to system event `{ev_name}`. This indicates potential regression or resource contention.\n"
    else:
        findings_bullets += "2. **Operational Correlation (Hypothesis):** No direct temporal overlap with logged deployments; anomaly likely driven by upstream traffic fluctuations or external factors.\n"
        
    findings_bullets += f"3. **Financial Impact Assessment:** Estimated aggregate revenue shortfall evaluated at `${revenue_risk:,.2f}` based on rolling baseline variance modeling."

    report_text = f"""### Incident Investigation Report
* **Generated At:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
* **Analyzed Metric:** `{metric_col}`
* **Total Statistical Anomalies:** `{stat_count}`
* **Estimated Financial Risk:** `${revenue_risk:,.2f}`

---
#### Key Findings & Hypotheses
{findings_bullets}

---
#### Actionable Recommendations
* Immediate review of service deployment logs and rollback protocols if regression is confirmed.
* Tune sensitivity thresholds or expand rolling windows if false positives persist.
"""
    st.markdown(report_text)
    st.download_button("Download Report (Markdown)", data=report_text, file_name="sentineliq_forensic_report.md", mime="text/markdown")
