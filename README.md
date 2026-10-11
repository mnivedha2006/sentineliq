# SentinelIQ: Enterprise Operational Intelligence Suite

A production-grade operational intelligence SaaS platform designed for real-time telemetry monitoring, dual-engine anomaly detection, and automated forensic post-mortem reporting.

**Live Demo**: [https://sentineliq-cxgbpkk3jrqnwhusvpkbzs.streamlit.app/](https://sentineliq-cxgbpkk3jrqnwhusvpkbzs.streamlit.app/)

---

##  Overview
Modern digital infrastructure requires high-precision monitoring to mitigate downtime and revenue loss. **SentinelIQ** bridges the gap between complex metric telemetry and executive-level decision-making. It provides a Security Operations Center (SOC) dashboard with deep diagnostic capabilities, machine learning anomaly flags, and financial risk modeling.

---

##  Key Features
* **Dual-Engine Anomaly Detection**: Combines statistical Rolling Median Absolute Deviation (MAD) with unsupervised Machine Learning (`Isolation Forest`) to flag high-precision operational anomalies.
* **Flexible Data Ingestion Modes**: 
  * *Default Demo Data*: Pre-configured enterprise telemetry streams.
  * *Upload Custom CSV*: Seamlessly drop in custom telemetry datasets directly via the interface.
  * *Generate Custom Simulation*: Adjust real-time simulation parameters and sliders on the fly to stress-test system resilience.
* **Business Impact Quantification**: Automatically calculates estimated financial revenue gaps during critical operational downtime events.
* **Automated Forensic Post-Mortem**: Generates dynamic markdown incident reports with instant management export capabilities.
* **Enterprise UI/UX**: Built with a native enterprise dark-mode theme, multi-tab layout, and high-performance Plotly visualizations.

---

##  Tech Stack
* **Language**: Python
* **Web Framework**: Streamlit
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-learn (Isolation Forest)
* **Visualizations**: Plotly (Dark Template)
* **Version Control & Cloud**: Git, GitHub, Streamlit Community Cloud

---

## 📊 Project Architecture
