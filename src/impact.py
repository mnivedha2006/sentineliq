import pandas as pd

def estimate_revenue_gap(df, anomaly_mask, revenue_col='revenue', baseline_window=24):
    # Calculate rolling baseline revenue
    baseline = df[revenue_col].rolling(window=baseline_window, min_periods=1).median()
    gap = baseline - df[revenue_col]
    
    # Only calculate positive gaps during anomalies
    total_gap = gap[anomaly_mask & (gap > 0)].sum()
    return total_gap
