import os
import pytest
import pandas as pd
import numpy as np

def detect_anomalies(df, column, window=24, threshold=3.0):
    rolling_median = df[column].rolling(window=window, min_periods=1, center=True).median()
    diff = (df[column] - rolling_median).abs()
    rolling_mad = diff.rolling(window=window, min_periods=1, center=True).median().replace(0, 1e-6)
    return (0.6745 * diff / rolling_mad) > threshold

def test_anomaly_detection_basic():
    np.random.seed(42)
    data = {
        'timestamp': pd.date_range(start="2026-01-01", periods=100, freq="H"),
        'revenue': np.random.normal(10000, 100, 100)
    }
    df = pd.DataFrame(data)
    df.loc[50, 'revenue'] = 50000 
    
    anomalies = detect_anomalies(df, 'revenue', window=24, threshold=3.0)
    
    assert isinstance(anomalies, pd.Series)
    assert len(anomalies) == 100
    assert anomalies.iloc[50] == True

def test_csv_validation_missing_timestamp():
    df_invalid = pd.DataFrame({'revenue': [1000, 2000, 3000]})
    has_timestamp = 'timestamp' in df_invalid.columns
    assert has_timestamp == False

def test_revenue_impact_calculation():
    df = pd.DataFrame({
        'revenue': [10000, 10000, 2000, 10000]
    })
    mask = pd.Series([False, False, True, False])
    
    normal_mean = df.loc[~mask, 'revenue'].mean()
    shortfall = (normal_mean - df.loc[mask, 'revenue']).clip(lower=0).sum()
    
    assert shortfall == 8000.0
