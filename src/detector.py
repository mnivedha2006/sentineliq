import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies_statistical(df, column='conversion_rate', window=24, threshold=3.0):
    # Rolling median and MAD (Median Absolute Deviation)
    rolling_median = df[column].rolling(window=window, min_periods=1).median()
    rolling_mad = (df[column] - rolling_median).abs().rolling(window=window, min_periods=1).median()
    
    # Handle zero MAD safely
    rolling_mad = rolling_mad.replace(0, 1e-6)
    
    # Modified Z-score using MAD
    modified_z = 0.6745 * (df[column] - rolling_median) / rolling_mad
    anomalies = modified_z.abs() > threshold
    return anomalies

def detect_anomalies_isolation_forest(df, column='conversion_rate'):
    model = IsolationForest(contamination=0.05, random_state=42)
    values = df[[column]].fillna(0)
    preds = model.fit_predict(values)
    return preds == -1
