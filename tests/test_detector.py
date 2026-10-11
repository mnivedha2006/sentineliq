import pandas as pd
import numpy as np
from src.detector import detect_anomalies_statistical

def test_detect_anomalies_statistical():
    # Create dummy time series data with an obvious spike
    data = {'conversion_rate': [0.03, 0.03, 0.03, 0.03, 0.50]} # 0.50 is an anomaly
    df = pd.DataFrame(data)
    
    anomalies = detect_anomalies_statistical(df, column='conversion_rate', window=3, threshold=2.0)
    
    # The last element should be flagged as an anomaly
    assert anomalies.iloc[-1] == True
