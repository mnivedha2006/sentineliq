import pandas as pd
from datetime import timedelta

def correlate_events(df_metrics, df_events, anomaly_timestamps, window_hours=3):
    correlations = []
    for anom_time in anomaly_timestamps:
        window_start = anom_time - timedelta(hours=window_hours)
        window_end = anom_time + timedelta(hours=window_hours)
        
        # Find events within the time window
        mask = (df_events['event_time'] >= window_start) & (df_events['event_time'] <= window_end)
        nearby_events = df_events[mask]
        
        for _, event in nearby_events.iterrows():
            correlations.append({
                'anomaly_time': anom_time,
                'event_time': event['event_time'],
                'event_type': event['event_type'],
                'affected_component': event['affected_component'],
                'description': event['description'],
                'time_diff_hours': (anom_time - event['event_time']).total_seconds() / 3600
            })
    return pd.DataFrame(correlations)
