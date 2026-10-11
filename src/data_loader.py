import pandas as pd

def load_metrics(filepath="data/demo_metrics.csv"):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def load_events(filepath="data/incident_events.csv"):
    df = pd.read_csv(filepath)
    df['event_time'] = pd.to_datetime(df['event_time'])
    return df
