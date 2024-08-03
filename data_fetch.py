import requests
import pandas as pd

def fetch_historical_data(api_url, symbol, interval):
    response = requests.get(f"{api_url}?symbol={symbol}&interval={interval}")
    data = response.json()
    df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def fetch_realtime_data(api_url, symbol):
    response = requests.get(f"{api_url}?symbol={symbol}")
    return response.json()