import websocket
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle

# 初始化全局變量
data_list = []
model = None
scaler = None

# 加載已訓練好的模型和縮放器
with open('sgd_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# 處理逐筆交易數據
def on_message(ws, message):
    global data_list, model, scaler

    # 解析數據
    data = json.loads(message)
    try:
        # 檢查數據格式是否包含'E'鍵
        if 'E' in data:
            event_time = datetime.fromtimestamp(data['E'] / 1000)
            price = float(data['p'])
            quantity = float(data['q'])

            # 將逐筆交易數據添加到列表中
            data_list.append([event_time, price, quantity])

            # 構建DataFrame
            df = pd.DataFrame(data_list, columns=['timestamp', 'price', 'quantity'])

            # 使用小寫的 's' 進行每秒重採樣
            df = df.resample('1s', on='timestamp').agg({
                'price': 'ohlc',
                'quantity': 'sum'
            }).dropna()

            if df.empty:
                print("Resampled DataFrame is empty. Skipping this iteration.")
                return

            # 特徵工程
            df['return'] = df['price']['close'].pct_change()
            df['5ma'] = df['price']['close'].rolling(window=5).mean()
            df['20ma'] = df['price']['close'].rolling(window=20).mean()
            df['std'] = df['price']['close'].rolling(window=5).std()
            df.dropna(inplace=True)

            if df.empty:
                print("Feature DataFrame is empty after dropna(). Skipping this iteration.")
                return

            # 提取特徵
            X = df[['return', '5ma', '20ma', 'std']]

            # 特徵縮放
            X_scaled = scaler.transform(X)

            # 預測下一筆交易的方向
            prediction = model.predict(X_scaled[-1].reshape(1, -1))
            direction = 'uP' if prediction[0] == 1 else 'dOwn'
            
            # 顯示預測結果
            prediction_time = event_time.strftime('%Y/%m/%d %H:%M:%S')
            next_time = (event_time + timedelta(seconds=1)).strftime('%Y/%m/%d %H:%M:%S')
            result = f"{prediction_time} to {next_time} : {direction}"
            print(result)
        else:
            print("Unexpected message format: Missing 'E' key.")
    except KeyError as e:
        print(f"KeyError: {e} - Skipping this iteration")
    except Exception as e:
        print(f"Error: {e}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    ws.send(json.dumps({
        "method": "SUBSCRIBE",
        "params": [
            "wifusdt@trade"  # 訂閱WIFUSDT的逐筆交易流
        ],
        "id": 1
    }))

if __name__ == "__main__":
    # 初始化WebSocket連接
    ws = websocket.WebSocketApp("wss://stream.binance.com:9443/ws",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
