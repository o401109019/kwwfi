import pandas as pd
import time
import requests
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

# 文件名
filename = 'wif_data.csv'

# 读取现有数据
try:
    existing_data = pd.read_csv(filename, parse_dates=['Open Time'])
    last_update_time = existing_data['Open Time'].max()
    last_update_timestamp = int(last_update_time.timestamp() * 1000)
except Exception as e:
    print(f"读取文件出错: {e}")
    exit(1)

# 当前时间
current_time = datetime.now(tz=timezone.utc)
current_timestamp = int(current_time.timestamp() * 1000)

# 下载数据的时间间隔（500个1分钟K线的数据）
interval = 500 * 60 * 1000  # 500分钟

# 台北时间的时区差（UTC+8）
taipei_tz = timezone(timedelta(hours=8))

# 获取每一列的最大小数位数
def get_decimal_places(series):
    decimal_places = series.astype(str).apply(lambda x: len(x.split('.')[1]) if '.' in x else 0)
    return decimal_places.max()

# 获取原始数据每列的小数位数
decimal_format = {col: f"{{:.{get_decimal_places(existing_data[col])}f}}" for col in existing_data.columns if existing_data[col].dtype == 'float64'}

# 初始化数据列表
all_data = []

# 获取数据
for current_time in tqdm(range(last_update_timestamp, current_timestamp, interval), desc="Downloading WIF/USDT Data"):
    try:
        url = f'https://api.binance.com/api/v3/klines?symbol=WIFUSDT&interval=1m&startTime={current_time}&endTime={current_time + interval - 1}&limit=500'
        response = requests.get(url)
        data = response.json()
        if data:
            df = pd.DataFrame(data, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                                             'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
                                             'Taker Buy Quote Asset Volume', 'Ignore'])
            # 转换时间为台北时间
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(taipei_tz)
            
            # 按原始数据格式化小数位数
            for col, fmt in decimal_format.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(float).map(lambda x: fmt.format(x))
                    except ValueError as e:
                        print(f"Error formatting column {col}: {e}")
                        continue
            
            all_data.append(df)
        time.sleep(0.1)  # 避免超过API限制
    except Exception as e:
        print(f"Error occurred: {e}")
        break

# 将新数据合并并保存为CSV
if all_data:
    new_data_df = pd.concat(all_data, ignore_index=True)
    combined_data_df = pd.concat([existing_data, new_data_df], ignore_index=True).drop_duplicates(subset=['Open Time'])
    combined_data_df.sort_values(by='Open Time', inplace=True)
    combined_data_df.to_csv(filename, index=False)
    print(f"数据已更新至 {filename}")
else:
    print("没有新数据需要更新")
