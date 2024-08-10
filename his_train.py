import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle

# 1. 讀取歷史數據，並解析時間列
df = pd.read_csv('wif_data.csv', parse_dates=['Open Time'])

# 2. 檢查並確認列名
print("列名如下：", df.columns)

# 3. 設置時間列為索引
df.set_index('Open Time', inplace=True)

# 4. 每秒生成一次特徵 (如果你想要按分鐘操作，也可以直接用現有數據)
# 不需要重採樣因為每筆數據已經是按分鐘記錄的
df['return'] = df['Close'].pct_change()  # 計算回報率
df['5ma'] = df['Close'].rolling(window=5).mean()  # 5分鐘移動平均線
df['20ma'] = df['Close'].rolling(window=20).mean()  # 20分鐘移動平均線
df['std'] = df['Close'].rolling(window=5).std()  # 5分鐘標準差
df.dropna(inplace=True)  # 移除缺失值

# 5. 預測目標變數
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # 定義目標變數
df.dropna(inplace=True)

# 6. 提取特徵和目標變數
X = df[['return', '5ma', '20ma', 'std']]
y = df['target']

# 7. 特徵縮放
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 8. 初始化SGDClassifier並進行訓練
# 將 loss 參數從 'log' 改為 'log_loss'
model = SGDClassifier(loss='log_loss', learning_rate='optimal', random_state=1)
model.fit(X_scaled, y)

# 9. 保存模型和縮放器
with open('sgd_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("模型已訓練並保存。")
