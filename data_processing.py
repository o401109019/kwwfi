# data_processing.py
import pandas as pd
import numpy as np

def calculate_features(data):
    data['moving_average'] = data['close'].rolling(window=5).mean()
    # 其他特征提取
    return data