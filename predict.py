# predict.py
from joblib import load
from data_processing import calculate_features

def predict(data):
    scaler = load('scaler.joblib')
    model = load('svm_model.joblib')

    features = calculate_features(data)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    return prediction