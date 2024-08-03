# train_model.py
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_scaled, y)

    dump(scaler, 'scaler.joblib')
    dump(model, 'svm_model.joblib')