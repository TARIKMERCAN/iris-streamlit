import joblib
import numpy as np
from sklearn.datasets import load_iris

model = joblib.load("app/model.joblib")
target_names = load_iris().target_names

def predict(features):
    X = np.array(features, dtype=float).reshape(1, -1)
    class_idx = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return class_idx, target_names[class_idx], proba
