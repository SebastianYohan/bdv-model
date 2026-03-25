import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf-model.pkl")
model = joblib.load(MODEL_PATH)

def predict_bdv(ppm):
    X = [[ppm]]
    return model.predict(X)[0]