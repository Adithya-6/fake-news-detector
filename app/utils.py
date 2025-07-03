import pickle
import os

def load_model():
    model_path = os.path.join("app", "model", "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_news(text):
    model = load_model()
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0].max()
    return "REAL" if pred == 1 else "FAKE", proba
