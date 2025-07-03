import pickle
import os

# ✅ Load the pipeline (model + vectorizer)
def load_model():
    with open("app/model/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# ✅ Predict using the pipeline
def predict_news(text):
    model = load_model()
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0].max()
    return "REAL" if pred == 1 else "FAKE", proba
