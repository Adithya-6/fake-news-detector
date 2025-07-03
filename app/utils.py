import pickle
import os

def load_model():
    with open("app/model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("app/model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_news(text):
    model, vectorizer = load_model()  # Only if you're still separating them
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0].max()
    return "REAL" if pred == 1 else "FAKE", proba

