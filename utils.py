"""
Shared helpers for the Streamlit app AND the training script.

clean_text lives here (single source of truth) so training and inference
always preprocess text identically; fetch_and_update.py imports it.
"""

import pickle
import re
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pkl"

# True.csv articles start with "CITY (Reuters) - ...". Strip it so the model
# doesn't learn the token "reuters" as a shortcut for "real".
_REUTERS_PREFIX = re.compile(r"^.{0,120}?\(reuters\)\s*-?\s*", re.IGNORECASE)


def clean_text(text):
    """Lowercase, strip Reuters prefix, URLs, @mentions, hashtags, punctuation."""
    text = "" if text is None else str(text)
    text = _REUTERS_PREFIX.sub("", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
    return text.lower().strip()


@lru_cache(maxsize=1)
def load_model():
    """Load the pickled sklearn pipeline once per process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python fetch_and_update.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_news(text):
    """Return ("REAL" | "FAKE", confidence in [0.5, 1.0]) for the predicted class."""
    model = load_model()
    proba = model.predict_proba([clean_text(text)])[0]  # O(L * nnz-features) per call
    idx = int(proba.argmax())
    label = "REAL" if model.classes_[idx] == 1 else "FAKE"  # 1 = real, 0 = fake
    return label, float(proba[idx])
