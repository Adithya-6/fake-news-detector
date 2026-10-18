"""
Retrain the fake-news classifier from local CSVs (data/True.csv, data/Fake.csv)
plus fresh real headlines from NewsAPI, then save the pipeline to
app/model/model.pkl.

Usage:
    export NEWSAPI_KEY="your_key"      # Windows: set NEWSAPI_KEY=your_key
    python fetch_and_update.py

Complexity (n = documents, L = avg tokens/doc, V = vocab size, k = LR iterations):
    cleaning + TF-IDF : O(n * L) time, O(nnz) memory
    LogisticRegression: O(k * nnz) time  (nnz ~ n * L)
"""

import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---- Paths are resolved relative to this file, so it works from any cwd ----
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "app" / "model" / "model.pkl"
LOG_PATH = BASE_DIR / "model_update_log.txt"

# FIX 1: the original `API_KEY = # ...` was a SyntaxError (and invites
# committing a secret). Read the key from an environment variable instead.
API_KEY = os.getenv("NEWSAPI_KEY", "")

MAX_PER_CLASS = 10_000  # cap per class to keep classes balanced and training fast

# FIX 2: True.csv articles start with "CITY (Reuters) - ...". After punctuation
# is stripped that becomes the token "reuters", which the model learns as a
# shortcut for "real" (and fails on NewsAPI / user input). Remove the prefix.
REUTERS_PREFIX = re.compile(r"^.{0,120}?\(reuters\)\s*-?\s*", re.IGNORECASE)


def clean_text(text):
    """Lowercase, strip URLs / @mentions / hashtags / punctuation."""
    text = "" if text is None else str(text)
    text = REUTERS_PREFIX.sub("", text)
    text = re.sub(r"http\S+|www\S+", "", text)  # URLs (https\S+ was redundant)
    text = re.sub(r"@\w+|#", "", text)          # mentions and hashtag symbols
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text) # punctuation / symbols
    return text.lower().strip()


def fetch_real_news():
    """Fetch current English headlines from NewsAPI; label them real (1)."""
    empty = pd.DataFrame({"text": [], "label": []})
    if not API_KEY:
        print("⚠️  NEWSAPI_KEY not set; skipping NewsAPI fetch.")
        return empty
    try:
        # FIX 3: timeout + HTTP status check; params dict instead of an f-string
        # so the key is URL-encoded properly.
        resp = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"language": "en", "pageSize": 100, "apiKey": API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"🚨 Error in NewsAPI: {exc}")
        return empty

    if payload.get("status") != "ok":  # NewsAPI reports errors in the JSON body
        print(f"🚨 NewsAPI error: {payload.get('message', 'unknown error')}")
        return empty

    texts = []
    for art in payload.get("articles", []):
        # FIX 4: NewsAPI returns explicit nulls, so .get("title", "") can still
        # be None and `None + " "` raised TypeError. Use `or ""`.
        title = art.get("title") or ""
        desc = art.get("description") or ""
        if desc:
            texts.append(clean_text(f"{title} {desc}"))
    print(f"📰 API fetched: {len(texts)} usable articles")
    return pd.DataFrame({"text": texts}).assign(label=1)


def load_local(csv_name, label):
    """Load a local CSV (needs 'title' and 'text' columns) with a fixed label."""
    df = pd.read_csv(DATA_DIR / csv_name)
    # FIX 5: astype(str) on NaN produced the literal word "nan" in the text.
    raw = df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
    return pd.DataFrame({"text": raw.map(clean_text)}).assign(label=label)


def tidy(df):
    """Drop empty and duplicate texts (duplicates leak between train and test)."""
    df = df[df["text"].str.len() > 0]
    return df.drop_duplicates(subset="text")


def retrain():
    real_api = tidy(fetch_real_news())
    real_local = tidy(load_local("True.csv", 1))
    fake_df = tidy(load_local("Fake.csv", 0))

    # FIX 6: the original concatenated API + local rows and then sampled, so
    # most fresh API articles were randomly thrown away. Keep every API row and
    # sample only the local rows to fill the per-class budget.
    local_quota = max(0, MAX_PER_CLASS - len(real_api))
    real_local = real_local.sample(min(len(real_local), local_quota), random_state=42)
    real_df = pd.concat([real_local, real_api], ignore_index=True)
    fake_df = fake_df.sample(min(len(fake_df), MAX_PER_CLASS), random_state=42)

    print(f"✅ Real samples: {len(real_df)} | Fake samples: {len(fake_df)}")
    df = pd.concat([real_df, fake_df], ignore_index=True)

    if df["label"].nunique() < 2:
        print("❌ Not enough class diversity. Abort.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], stratify=df["label"], test_size=0.2, random_state=42
    )

    # FIX 7: GradientBoostingClassifier on a ~100k-feature sparse TF-IDF matrix is
    # extremely slow, and the README says Logistic Regression. Use LR.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7, min_df=2,
                                  ngram_range=(1, 2), max_features=100_000,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    # FIX 8: the original discarded the test split, so accuracy was never checked.
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"🎯 Hold-out accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["fake", "real"]))

    # Write to a temp file and swap it in, so the Streamlit app never reads a
    # half-written model.pkl.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(pipeline, f)
    os.replace(tmp_path, MODEL_PATH)

    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(
            f"✅ Retrained on {datetime.now()} | Real: {len(real_df)} | "
            f"Fake: {len(fake_df)} | Acc: {acc:.4f}\n"
        )
    print("✅ Model saved successfully!")


if __name__ == "__main__":
    retrain()
