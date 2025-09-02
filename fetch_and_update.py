import pandas as pd
import requests
import pickle
import os
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier  # Better than LogisticRegression

API_KEY = # ✅ Replace with your API key

# 🔹 Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    return text.lower().strip()

# 🔹 Fetch real news from API
def fetch_real_news():
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=100&apiKey={API_KEY}"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        print(f"📰 API fetched: {len(articles)} articles")

        texts = [clean_text(a.get("title", "") + " " + a.get("description", ""))
                 for a in articles if a.get("description")]

        return pd.DataFrame({"text": texts, "label": 1})
    except Exception as e:
        print("🚨 Error in NewsAPI:", e)
        return pd.DataFrame(columns=["text", "label"])

# 🔹 Load local fake/real data
def load_real_news():
    df = pd.read_csv("data/True.csv")
    df["text"] = df["title"].astype(str) + " " + df["text"].astype(str)
    df["text"] = df["text"].apply(clean_text)
    return df[["text"]].assign(label=1)

def load_fake_news():
    df = pd.read_csv("data/Fake.csv")
    df["text"] = df["title"].astype(str) + " " + df["text"].astype(str)
    df["text"] = df["text"].apply(clean_text)
    return df[["text"]].assign(label=0)

# 🔹 Retrain pipeline
def retrain():
    real_local = load_real_news()
    real_api = fetch_real_news()
    fake_df = load_fake_news()

    real_df = pd.concat([real_local, real_api], ignore_index=True)
    print(f"✅ Real samples: {len(real_df)} | Fake samples: {len(fake_df)}")

    # ⚖️ Sample for balance (max 10000 per class)
    real_df = real_df.sample(min(len(real_df), 10000), random_state=42)
    fake_df = fake_df.sample(min(len(fake_df), 10000), random_state=42)

    df = pd.concat([real_df, fake_df], ignore_index=True).dropna()

    if df["label"].nunique() < 2:
        print("❌ Not enough class diversity. Abort.")
        return

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', GradientBoostingClassifier())
    ])

    X_train, _, y_train, _ = train_test_split(df["text"], df["label"], stratify=df["label"], test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    os.makedirs("app/model", exist_ok=True)
    with open("app/model/model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    with open("model_update_log.txt", "a", encoding="utf-8") as log:
        log.write(f"✅ Retrained on {datetime.now()} | Real: {len(real_df)} | Fake: {len(fake_df)}\n")

    print("✅ Model saved successfully!")

# 🔹 Run manually
if __name__ == "__main__":
    retrain()
