# 📰 Fake News Detector

A **Streamlit-powered web app** that detects whether a news headline or article is **real or fake** using Machine Learning (Logistic Regression + TF-IDF).

---

## 🚀 Features

- ✅ Real-time prediction of news authenticity
- 📈 Confidence score with visual bar
- 🧠 Logistic Regression with TF-IDF
- 🔁 Auto-retraining using latest real news from NewsAPI
- 📦 Streamlit frontend with modern UI

---

## 🧠 Tech Stack

- Python 3.x
- Streamlit
- scikit-learn
- pandas
- NewsAPI
- Logistic Regression (binary classification)

---

## 🧪 Run Locally

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # On Windows
# OR
source venv/bin/activate     # On Mac/Linux

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app/app.py

🔑 Setup
You need a NewsAPI Key to fetch real-time news for retraining:

Sign up at https://newsapi.org

Get your API key

Add it in fetch_and_update.py:

API_KEY = "your_newsapi_key_here"
📜 License
This project is open-source and free to use for educational purposes.
