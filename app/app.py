# app/app.py

import streamlit as st
from utils import predict_news

st.set_page_config(page_title="🧠 Fake News Detector", layout="wide")

# -------------- Custom CSS --------------
st.markdown("""
    <style>
    .main {
        background-color: var(--background-color);
    }
    .stTextArea textarea {
        background-color: var(--text-color);
        border-radius: 0.5rem !important;
    }
    .title {
        font-size: 2.8rem;
        font-weight: bold;
        color: var(--text-color);
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        background-color: #ddd;
        overflow: hidden;
        margin-top: 8px;
    }
    .confidence-fill {
        height: 100%;
        color: white;
        text-align: center;
        font-size: 13px;
        line-height: 20px;
    }
    .input-preview {
        background-color: #f2f2f2;
        padding: 15px;
        border-left: 5px solid #0099ff;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    @media (prefers-color-scheme: dark) {
        .input-preview {
            background-color: #1e1e1e;
            border-left: 5px solid #3399ff;
            color: #eeeeee;
        }
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -------------- Title --------------
st.markdown("<h1 class='title'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.subheader("🔍 Paste a news headline or article to find out if it's real or fake")

# -------------- Layout --------------
col1, col2 = st.columns([1, 1])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/7391/7391843.png", width=250)

with col2:
    user_input = st.text_area("✏️ Your news text", height=220)

    if st.button("🚀 Detect Now"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some news content.")
        elif len(user_input.split()) < 5:
            st.warning("⚠️ Please enter more detailed news content (at least 5 words).")
        else:
            label, confidence = predict_news(user_input)

            # Display input
            st.markdown("#### 📝 Your Input:")
            st.markdown(f"<div class='input-preview'>{user_input}</div>", unsafe_allow_html=True)

            # Result
            label_color = "#28a745" if label == "REAL" else "#dc3545"
            emoji = "✅" if label == "REAL" else "❌"
            st.markdown(f"""
                <h4 style='color:{label_color}; margin-top:10px'>
                    {emoji} This news is <b>{label}</b>
                </h4>
            """, unsafe_allow_html=True)

            # Confidence Bar
            confidence_percent = f"{confidence * 100:.2f}%"
            st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{confidence*100}%; background-color:{label_color}">
                        {confidence_percent} confident
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Uncertainty warning
            if confidence < 0.65:
                st.warning("🤔 The model is unsure. Please verify this news with trusted sources.")

# -------------- Footer --------------
st.markdown("""
<br><hr>
<div style='text-align:center; color:gray; font-size:14px'>
    By <b>Adithya</b> using Streamlit and Logistic Regression.
</div>
""", unsafe_allow_html=True)
