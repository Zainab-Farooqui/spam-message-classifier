import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
    }
    .subtitle {
        text-align: center;
        color: #bbbbbb;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown('<div class="title">📩 Spam Message Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether a message is Spam or Not</div>', unsafe_allow_html=True)

# -------------------- LOAD & TRAIN MODEL --------------------
df = pd.read_csv("dataset/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# -------------------- INPUT --------------------
input_msg = st.text_area("✍️ Enter your message here:", height=150)

# -------------------- BUTTON --------------------
if st.button("🔍 Predict"):
    if input_msg.strip() == "":
        st.warning("Please enter a message first!")
    else:
        msg_vec = vectorizer.transform([input_msg])
        prediction = model.predict(msg_vec)

        # -------------------- RESULT --------------------
        if prediction[0] == 1:
            st.error("Spam Message Detected!")
        else:
            st.success("This is a Safe Message!")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("💡 Built with Machine Learning + Streamlit")