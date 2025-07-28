import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import os

# Add local nltk_data path before anything else
nltk.data.path.append('./nltk_data')

# Optional: You can remove this in production
nltk.download('punkt')  # only runs if not found in nltk_data

ps = PorterStemmer()


# Function to transform input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Enhanced Styling
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F9FAFC;
    }

    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        color: #2C3E50;
        padding: 0.5rem 0;
    }

    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #7F8C8D;
        margin-bottom: 2rem;
    }

    .stTextArea label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2C3E50;
    }

    .stButton button {
        background: linear-gradient(90deg, #4A00E0, #8E2DE2);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background: linear-gradient(90deg, #8E2DE2, #4A00E0);
        transform: scale(1.02);
    }

    .result {
        font-size: 2rem;
        font-weight: 700;
        color: #E74C3C;
        text-align: center;
        margin-top: 2rem;
        background-color: #FDEDEC;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }

    .not-spam {
        font-size: 2rem;
        font-weight: 700;
        color: #27AE60;
        text-align: center;
        margin-top: 2rem;
        background-color: #E9F7EF;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }

    .footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        color: #BDC3C7;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">📩 Email / SMS Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Built with Streamlit and Machine Learning</div>', unsafe_allow_html=True)

# Input Field
input_sms = st.text_area("✉️ Enter the message to check for spam")

# Predict Button
if st.button('🔍 Predict'):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Output
    if result == 1:
        st.markdown('<div class="result">🚫 This message is <b>SPAM</b>.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="not-spam">✅ This message is <b>NOT SPAM</b>.</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 Aman Verma • Streamlit Spam Detector</div>', unsafe_allow_html=True)
