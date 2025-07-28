import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Setup NLTK download path (for compatibility)
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data if missing
for resource in ['punkt', 'stopwords']:
    try:
        if resource == 'punkt':
            nltk.data.find('tokenizers/punkt')
        else:
            nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Make sure 'vectorizer.pkl' and 'model.pkl' are in the same folder.")
    st.stop()

# Preprocessing
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

# Streamlit UI
st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter the message:")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        
        if result == 1:
            st.error("🚫 Spam")
        else:
            st.success("✅ Not Spam")
