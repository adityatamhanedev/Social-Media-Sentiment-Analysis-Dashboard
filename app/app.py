import streamlit as st
import pickle
from src.preprocess import clean_text
from src.feature_engineering import get_vectorizer
from src.predict import SentimentPredictor

st.title("Social Media Sentiment Analysis")

text_input = st.text_area("Enter text")

if st.button("Predict"):
    with open('models/sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)

    vectorizer = get_vectorizer()
    cleaned = clean_text(text_input)
    vec = vectorizer.fit_transform([cleaned])


predictor = SentimentPredictor(
    "models/sentiment_model.pkl",
    "models/vectorizer.pkl"
)

if st.button("Predict"):
    result = predictor.predict_with_confidence(text_input)
    st.write(result)

    st.write("Sentiment:", pred)