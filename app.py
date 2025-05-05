import streamlit as st
import joblib

st.title("Netflix Review Sentiment Analyzer")

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

user_input = st.text_area("Enter a Netflix review:")

if st.button("Analyze Sentiment"):
    if user_input:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        st.success(f"Predicted Sentiment: {prediction}")
    else:
        st.warning("Please enter some text.")
