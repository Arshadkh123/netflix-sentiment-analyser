import streamlit as st
import pandas as pd
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Page config ---
st.set_page_config(page_title="Netflix Sentiment Analyzer", layout="centered")

# --- Dummy credentials ---
users = {
    "user1": "password123",
    "user2": "abc123"
}

# --- Session state initialization ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# --- Login screen ---
def login_screen():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid credentials")

# --- Sentiment prediction page ---
def sentiment_home():
    st.sidebar.title("📂 Menu")
    choice = st.sidebar.radio("Navigate", ["🏠 Sentiment Analyzer", "🔓 Logout"])

    if choice == "🏠 Sentiment Analyzer":
        st.title("🎬 Netflix Review Sentiment Analyzer")
        st.markdown("Enter your review below to predict if it's **Positive**, **Negative**, or **Neutral**.")

        review = st.text_area("Enter Review")
        if st.button("Analyze"):
            if not review.strip():
                st.warning("Please enter a valid review.")
            else:
                review_vector = vectorizer.transform([review])
                prediction = model.predict(review_vector)[0]

                if prediction == "Positive":
                    st.success(f"✅ Sentiment: {prediction}")
                elif prediction == "Negative":
                    st.error(f"❌ Sentiment: {prediction}")
                else:
                    st.info(f"ℹ️ Sentiment: {prediction}")

    elif choice == "🔓 Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("Logged out successfully.")
        st.stop()  # End execution cleanly

# --- App control flow ---
if st.session_state.logged_in:
    sentiment_home()
else:
    login_screen()
