import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Load dataset
df = pd.read_csv("netflix_reviews.csv")
df.dropna(subset=["content", "score"], inplace=True)

# Label reviews
def label_sentiment(score):
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["score"].apply(label_sentiment)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X = vectorizer.fit_transform(df["content"])
y = df["sentiment"]

# Train model
model = LinearSVC()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved.")
