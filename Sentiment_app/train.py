import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Better dataset (with Neutral)
X_train = [
    "I love this product",
    "This is amazing",
    "Very bad experience",
    "I hate this",
    "Worst purchase ever",
    "Absolutely fantastic",
    "hello",
    "hi",
    "bye",
    "okay",
    "thanks"
]

y_train = [
    "Positive", "Positive", "Negative", "Negative", "Negative", "Positive",
    "Neutral", "Neutral", "Neutral", "Neutral", "Neutral"
]

# TF-IDF (better than CountVectorizer)
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vect, y_train)

# Save files
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Training complete!")