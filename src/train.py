import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load cleaned data
df = pd.read_csv("data/data.csv")

X = df["text"]
y = df["label"]

# Convert text → numbers
vectorizer = TfidfVectorizer(stop_words="english")
X_transformed = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_transformed, y)

# Predict
preds = model.predict(X_transformed)

# Accuracy
acc = accuracy_score(y, preds)
print("Accuracy:", acc)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)