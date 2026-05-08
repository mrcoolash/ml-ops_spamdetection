import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import mlflow
import mlflow.sklearn

# Start MLflow experiment
with mlflow.start_run():

    # Load dataset
    df = pd.read_csv("data/data.csv")

    X = df["text"]
    y = df["label"]

    # Convert text into vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    X_transformed = vectorizer.fit_transform(X)

    # Hyperparameter
    max_iter = 200

    # Train model
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_transformed, y)

    # Predictions
    preds = model.predict(X_transformed)

    # Accuracy
    acc = accuracy_score(y, preds)

    print("Accuracy:", acc)

    # Log parameter
    mlflow.log_param("max_iter", max_iter)

    # Log metric
    mlflow.log_metric("accuracy", acc)

    # Save model locally
    with open("models/model.pkl", "wb") as f:
        pickle.dump((model, vectorizer), f)

    # Log model in MLflow
    mlflow.sklearn.log_model(model, "spam_classifier")