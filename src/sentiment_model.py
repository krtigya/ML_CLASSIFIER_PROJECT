# src/sentiment_model.py

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss





# LightGBM
from lightgbm import LGBMClassifier
from pipeline import *

def train_sentiment_model():
    """Train and compare multiple sentiment classification models (including LightGBM) using cleaned reviews from DB."""

    print("[INFO] Loading cleaned reviews from database...")
    nlp_df = load_nlp_processed_reviews()

    # Step 2: Filter required columns
    df = filter_columns(nlp_df)

    if df.empty:
        print("[ERROR] No clean reviews found for training!")
        return

    if "cleaned_text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Database must have 'cleaned_text' and 'sentiment' columns.")

    df = df.dropna(subset=["cleaned_text", "sentiment"])
    df["cleaned_text"] = df["cleaned_text"].astype(str)
    df = df[df["cleaned_text"].str.strip() != ""]
    df = df.reset_index(drop=True)

    if df["sentiment"].dtype == object:
        df["sentiment"] = df["sentiment"].map({"POSITIVE": 1, "NEGATIVE": 0}).fillna(0).astype(int)

    print("[INFO] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["sentiment"], test_size=0.2, random_state=42
    )

    print("[INFO] Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # --- Define candidate models with tuned hyperparameters ---
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver='liblinear',
            C=2.0,
            penalty='l2',
            random_state=42,

        ),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(
            C=1.0,
            loss='hinge',
            max_iter=2000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=6,
            min_samples_leaf=4,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=40,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
    }

    results = {}
    best_model = None
    best_acc = 0

    print("[INFO] Training multiple algorithms...")
    for name, model in models.items():
        print(f"\n[MODEL] Training {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        # Compute accuracy
        acc = accuracy_score(y_test, y_pred)

        # Compute log loss if possible
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_vec)
                lloss = log_loss(y_test, y_proba)
            else:
                lloss = None
        except Exception:
            lloss = None

        results[name] = {
            "accuracy": acc,
            "log_loss": lloss,
        }

        print(f"→ Accuracy: {acc * 100:.2f}% | Log Loss: {lloss if lloss else 'N/A'}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    # --- Model comparison summary ---
    print("\n=== Model Comparison ===")
    for name, metrics in results.items():
        print(f"{name:20s} : Accuracy = {metrics['accuracy'] * 100:.2f}% | Log Loss = {metrics['log_loss'] if metrics['log_loss'] else 'N/A'}")

    # --- Save best model ---
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_sentiment_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\n[SUCCESS] Best Model: {type(best_model).__name__} with accuracy {best_acc * 100:.2f}%")
    print("[INFO] Model and vectorizer saved successfully.")
    print("[DONE] Training complete.")


if __name__ == "__main__":
    train_sentiment_model()
