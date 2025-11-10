# src/sentiment_model.py

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import sys 

# --- FIX: Ensure UTF-8 Encoding for Console Output
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass 


# --- CONFIGURATION ---
NLP_PROCESSED_CSV = os.path.join("data", "nlp_processed_reviews.csv")
MODEL_PATH = os.path.join("models", "best_sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")


def load_nlp_processed_reviews():
    """Load the model-ready CSV created by preprocessing.py."""
    if not os.path.exists(NLP_PROCESSED_CSV):
        raise FileNotFoundError(f"Model-ready data not found at: {NLP_PROCESSED_CSV}. Run preprocessing.py first.")
    
    df = pd.read_csv(NLP_PROCESSED_CSV)
    print(f"[INFO] Loaded {len(df)} NLP-processed reviews from CSV.")

    if "cleaned_text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("The loaded CSV must contain 'cleaned_text' and 'sentiment' columns.")
    
    return df


def train_sentiment_model():
    """Train sentiment classification model using Logistic Regression only."""

    print("[INFO] Loading cleaned reviews...")
    df = load_nlp_processed_reviews()
    
    if df.empty:
        print("[ERROR] No clean reviews found for training!")
        return

    # Clean up and prepare data
    df = df.dropna(subset=["cleaned_text", "sentiment"])
    df["cleaned_text"] = df["cleaned_text"].astype(str)
    df = df[df["cleaned_text"].str.strip() != ""]
    df = df.reset_index(drop=True)

    # Encode sentiment (POSITIVE: 1, NEGATIVE: 0)
    if df["sentiment"].dtype == object:
        df["sentiment"] = df["sentiment"].map({"POSITIVE": 1, "NEGATIVE": 0}).fillna(0).astype(int)

    print(f"[INFO] Final dataset size: {len(df)}")
    print("[INFO] Splitting dataset (80/20 train/test)...")
    
    X = df["cleaned_text"]
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- TF-IDF Vectorization ---
    print("\n[VECTORIZER] Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # --- Train Logistic Regression ---
    print("\n[MODEL] Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        C=2.0,
        penalty='l2',
        random_state=42
    )

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    lloss = None
    try:
        y_proba = model.predict_proba(X_test_vec)
        lloss = log_loss(y_test, y_proba)
    except Exception:
        pass

    print(f"\n Training Complete. Model Metrics on Test Set:")
    print(f"→ Accuracy: {acc * 100:.2f}%")
    print(f"→ Log Loss: {lloss:.4f}" if lloss else "→ Log Loss: N/A")

    # Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
        
    print(f"\n Model saved to: {MODEL_PATH}")
    print(f"\n Vectorizer saved to: {VECTORIZER_PATH}")

if __name__ == "__main__":
    train_sentiment_model()