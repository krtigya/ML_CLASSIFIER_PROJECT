import os
import pickle
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')


MODEL_PATH = "models/best_sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def load_model_and_vectorizer():
    """Load trained sentiment model and TF-IDF vectorizer from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model or vectorizer not found. Train your model first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_sentiment(reviews):
    """
    Predict sentiment for one or multiple new reviews.
    :param reviews: list, Series, or single text string
    :return: DataFrame with predictions
    """
    model, vectorizer = load_model_and_vectorizer()

    # Convert input to list of strings
    if isinstance(reviews, pd.Series):
        reviews = reviews.tolist()
    elif isinstance(reviews, pd.DataFrame):
        # Take the first column if user passed a DataFrame
        reviews = reviews.iloc[:, 0].tolist()
    elif isinstance(reviews, str):
        reviews = [reviews]

    # Clean up and vectorize
    reviews_df = pd.DataFrame({"review": [str(r).strip() for r in reviews]})
    X_vec = vectorizer.transform(reviews_df["review"])

    # Predict
    y_pred = model.predict(X_vec)
    reviews_df["predicted_sentiment"] = ["POSITIVE" if p == 1 else "NEGATIVE" for p in y_pred]

    return reviews_df


if __name__ == "__main__":
    # Load your new CSV file
    csv_path = r"C:\Users\Lenovo\Downloads\ML_CLASSIFIER_PROJECT\data\synthetic_prediction_data.csv"
    predictions_reviews = pd.read_csv(csv_path)

    # Extract text column safely
    if 'content' not in predictions_reviews.columns:
        raise ValueError(f"'content' column not found in {csv_path}")

    new_reviews = predictions_reviews['content']

    predictions = predict_sentiment(new_reviews)

    print("\n=== Sentiment Predictions ===")
    print(predictions.to_string(index=False).encode('utf-8', errors='ignore').decode('utf-8'))

    # Optional: Save predictions to CSV
    output_path = os.path.join(os.path.dirname(csv_path), "predicted_sentiments.csv")
    predictions.to_csv(output_path, index=False)
    print(f"\n[INFO] Predictions saved to: {output_path}")
