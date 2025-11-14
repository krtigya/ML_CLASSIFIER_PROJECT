# run_prediction_pipeline.py (Consolidated Prediction Script)

import os
import pickle
from tkinter import _test
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
import sys
import locale
from preprocessing_for_prediction import apply_preprocessing_to_series 


#  ROBUST UTF-8 

def ensure_utf8_encoding():
    try:
        if sys.platform.startswith('win'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

ensure_utf8_encoding()



# Configuration (Use relative path for portability)
INPUT_DATA_PATH = os.path.join("data", "testing_data.csv") 
MODEL_PATH = os.path.join("models", "best_sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")


# Step 1: 

def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer."""
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError("Trained model or vectorizer not found. Run sentiment_model.py first.")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

        print("Model and Vectorizer loaded successfully.") 
        return model, vectorizer
    except Exception as e:
        print(f" Error loading model/vectorizer: {e}")
        return None, None


def load_raw_data(path):
    """Load raw data from CSV, inferring the review column."""
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Loaded {len(df)} raw reviews from: {path}")
        
        # Identify columns that are likely IDs based on name
        id_cols = [col for col in df.columns if 'id' in col.lower() or 'uuid' in col.lower()]
        
        # 1. Search for keywords first (prioritizing non-ID columns)
        review_cols = [col for col in df.columns if col not in id_cols and ('review' in col.lower() or 'text' in col.lower() or 'content' in col.lower())]

        raw_review_col = None
        if review_cols:
            # Use the first non-ID column found with a keyword
            raw_review_col = review_cols[0]
        elif len(df.columns) >= 2:
            # 2. Fallback: No keyword found. ASSUME the first column is ID and use the second column.
            # This is the fix to skip the ID column confirmed by the UUID output.
            print(f"[WARNING] No text column found by keyword. Assuming review text is the SECOND column: '{df.columns[1]}'.")
            raw_review_col = df.columns[1]
        elif len(df.columns) == 1:
            # 3. Last Fallback: Use the only column available.
            print(f"[WARNING] Only one column found. Using it as review text: '{df.columns[0]}'.")
            raw_review_col = df.columns[0]
        else:
             raise ValueError("Could not find a suitable text column in the input CSV.")
             
        # Final rename
        df = df.rename(columns={raw_review_col: 'raw_review'})
        
        return df.reset_index()
    except Exception as e:
        print(f" [ERROR] Failed to load data from {path}: {e}") 
        return None


# Step 2 & 3: Preprocessing and Prediction 

def run_prediction_pipeline(data_path):
    """Orchestrates the prediction steps: Load, Preprocess, Vectorize, Predict."""
    
    # 1. Load Trained Assets
    model, vectorizer = load_model_and_vectorizer()
    if model is None:
        return None, 0.0
    
    # 2. Load Raw Data
    raw_df = load_raw_data(data_path)
    if raw_df is None:
        return None, 0.0

    # 3. Preprocessing (Using the centralized function)
    print("\n[STEP 1] Applying Preprocessing (Filtering/Cleaning)...")
    df_preprocessed = apply_preprocessing_to_series(raw_df['raw_review']) 
    
    # Merge preprocessed data back with original data for full context
    df = pd.merge(
        raw_df, 
        df_preprocessed.reset_index(),
        on='index', 
        how='inner',
        suffixes=('_raw', '_preprocessed')
    )

    # Clean up columns
    df = df.drop(columns=['index', 'raw_review_preprocessed']) 
    df = df.rename(columns={'raw_review_raw': 'raw_review'})


    if df.empty:
        print("[WARNING] No clean English reviews left for prediction. Exiting.")
        return None, 0.0
        
    # 4. Vectorize Cleaned Text
    print("[STEP 2] Applying TF-IDF transformation...")
    X_vec = vectorizer.transform(df["cleaned_text"])
    
    # 5. Predict Sentiment
    print("[STEP 3] Predicting sentiment...")
    y_proba = model.predict_proba(X_vec)
    y_pred = np.argmax(y_proba, axis=1) # 1=POSITIVE, 0=NEGATIVE
    
    df["predicted_label"] = np.where(y_pred == 1, "POSITIVE", "NEGATIVE")
    df["confidence"] = np.max(y_proba, axis=1).round(4)

    # 6. Evaluation (if labels are present)
    accuracy = 0.0
    label_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower()]
    
    if label_cols:
        actual_col = label_cols[0]
        
        # Standardize labels
        df['actual_sentiment'] = df[actual_col].astype(str).str.upper()
        df['actual_sentiment'] = df['actual_sentiment'].replace({
            'POS': 'POSITIVE', 
            'NEG': 'NEGATIVE',
            'NEGATIVEATIVE': 'NEGATIVE'
        })
        
        eval_df = df[df['actual_sentiment'].isin(['POSITIVE', 'NEGATIVE'])].copy()
        
        if not eval_df.empty:
            y_true = eval_df['actual_sentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0}).values
            
            # Predict labels on the evaluation subset
            y_pred_model = model.predict(vectorizer.transform(eval_df["cleaned_text"]))
            
            accuracy = accuracy_score(y_true, y_pred_model)
            print(f"\n EVALUATION COMPLETE: Accuracy on {len(eval_df)} Labeled Data Points: {accuracy * 100:.2f}%")
        else:
            print("[WARNING] No valid 'POSITIVE'/'NEGATIVE' labels found for accuracy calculation.")
            
    # Select final columns to display/save
    final_cols = ["raw_review", "predicted_label", "confidence"]
    if 'actual_sentiment' in df.columns:
        final_cols.append("actual_sentiment")
        
    return df[final_cols], accuracy

# print("\n[CONFUSION MATRIX]")
# cm = confusion_matrix(_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NEGATIVE", "POSITIVE"])
# disp.plot(cmap='Blues', values_format='d')
# plt.title("Confusion Matrix - Sentiment Model")
# plt.show()


# Step 4: Main execution

if __name__ == "__main__":
    try:
        print("\n=== Starting Consolidated Prediction Pipeline ===\n")
        
        predictions_df, accuracy = run_prediction_pipeline(INPUT_DATA_PATH)

        if predictions_df is not None and not predictions_df.empty:
            # Print and save results
            print("\n\n=== Prediction Results Sample ===")
            pd.set_option("display.max_colwidth", 100)
            print(predictions_df.head(10).to_markdown(index=False))

            output_path = os.path.join("data", "sentiment_predictions_final.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            predictions_df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"\n All predictions saved to: {output_path}")

    except Exception as e:
        print(f"\n [CRITICAL ERROR] Failed to run prediction pipeline: {e}")