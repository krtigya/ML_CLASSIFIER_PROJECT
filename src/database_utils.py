# db.py — Database Utility Functions
# -----------------------------------
# Handles database connection, saving, and loading
# for clean reviews and suggestion reviews.

import sqlite3
import pandas as pd
import os
import sys
import locale

# --- Ensure UTF-8 output ---
def ensure_utf8_encoding():
    try:
        encoding = locale.getpreferredencoding(False)
        if encoding.lower() != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

ensure_utf8_encoding()

# --- Database Setup ---
DB_FOLDER = "data"
DB_NAME = "reviews.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)
os.makedirs(DB_FOLDER, exist_ok=True)

def get_connection():
    """Return a SQLite connection to the database."""
    return sqlite3.connect(DB_PATH)

def save_to_db(clean_df, suggest_df):
    """Save clean and suggestion DataFrames into the database."""
    conn = get_connection()
    clean_df.to_sql("clean_reviews", conn, if_exists="replace", index=False)
    suggest_df.to_sql("suggestions", conn, if_exists="replace", index=False)
    conn.close()
    print("\n✅ Data saved into database tables: clean_reviews, suggestions")

def load_clean_reviews():
    """Load clean (positive/negative) reviews from the database."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM clean_reviews", conn)
    conn.close()
    return df

def load_suggestions():
    """Load suggestion reviews from the database."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM suggestions", conn)
    conn.close()
    return df

# --- Prepare data for sentiment model ---
def prepare_clean_reviews_for_model():
    """
    Return a DataFrame for sentiment model training.
    Columns: cleaned_text, sentiment
    Ensures no NaNs and all text is string.
    """
    conn = get_connection()
    try:
        df = pd.read_sql("SELECT cleaned_text, sentiment FROM nlp_processed_reviews", conn)
    except Exception:
        # Fallback: read from CSV if table not available
        csv_path = "data/nlp_processed_reviews.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df[['cleaned_text', 'sentiment']]
        else:
            df = pd.DataFrame(columns=['cleaned_text', 'sentiment'])
    finally:
        conn.close()

    # --- Clean data for model ---
    if not df.empty:
        # Drop rows with missing values
        df = df.dropna(subset=['cleaned_text', 'sentiment'])
        # Ensure all text is string
        df['cleaned_text'] = df['cleaned_text'].astype(str)
        # Remove empty strings
        df = df[df['cleaned_text'].str.strip() != ""]
        # Reset index
        df = df.reset_index(drop=True)

    return df


def save_nlp_processed_reviews(df):
    """Save NLP-processed reviews into the database."""
    conn = get_connection()
    df.to_sql("nlp_processed_reviews", conn, if_exists="replace", index=False)
    conn.close()
    print("✅ NLP-processed reviews saved into 'nlp_processed_reviews' table.")


# --- Optional test when running this script directly ---
if __name__ == "__main__":
    conn = get_connection()
    print("[INFO] Database connected successfully at:", DB_PATH)
    conn.close()

    df_model = prepare_clean_reviews_for_model()
    print("[INFO] Data ready for sentiment analysis model:")
    print(df_model.head())
