"""
pipeline.py — Save NLP-processed reviews for model training
-----------------------------------------------------------
This script loads the NLP-processed CSV, filters the required columns,
saves them into the database, and makes them ready for model training.
"""

import os
import sqlite3

import pandas as pd
from database_utils import save_nlp_processed_reviews

DB_FOLDER = "data"
DB_NAME = "nlp_final.db"
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)
os.makedirs(DB_FOLDER, exist_ok=True)



# 1️ Load NLP-processed reviews
def load_nlp_processed_reviews():
    csv_path = "data/nlp_processed_reviews.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Make sure NLP-processed CSV exists.")
    
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} NLP-processed reviews from CSV")
    return df

# 2️ Filter required columns

def filter_columns(df):
    possible_cols = ["review_text", "cleaned_text", "sentiment"]
    available_cols = [c for c in possible_cols if c in df.columns]

    if not {"cleaned_text", "sentiment"}.issubset(available_cols):
        raise ValueError(f"Missing essential columns in CSV. Found only: {available_cols}")

    filtered_df = df[available_cols].copy()
    print(f"[INFO] Using available columns: {available_cols}")
    print(f"[INFO] Filtered down to {len(filtered_df)} rows.")
    return filtered_df


# 3️ Save filtered data to database

def get_connection():
    """Return a SQLite connection to the database."""
    return sqlite3.connect(DB_PATH)


def save_nlp_processed_reviews(df):
    """Save NLP-processed reviews into the database."""
    conn = get_connection()
    df.to_sql("nlp_final", conn, if_exists="replace", index=False)
    conn.close()
    print("✅ NLP-processed reviews saved into 'nlp_processed_reviews' table.")


def save_to_database(df):
    save_nlp_processed_reviews(df)
    print("[INFO] NLP-processed reviews saved to 'nlp_processed_reviews' table for model training")


# 4️ Backup CSV

def backup_csv(df):
    csv_path = "data/nlp_processed_reviews_filtered.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[INFO] Backup CSV saved to {csv_path}")


# 5️⃣ Main execution

if __name__ == "__main__":
    print("\n Preparing NLP-processed reviews for model training...\n")

    # Step 1: Load CSV
    nlp_df = load_nlp_processed_reviews()

    # Step 2: Filter required columns
    filtered_df = filter_columns(nlp_df)

    # Step 3: Save to database
    save_to_database(filtered_df)

    # Step 4: Optional backup
    backup_csv(filtered_df)

    print("\n NLP-processed data saved to DB and ready for sentiment_model.py training!\n")
