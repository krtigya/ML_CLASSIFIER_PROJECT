# src/preprocessing.py
import pandas as pd
import re
import emoji
import os
import nltk
from database_utils import get_connection, ensure_reviews_table
from suggestion_extractor import extract_suggestions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# --- Ensure NLTK resources are available ---
print("[INFO] Ensuring NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))

# --- Text cleaning ---
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def clean_text_safe(text):
    """Clean text safely; return empty string if any exception occurs."""
    try:
        if pd.isna(text):
            return ""
        text = text.lower()
        text = remove_emojis(text)
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in STOPWORDS]
        return ' '.join(tokens)
    except Exception as e:
        print(f"[WARNING] Failed to clean text: {text} | Error: {e}")
        return ""

# --- Sentiment labeling ---
def label_review(rating):
    try:
        rating = int(rating)
        if rating > 3:
            return "positive"
        elif rating < 3:
            return "negative"
        else:
            return "neutral"
    except:
        return "unknown"

# --- Suggestion detection ---
SUGGESTION_KEYWORDS = ['please', 'fix', 'need', 'should', 'improve', 'suggest', 'update', 'bug', 'problem']

def detect_suggestion_safe(text):
    try:
        text_lower = str(text).lower()
        for keyword in SUGGESTION_KEYWORDS:
            if keyword in text_lower:
                return 1
        return 0
    except Exception as e:
        print(f"[WARNING] Failed to detect suggestion: {text} | Error: {e}")
        return 0

# --- Nepali review detection ---
def is_nepali(text):
    """Return True if the text contains Devanagari characters (Nepali/Hindi script)."""
    if pd.isna(text):
        return False
    return any('\u0900' <= c <= '\u097F' for c in text)

# --- Main preprocessing function ---
def preprocess_from_db(db_path, output_csv):
    print(f"[INFO] Connecting to database: {db_path}")
    conn = get_connection()
    ensure_reviews_table(conn)

    # Read reviews table
    try:
        df = pd.read_sql_query("SELECT id, review_text, rating, date FROM reviews", conn)
    except Exception as e:
        print(f"[ERROR] Failed to read 'reviews' table: {e}")
        conn.close()
        return

    if df.empty:
        print("[WARNING] No reviews found in the database. Please run data_ingestion.py first.")
        conn.close()
        return

    # Skip Nepali reviews
    num_nepali = df['review_text'].apply(is_nepali).sum()
    print(f"[INFO] Skipping {num_nepali} Nepali reviews...")
    df = df[~df['review_text'].apply(is_nepali)]

    if df.empty:
        print("[WARNING] No English reviews left after skipping Nepali text.")
        conn.close()
        return

    # Cleaning, labeling, and suggestion detection
    print("[INFO] Cleaning, tokenizing, and labeling reviews...")
    df['cleaned_review'] = df['review_text'].apply(clean_text_safe)
    df['label'] = df['rating'].apply(label_review)
    df['is_suggestion'] = df['review_text'].apply(detect_suggestion_safe)
    df['suggestion_text'] = df['review_text'].apply(extract_suggestions)


    # Remove empty cleaned reviews
    df = df[df['cleaned_review'].str.strip() != '']

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df[['id', 'rating', 'cleaned_review', 'label', 'is_suggestion']].to_csv(output_csv, index=False)

    # Save to SQLite
    df.to_sql('cleaned_reviews', conn, if_exists='replace', index=False)

    conn.close()
    print("[DONE] Preprocessing complete! Cleaned and labeled reviews saved to CSV and database.")

if __name__ == "__main__":
    db_path = "data/reviews.db"
    output_csv = "data/cleaned_reviews.csv"
    preprocess_from_db(db_path, output_csv)
