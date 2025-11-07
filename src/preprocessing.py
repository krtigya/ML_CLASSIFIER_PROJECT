# src/preprocessing.py
import pandas as pd
import re
import emoji
import os
import nltk
# FIX: Removed the undefined 'ensure_reviews_table'
from database_utils import get_connection 
from suggestion_extractor import extract_suggestions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# --- Ensure NLTK resources are available ---
print("[INFO] Ensuring NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


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
    except Exception:
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
        
# --- Suggestion detection keywords ---
SUGGESTION_KEYWORDS = ['please', 'fix', 'need', 'should', 'improve', 'suggest', 'update', 'bug', 'problem']

def detect_suggestion_safe(text):
    """Return 1 if text contains suggestion keywords, 0 otherwise."""
    try:
        text_lower = str(text).lower()
        for keyword in SUGGESTION_KEYWORDS:
            if keyword in text_lower:
                return 1
        return 0
    except Exception:
        return 0

# --- Nepali review detection ---
def is_nepali(text):
    """Return True if the text contains Devanagari characters (Nepali/Hindi script)."""
    if pd.isna(text):
        return False
    # Check for Devanagari Unicode range
    return any('\u0900' <= c <= '\u097F' for c in text)

# --- Main preprocessing function ---
def run_preprocessing(output_csv="data/cleaned_reviews.csv", nlp_csv="data/nlp_processed_reviews.csv"):
    
    # FIX: Removed ensure_reviews_table call, as data_ingestion created the table.
    conn = get_connection()
    
    # Read reviews table
    try:
        # We assume the 'reviews' table was successfully created by data_ingestion.py
        df = pd.read_sql_query("SELECT id, review_text, rating, date FROM reviews", conn)
    except Exception as e:
        print(f"[ERROR] Failed to read 'reviews' table. Have you run data_ingestion.py? Error: {e}")
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

    # --- 1. Save all cleaned data (for Analytics) ---
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df[['id', 'rating', 'cleaned_review', 'label', 'is_suggestion', 'suggestion_text']].to_csv(output_csv, index=False)

    # Save to SQLite
    df.to_sql('cleaned_reviews', conn, if_exists='replace', index=False)
    print(f"[DONE] Cleaned reviews (including 'neutral') saved to {output_csv} and database table 'cleaned_reviews'.")

    # --- 2. Create and Save NLP-Processed Data (for Model Training) ---
    nlp_df = df.copy()

    # Filter out neutral reviews (typically not used in binary classification)
    nlp_df = nlp_df[nlp_df['label'].isin(['positive', 'negative'])].copy()
    
    # Rename columns to match what sentiment_model.py expects
    nlp_df = nlp_df.rename(columns={
        'cleaned_review': 'cleaned_text',
        'label': 'sentiment'
    })

    # Convert labels to uppercase
    nlp_df['sentiment'] = nlp_df['sentiment'].str.upper()

    # Select only the columns needed for training
    nlp_df = nlp_df[['cleaned_text', 'sentiment']]

    # Save to CSV
    os.makedirs(os.path.dirname(nlp_csv), exist_ok=True)
    nlp_df.to_csv(nlp_csv, index=False)

    print(f"[DONE] NLP-Processed reviews (POS/NEG only) saved to {nlp_csv}.")
    
    conn.close()


if __name__ == "__main__":
    run_preprocessing()