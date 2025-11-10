# preprocessing_for_prediction.py 
# All the preprocessing functions needed for prediction pipeline
# Safe preprocessing for prediction (no database operations)

import re
import emoji
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Ensure NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def clean_text_safe(text):
    """Clean text safely; return empty string if any exception occurs."""
    try:
        if pd.isna(text):
            return ""
        text = str(text).lower() # Ensure it's a string
        text = remove_emojis(text)
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in STOPWORDS]
        return ' '.join(tokens)
    except Exception:
        return ""

def is_nepali(text):
    """Return True if the text contains Devanagari characters (Nepali/Hindi)."""
    if pd.isna(text):
        return False
    return any('\u0900' <= c <= '\u097F' for c in text)

# --- Central function to apply preprocessing to a Pandas Series or list ---
def apply_preprocessing_to_series(data_series):
    """
    Applies cleaning and Nepali filtering to a Series of raw reviews.
    
    Returns a DataFrame with the original reviews (raw_review) and
    the clean text (cleaned_text).
    """
    df = pd.Series(data_series).to_frame(name='raw_review')
    
    # 1. Filter Nepali reviews
    df = df[~df['raw_review'].apply(is_nepali)]
    
    # 2. Apply cleaning
    df['cleaned_text'] = df['raw_review'].apply(clean_text_safe)
    
    # 3. Remove rows with empty cleaned text
    df = df[df['cleaned_text'].str.strip() != '']
    
    return df