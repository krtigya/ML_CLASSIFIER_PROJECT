# suggestion_extractor.py (Refactored fix)

import pandas as pd
import re
import os
import sys

#this metho helps to Fix Windows emoji print issue
sys.stdout.reconfigure(encoding='utf-8')

# CONFIGURATION

CLEAN_DATA_PATH = os.path.join("data", "cleaned_reviews.csv")
OUTPUT_PATH = os.path.join("data", "suggestion_reviews.csv")


# KEYWORD PATTERNS

SUGGESTION_KEYWORDS = [
    r"\bshould\b",
    r"\bneed to\b",
    r"\bneeds\b",
    r"\bmust\b",
    r"\bplease\b",
    r"\badd\b",
    r"\bfix\b",
    r"\bimprove\b",
    r"\bmake it\b",
    r"\bwould be better\b",
    r"\bhope you\b",
    r"\btry to\b",
]

SUGGESTION_REGEX = re.compile("|".join(SUGGESTION_KEYWORDS), re.IGNORECASE)


# FUNCTION TO EXTRACT SUGGESTION SENTENCES

def extract_suggestions(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Split text into sentences
    sentences = re.split(r"[.!?]", text)
    suggestions = [s.strip() for s in sentences if SUGGESTION_REGEX.search(s)]
    
    return " | ".join(suggestions)


# MAIN PIPELINE

def extract_suggestion_phrases():
    print(f"🧾 Reading cleaned data from: {CLEAN_DATA_PATH}")
    df = pd.read_csv(CLEAN_DATA_PATH)
    
    print("🔍 Extracting suggestion phrases from reviews...")
    
    # FIX: Change 'clean_text' to 'cleaned_review' to match preprocessing.py output
    if 'cleaned_review' not in df.columns:
        raise ValueError("Missing 'cleaned_review' column in CSV. Run preprocessing.py first.")
        
    df["suggestion_text"] = df["cleaned_review"].apply(extract_suggestions)
    
    # Mark whether review contains a suggestion
    df["has_suggestion"] = df["suggestion_text"].apply(lambda x: bool(x.strip()))
    
    print("💾 Saving suggestion results...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    print(f" Extraction complete! File saved at: {OUTPUT_PATH}")
    print(f" Total reviews with suggestions: {df['has_suggestion'].sum()} / {len(df)}")
    
    return df


if __name__ == "__main__":
    extract_suggestion_phrases()