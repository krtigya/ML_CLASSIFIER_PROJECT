
import argparse
import datetime as dt
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pickle

# ========== Project Imports ==========
from pipeline import filter_columns, load_nlp_processed_reviews
from test_sentiment_model import load_model_and_vectorizer  # reuse existing loader

# ========== Logging Setup ==========
LOGGER_NAME = "sentiment_infer"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# ========== Utility ==========
RUNS_DIR = Path("logs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def start_run_logger(run_id: str) -> Path:
    """Create a timestamped log file for each prediction run."""
    log_path = RUNS_DIR / f"predict_run_{run_id}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info("Run started: %s", run_id)
    return log_path


def read_input(input_csv: Optional[str], input_db: Optional[str], input_table: Optional[str]) -> pd.DataFrame:
    """Load input reviews from CSV or SQLite database."""
    if input_csv:
        df = pd.read_csv(input_csv)
        logger.info("Loaded CSV: %s [rows=%d]", input_csv, len(df))
    elif input_db and input_table:
        con = sqlite3.connect(input_db)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {input_table}", con)
            logger.info("Loaded table %s from DB %s [rows=%d]", input_table, input_db, len(df))
        finally:
            con.close()
    else:
        raise ValueError("Provide --input-csv or (--input-db and --input-table).")

    # Standardize text column to 'content'
    for c in ["content", "review_text", "review", "text", "body", "message"]:
        if c in df.columns:
            if c != "content":
                df = df.rename(columns={c: "content"})
            return df

    raise ValueError("No valid text column found. Expected one of: content, review_text, review, text, body, message.")


def basic_checks(cleaned: pd.Series, vocab_set: set) -> dict:
    """Perform basic checks between incoming data and training vocabulary."""
    lengths = cleaned.str.split().map(len)
    oov = cleaned.str.split().map(lambda toks: sum(1 for t in toks if t not in vocab_set) / max(1, len(toks)))

    return {
        "doc_count": int(lengths.shape[0]),
        "len_mean": float(np.nanmean(lengths)),
        "len_p50": float(np.nanpercentile(lengths, 50)),
        "len_p95": float(np.nanpercentile(lengths, 95)),
        "oov_mean": float(np.nanmean(oov)),
        "oov_p95": float(np.nanpercentile(oov, 95)),
        "empty_docs": int((lengths == 0).sum())
    }


def predict_scores(model, X) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate predictions and optional confidence scores."""
    yhat = model.predict(X)
    score = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            score = proba.max(axis=1)
        except Exception:
            score = None
    elif hasattr(model, "decision_function"):
        try:
            df = model.decision_function(X)
            if np.ndim(df) == 1:
                mn, mx = np.min(df), np.max(df)
                score = (df - mn) / (mx - mn + 1e-9)
            else:
                score = df.max(axis=1)
        except Exception:
            score = None
    return yhat, score


def main():
    parser = argparse.ArgumentParser(description="Production sentiment scoring orchestrator")
    io = parser.add_mutually_exclusive_group(required=True)
    io.add_argument("--input-csv", type=str, help="Path to CSV with text column")
    io.add_argument("--input-db", type=str, help="Path to SQLite DB")
    parser.add_argument("--input-table", type=str, help="SQLite table name if using --input-db")

    parser.add_argument("--model", type=str, default="models/best_sentiment_model.pkl")
    parser.add_argument("--vectorizer", type=str, default="models/tfidf_vectorizer.pkl")
    parser.add_argument("--out-csv", type=str, default="data/predicted_sentiments.csv")

    parser.add_argument("--save-cleaned", action="store_true", help="Include cleaned text in output")
    args = parser.parse_args()

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = start_run_logger(run_id)

    # Step 1 — Read input
    df = read_input(args.input_csv, args.input_db, args.input_table)
    logger.info("Columns: %s", list(df.columns))

    # Step 2 — Clean text (simple normalization)
    df["_cleaned_text"] = df["content"].astype(str).str.lower().str.replace(r"[^a-z\s]", "", regex=True)

    # Step 3 — Load model + vectorizer
    model, vectorizer = load_model_and_vectorizer()

    # Step 4 — Sanity checks
    vocab_set = set(vectorizer.vocabulary_.keys())
    stats = basic_checks(df["_cleaned_text"], vocab_set)
    logger.info("Stats: %s", json.dumps(stats, ensure_ascii=False))

    # Step 5 — Predict
    X = vectorizer.transform(df["_cleaned_text"])
    yhat, score = predict_scores(model, X)

    # Step 6 — Save results
    df_out = pd.DataFrame({
        "content": df["content"],
        "predicted_label": ["POSITIVE" if y == 1 else "NEGATIVE" for y in yhat],
        "predicted_score": score,
        "run_id": run_id
    })
    if args.save_cleaned:
        df_out.insert(1, "_cleaned_text", df["_cleaned_text"])

    Path(os.path.dirname(args.out_csv)).mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    logger.info("CSV saved: %s", args.out_csv)
    logger.info("Run complete: %s", run_id)


if __name__ == "__main__":
    # Hardcoded prediction CSV path
    prediction_csv = r"C:\Users\Lenovo\Downloads\ML_CLASSIFIER_PROJECT\data\synthetic_prediction_data.csv"
    
    # Optional: output CSV path
    output_csv = r"C:\Users\Lenovo\Downloads\ML_CLASSIFIER_PROJECT\data\synthetic_prediction_results.csv"
    
    # Run the orchestrator logic
    import sys
    sys.argv = [
        "make_predictions.py",  # dummy script name
        "--input-csv", prediction_csv,
        "--out-csv", output_csv,
        "--save-cleaned"  # optional, include cleaned text
    ]
    
    main()

