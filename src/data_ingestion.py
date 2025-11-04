# src/data_ingestion.py

from google_play_scraper import reviews, Sort
from database_utils import insert_review, create_tables



def fetch_reviews_from_playstore(app_id="np.com.worldlink.worldlinkapp", count=11047):
    """Fetch reviews from Google Play Store."""
    print("[INFO] Fetching reviews from Google Play...")

    result, _ = reviews(
        app_id,
        lang="en",
        country="np",
        sort=Sort.NEWEST,
        count=count
    )

    print(f"[SUCCESS] Fetched {len(result)} reviews.")
    return result


def save_reviews_to_db(reviews_list):
    """Save reviews to the database."""
    create_tables()
    for r in reviews_list:
        insert_review(
            review_text=r.get("content", ""),
            rating=r.get("score"),
            date=str(r.get("at"))
        )
    print("[INFO] Reviews saved to database successfully.")


if __name__ == "__main__":
    data = fetch_reviews_from_playstore()
    save_reviews_to_db(data)
