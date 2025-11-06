import os
import csv
import time
import argparse
import requests
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "movies-ratings2.p.rapidapi.com",
}

# TMDB config (optional — script will still run for ratings if TMDB_BEARER is missing)
TMDB_BEARER = os.getenv("TMDB_BEARER")
TMDB_HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_BEARER}"
} if TMDB_BEARER else None


def fetch_imdb_ratings(imdb_id, retries=3, backoff=1.5, headers=None):
    """Fetch ratings for an imdb id via RapidAPI.
    Returns parsed JSON on success or an empty dict on 404 / not found.
    Raises requests.exceptions.HTTPError for non-404 HTTP errors after retries.
    """
    if not imdb_id:
        return {}

    headers = headers or RAPIDAPI_HEADERS
    url = "https://movies-ratings2.p.rapidapi.com/ratings"
    params = {"id": str(imdb_id)}

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError:
                return {}
        except requests.exceptions.HTTPError as e:
            # If rating not found, return empty instead of error
            if e.response is not None and e.response.status_code == 404:
                return {}
            if attempt >= retries:
                raise
            time.sleep(backoff ** attempt)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt >= retries:
                raise
            time.sleep(backoff ** attempt)


def fetch_tmdb_movie_details(tmdb_id, retries=3, backoff=1.5, headers=None):
    """Fetch TMDB movie details (requires TMDB_BEARER to be set).
    Returns parsed JSON or empty dict on 404/not found.
    """
    if not tmdb_id:
        return {}
    headers = headers or TMDB_HEADERS
    if not headers:
        # no credentials configured
        return {}

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError:
                return {}
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return {}
            if attempt >= retries:
                raise
            time.sleep(backoff ** attempt)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt >= retries:
                raise
            time.sleep(backoff ** attempt)


def find_imdb_column(row):
    """Try common column names to find the imdb id in the CSV row."""
    candidates = ["imdb_id", "imdbId", "imdb", "imdb id", "imdbId"]
    for c in candidates:
        if c in row and row[c]:
            return row[c]
    # fallback: try any column that looks like tt + digits
    for k, v in row.items():
        if isinstance(v, str) and v.startswith("tt"):
            return v
    return None


RATING_COLUMNS = [
    "score",
    "reviewsCount",
    "metascore",
    "userScore",
    "averageScore",
    "tomatometer",
    "tomatoReviewsCount",
    "averageScore2",
    "letterBoxd",
    "origin_country",
    "imdb_popularity",
]

# We'll also add TMDB 'revenue' as a new column if missing
RATING_COLUMNS.append("revenue")


def enrich_row_with_ratings(row, ratings, columns_to_add):
    # ratings may be {} or a dict containing 'ratings'
    r = ratings.get("ratings", {}) if isinstance(ratings, dict) else {}

    # helper to set only when requested
    def set_if_requested(key, value):
        if key in columns_to_add:
            row[key] = value if value is not None else ""

    set_if_requested("score", (r.get("imdb") or {}).get("score"))
    set_if_requested("reviewsCount", (r.get("imdb") or {}).get("reviewsCount"))
    set_if_requested("metascore", (r.get("metacritic") or {}).get("metascore"))
    set_if_requested("userScore", (r.get("metacritic") or {}).get("userScore"))
    set_if_requested("averageScore", (r.get("average") or {}).get("score"))
    set_if_requested("tomatometer", (r.get("rotten_tomatoes") or {}).get("tomatometer"))
    set_if_requested("tomatoReviewsCount", (r.get("rotten_tomatoes") or {}).get("reviewsCount"))
    set_if_requested("averageScore2", (r.get("rotten_tomatoes") or {}).get("averageScore"))
    set_if_requested("letterBoxd", (r.get("letterboxd") or {}).get("score"))
    set_if_requested("origin_country", (r.get("media") or {}).get("origin_country"))
    # popularity fallback
    set_if_requested("imdb_popularity", ratings.get("media", {}).get("popularity") if isinstance(ratings, dict) else None)

    # Ensure strings for CSV for the columns we added (None -> empty string)
    for k in columns_to_add:
        if row.get(k) is None:
            row[k] = ""
        else:
            row[k] = str(row[k])


def main():
    parser = argparse.ArgumentParser(description="Update CSV with RapidAPI movie rating fields using imdb ids")
    parser.add_argument("--csv", default="PRMoviesDB.csv", help="Input CSV file")
    parser.add_argument("--out", help="Output CSV file (default: input with _updated.csv suffix)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite input CSV in-place (writes a temp file then replaces)")
    parser.add_argument("--delay", type=float, default=0.001, help="Seconds to wait between API requests (default 0.001)")
    parser.add_argument("--start", type=int, default=0, help="Row index to start at (0-based, inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Row index to stop at (0-based, exclusive)")
    parser.add_argument("--force", action="store_true", help="Refetch and overwrite existing rating columns")
    args = parser.parse_args()

    input_path = args.csv
    if not os.path.exists(input_path):
        print(f"Input CSV not found: {input_path}")
        return

    out_path = args.out or (input_path[:-4] + "_updated.csv" if input_path.lower().endswith('.csv') else input_path + "_updated.csv")

    # Read all rows first (small-medium CSV expected)
    with open(input_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        orig_fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Prepare fieldnames for output (append rating columns if missing)
    new_fieldnames = list(orig_fieldnames)
    for c in RATING_COLUMNS:
        if c not in new_fieldnames:
            new_fieldnames.append(c)

    # Determine which rating columns are not already present in the CSV
    columns_to_add = [c for c in RATING_COLUMNS if c not in orig_fieldnames]
    if not columns_to_add:
        print("No new rating columns to add — all rating columns already exist in CSV. Exiting.")
        return

    total = len(rows)
    start = max(0, args.start)
    end = args.end if (args.end is not None) else total
    end = min(end, total)

    print(f"Processing rows {start}..{end} of {total} (delay={args.delay}s) — adding columns: {columns_to_add}\n")

    updated = 0
    skipped = 0
    errors = 0

    for idx in range(start, end):
        row = rows[idx]

        imdb_id = find_imdb_column(row)
        if not imdb_id:
            print(f"[{idx}] no imdb id found — skipping")
            skipped += 1
            continue

        try:
            ratings = fetch_imdb_ratings(imdb_id)
            enrich_row_with_ratings(row, ratings, columns_to_add)

            # If revenue is requested, try to extract it from the RapidAPI response
            if 'revenue' in columns_to_add:
                rev = None
                # Try a few likely locations in the RapidAPI payload
                if isinstance(ratings, dict):
                    rev = ratings.get('revenue')
                    if rev is None:
                        rev = ratings.get('media', {}).get('revenue')
                # also check nested 'ratings' payload
                if rev is None and isinstance(ratings, dict):
                    nested = ratings.get('ratings', {})
                    if isinstance(nested, dict):
                        rev = nested.get('revenue') or nested.get('media', {}).get('revenue')

                row['revenue'] = str(rev) if rev is not None else ''
            updated += 1
        except Exception as e:
            print(f"[{idx}] imdb={imdb_id} — error fetching ratings: {e}")
            errors += 1

        time.sleep(args.delay)

    # Write out CSV
    tmp_out = out_path + ".tmp"
    with open(tmp_out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=new_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if args.inplace:
        # Replace original file
        backup = input_path + ".bak"
        os.replace(input_path, backup)
        os.replace(tmp_out, input_path)
        print(f"Wrote updated CSV in-place to {input_path} (backup at {backup})")
    else:
        os.replace(tmp_out, out_path)
        print(f"Wrote updated CSV to {out_path}")

    print(f"done — updated={updated}, skipped={skipped}, errors={errors}")


if __name__ == '__main__':
    main()
