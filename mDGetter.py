import os
import re
import requests
import argparse
import json
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Load API credentials from environment variables
TMDB_BEARER = os.getenv("TMDB_BEARER")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

TMDB_HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_BEARER}"
}

RAPIDAPI_HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "movies-ratings2.p.rapidapi.com"
}


def search_movie_on_tmdb(title, include_adult=False, language="en-US", page=1):
    """
    Search TMDB for a movie title with optional filters.
    Returns the first result's JSON.
    """
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "query": title,
        "include_adult": str(include_adult).lower(),
        "language": language,
        "page": page
    }
    resp = requests.get(url, headers=TMDB_HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No TMDB results found for '{title}'")
    return results[0]


def fetch_tmdb_movie_details(tmdb_id):
    """
    Fetch detailed TMDB info (including genres, runtime, budget, imdb_id).
    """
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    resp = requests.get(url, headers=TMDB_HEADERS)
    resp.raise_for_status()
    return resp.json()


def fetch_imdb_ratings(imdb_id):
    """
    Fetch IMDb-based ratings via RapidAPI (score, metascore, tomatometer, etc.).
    Returns empty dict if no ratings available or on 404.
    """
    if not imdb_id:
        return {}
    url = "https://movies-ratings2.p.rapidapi.com/ratings"
    params = {"id": str(imdb_id)}
    try:
        resp = requests.get(url, headers=RAPIDAPI_HEADERS, params=params)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # If rating not found, return empty instead of error
        if e.response is not None and e.response.status_code == 404:
            return {}
        # Other HTTP errors should bubble up
        raise
    # Successful fetch — return the parsed JSON
    try:
        return resp.json()
    except ValueError:
        return {}


def gather_movie_data(title):
    """
    Given a movie title, fetch and combine data from TMDB and IMDb (via RapidAPI).
    Returns a dict of key movie features.
    """
    tmdb_result = search_movie_on_tmdb(title)
    tmdb_id = tmdb_result.get("id")
    details = fetch_tmdb_movie_details(tmdb_id)

    imdb_id = details.get("imdb_id")
    ratings = fetch_imdb_ratings(imdb_id)
    # Normalize RapidAPI ratings (they are nested under 'ratings')
    r = ratings.get('ratings', {}) if isinstance(ratings, dict) else {}

    movie_data = {
        "tmdb_id": tmdb_id,
        "imdb_id": imdb_id,
        "title": details.get("title"),
        "release_date": details.get("release_date"),
        "budget": details.get("budget"),
        "revenue": details.get("revenue"),
        "popularity": details.get("popularity"),
        "vote_average": details.get("vote_average"),
        "vote_count": details.get("vote_count"),
        "runtime": details.get("runtime"),
        "genres": [g.get("name") for g in details.get("genres", [])],
        "spoken_languages": [l.get("english_name") for l in details.get("spoken_languages", [])],
        "production_countries": [c.get("iso_3166_1") for c in details.get("production_countries", [])],
        # IMDb/RapidAPI ratings (nested under ratings)
        "score": (r.get("imdb") or {}).get("score"),
        "reviewsCount": (r.get("imdb") or {}).get("reviewsCount"),
        "metascore": (r.get("metacritic") or {}).get("metascore"),
        "userScore": (r.get("metacritic") or {}).get("userScore"),
        "averageScore": (r.get("average") or {}).get("score"),
        "tomatometer": (r.get("rotten_tomatoes") or {}).get("tomatometer"),
        "tomatoReviewsCount": (r.get("rotten_tomatoes") or {}).get("reviewsCount"),
        "averageScore2": (r.get("rotten_tomatoes") or {}).get("averageScore"),
        "letterBoxd": (r.get("letterboxd") or {}).get("score"),
        "origin_country": (r.get("media") or {}).get("origin_country"),
        # popularity isn't provided as a single top-level field; fall back to media.popularity if present
        "imdb_popularity": ratings.get("media", {}).get("popularity") if isinstance(ratings, dict) else None
    }
    return movie_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch movie data from TMDB and IMDb (via RapidAPI)"
    )
    parser.add_argument(
        "title",
        nargs="*",
        help="Movie title to search for (omit for interactive prompt)"
    )
    args = parser.parse_args()

    if args.title:
        title = " ".join(args.title)
    else:
        title = input("Enter movie title: ")

    try:
        data = gather_movie_data(title)
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error fetching data for '{title}': {e}")
