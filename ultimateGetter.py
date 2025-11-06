import os
import pandas as pd
from mDGetter import gather_movie_data
from dotenv import load_dotenv
import time

# Load environment variables from .env if present
load_dotenv()

def update_movie_database(csv_path, start_row=3492):
    """
    Update the movie database with TMDB and IMDb data starting from a specific row.
    Only processes rows that have a title but missing other data.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    total_rows = len(df) - start_row
    processed = 0
    updated = 0

    print(f"Starting to process {total_rows} rows from row {start_row}")

    # Process each row starting from start_row
    for idx in range(start_row, len(df)):
        title = df.loc[idx, 'FILM TITLE']
        
        # Skip if we already have TMDB data (checking tmdb_id)
        if pd.notna(df.loc[idx, 'tmdb_id']):
            processed += 1
            continue

        try:
            print(f"\nProcessing [{processed}/{total_rows}]: {title}")
            
            # Gather movie data
            movie_data = gather_movie_data(title)
            
            # Update the dataframe with new data
            df.loc[idx, 'tmdb_id'] = movie_data.get('tmdb_id')
            df.loc[idx, 'imdb_id'] = movie_data.get('imdb_id')
            df.loc[idx, 'budget'] = movie_data.get('budget')
            df.loc[idx, 'revenue'] = movie_data.get('revenue')
            df.loc[idx, 'popularity'] = movie_data.get('popularity')
            df.loc[idx, 'vote_average'] = movie_data.get('vote_average')
            df.loc[idx, 'vote_count'] = movie_data.get('vote_count')
            df.loc[idx, 'runtime'] = movie_data.get('runtime')
            df.loc[idx, 'genres'] = str(movie_data.get('genres'))
            df.loc[idx, 'spoken_languages'] = str(movie_data.get('spoken_languages'))
            df.loc[idx, 'origin_country'] = str(movie_data.get('production_countries'))
            df.loc[idx, 'score'] = movie_data.get('score')
            df.loc[idx, 'reviewsCount'] = movie_data.get('reviewsCount')
            df.loc[idx, 'metascore'] = movie_data.get('metascore')
            df.loc[idx, 'userScore'] = movie_data.get('userScore')
            df.loc[idx, 'averageScore'] = movie_data.get('averageScore')
            df.loc[idx, 'tomatometer'] = movie_data.get('tomatometer')
            df.loc[idx, 'tomatoReviewsCount'] = movie_data.get('tomatoReviewsCount')
            df.loc[idx, 'averageScore2'] = movie_data.get('averageScore2')
            df.loc[idx, 'letterBoxd'] = movie_data.get('letterBoxd')
            df.loc[idx, 'imdb_popularity'] = movie_data.get('imdb_popularity')
            
            updated += 1
            
            # Save after every successful update
            if updated % 10 == 0:  # Save every 10 updates
                df.to_csv(csv_path, index=False)
                print(f"Saved progress - {updated} records updated so far")
            
            # Add a small delay to avoid hitting API rate limits
            time.sleep(0.5)  # 0.5 seconds between API calls
            
        except Exception as e:
            print(f"Error processing '{title}': {str(e)}")
        
        processed += 1
        
        # Print progress
        if processed % 5 == 0:
            print(f"Progress: {processed}/{total_rows} rows processed, {updated} updated")
    
    # Final save
    df.to_csv(csv_path, index=False)
    print(f"\nComplete! Processed {processed} rows, updated {updated} records")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Update movie database with TMDB and IMDb data"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV file to update"
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=2124,
        help="Row number to start processing from (default: 2124)"
    )
    
    args = parser.parse_args()
    
    update_movie_database(args.csv, args.start_row)