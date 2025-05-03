import requests
import pandas as pd
import boto3
import time
from datetime import datetime, timedelta
from airflow.models import Variable

TMDB_API_KEY = Variable.get("tmdb_api_key")
S3_BUCKET = "movie-appeal-prediction-final"

def fetch_movies(execution_date, **kwargs):
    date_str = execution_date
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')

    # Fetch movies from the past 30 days
    movies = []
    target_count = 500
    start_date = (date_obj - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 days before
    end_date = date_obj.strftime('%Y-%m-%d')

    for page in range(1, 51):  # Up to 50 pages to get ~1000 movies
        url = (
            f"https://api.themoviedb.org/3/discover/movie"
            f"?api_key={TMDB_API_KEY}"
            f"&page={page}"
            f"&primary_release_date.gte={start_date}"
            f"&primary_release_date.lte={end_date}"
            f"&with_release_type=3|4"
            f"®ion=US"
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            movies.extend(data['results'])
            print(f"Fetched page {page} for {start_date} to {end_date}: {len(data['results'])} movies")
        except requests.exceptions.RequestException as e:
            print(f"Error on page {page}: {e}")
            raise
        time.sleep(0.25)
        if len(movies) >= target_count:
            break

    if not movies:
        raise ValueError(f"No movies fetched for date {date_str}")

    # Convert to DataFrame and deduplicate by 'id'
    df = pd.DataFrame(movies).drop_duplicates(subset=['id'])
    print(f"Total movies after deduplication for {date_str}: {len(df)}")

    s3_key = f"raw-movies/{date_str}/movies.json"
    local_file = f"/tmp/movies_{date_str}.json"
    df.to_json(local_file, orient='records')

    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_file, S3_BUCKET, s3_key)
        print(f"Uploaded to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise

    import os
    os.remove(local_file)

    return s3_key

if __name__ == "__main__":
    test_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    fetch_movies(test_date)
