import pandas as pd
import requests
import boto3
import csv
import sys
import os
import time
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

S3_BUCKET = "bucket"
FLASK_PREDICT_ENDPOINT = "http://host.docker.internal:5000/predict"

# Configure retries for requests
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))

# Initialize Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="eu-central-1")
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

def enhance_overview(overview):
    # Updated prompt to generate only the enhanced overview
    prompt = (
        "Rewrite the following movie overview to be more exciting, dramatic, and appealing, emphasizing its thrilling sci-fi elements in 2-3 sentences. "
        "Output only the rewritten overview, without any introductory text or explanations:\n\n"
        f"{overview}"
    )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }

    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        result = json.loads(response['body'].read())
        enhanced_overview = result['content'][0]['text']
        return enhanced_overview
    except Exception as e:
        print(f"Error enhancing overview: {e}")
        return overview  # Fallback to original overview if Bedrock fails

def make_predictions(execution_date):
    date_str = execution_date
    input_path = f"s3://{S3_BUCKET}/processed-movies/{date_str}/movies.csv"
    output_path = f"s3://{S3_BUCKET}/predictions/{date_str}/predictions.parquet"

    # Read the processed CSV from S3
    s3_client = boto3.client('s3')
    s3_key = f"processed-movies/{date_str}/movies.csv"
    local_file = f"/tmp/movies_{date_str}.csv"
    try:
        s3_client.download_file(S3_BUCKET, s3_key, local_file)
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise

    # Define column names (removed overview_summary, keep only overview)
    embedding_cols = [f"embedding_{i}" for i in range(384)]
    columns = ['id', 'title', 'sci_fi_appeal', 'popularity', 'vote_count', 'vote_average', 'release_year'] + embedding_cols + ['overview']
    df = pd.read_csv(local_file, header=None, names=columns, quoting=csv.QUOTE_ALL, on_bad_lines='skip')
    print(f"Read {len(df)} movies from S3")

    # Prepare predictions and enhance overviews where applicable
    predictions = []
    for idx, row in df.iterrows():
        # Prepare input for the Flask /predict endpoint
        input_data = {
            'popularity': float(row['popularity']),
            'vote_count': float(row['vote_count']),
            'vote_average': float(row['vote_average']),
            'release_year': int(row['release_year'])
        }
        for i in range(384):
            input_data[f'embedding_{i}'] = float(row[f'embedding_{i}'])

        # Call the Flask /predict endpoint with retries and throttling
        sci_fi_appeal_pred = 0.0
        for attempt in range(3):
            try:
                response = session.post(FLASK_PREDICT_ENDPOINT, json=input_data, timeout=10)
                response.raise_for_status()
                sci_fi_appeal_pred = response.json()['sci_fi_appeal_pred']
                break
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed for movie {row['id']}: {e}")
                if attempt == 2:
                    print(f"Error predicting for movie {row['id']}: {e}")
                time.sleep(2 ** attempt)
        time.sleep(0.1)

        # Enhance overview if sci_fi_appeal_pred > 0.6, otherwise keep original
        final_overview = row['overview']
        if sci_fi_appeal_pred > 0.6:
            print(f"Enhancing overview for movie {row['id']} with sci_fi_appeal_pred {sci_fi_appeal_pred}")
            final_overview = enhance_overview(row['overview'])
            time.sleep(0.2)

        # Collect prediction
        predictions.append({
            'id': row['id'],
            'title': row['title'],
            'sci_fi_appeal_pred': sci_fi_appeal_pred,
            'overview_summary': final_overview
        })

    # Create output DataFrame
    pred_df = pd.DataFrame(predictions)
    print(f"Generated predictions for {len(pred_df)} movies")

    # Write to S3 as Parquet
    local_parquet = f"/tmp/predictions_{date_str}.parquet"
    pred_df.to_parquet(local_parquet, index=False, engine='pyarrow')

    s3_output_key = f"predictions/{date_str}/predictions.parquet"
    try:
        s3_client.upload_file(local_parquet, S3_BUCKET, s3_output_key)
        print(f"Uploaded predictions to {output_path}")
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise

    # Clean up
    os.remove(local_file)
    os.remove(local_parquet)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_predictions.py <execution_date>")
        sys.exit(1)
    execution_date = sys.argv[1]
    make_predictions(execution_date)
