import requests
import json
import boto3
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, when, udf
from pyspark.sql.types import ArrayType, FloatType, StringType

# Constants
S3_BUCKET = "movie-appeal-prediction-final"
FLASK_EMBEDDING_ENDPOINT = "http://host.docker.internal:5000/embed"

# Initialize Spark
spark = SparkSession.builder \
    .appName("process_movies_task_2") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

def get_embeddings(overview):
    """Call local Flask endpoint to get embeddings for the overview."""
    try:
        response = requests.post(FLASK_EMBEDDING_ENDPOINT, json={"inputs": overview})
        response.raise_for_status()
        embeddings = response.json()["embeddings"]  # Expecting 384 floats
        return embeddings
    except requests.exceptions.RequestException as e:
        print(f"Error getting embeddings: {e}")
        raise

# Register UDF
get_embeddings_udf = udf(get_embeddings, ArrayType(FloatType()))

def process_movies(execution_date):
    """Process daily movies: clean, add embeddings and summaries, save to S3."""
    date_str = execution_date
    temp_output_path = f"s3a://{S3_BUCKET}/processed-movies/{date_str}/temp"
    final_output_path = f"s3a://{S3_BUCKET}/processed-movies/{date_str}/movies.csv"
    old_output_dir = f"s3://{S3_BUCKET}/processed-movies/{date_str}/movies.csv"
    
    # Read JSON from S3
    df = spark.read.json(f"s3a://{S3_BUCKET}/raw-movies/{date_str}/movies.json", multiLine=True)
    print(f"Initial record count: {df.count()}")
    
    # Drop unnecessary columns
    drop_cols = ['adult', 'backdrop_path', 'original_language', 'original_title', 'poster_path', 'video']
    df = df.drop(*drop_cols)
    
    # Drop NA and filter
    df = df.dropna(subset=['vote_average', 'release_date', 'overview'])
    df = df.filter(col('vote_count') >= 10)
    df = df.filter(col('overview') != '')
    print(f"After filtering: {df.count()}")
    
    # Fix column order
    df = df.select('id', 'title', 'overview', 'release_date', 'genre_ids', 'popularity', 'vote_average', 'vote_count')
    
    # Extract year, compute sci_fi columns
    df = df.withColumn('release_year', year(col('release_date')))
    df = df.withColumn('is_sci_fi', when(col('genre_ids').cast('string').contains('878'), 1).otherwise(0))
    df = df.withColumn('sci_fi_appeal', when((col('vote_average') > 6) & (col('is_sci_fi') == 1), 1).otherwise(0))
    df = df.drop('genre_ids', 'release_date')
    
    # Add embeddings
    df = df.withColumn('overview_embedding', get_embeddings_udf(col('overview')))

    # Select final columns
    df = df.select('id', 'title', 'sci_fi_appeal', 'popularity', 'vote_count', 'vote_average', 'release_year', 'overview_embedding', 'overview')
    
    # Flatten embeddings for CSV output
    embedding_cols = [col('overview_embedding')[i].alias(f'embedding_{i}') for i in range(384)]
    df = df.select('id', 'title', 'sci_fi_appeal', 'popularity', 'vote_count', 'vote_average', 'release_year', *embedding_cols, 'overview')
    
    # Write to a temporary directory
    df.coalesce(1).write.csv(temp_output_path, header=False, mode='overwrite')
    print(f"Written temporary CSV to {temp_output_path}")
    
    # Rename the part file to movies.csv
    s3_client = boto3.client('s3')
    prefix = f"processed-movies/{date_str}/temp/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    part_file_key = None
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.csv'):
            part_file_key = obj['Key']
            break
    
    if not part_file_key:
        raise Exception("No part file found in temporary directory")
    
    # Copy the part file to the final location and delete the temp directory
    final_s3_key = f"processed-movies/{date_str}/movies.csv"
    s3_client.copy_object(Bucket=S3_BUCKET, CopySource={'Bucket': S3_BUCKET, 'Key': part_file_key}, Key=final_s3_key)
    print(f"Copied {part_file_key} to {final_output_path}")
    
    # Delete the temporary directory
    s3_client.delete_object(Bucket=S3_BUCKET, Key=part_file_key)
    s3_client.delete_object(Bucket=S3_BUCKET, Key=f"{prefix}_SUCCESS")
    print(f"Cleaned up temporary directory {temp_output_path}")
    
    # Delete the old movies.csv directory if it exists
    old_dir_prefix = f"processed-movies/{date_str}/movies.csv/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=old_dir_prefix)
    if 'Contents' in response:
        objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
        if objects_to_delete:
            s3_client.delete_objects(Bucket=S3_BUCKET, Delete={'Objects': objects_to_delete})
            print(f"Deleted old directory {old_output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: spark-submit process_movies.py <execution_date>")
        sys.exit(1)
    execution_date = sys.argv[1]
    process_movies(execution_date)
