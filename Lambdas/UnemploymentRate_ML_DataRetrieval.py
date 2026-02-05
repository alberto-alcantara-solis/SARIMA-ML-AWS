#Automated triggered Lambda function (7:30 on 21th of each month) to retrieve unemployment rate data from Eurostat, decompress it, and store it in S3 for ML processing.

import os
import gzip
import boto3
import urllib.request
from io import BytesIO

DATA_URL = os.environ["DATA_URL"]

S3_BUCKET = os.environ["DATA_BUCKET"]
S3_KEY = os.environ["DATA_KEY"]

s3 = boto3.client("s3")


def lambda_handler(event, context):
    # Download compressed data
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        compressed_data = response.read()

    # Decompress gzip (Eurostat uses gzip)
    with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as gz:
        tsv_data = gz.read()

    # Upload TSV to S3
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_KEY,
        Body=tsv_data,
        ContentType="text/tab-separated-values"
    )

    return {
        "status": "success",
        "bucket": S3_BUCKET,
        "key": S3_KEY,
        "size_bytes": len(tsv_data),
    }
