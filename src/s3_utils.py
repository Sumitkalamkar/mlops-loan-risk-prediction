import boto3
import os
from datetime import datetime

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "ap-south-1"

BUCKET_NAME = "mlops-loan-risk-gfg"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def upload_model(file_path, model_name="model.pkl"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version_path = f"models/{timestamp}/{model_name}"

    s3.upload_file(file_path, BUCKET_NAME, version_path)

    print(f"Uploaded to S3: {version_path}")
    return version_path

def download_latest_model(local_path="models/latest_model.pkl"):
    # Download the latest model.pkl by name
    return download_latest_file("model.pkl", local_path)


def download_latest_file(file_name, local_path):
    """Download the latest file with name `file_name` under the models/ prefix.

    Example: download_latest_file('scaler.pkl', 'models/scaler.pkl')
    """
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="models/")

    contents = response.get("Contents", [])
    all_files = [obj["Key"] for obj in contents if obj["Key"].endswith(file_name)]

    if not all_files:
        raise FileNotFoundError(f"No files named {file_name} found in S3 bucket {BUCKET_NAME} under 'models/'.")

    latest = sorted(all_files)[-1]

    # ensure local directory exists
    local_dir = os.path.dirname(local_path)
    if local_dir and not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    s3.download_file(BUCKET_NAME, latest, local_path)

    print(f"Downloaded: {latest}")
    return local_path


def list_versions(file_name):
    """List all available versions of a file (sorted by timestamp, newest first).
    
    Example: list_versions('model.pkl') -> ['20260430-120000', '20260429-150000', ...]
    """
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="models/")
    contents = response.get("Contents", [])
    
    versions = {}
    for obj in contents:
        key = obj["Key"]
        if key.endswith(file_name):
            # Extract timestamp from path like "models/20260430-120000/model.pkl"
            parts = key.split("/")
            if len(parts) >= 2:
                timestamp = parts[1]
                versions[timestamp] = key
    
    return sorted(versions.keys(), reverse=True), versions


def download_file_by_version(file_name, version_timestamp, local_path):
    """Download a specific version of a file by timestamp.
    
    Args:
        file_name: e.g., 'model.pkl'
        version_timestamp: e.g., '20260430-120000'
        local_path: where to save locally
    
    Example: download_file_by_version('scaler.pkl', '20260430-120000', 'models/scaler.pkl')
    """
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="models/")
    contents = response.get("Contents", [])
    
    target_key = None
    for obj in contents:
        key = obj["Key"]
        if key.endswith(file_name) and version_timestamp in key:
            target_key = key
            break
    
    if not target_key:
        raise FileNotFoundError(
            f"No file {file_name} with version {version_timestamp} found in S3 bucket {BUCKET_NAME}."
        )
    
    # ensure local directory exists
    local_dir = os.path.dirname(local_path)
    if local_dir and not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    
    s3.download_file(BUCKET_NAME, target_key, local_path)
    print(f"Downloaded version {version_timestamp}: {target_key}")
    return local_path