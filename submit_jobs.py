# submit_jobs.py
import boto3

batch  = boto3.client("batch", region_name="eu-central-1")
s3     = boto3.client("s3",    region_name="eu-central-1")

S3_INPUT_BUCKET  = "your-s3-bucket"
S3_OUTPUT_BUCKET = "your-s3-bucket"
TFRECORD_PREFIX  = "tfrecords/"
OUTPUT_PREFIX    = "parquet/run-001"
JOB_QUEUE        = "parquet-worker-queue"
JOB_DEFINITION   = "parquet-worker"

# alle TFRecord-Keys listen
paginator = s3.get_paginator("list_objects_v2")
keys = sorted([
    obj["Key"]
    for page in paginator.paginate(Bucket=S3_INPUT_BUCKET, Prefix=TFRECORD_PREFIX)
    for obj in page.get("Contents", [])
    if obj["Key"].endswith(".tfrecord")
])

print(f"{len(keys)} TFRecords gefunden — submitte {len(keys)} Jobs...")

for shard_index, key in enumerate(keys):
    response = batch.submit_job(
        jobName=f"parquet-worker-{shard_index:05d}",
        jobQueue=JOB_QUEUE,
        jobDefinition=JOB_DEFINITION,
        containerOverrides={
            "environment": [
                {"name": "SHARD_INDEX",      "value": str(shard_index)},
                {"name": "TOTAL_SHARDS",     "value": str(len(keys))},
                {"name": "S3_INPUT_BUCKET",  "value": S3_INPUT_BUCKET},
                {"name": "S3_INPUT_KEY",     "value": key},
                {"name": "S3_BUCKET",        "value": S3_OUTPUT_BUCKET},
                {"name": "S3_PREFIX",        "value": OUTPUT_PREFIX},
            ]
        },
    )
    print(f"  submitted shard {shard_index:05d} ({key}): {response['jobId']}")