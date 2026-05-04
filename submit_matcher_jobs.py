# submit_matcher_jobs.py
import boto3

batch = boto3.client("batch", region_name="eu-central-1")

S3_INPUT_BUCKET = "womd-features"
S3_OUTPUT_BUCKET = "womd-features"
INPUT_PREFIX    = "parquet/run-001"
RESULTS_PREFIX  = "results"
TOTAL_SHARDS    = 50
OSC_FILE        = "change_lane.osc"
RUN_ID          = "run-001"
JOB_QUEUE       = "parquet-worker-queue"
JOB_DEFINITION  = "matcher-worker:1"

for shard_index in range(TOTAL_SHARDS):
    response = batch.submit_job(
        jobName=f"matcher-{OSC_FILE.replace('.', '-')}-{shard_index:05d}",
        jobQueue=JOB_QUEUE,
        jobDefinition=JOB_DEFINITION,
        containerOverrides={
            "environment": [
                {"name": "JOB_COMPLETION_INDEX", "value": str(shard_index)},
                {"name": "MAX_SHARD_INDEX",      "value": str(TOTAL_SHARDS - 1)},
                {"name": "BASE_PREFIX_ROOT",     "value": INPUT_PREFIX},
                {"name": "INPUT_BUCKET",         "value": S3_INPUT_BUCKET},
                {"name": "RESULTS_BUCKET",       "value": S3_OUTPUT_BUCKET},
                {"name": "RESULTS_PREFIX",       "value": RESULTS_PREFIX},
                {"name": "RUN_ID",               "value": RUN_ID},
                {"name": "OSC_FILE",             "value": OSC_FILE},
                {"name": "FPS",                  "value": "10"},
                {"name": "USE_SED",              "value": "true"},
            ]
        },
    )
    print(f"  matcher shard {shard_index:05d}: {response['jobId']}")

print("Alle Matcher-Jobs submitted.")