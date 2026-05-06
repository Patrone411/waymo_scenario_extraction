# Waymo Scenario Extraction Pipeline

A distributed data engineering pipeline that processes the [Waymo Open Dataset](https://waymo.com/open/) at scale — extracting structured features from raw sensor recordings and matching them against formally defined traffic scenarios using OpenSCENARIO 2.0.

---

## Architecture

```
S3 (raw TFRecords)
  s3://womd/waymo-training/*.tfrecord
        │
        ▼
┌─────────────────────────────────────────────┐
│   Stage 1: Feature Extraction (worker.py)   │
│   AWS Batch — 1 job per TFRecord shard      │
│   TFRecord parsing → feature extraction     │
│   → one Parquet file per scene              │
└─────────────────────────────────────────────┘
        │
        ▼
S3  s3://womd-features/parquet/run-001/{shard:05d}/scenes/{scene_id}.parquet
        │
        ▼
┌─────────────────────────────────────────────┐
│   Stage 2: Scenario Matching                │
│   (scenario_extraction/run_matching.py)     │
│   AWS Batch — 1 job per shard per scenario  │
│   OSC2 parsing → binding → hit extraction   │
└─────────────────────────────────────────────┘
        │
        ▼
S3  s3://womd-features/results/
      match_hits/scenario={scenario}/run_id={run_id}/shard={N:05d}.parquet
      match_actor_frames/...
      match_pair_frames/...
```

Both stages run independently and are horizontally scalable — processing 1,000 shards in parallel is the same operational effort as processing 1.

---

## What it does

### Stage 1: Feature Extraction (worker.py)

Reads Waymo TFRecord shards and extracts structured per-scene features:

- **Road segments** with lane count, geometry (reference line, target/left/right polygons, centerlines in GeoJSON)
- **Actor time series** — Cartesian (x, y, yaw, speed) and Frenet-frame (s, t, s_dot, t_dot, yaw_delta, osc_lane_id) at 10 Hz over 91 timesteps
- **Pairwise actor interactions** — relative position, euclidean distance, TTC — stored in a sparse interval encoding that reduces storage by ~80% vs. naive serialization
- **Environment elements** — traffic lights and crosswalks with their s/t projections onto the segment reference line

Output: one Parquet file per scene, flat schema, Snappy-compressed.

### Stage 2: Scenario Matching (run_matching.py)

Matches extracted scenes against traffic scenarios formally specified in OpenSCENARIO 2.0:

- Parses and compiles `.osc` files into constraint graphs
- Binds actor roles (e.g. `ego_vehicle`, `npc`) to real actors in each scene
- Evaluates temporal constraints across all candidate windows
- Writes hit results as three Parquet tables partitioned by scenario and run_id — queryable via AWS Athena or DuckDB without any additional ETL

Output tables per shard:
- `match_hits` — one row per matched window (scene, segment, roles, t0, t1)
- `match_actor_frames` — scalar feature values for each actor at t0 and t1
- `match_pair_frames` — pairwise interaction values (distance, TTC, relative position) at t0 and t1

---

## Key Design Decisions

### Why Parquet over Pickle?

The original pipeline serialized scene features as Python pickles stored on S3. Parquet with a flat, explicit schema solves all resulting problems:

- Columnar compression: ~3× smaller than equivalent pickle files
- Queryable without full deserialization via Athena or DuckDB
- Schema stability across shards — inconsistencies are caught at write time
- Native interoperability with Spark, pandas, and ML tooling

### Why one Parquet file per scene?

The matcher's access pattern is exactly one scene at a time. One file per scene means one S3 GET, one `pq.read_table()`, zero joins — matching the natural unit of work.

### Why sparse inter-actor encoding?

In a typical Waymo scene with 55 actors, 55 × 54 = 2,970 directed actor pairs exist but the vast majority have no valid overlapping timesteps. The sparse format stores only contiguous valid intervals, reducing the inter-actor payload by ~80%:

```python
# instead of: [None, None, 1.2, 1.3, 1.1, None, None, ...]  (91 elements)
# stored as:  {"intervals": [[2, 4]], "data": [1.2, 1.3, 1.1]}
```

### Why SQL-queryable match results?

Earlier versions wrote per-shard JSONL stats files that required a separate merge step before any analysis. The new design writes Parquet directly with Hive-style partitioning, making all 50 shards immediately queryable as a single logical table via Athena or DuckDB — no merge step required.

---

## Repository Structure

```
waymo_scenario_extraction/
├── worker.py                        # Stage 1: TFRecord → Parquet
├── submit_jobs.py                   # Submit Stage 1 worker jobs to AWS Batch
├── submit_matcher_jobs.py           # Submit Stage 2 matcher jobs to AWS Batch
├── docker/
│   ├── worker.Dockerfile            # Container image for Stage 1
│   └── matcher.Dockerfile           # Container image for Stage 2
├── k8s/
│   ├── worker-job.yaml              # Kubernetes Job spec (alternative to Batch)
│   └── matcher-job.yaml
├── feature_extraction/              # Core feature extraction library
│   ├── pipeline.py                  # process_scenario() entry point
│   └── tools/
│       └── scenario.py              # Scenario class, features_description
├── external/
│   └── waymo_motion_scenario_mining/ # Git submodule
├── scenario_extraction/
│   ├── run_matching.py              # Stage 2: Parquet → match results
│   ├── parquet_source.py            # Parquet reader, TagFeatures reconstruction
│   └── results_writer.py            # Writes match results as Parquet to S3
├── requirements_worker.txt
├── requirements_matcher.txt
└── notebooks/
    └── scenario_analysis.ipynb      # SQL analysis + visualisation
```

---

## Prerequisites

- Python 3.10+
- Docker
- AWS CLI configured (`aws configure`)
- AWS account with S3, ECR, and Batch permissions
- Access to the [Waymo Open Dataset](https://waymo.com/open/)

---

## Local Development

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/Patrone411/waymo_scenario_extraction.git
cd waymo_scenario_extraction
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

```bash
# Stage 1 (feature extraction)
pip install -r requirements_worker.txt

# Stage 2 (scenario matching)
pip install -r requirements_matcher.txt
```

### 3. Run Stage 1 locally (single shard)

Place a TFRecord file under `data/`:

```
data/training_tfexample.tfrecord-00000-of-01000
```

```bash
LOCAL_MODE=1 \
LOCAL_INPUT=data \
LOCAL_OUTPUT=test_output \
SHARD_INDEX=0 \
TOTAL_SHARDS=1000 \
python worker.py
```

Output lands at `test_output/00000/scenes/*.parquet`.

### 4. Run Stage 2 locally (against local Parquet files)

```bash
cd scenario_extraction

python run_matching.py \
  --osc_file change_lane.osc \
  --base_prefix ../test_output/00000 \
  --local true \
  --run_id run-001 \
  --out_dir ../test_matcher_out \
  --results_prefix results
```

Results land at `test_matcher_out/change_lane.osc/manual/`:
```
match_hits/shard_00000.parquet
match_actor_frames/shard_00000.parquet
match_pair_frames/shard_00000.parquet
```

### 5. Test the Docker image locally (with S3 input)

```bash
# build
docker build -f docker/worker.Dockerfile -t parquet-worker:latest .

# test: read one TFRecord from S3, write 3 scenes locally
docker run --rm \
  -e LOCAL_MODE=0 \
  -e SHARD_INDEX=0 \
  -e TOTAL_SHARDS=50 \
  -e S3_INPUT_BUCKET=womd \
  -e S3_INPUT_KEY=waymo-training/training_tfexample.tfrecord-00000-of-01000 \
  -e S3_BUCKET=womd-features \
  -e S3_PREFIX=parquet/run-001 \
  -e AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id) \
  -e AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key) \
  -e AWS_DEFAULT_REGION=eu-central-1 \
  parquet-worker:latest
```

---

## AWS Batch Deployment

### One-time setup

#### 1. ECR repositories

```bash
aws ecr create-repository --repository-name parquet-worker --region eu-central-1
aws ecr create-repository --repository-name matcher-worker  --region eu-central-1
```

#### 2. IAM roles

```bash
# Job role: S3 read/write access
cat > /tmp/batch-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]
}
EOF

aws iam create-role --role-name BatchJobRole \
  --assume-role-policy-document file:///tmp/batch-trust.json

cat > /tmp/batch-s3-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
    "Resource": ["arn:aws:s3:::womd", "arn:aws:s3:::womd/*",
                 "arn:aws:s3:::womd-features", "arn:aws:s3:::womd-features/*"]}]
}
EOF

aws iam put-role-policy --role-name BatchJobRole \
  --policy-name S3Access \
  --policy-document file:///tmp/batch-s3-policy.json

# Execution role: ECR pull access
cat > /tmp/ecs-execution-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]
}
EOF

aws iam create-role --role-name BatchExecutionRole \
  --assume-role-policy-document file:///tmp/ecs-execution-trust.json

aws iam attach-role-policy --role-name BatchExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

aws iam attach-role-policy --role-name BatchExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

# Batch service role
cat > /tmp/batch-service-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{"Effect": "Allow", "Principal": {"Service": "batch.amazonaws.com"}, "Action": "sts:AssumeRole"}]
}
EOF

aws iam create-role --role-name AWSBatchServiceRole \
  --assume-role-policy-document file:///tmp/batch-service-trust.json

aws iam attach-role-policy --role-name AWSBatchServiceRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole
```

#### 3. Compute environment and job queue

```bash
# get default VPC subnet and security group IDs first
aws ec2 describe-subnets \
  --filters "Name=default-for-az,Values=true" \
  --query 'Subnets[*].[SubnetId,AvailabilityZone]' \
  --output table --region eu-central-1

aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=default" \
  --query 'SecurityGroups[*].[GroupId,VpcId]' \
  --output table --region eu-central-1

# replace subnet and security group IDs below
aws batch create-compute-environment \
  --compute-environment-name parquet-worker-env \
  --type MANAGED --state ENABLED \
  --service-role arn:aws:iam::<account-id>:role/AWSBatchServiceRole \
  --compute-resources '{
    "type": "EC2",
    "allocationStrategy": "BEST_FIT_PROGRESSIVE",
    "minvCpus": 0, "maxvCpus": 200,
    "instanceTypes": ["optimal"],
    "subnets": ["<subnet-id>"],
    "securityGroupIds": ["<sg-id>"],
    "instanceRole": "arn:aws:iam::<account-id>:instance-profile/ecsInstanceRole"
  }' --region eu-central-1

aws batch create-job-queue \
  --job-queue-name parquet-worker-queue \
  --state ENABLED --priority 1 \
  --compute-environment-order '[{"order": 1, "computeEnvironment": "parquet-worker-env"}]' \
  --region eu-central-1
```

#### 4. Job definitions

```bash
# Stage 1: worker
aws batch register-job-definition \
  --job-definition-name parquet-worker \
  --type container \
  --container-properties '{
    "image": "<account-id>.dkr.ecr.eu-central-1.amazonaws.com/parquet-worker:latest",
    "resourceRequirements": [{"type": "VCPU", "value": "2"}, {"type": "MEMORY", "value": "4096"}],
    "jobRoleArn": "arn:aws:iam::<account-id>:role/BatchJobRole",
    "executionRoleArn": "arn:aws:iam::<account-id>:role/BatchExecutionRole",
    "environment": [
      {"name": "S3_BUCKET",    "value": "womd-features"},
      {"name": "S3_PREFIX",    "value": "parquet/run-001"},
      {"name": "TOTAL_SHARDS", "value": "50"}
    ]
  }' --region eu-central-1

# Stage 2: matcher
aws batch register-job-definition \
  --job-definition-name matcher-worker \
  --type container \
  --container-properties '{
    "image": "<account-id>.dkr.ecr.eu-central-1.amazonaws.com/matcher-worker:latest",
    "resourceRequirements": [{"type": "VCPU", "value": "2"}, {"type": "MEMORY", "value": "4096"}],
    "jobRoleArn": "arn:aws:iam::<account-id>:role/BatchJobRole",
    "executionRoleArn": "arn:aws:iam::<account-id>:role/BatchExecutionRole"
  }' --region eu-central-1
```

---

### Stage 1: Run feature extraction

#### Build and push the worker image

```bash
docker build -f docker/worker.Dockerfile -t parquet-worker:latest .

aws ecr get-login-password --region eu-central-1 | \
  docker login --username AWS \
  --password-stdin <account-id>.dkr.ecr.eu-central-1.amazonaws.com

docker tag parquet-worker:latest \
  <account-id>.dkr.ecr.eu-central-1.amazonaws.com/parquet-worker:latest

docker push \
  <account-id>.dkr.ecr.eu-central-1.amazonaws.com/parquet-worker:latest
```

#### Submit jobs (from AWS CloudShell or local with AWS CLI)

Edit `submit_jobs.py` and set:

```python
S3_INPUT_BUCKET = "womd"                # bucket containing TFRecords
S3_OUTPUT_BUCKET = "womd-features"     # bucket for Parquet output
TFRECORD_PREFIX  = "waymo-training/"   # S3 prefix of TFRecord files
OUTPUT_PREFIX    = "parquet/run-001"   # output prefix
JOB_QUEUE        = "parquet-worker-queue"
JOB_DEFINITION   = "parquet-worker:1"
```

Then:

```bash
python submit_jobs.py
```

#### Monitor progress

```bash
for STATUS in RUNNABLE STARTING RUNNING SUCCEEDED FAILED; do
  COUNT=$(aws batch list-jobs \
    --job-queue parquet-worker-queue \
    --job-status $STATUS \
    --region eu-central-1 \
    --query 'length(jobSummaryList)' --output text)
  echo "$STATUS: $COUNT"
done

# check output on S3
aws s3 ls s3://womd-features/parquet/run-001/ --recursive | wc -l
```

Expected output layout after completion:

```
s3://womd-features/parquet/run-001/
  00000/scenes/104b4a3e67b26ce1.parquet   (~50–500 KB per scene)
  00000/scenes/111ad99bc19e2b28.parquet
  ...
  00049/scenes/...
```

---

### Stage 2: Run scenario matching

#### Build and push the matcher image

```bash
docker build -f docker/matcher.Dockerfile -t matcher-worker:latest .

docker tag matcher-worker:latest \
  <account-id>.dkr.ecr.eu-central-1.amazonaws.com/matcher-worker:latest

docker push \
  <account-id>.dkr.ecr.eu-central-1.amazonaws.com/matcher-worker:latest
```

#### Submit jobs

Edit `submit_matcher_jobs.py` and set:

```python
S3_INPUT_BUCKET  = "womd-features"
S3_OUTPUT_BUCKET = "womd-features"
INPUT_PREFIX     = "parquet/run-001"
RESULTS_PREFIX   = "results"
TOTAL_SHARDS     = 50
RUN_ID           = "run-001"
JOB_QUEUE        = "parquet-worker-queue"
JOB_DEFINITION   = "matcher-worker:1"

OSC_FILES = [
    "change_lane.osc",
    "cross.osc",
    "ccrb.osc",
]
```

Then submit 150 jobs (50 shards × 3 scenarios):

```bash
python submit_matcher_jobs.py
```

#### Monitor progress

```bash
for STATUS in RUNNABLE STARTING RUNNING SUCCEEDED FAILED; do
  COUNT=$(aws batch list-jobs \
    --job-queue parquet-worker-queue \
    --job-status $STATUS \
    --region eu-central-1 \
    --query 'length(jobSummaryList)' --output text)
  echo "$STATUS: $COUNT"
done

# check results on S3
aws s3 ls s3://womd-features/results/ --recursive | wc -l
```

Expected output layout after completion:

```
s3://womd-features/results/
  match_hits/
    scenario=change_lane.osc/run_id=run-001/shard=00000.parquet
    scenario=change_lane.osc/run_id=run-001/shard=00001.parquet
    ...
  match_actor_frames/
    scenario=change_lane.osc/run_id=run-001/shard=00000.parquet
    ...
  match_pair_frames/
    ...
```

---

## Querying Results

### With DuckDB (local, no setup)

```python
import duckdb
import pyarrow.dataset as ds
import pyarrow.fs as pafs

s3 = pafs.S3FileSystem(region="eu-central-1")

hits = ds.dataset(
    "womd-features/results/match_hits",
    filesystem=s3, format="parquet", partitioning="hive"
).to_table().to_pandas()

con = duckdb.connect()

# hits per scenario
con.execute("""
    SELECT scenario, COUNT(*) as n_hits
    FROM hits GROUP BY scenario ORDER BY n_hits DESC
""").df()

# average ego speed at scenario start
actor_frames = ds.dataset(
    "womd-features/results/match_actor_frames",
    filesystem=s3, format="parquet", partitioning="hive"
).to_table().to_pandas()

con.execute("""
    SELECT scenario,
           AVG(speed * 3.6) as avg_speed_kmh,
           STDDEV(speed * 3.6) as std_speed_kmh
    FROM actor_frames
    WHERE role = 'ego_vehicle' AND frame = 'start'
    GROUP BY scenario
""").df()
```

### With AWS Athena

Create tables once in the Athena Query Editor:

```sql
CREATE EXTERNAL TABLE match_hits (
    run_id STRING, scenario STRING, shard_index INT,
    scene_id STRING, segment_id STRING, block_label STRING,
    roles_json STRING, t0 INT, t1 INT,
    n_windows INT, n_possible_windows INT, source_uri STRING
)
PARTITIONED BY (scenario STRING, run_id STRING)
STORED AS PARQUET
LOCATION 's3://womd-features/results/match_hits/'
TBLPROPERTIES ('parquet.compress'='SNAPPY');

MSCK REPAIR TABLE match_hits;
```

Then query directly:

```sql
SELECT scenario, COUNT(*) as n_hits
FROM match_hits
GROUP BY scenario;
```

---

## Retrieving Full Timeseries for a Hit

Match results store `scene_id`, `segment_id`, `t0`, `t1`, and `source_uri` — enough to retrieve the full timeseries from the feature Parquet files without re-running the pipeline:

```python
import json
import pyarrow.parquet as pq
from shapely.geometry import shape
import matplotlib.pyplot as plt

# load hit
hit = hits[hits["scene_id"] == "371c84dd9c5f5600"].iloc[0]
roles = json.loads(hit["roles_json"])
t0, t1 = hit["t0"], hit["t1"]

# load feature Parquet for this scene (shard 00001 in this example)
df  = pq.read_table(
    f"s3://womd-features/parquet/run-001/00001/scenes/{hit['scene_id']}.parquet"
).to_pandas()

row    = df[df["segment_id"] == hit["segment_id"]].iloc[0]
actors = json.loads(row["actors_json"])

ego_id = roles["ego_vehicle"]
npc_id = roles["npc"]

# full timeseries t0..t1
ego_x = actors["actor_ts"][ego_id]["x"][t0:t1+1]
ego_y = actors["actor_ts"][ego_id]["y"][t0:t1+1]
npc_x = actors["actor_ts"][npc_id]["x"][t0:t1+1]
npc_y = actors["actor_ts"][npc_id]["y"][t0:t1+1]

# road geometry
target_polygon = shape(json.loads(row["target_polygon_json"]))
reference_line = shape(json.loads(row["reference_line_json"]))

# plot
fig, ax = plt.subplots(figsize=(10, 6))
x, y = target_polygon.exterior.xy
ax.fill(x, y, alpha=0.2, color="steelblue", label="target lane")
ax.plot(*reference_line.xy, color="black", linewidth=2, label="reference line")
ax.plot(ego_x, ego_y, "bo-", markersize=3, label="ego")
ax.plot(npc_x, npc_y, "ro-", markersize=3, label="npc")
ax.set_aspect("equal")
ax.legend()
plt.show()
```

---

## Supported Scenarios (OSC2)

| File | Description |
|------|-------------|
| `change_lane.osc` | NPC performs a lane change in front of the ego vehicle |
| `cross.osc` | Pedestrian or cyclist crosses the ego vehicle's path |
| `ccrb.osc` | Cut-in with close range braking (TTC-constrained) |

New scenarios can be added by dropping an `.osc` file into `osc2_parser/osc/` — no code changes required.

---

## Stack

| Component | Technology |
|-----------|------------|
| Data format | Apache Parquet (PyArrow), Snappy compression |
| Feature extraction | Python, TensorFlow (TFRecord parsing), Shapely |
| Scenario specification | OpenSCENARIO 2.0, custom OSC2 parser (ANTLR) |
| Distributed processing | AWS Batch |
| Storage | Amazon S3 |
| Containerization | Docker, Amazon ECR |
| Result analysis | DuckDB, AWS Athena, Jupyter |

---

## Data Source

This project processes data from the [Waymo Open Dataset](https://waymo.com/open/). Access requires registration with Waymo. TFRecord files are not included in this repository.

---

## License

MIT
