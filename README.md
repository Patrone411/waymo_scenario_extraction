# Waymo Scenario Extraction Pipeline

A distributed data engineering pipeline that processes the [Waymo Open Dataset](https://waymo.com/open/) at scale — extracting structured features from raw sensor recordings and matching them against formally defined traffic scenarios using OpenSCENARIO 2.0.

---

## Architecture

```
S3 / GCS (raw TFRecords)
        │
        ▼
┌───────────────────┐
│   Worker Job      │  AWS Batch — 1 job per TFRecord shard
│   (worker.py)     │  TFRecord parsing → feature extraction → Parquet
└───────────────────┘
        │
        ▼
S3  parquet/run-xxx/{shard}/scenes/{scene_id}.parquet
        │
        ▼
┌───────────────────┐
│   Matcher Job     │  AWS Batch / Kubernetes — 1 job per shard
│ (run_matching.py) │  OSC2 parsing → scenario matching → statistics
└───────────────────┘
        │
        ▼
S3  results/{scenario}/  →  stats_shard.json  |  example_windows.jsonl
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
- Collects hit/baseline statistics and parameter histograms
- Generic parameter extractor automatically derives measurable quantities (speed, distance, lane index, TTC) directly from OSC2 modifier definitions — no scenario-specific extraction code required

---

## Key Design Decisions

### Why Parquet over Pickle?

The original pipeline serialized scene features as Python pickles stored on S3. This worked locally but created several problems at scale:

- Pickle files are not queryable without full deserialization
- Schema inconsistencies across shards were silent and hard to debug
- No columnar compression — pickle files are ~3× larger than equivalent Parquet
- No interoperability with Spark, Athena, or downstream ML tooling

Parquet with a flat, explicit schema solves all of these. The scalar columns (`scene_id`, `segment_id`, `num_lanes`, `valid`) support predicate pushdown for fast scene filtering without reading actor blobs.

### Why one Parquet file per scene?

The matcher's access pattern is exactly one scene at a time — it loads all segments for a scene, runs binding evaluation, then moves on. A normalized multi-table layout would require five joined scans per scene. One file per scene means one S3 GET, one `pq.read_table()`, zero joins.

### Why sparse inter-actor encoding?

In a typical Waymo scene with 55 actors, there are 55 × 54 = 2,970 directed actor pairs. The vast majority have no valid overlapping timesteps — both actors are rarely present in the same segment simultaneously. Storing 91-element arrays of `None` for each pair wastes ~40× space.

The sparse format stores only contiguous valid intervals:

```python
# instead of: [None, None, 1.2, 1.3, 1.1, None, None, ...]  (91 elements)
# stored as:  {"intervals": [[2, 4]], "data": [1.2, 1.3, 1.1]}
```

This reduces the inter-actor payload from ~2,970 full arrays to typically 20–80 non-empty pairs per scene.

### Why a generic OSC2 parameter extractor?

Earlier versions required a hand-written Python extractor for each new scenario type (`stats_extractors_change_lane.py`, `stats_extractors_cross.py`, etc.). The generic extractor (`generic_window_extractor.py`) introspects the lowered OSC2 call structure at runtime and automatically maps modifier arguments to `TagFeatures` fields:

```
speed: {range: [30, 120], unit: kilometer_per_hour}  →  feats.speed[actor_id][t0]
distance: {range: [5, 50], unit: meter}              →  feats.rel_distance[(a, b)][t0]
lane: -1                                              →  feats.lane_idx[actor_id][t0]
```

Adding a new scenario requires only an `.osc` file — no Python changes.

---

## Repository Structure

```
waymo_scenario_extraction/
├── worker.py                   # Stage 1: TFRecord → Parquet (AWS Batch job)
├── submit_jobs.py              # Submit worker jobs to AWS Batch
├── docker/
│   ├── worker.Dockerfile       # Container image for worker
│   └── matcher.Dockerfile      # Container image for matcher
├── k8s/
│   ├── worker-job.yaml         # Kubernetes Job spec (alternative to Batch)
│   └── matcher-job.yaml
├── feature_extraction/         # Core feature extraction library
│   ├── pipeline.py             # process_scenario() entry point
│   └── tools/
│       └── scenario.py         # Scenario class, features_description
├── scenario_extraction/
│   ├── run_matching.py         # Stage 2: Parquet → scenario match results
│   ├── parquet_source.py       # Parquet reader, TagFeatures reconstruction
│   ├── generic_window_extractor.py  # OSC2-driven parameter extraction
│   └── worker_source.py        # Legacy source interface
├── requirements_worker.txt
├── requirements_matcher.txt
└── tests/
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- AWS credentials configured (`aws configure` or IAM role)
- Access to Waymo Open Dataset TFRecords

### Local test run (single shard)

```bash
# 1. install dependencies
pip install -r requirements_worker.txt

# 2. run worker locally against a single TFRecord
LOCAL_MODE=1 \
LOCAL_INPUT=data \
LOCAL_OUTPUT=test_output \
SHARD_INDEX=0 \
TOTAL_SHARDS=1000 \
python worker.py

# 3. run matcher against the produced Parquet files
cd scenario_extraction
python run_matching.py \
  --osc_file change_lane.osc \
  --base_prefix ../test_output/00000 \
  --local true \
  --out_dir ../test_matcher_out
```

### AWS Batch deployment

```bash
# 1. build and push worker image to ECR
docker build -f docker/worker.Dockerfile -t parquet-worker .
docker tag parquet-worker:latest \
  <account>.dkr.ecr.<region>.amazonaws.com/parquet-worker:latest
docker push \
  <account>.dkr.ecr.<region>.amazonaws.com/parquet-worker:latest

# 2. create Batch compute environment, job queue, and job definition
#    (see docs/batch-setup.md for full CLI commands)

# 3. submit one job per TFRecord shard
python submit_jobs.py
```

### Scenario matching at scale

Once Parquet files are on S3, submit matcher jobs — one per shard:

```bash
python submit_jobs.py --stage matcher --osc_file change_lane.osc
```

Results per shard land at:

```
s3://<bucket>/results/change_lane.osc/<shard>/stats_shard.json
s3://<bucket>/results/change_lane.osc/<shard>/example_windows.jsonl
```

Merge all shard stats into a single result with:

```bash
python scenario_extraction/merge_stats.py \
  --input_prefix results/change_lane.osc \
  --output results/change_lane_merged.json
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
| Distributed processing | AWS Batch (primary), Kubernetes (alternative) |
| Storage | Amazon S3 |
| Containerization | Docker |
| Statistics | Custom mergeable shard statistics with parameter histograms |

---

## Data Source

This project processes data from the [Waymo Open Dataset](https://waymo.com/open/). Access requires registration with Waymo. TFRecord files are not included in this repository.

---

## License

MIT