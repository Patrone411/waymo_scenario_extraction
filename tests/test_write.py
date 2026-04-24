# test_write.py
import json
from pathlib import Path
from feature_extraction.worker import (
    stream_tfrecord,
    make_serializable,
    scene_to_parquet_rows,
    SCENE_SCHEMA,
    process_shard,
)
from feature_extraction.pipeline import process_scenario
from feature_extraction.tools.scenario import Scenario
import pyarrow as pa
import pyarrow.parquet as pq

TFRECORD_PATH = "data/training_tfexample.tfrecord-00000-of-01000"
OUT_DIR       = Path("test_output/scenes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for example in stream_tfrecord(TFRECORD_PATH):
    try:
        scenario = Scenario(example)
        scenario.setup()
        result = process_scenario(scenario)
        if result is None:
            continue

        result = make_serializable(result)
        rows   = scene_to_parquet_rows(result)
        if not rows:
            continue

        scene_id = result["scene_id"]
        table    = pa.Table.from_pylist(rows, schema=SCENE_SCHEMA)
        out_path = OUT_DIR / f"{scene_id}.parquet"
        pq.write_table(table, out_path, compression="snappy")

        print(f"wrote {out_path}  ({len(rows)} segments, {out_path.stat().st_size / 1024:.1f} KB)")
        break   # just one scene for now

    except Exception as e:
        import traceback
        traceback.print_exc()
        continue