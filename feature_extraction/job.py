from pyspark.sql import SparkSession
from feature_extraction.pipeline import process_scenario
from feature_extraction.tools.scenario import Scenario

from feature_extraction.tools.scenario import features_description
import tensorflow as tf
import os

import json
import numpy as np
from shapely.geometry import mapping

def make_serializable(obj):
    # plain Python float NaN/Inf
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None
    # numpy scalar float NaN/Inf
    if isinstance(obj, np.floating):
        if obj != obj or obj == np.inf or obj == -np.inf:
            return None
        return float(obj)
    # numpy integer
    if isinstance(obj, np.integer):
        return int(obj)
    # numpy array — recurse element by element so NaN/Inf get caught above
    if isinstance(obj, np.ndarray):
        return [make_serializable(x) for x in obj.flat] if obj.ndim == 1 \
            else [make_serializable(row) for row in obj]
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, '__geo_interface__'):
        return mapping(obj)
    if isinstance(obj, dict):
        return {
            int(k) if isinstance(k, np.integer)
            else float(k) if isinstance(k, np.floating)
            else str(k) if not isinstance(k, (str, int, float, bool, type(None)))
            else k
            : make_serializable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    return obj

python_exec = os.popen("which python").read().strip()

# -----------------------------
# TFRECORD PARSER
# -----------------------------
def parse_example(serialized_example):

    example = tf.io.parse_single_example(
        serialized_example,
        features_description
    )

    # convert tensors → python types
    return {k: v.numpy() for k, v in example.items()}


# -----------------------------
# TFRECORD STREAM READER
# -----------------------------
def load_tfrecord(path):

    dataset = tf.data.TFRecordDataset(path)

    for raw in dataset:
        yield parse_example(raw)

_saved_example = False  # module-level flag
# -----------------------------
# SPARK PARTITION LOGIC
# -----------------------------
def test_partition(paths):
    global _saved_example

    for path in paths:
        for example in load_tfrecord(path):
            try:
                scenario = Scenario(example)
                scenario.setup()
                result = process_scenario(scenario)
                if result is None:
                    continue

                clean = make_serializable(result)

                # save first successful result to disk for inspection
                if not _saved_example:
                    with open("example_output.json", "w") as f:
                        json.dump(clean, f, indent=2, allow_nan=False)
                    print("Saved example_output.json")
                    _saved_example = True

                yield (json.dumps(clean, allow_nan=False),)

            except Exception as e:
                print("ERROR:", e)
                continue


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    path = "data/training_tfexample.tfrecord-00000-of-01000"
    for i, example in enumerate(load_tfrecord(path)):
        print(f"[{i}] processing example...")
        try:
            scenario = Scenario(example)
            scenario.setup()
            result = process_scenario(scenario)
            print(f"[{i}] result: {type(result)} — {'None' if result is None else list(result.keys())}")
            if result is None:
                continue
            clean = make_serializable(result)
            with open("example_output.json", "w") as f:
                json.dump(clean, f, indent=2, allow_nan=False)
            print(f"Saved example_output.json at example {i}")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()  # full stack trace, not just the message
            continue

    else:
        print("Exhausted all examples — none produced a result")