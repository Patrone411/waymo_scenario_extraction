from pyspark.sql import SparkSession
from feature_extraction.pipeline import process_scenario
from feature_extraction.tools.scenario import Scenario

from feature_extraction.tools.scenario import features_description
import tensorflow as tf
import os

os.environ["PYSPARK_PYTHON"] = r"C:\Users\I010444\AppData\Local\anaconda3\envs\waymo\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\I010444\AppData\Local\anaconda3\envs\waymo\python.exe"
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


# -----------------------------
# SPARK PARTITION LOGIC
# -----------------------------
def test_partition(paths):

    for path in paths:

        for example in load_tfrecord(path):

            try:
                result = process_scenario(example)
                print(result)
                if result is None:
                    continue

                yield {
                    "data": result   # 🔥 GANZES DICT
                }

            except Exception as e:
                print("ERROR:", e)
                continue

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    dataset = tf.data.TFRecordDataset("data/tfexample.tfrecord-00000-of-01000")

    for raw in dataset:
        example = parse_example(raw)
        scenario = Scenario(example)
        scenario.setup()
        result = process_scenario(scenario)
        print(result)

    """spark = SparkSession.builder \
        .master("local[*]") \
        .appName("scenario-construction-test") \
        .getOrCreate()

    # 🔥 IMPORTANT: DO NOT load data into memory

    # Instead pass file paths to Spark
    paths = [
        "data/tfexample.tfrecord-00000-of-01000"
    ]

    # Spark distributes file paths
    rdd = spark.sparkContext.parallelize(paths, numSlices=1)

    # run pipeline
    result = rdd.mapPartitions(test_partition)

    # convert to DataFrame
    out = spark.createDataFrame(result)

    out.show(truncate=False)

    spark.stop()"""