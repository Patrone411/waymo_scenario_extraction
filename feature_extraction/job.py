

from core.spark import create_spark
from core.io import read_tfrecord
from pipelines.feature_extraction.processor import process_scene


def process_partition(rows):

    for row in rows:
        yield process_scene(row.asDict())


if __name__ == "__main__":

    spark = create_spark("feature_extraction")

    df = read_tfrecord(spark, "data/scenes/")

    df = df.repartition(300)

    result = df.rdd.mapPartitions(process_partition)

    spark.createDataFrame(result) \
        .write.mode("overwrite") \
        .parquet("output/features/")