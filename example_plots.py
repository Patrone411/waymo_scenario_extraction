from scenario_plot import plot_hit, plot_hits_grid
import pyarrow.dataset as ds
import pyarrow.fs as pafs

# Hits von S3 laden
s3   = pafs.S3FileSystem(region="eu-central-1")
hits = ds.dataset("womd-features/results/match_hits", filesystem=s3,
                   format="parquet", partitioning="hive").to_table().to_pandas()

# einen Hit plotten
hit = hits[hits["scenario"] == "change_lane.osc"].iloc[2]
print(hits[hits["scenario"] == "change_lane.osc"].head(9)[["scene_id", "shard_index", "source_uri"]])

plot_hit(hit, scenes_dir="s3://womd-features/parquet/run-001/00000/scenes")

# Grid mit den ersten 9 Hits
plot_hits_grid(
    hits[hits["scenario"] == "change_lane.osc"].head(9),
    scenes_dir="s3://womd-features/parquet/run-001/00000/scenes",
    #save_path="change_lane_grid.png",
)