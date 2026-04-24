# test_read.py
import io, json
from pathlib import Path
import pyarrow.parquet as pq
from scenario_extraction.parquet_source import ParquetSource, SegmentFeatures, _decode_sparse

# ── read directly from local file (no S3 needed) ──────────────────────────
path = next(Path("test_output/scenes").glob("*.parquet"))
df   = pq.read_table(path).to_pandas()

print(f"\nfile: {path.name}")
print(f"rows (segments): {len(df)}")
print(f"columns: {list(df.columns)}\n")

for _, row in df.iterrows():
    feats = SegmentFeatures(row.to_dict())
    print(f"─── segment {feats.seg_id} ───")
    print(f"  num_lanes:    {feats.num_lanes}")
    print(f"  num_segments: {feats.num_segments}")
    print(f"  valid:        {feats.valid}")
    print(f"  T:            {feats.T}")
    print(f"  actor_ids:    {feats.actor_ids}")
    print(f"  length_m:     {feats.length_m:.1f}")

    # check a few actor time series
    for actor_id in feats.actor_ids[:2]:
        ts = feats.actor_ts.get(actor_id) or {}
        x  = ts.get("x") or []
        n_valid = sum(1 for v in x if v is not None)
        print(f"  actor {actor_id}: {n_valid}/91 valid x timesteps")

    # check sparse inter-actor
    pairs = list(feats._inter_actor_sparse.keys())
    print(f"  non-empty inter-actor pairs: {len(pairs)}")
    if pairs:
        key = pairs[0]
        a, b = key.split("|", 1)
        ttc  = feats.get_inter_actor(a, b, "ttc")
        n_ttc = sum(1 for v in ttc if v is not None)
        dist  = feats.get_inter_actor(a, b, "eucl_distance")
        n_dist = sum(1 for v in dist if v is not None)
        print(f"  example pair {key}: {n_ttc} valid ttc, {n_dist} valid dist timesteps")

    # check geometry loaded
    print(f"  reference_line coords: {len((feats.reference_line or {}).get('coordinates') or [])}")
    print(f"  tl_results: {len(feats.tl_results)}")
    print(f"  cw_results: {len(feats.cw_results)}")
    print()