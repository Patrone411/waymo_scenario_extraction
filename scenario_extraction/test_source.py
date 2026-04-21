# test_source.py
import io
from pathlib import Path
import pyarrow.parquet as pq
from parquet_source import SegmentFeatures, SegmentMeta, SceneResult


class LocalParquetSource:
    """Same interface as ParquetSource but reads from a local directory."""
    def __init__(self, scenes_dir: str, min_lanes=None):
        self.scenes_dir = Path(scenes_dir)
        self.min_lanes  = min_lanes

    def __iter__(self):
        for path in sorted(self.scenes_dir.glob("*.parquet")):
            df = pq.read_table(path).to_pandas()
            feats_by_seg = {}
            meta_by_id   = {}
            for _, row in df.iterrows():
                num_lanes = int(row.get("num_lanes") or 0)
                if self.min_lanes and num_lanes < self.min_lanes:
                    continue
                feats  = SegmentFeatures(row.to_dict())
                seg_id = feats.seg_id
                meta   = SegmentMeta(
                    source_uri=str(path),
                    seg_id=seg_id,
                    num_lanes=num_lanes,
                    length_m=feats.length_m,
                )
                feats_by_seg[seg_id] = feats
                meta_by_id[seg_id]   = meta
            if feats_by_seg:
                yield SceneResult(feats_by_seg, meta_by_id)


# ── simulate what run_one_prefix does ─────────────────────────────────────
"""src = LocalParquetSource("test_output/scenes", min_lanes=2)

for scene_result in src:
    print(f"scene has {len(scene_result.feats_by_seg)} segments")
    for seg_id, feats in scene_result.feats_by_seg.items():
        print(f"  {seg_id}: {len(feats.actor_ids)} actors, {len(feats._inter_actor_sparse)} pairs")
    meta = next(iter(scene_result.seg_meta_by_id.values()))
    print(f"  source_uri: {meta.source_uri}")
    break"""