"""
azure_results_writer.py

Azure-Pendant zu `ResultsWriter` aus run_matching.py.
Sammelt Hit-Rows waehrend eines Shard-Runs und schreibt am Ende
drei Parquet-Tabellen nach Azure Blob Storage (oder optional lokal).

Blob-Partitionierung (Hive-kompatibel, z.B. fuer Athena/Synapse):
  {prefix}/{table}/scenario={scenario}/run_id={run_id}/shard={N:05d}.parquet

Nutzt dieselben Schemas (HITS_SCHEMA, ACTOR_FRAMES_SCHEMA, PAIR_FRAMES_SCHEMA)
wie die S3-Variante in run_matching.py, damit Downstream-Queries (Athena/DuckDB
Aequivalent) unveraendert funktionieren.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from run_matching import HITS_SCHEMA, ACTOR_FRAMES_SCHEMA, PAIR_FRAMES_SCHEMA, _safe_val, _safe_pair, _safe_pair_str


def get_blob_credential(account_key: Optional[str]):
    """
    Authentication, analog zu worker.py's get_blob_credential():
    1. account_key gesetzt (AZURE_STORAGE_KEY) -> Storage Account Key
       (z.B. bestehende CI-Pipeline)
    2. Kein Storage Key -> Azure Workload Identity
       (z.B. AKS)
    """
    if account_key:
        print("[azure] authentication: Storage Account Key", flush=True)
        return account_key

    print("[azure] authentication: Azure Workload Identity", flush=True)

    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


class AzureResultsWriter:
    """
    Analog zu ResultsWriter, schreibt aber nach Azure Blob Storage
    (account_name + AZURE_STORAGE_KEY, wie worker.py) statt S3.

    Ohne gesetzten account_key faellt die Auth auf Workload Identity
    (DefaultAzureCredential) zurueck, wie in worker.py.

    Bei gesetztem `local_dir` wird stattdessen lokal geschrieben (fuer
    lokale Testlaeufe / Debugging ohne Azure-Zugriff).
    """

    def __init__(
        self,
        run_id: str,
        scenario: str,
        shard_index: int,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        container: str = "results",
        prefix: str = "results",
        local_dir: Optional[str] = None,
    ):
        self.run_id      = run_id
        self.scenario    = scenario
        self.shard_index = shard_index

        self.account_name = account_name
        self.account_key  = account_key
        self.container    = container
        self.prefix       = prefix.rstrip("/")
        self.local_dir    = local_dir

        self._hits:         list = []
        self._actor_frames: list = []
        self._pair_frames:  list = []

        self._blob_service = None  # lazy init, siehe _blob_service_client()

    # -----------------------------------------------------------------
    # Hit sammeln (identische Logik/Felder wie ResultsWriter.add_hit)
    # -----------------------------------------------------------------
    def add_hit(
        self,
        *,
        scene_id: str,
        segment_id: str,
        block_label: str,
        roles: dict,
        t0: int,
        t1: int,
        n_windows: int,
        n_possible_windows: int,
        source_uri: str,
        feats,
    ) -> None:
        base = dict(
            run_id=self.run_id,
            scenario=self.scenario,
            scene_id=scene_id,
            segment_id=segment_id,
            t0=int(t0),
            t1=int(t1),
        )

        self._hits.append({
            **base,
            "shard_index":        self.shard_index,
            "block_label":        block_label,
            "roles_json":         json.dumps(roles),
            "n_windows":          int(n_windows),
            "n_possible_windows": int(n_possible_windows),
            "source_uri":         source_uri,
        })

        role_list = list(roles.items())

        for frame_label, t in [("start", t0), ("end", t1)]:

            for role, actor_id in role_list:
                self._actor_frames.append({
                    **base,
                    "role":        role,
                    "actor_id":    actor_id,
                    "frame":       frame_label,
                    "t":           int(t),
                    "x":           _safe_val(feats.x,         actor_id, t),
                    "y":           _safe_val(feats.y,         actor_id, t),
                    "yaw":         _safe_val(feats.yaw,       actor_id, t),
                    "speed":       _safe_val(feats.speed,     actor_id, t),
                    "accel":       _safe_val(feats.accel,     actor_id, t),
                    "s":           _safe_val(feats.s,         actor_id, t),
                    "t_lat":       _safe_val(feats.t,         actor_id, t),
                    "s_dot":       _safe_val(feats.s_dot,     actor_id, t),
                    "t_dot":       _safe_val(feats.t_dot,     actor_id, t),
                    "yaw_delta":   _safe_val(feats.yaw_delta, actor_id, t),
                    "osc_lane_id": _safe_val(feats.lane_idx,  actor_id, t),
                })

            for i in range(len(role_list)):
                for j in range(len(role_list)):
                    if i == j:
                        continue
                    role_a, actor_a = role_list[i]
                    role_b, actor_b = role_list[j]
                    self._pair_frames.append({
                        **base,
                        "role_a":       role_a,
                        "role_b":       role_b,
                        "actor_a":      actor_a,
                        "actor_b":      actor_b,
                        "frame":        frame_label,
                        "t":            int(t),
                        "rel_distance": _safe_pair(
                            feats.rel_distance, actor_a, actor_b, t),
                        "ttc":          _safe_pair(
                            feats.ttc, actor_a, actor_b, t),
                        "rel_position": _safe_pair_str(
                            feats.rel_position, actor_a, actor_b, t),
                        "lat_rel":      _safe_pair_str(
                            feats.lat_rel, actor_a, actor_b, t),
                    })

    # -----------------------------------------------------------------
    # Flush: drei Tabellen nach Azure Blob Storage (oder lokal) schreiben
    # -----------------------------------------------------------------
    def flush(self) -> dict:
        written = {}
        for table_name, rows, schema in [
            ("match_hits",         self._hits,         HITS_SCHEMA),
            ("match_actor_frames", self._actor_frames, ACTOR_FRAMES_SCHEMA),
            ("match_pair_frames",  self._pair_frames,  PAIR_FRAMES_SCHEMA),
        ]:
            if not rows:
                print(f"[results] {table_name}: keine Rows", flush=True)
                continue

            table = pa.Table.from_pylist(rows, schema=schema)

            if self.local_dir:
                out = Path(self.local_dir) / table_name
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"shard_{self.shard_index:05d}.parquet"
                pq.write_table(table, path, compression="snappy")
                written[table_name] = str(path)
            else:
                blob_path = (
                    f"{self.prefix}/{table_name}"
                    f"/scenario={self.scenario}"
                    f"/run_id={self.run_id}"
                    f"/shard={self.shard_index:05d}.parquet"
                )
                buf = io.BytesIO()
                pq.write_table(table, buf, compression="snappy")
                buf.seek(0)

                blob_client = self._blob_service_client().get_blob_client(
                    container=self.container,
                    blob=blob_path,
                )
                blob_client.upload_blob(buf, overwrite=True)
                written[table_name] = (
                    f"azure://{self.account_name}/{self.container}/{blob_path}"
                )

            print(f"[results] {len(rows):>6} rows -> {written[table_name]}", flush=True)

        return written

    # -----------------------------------------------------------------
    # Azure-Client (lazy, analog zu ResultsWriter._s3)
    # -----------------------------------------------------------------
    def _blob_service_client(self):
        if self._blob_service is not None:
            return self._blob_service

        from azure.storage.blob import BlobServiceClient

        if not self.account_name:
            raise ValueError(
                "AzureResultsWriter benoetigt account_name "
                "(oder local_dir fuer lokale Laeufe ohne Azure-Zugriff)."
            )

        account_url = f"https://{self.account_name}.blob.core.windows.net"
        self._blob_service = BlobServiceClient(
            account_url=account_url,
            credential=get_blob_credential(self.account_key),
        )
        return self._blob_service