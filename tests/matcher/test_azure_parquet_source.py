import io
from unittest.mock import MagicMock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scenario_extraction.parquet_source import AzureParquetSource

import os

from scenario_extraction.parquet_source import AzureParquetSource


def test_azure_parquet_source_reads_real_blob():
    source = AzureParquetSource(
        account_name=os.environ["AZURE_STORAGE_ACCOUNT"],
        account_key=os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
        container="parquets",
        base_prefix="parquet/test_feature_parquets/00000",
    )

    result = next(iter(source), None)

    assert result is not None, "No usable scene found in Azure"

    assert result.feats_by_seg
    assert result.seg_meta_by_id

