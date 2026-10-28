"""Writing ARC's tables (the VDT and AP databases, the curve files and the representative cross sections) as CSV or
Parquet.

Parquet is written with brotli at level 5, with dictionary encoding only on the columns where under 30% of the values
differ. pyarrow's default gives every column a dictionary, which for columns whose values mostly differ (water surface
elevations, top widths, most discharges) stores every value plus an index for every row. The files hold the same
columns and values either way; they're just smaller and faster to write.

A run's Parquet files and rasters also say what made them (run_metadata): the ARC version, and the inputs as JSON.
CSV files have nowhere to say it; the decision log and the log file do (Print_Decision_Log, Log_File).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DICTIONARY_FRACTION = 0.3  # a column is dictionary-encoded when fewer than this fraction of its values differ
BROTLI_LEVEL = 5


def run_metadata(configs) -> dict[str, str]:
    """What made a run's outputs, for their Parquet key-value metadata and their rasters' tags: arc_version, ARC's
    version, and arc_configs, the inputs as JSON (Configs.from_mapping(json.loads(...)) makes them again)."""
    from arc import __version__
    return {"arc_version": __version__, "arc_configs": json.dumps(configs.as_mapping(), default=str)}


def with_metadata(table: pa.Table, metadata: dict[str, str] | None) -> pa.Table:
    """The table with these keys and values added to its schema's metadata."""
    if not metadata:
        return table
    return table.replace_schema_metadata({**(table.schema.metadata or {}),
                                          **{k.encode(): v.encode() for k, v in metadata.items()}})


def write_table(df: pd.DataFrame, path: os.PathLike, metadata: dict[str, str] | None = None) -> None:
    """Write a table as CSV or, for a .parquet path, Parquet with these keys and values in its metadata (see the notes
    above)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith('.parquet'):
        repeated = [col for col in df.columns if df[col].nunique() < DICTIONARY_FRACTION * len(df)]
        table = with_metadata(pa.Table.from_pandas(df, preserve_index=False), metadata)
        pq.write_table(table, path, compression='brotli', compression_level=BROTLI_LEVEL, use_dictionary=repeated)
    else:
        df.to_csv(path, index=False)
