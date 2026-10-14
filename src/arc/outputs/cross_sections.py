"""The cross-section file (XS_Out_File): each stream cell's cross section, as tab-separated text or as Parquet.

The file has a row per cross section. Its profiles and Manning's n run out from the stream cell on each side; side 1
is towards the higher rows (or, along a row, the higher columns). The text file prints them as numpy prints arrays,
to 6 decimals. A .parquet file keeps them as lists of float64, exactly as ARC used them: nothing to parse, and not
rounded.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from arc import LOG

XS_EXPORT_COLUMNS = ['COMID', 'Row', 'Col', 'XS1_Profile', 'Ordinate_Dist', 'Manning_N_Raster1', 'XS2_Profile',
                     'Manning_N_Raster2', 'r1', 'c1', 'r2', 'c2']
_PROFILES = ('XS1_Profile', 'XS2_Profile')
_ROUGHNESS = ('Manning_N_Raster1', 'Manning_N_Raster2')
_PARQUET_SCHEMA = pa.schema([(name, pa.list_(pa.float64()) if name in _PROFILES + _ROUGHNESS
                              else pa.float64() if name == 'Ordinate_Dist' else pa.int64())
                             for name in XS_EXPORT_COLUMNS])
# The elevations' bytes split into streams (their high bytes barely change along a profile) and the n values, which
# take a few values, dictionary-encoded, then zstd: a 1° tile's 43,000 cross sections take 43 MB, against 131 MB of
# text. A list's values are addressed by their path, '<column>.list.element'; the bare name is silently ignored.
_PARQUET_OPTIONS = dict(compression='zstd', compression_level=3,
                        use_byte_stream_split=[f'{name}.list.element' for name in _PROFILES],
                        use_dictionary=[name for name in XS_EXPORT_COLUMNS if name not in _PROFILES + _ROUGHNESS]
                        + [f'{name}.list.element' for name in _ROUGHNESS])


class CrossSectionRecord(NamedTuple):
    """One row of the cross-section file. The profiles and n run out from the stream cell, which starts both, and
    (r1, c1) and (r2, c2) are the cells at the end of each side."""
    comid: int
    row: int
    col: int
    xs1_profile: np.ndarray
    ordinate_dist: float
    manning_n_raster1: np.ndarray
    xs2_profile: np.ndarray
    manning_n_raster2: np.ndarray
    r1: int
    c1: int
    r2: int
    c2: int


def format_array(values: np.ndarray) -> str:
    """The text np.array2string(values, precision=6, max_line_width=np.inf, threshold=np.inf, floatmode='fixed')
    gives, built directly for a 1-D float array without exponents, which is several times faster.

    Every number is rounded to 6 decimals and padded on the left to the widest whole part, as numpy does. Arrays
    numpy would print with exponents (values of 1e8 or more, or a smallest non-zero value below 0.0001 or under a
    thousandth of the largest), or without any finite values, go to np.array2string itself.
    """
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating) or values.size == 0:
        return _array2string(values)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        return _array2string(values)
    magnitudes = np.abs(finite_values[finite_values != 0])
    if magnitudes.size:
        largest, smallest = magnitudes.max(), magnitudes.min()
        with np.errstate(over='ignore'):
            if largest >= 10.0 ** min(8, np.finfo(values.dtype).precision) or smallest < 0.0001 \
                    or largest / smallest > 1000.0:
                return _array2string(values)
    texts = ['%.6f' % x for x in values.tolist()]
    whole = max(len(text) - 7 for text, is_finite in zip(texts, finite) if is_finite)  # the part before '.'
    width = whole + 7
    parts = []
    for text, is_finite in zip(texts, finite):
        if not is_finite:
            text = 'nan' if text == 'nan' else ('-inf' if text.startswith('-') else 'inf')
        parts.append(text.rjust(width))
    return '[' + ' '.join(parts) + ']'


def _array2string(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, max_line_width=np.inf, threshold=np.inf, floatmode='fixed')


def cross_section_dataframe(records: list[CrossSectionRecord]) -> pd.DataFrame:
    """The cross-section file's table, with the arrays as text."""
    records = [record for record in records if record is not None]
    df = pd.DataFrame([record._asdict() for record in records])
    if df.empty:
        return pd.DataFrame(columns=XS_EXPORT_COLUMNS)
    df.columns = XS_EXPORT_COLUMNS
    for col in df.columns:
        df[col] = df[col].apply(lambda x: format_array(x) if isinstance(x, np.ndarray) else x)
    return df


def cross_section_table(records: list[CrossSectionRecord]) -> pa.Table:
    """The cross-section file's table for Parquet, with the profiles and n as lists of float64."""
    records = [record for record in records if record is not None]
    values = zip(*records) if records else [()] * len(XS_EXPORT_COLUMNS)
    columns = [_list_array(column) if pa.types.is_list(field.type) else pa.array(column, type=field.type)
               for field, column in zip(_PARQUET_SCHEMA, values)]
    return pa.Table.from_arrays(columns, schema=_PARQUET_SCHEMA)


def _list_array(arrays) -> pa.ListArray:
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum(np.fromiter((a.size for a in arrays), dtype=np.int64, count=len(arrays)), out=offsets[1:])
    values = np.concatenate(arrays).astype(np.float64, copy=False) if arrays else np.empty(0)
    # int32 offsets, which pa.array refuses to overflow
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.array(values, type=pa.float64()))


def write_cross_sections(records: list[CrossSectionRecord], path: os.PathLike) -> pd.DataFrame:
    """Write the cross-section file, tab-separated or, for a .parquet path, Parquet, and return its table (with the
    arrays as text or as arrays)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith('.parquet'):
        table = cross_section_table(records)
        pq.write_table(table, path, **_PARQUET_OPTIONS)
        df = table.to_pandas()
    else:
        df = cross_section_dataframe(records)
        df.to_csv(path, index=False, sep='\t')
    LOG.info('Finished writing ' + str(path))
    return df
