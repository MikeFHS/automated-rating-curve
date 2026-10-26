"""Each stream cell's rating curve, and the VDT and AP databases written from them.

A RatingCurves holds, per stream cell, its metadata (COMID, row, column, elevation, baseflow, slope and
cross-section angle), and per flow increment its discharge, velocity, top width, water surface elevation and wetted
perimeter. A cell without a rating curve is NaN throughout, but with the reach-average curve file arc.pipeline fills
in the metadata of a cell without a usable cross section (its slope 0, its baseflow left NaN), for the reach's power
laws to be given to it too.

With Valley_Sections it also holds each cell's valley slope (Valley_Slope, after XS_Angle in the VDT database): the
slope its overbanks flow at, the stream's slope times the cross section's slope ratio (arc.xsection.valley.slope_ratio),
or the stream's slope where its overbanks follow the stream. Over a meandering river's floodplain the water runs down
the valley, and Curve2Flood takes its velocity there from it. Without Valley_Sections there's no such column.

The elevation (Elev) is that of the centre of the cross section the rating curve was worked out on: the carved
channel bed with bathymetry, and otherwise the DEM (or a manual cross section's own). So a water surface elevation
less Elev is the depth of the water there. The curve file calls it BaseElev.
"""
from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import pandas as pd

from arc import LOG
from arc.outputs.tables import write_table

METADATA_COLUMNS = ['COMID', 'Row', 'Col', 'Elev', 'QBaseflow', 'Slope', 'XS_Angle']
VALLEY_SLOPE_COLUMN = 'Valley_Slope'
INCREMENT_FIELDS = ['q', 'v', 't', 'wse', 'p']
(COMID, ROW, COL, ELEV, QBASEFLOW, SLOPE, XS_ANGLE) = range(7)
(Q, V, T, WSE, P) = range(5)


class RatingCurves(NamedTuple):
    """Each stream cell's rating curve (see the notes above)."""
    metadata: np.ndarray  # (cells, 7), in the order of METADATA_COLUMNS
    increments: np.ndarray  # (cells, increments, 5), in the order of INCREMENT_FIELDS
    valley_slopes: np.ndarray | None = None  # (cells,), with Valley_Sections

    @classmethod
    def empty(cls, cells: int, increments: int, valley_slopes: bool = False) -> RatingCurves:
        return cls(np.full((cells, len(METADATA_COLUMNS)), np.nan),
                   np.full((cells, increments, len(INCREMENT_FIELDS)), np.nan),
                   np.full(cells, np.nan) if valley_slopes else None)

    @property
    def number_of_increments(self) -> int:
        return self.increments.shape[1]

    def columns(self) -> list[str]:
        valley = [VALLEY_SLOPE_COLUMN] if self.valley_slopes is not None else []
        return METADATA_COLUMNS + valley + [f"{prefix}_{i}" for i in range(1, self.number_of_increments + 1)
                                            for prefix in INCREMENT_FIELDS]

    def output_array(self) -> np.ndarray:
        """One row per cell: the metadata, the valley slope if there are valley slopes, then q, v, t, wse and p for each
        increment in turn."""
        valley = [self.valley_slopes[:, None]] if self.valley_slopes is not None else []
        return np.concatenate([self.metadata, *valley, self.increments.reshape(self.increments.shape[0], -1)], axis=1)

    def has_vdt_data(self) -> bool:
        """Whether any cell has its last increment."""
        return bool(np.any(~np.isnan(self.increments[:, -1, P]))) if self.number_of_increments else False


def vdt_dataframe(curves: RatingCurves) -> pd.DataFrame:
    """The VDT database: the cells with complete rating curves, without their wetted perimeters, and without rows
    that repeat or have a negative discharge, velocity or top width. Values are rounded to 3 decimals, but the slopes
    to 8 and the cross-section angle to 7."""
    vdt_df = pd.DataFrame(curves.output_array(), columns=curves.columns())
    vdt_df = vdt_df.drop(columns=[col for col in vdt_df.columns if col.startswith('p_')])
    vdt_df = vdt_df.dropna()
    vdt_df = vdt_df.drop_duplicates()
    for col in ['COMID', 'Row', 'Col']:
        vdt_df[col] = vdt_df[col].astype(int)
    slopes = [col for col in ('Slope', VALLEY_SLOPE_COLUMN) if col in vdt_df.columns]
    for col in vdt_df.columns:
        if col not in ('XS_Angle', *slopes):
            vdt_df[col] = vdt_df[col].round(3)
    vdt_df['XS_Angle'] = vdt_df['XS_Angle'].round(7)
    vdt_df[slopes] = vdt_df[slopes].round(8)
    cols_to_check = [col for col in vdt_df.columns if col.startswith('q') or col.startswith('t') or col.startswith('v')]
    return vdt_df.loc[~(vdt_df[cols_to_check] < 0).any(axis=1)]


def write_vdt(curves: RatingCurves, path: os.PathLike) -> pd.DataFrame:
    """Write the VDT database, as CSV or, for a .parquet path, Parquet, and return it."""
    vdt_df = vdt_dataframe(curves)
    write_table(vdt_df, path)
    LOG.info('Finished writing ' + str(path))
    return vdt_df


def ap_dataframe(curves: RatingCurves) -> pd.DataFrame:
    """The AP database: each complete rating curve's discharge, area (discharge over velocity, 0 where the velocity
    is 0) and wetted perimeter, rounded to 3 decimals, without rows that repeat or have a negative value."""
    n = curves.number_of_increments
    ap_df = pd.DataFrame(curves.output_array(), columns=curves.columns())
    ap_df = ap_df.drop(columns=['Elev', 'QBaseflow', 'Slope', 'XS_Angle']
                       + [col for col in ap_df.columns if col == VALLEY_SLOPE_COLUMN or col.startswith('t_')
                          or col.startswith('wse_')])
    ap_df = ap_df.dropna()
    ap_df = ap_df.drop_duplicates()
    for col in ['COMID', 'Row', 'Col']:
        ap_df[col] = ap_df[col].astype(int)
    for i in range(1, n + 1):
        ap_df[f'a_{i}'] = ap_df[f'q_{i}'].div(ap_df[f'v_{i}'], fill_value=0)
        ap_df.loc[ap_df[f'v_{i}'] == 0, f'a_{i}'] = 0
    column_order = ['COMID', 'Row', 'Col'] + [col for i in range(1, n + 1) for col in (f'q_{i}', f'a_{i}', f'p_{i}')]
    ap_df = ap_df[column_order].round(3)
    cols_to_check = [col for col in ap_df.columns if col.startswith('q') or col.startswith('a') or col.startswith('p')]
    return ap_df.loc[~(ap_df[cols_to_check] < 0).any(axis=1)]


def write_ap(curves: RatingCurves, path: os.PathLike) -> pd.DataFrame:
    """Write the AP database, as CSV or, for a .parquet path, Parquet, and return it."""
    ap_df = ap_dataframe(curves)
    write_table(ap_df, path)
    LOG.info('Finished writing ' + str(path))
    return ap_df
