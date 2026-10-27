"""The curve file: power laws fitted to each stream cell's rating curve, or to each reach's.

Depth, top width and velocity are each fitted as a * Q**b by least squares (scipy's curve_fit). The depth is the
water surface elevation less BaseElev, the elevation of the centre of the cross section (Elev in the VDT database).
qmax is the flow file's maximum-flow column, indexed by its IDs, which goes in the QMax column.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from numba import njit
from numba.core.errors import TypingError

from arc import LOG
from arc.outputs.rating_curves import METADATA_COLUMNS, WSE, Q, RatingCurves, T, V
from arc.outputs.tables import write_table

FAILED_FIT = -9999.9  # the coefficient, power and R² when a fit fails
# Initial guesses for a and b
TOP_WIDTH_GUESS = (12.0, 0.3)
VELOCITY_GUESS = (1.0, 0.3)
DEPTH_GUESS = (0.2, 0.5)
_METADATA = ['COMID', 'Row', 'Col', 'BaseElev', 'QBaseflow', 'Slope', 'XS_Angle']


@njit(cache=True)
def _power(d_value: np.ndarray, d_coefficient: float, d_power: float):
    """a * x**b, compiled."""
    return d_coefficient * (d_value ** d_power)


def fit_power_law(x: np.ndarray, y: np.ndarray, init_guess=(1.0, 1.0)) -> tuple[float, float, float]:
    """a, b and R² of y = a * x**b. R² isn't worked out, so it's -9999.9, and so are a and b when the fit doesn't
    converge."""
    coefficient, power, r2 = FAILED_FIT, FAILED_FIT, FAILED_FIT
    try:
        coefficient, power = _least_squares(x, y, init_guess)
    except TypingError as e:
        LOG.error(e)
    except RuntimeError:
        pass
    return coefficient, power, r2


def _least_squares(x: np.ndarray, y: np.ndarray, init_guess) -> np.ndarray:
    """a and b, as scipy's curve_fit(_power, x, y, p0=init_guess) fits them. This function gives the same numbers, 
    in half the time, without the covariance and the checks on it that curve_fit works out and fit_power_law doesn't keep. 
    It raises ValueError for values that aren't finite, TypeError for fewer than two of them, and RuntimeError where lmdif
    fails. With a SciPy that doesn't have lmdif where this looks for it, curve_fit itself."""
    try:
        from scipy.optimize._minpack import (
            _lmdif,  # here, as importing it takes a quarter of a second
        )
    except ImportError:
        from scipy.optimize import curve_fit
        return curve_fit(_power, x, y, p0=list(init_guess))[0]
    x, y = np.asarray_chkfinite(x, float), np.asarray_chkfinite(y, float)
    p0 = np.atleast_1d(np.asarray(init_guess, dtype=float))
    if p0.size > y.size:
        raise TypeError(f"The number of func parameters={p0.size} must not exceed the number of data points={y.size}")

    def residuals(params):
        return _power(x, *params) - y  # as curve_fit's

    # leastsq's defaults: ftol, xtol, gtol, maxfev, epsfcn (the residuals' type's epsilon), factor and diag
    popt, info = _lmdif(residuals, p0, (), 0, 1.49012e-08, 1.49012e-08, 0.0, 200 * (p0.size + 1),
                        np.finfo(np.float64).eps, 100, None)
    if info not in (1, 2, 3, 4):
        raise RuntimeError(f"Optimal parameters not found: MINPACK's lmdif returned {info}.")
    
    return popt


def _round_metadata(df: pd.DataFrame) -> pd.DataFrame:
    # Most columns are rounded to 3 decimals, but the angle and slope keep more precision
    for col in df.columns:
        if col not in ('Slope', 'XS_Angle'):
            df[col] = df[col].round(3)
    df['XS_Angle'] = df['XS_Angle'].round(7)
    df['Slope'] = df['Slope'].round(8)
    return df


def curve_file_dataframe(curves: RatingCurves, qmax: pd.Series) -> pd.DataFrame:
    """Each cell's power laws, for the cells whose metadata is complete and whose coefficients are all positive."""
    df = pd.DataFrame(curves.metadata, columns=METADATA_COLUMNS).rename(columns={'Elev': 'BaseElev'})[_METADATA]
    df = df.dropna()
    for col in ['COMID', 'Row', 'Col']:
        df[col] = df[col].astype(int)
    df = df.rename(columns={'QBaseflow': 'QMax'})
    df['QMax'] = df['COMID'].map(qmax)
    df = _round_metadata(df)

    fits = {name: [] for name in ('depth_a', 'depth_b', 'tw_a', 'tw_b', 'vel_a', 'vel_b')}
    for i in df.index:
        q = curves.increments[i, :, Q]
        v = curves.increments[i, :, V]
        t = curves.increments[i, :, T]
        depth = curves.increments[i, :, WSE] - df.loc[i, 'BaseElev']
        mask = ~np.isnan(q)
        if not mask.all():
            q, t, v, depth = q[mask], t[mask], v[mask], depth[mask]
        tw_a, tw_b, _ = fit_power_law(q, t, TOP_WIDTH_GUESS)
        vel_a, vel_b, _ = fit_power_law(q, v, VELOCITY_GUESS)
        depth_a, depth_b, _ = fit_power_law(q, depth, DEPTH_GUESS)
        for name, value in (('depth_a', depth_a), ('depth_b', depth_b), ('tw_a', tw_a), ('tw_b', tw_b),
                            ('vel_a', vel_a), ('vel_b', vel_b)):
            fits[name].append(value)

    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(fits).round(3)], axis=1)
    return df.loc[(df['depth_a'] > 0) & (df['tw_a'] > 0) & (df['vel_a'] > 0)]


def reach_average_curve_file_dataframe(curves: RatingCurves, vdt_df: pd.DataFrame, qmax: pd.Series) -> pd.DataFrame:
    """Each reach's power laws, fitted to all of its cells' rating curves in the VDT database, alongside each of
    its cells' metadata. As in curve_file_dataframe, a reach whose fit fails, or gives a coefficient that isn't
    positive, is left out."""
    n = curves.number_of_increments
    df = pd.DataFrame(curves.metadata, columns=METADATA_COLUMNS).dropna(how='all')
    df = df.rename(columns={'Elev': 'BaseElev'})[_METADATA]
    for col in ['COMID', 'Row', 'Col']:
        df[col] = df[col].astype(int)
    df = df.rename(columns={'QBaseflow': 'QMax'})
    df['QMax'] = df['COMID'].map(qmax)
    df = _round_metadata(df)

    q_columns = [f'q_{i}' for i in range(1, n + 1)]
    t_columns = [f't_{i}' for i in range(1, n + 1)]
    v_columns = [f'v_{i}' for i in range(1, n + 1)]
    wse_columns = [f'wse_{i}' for i in range(1, n + 1)]
    fits = {name: [] for name in ('COMID', 'depth_a', 'depth_b', 'tw_a', 'tw_b', 'vel_a', 'vel_b')}
    for comid, group in vdt_df.groupby("COMID"):
        # The reach's cells that are in the VDT database, matched on their row and column
        group_index = pd.MultiIndex.from_arrays([group["Row"].values, group["Col"].values], names=["Row", "Col"])
        matching = df[(df["COMID"] == comid) & (pd.MultiIndex.from_frame(df[["Row", "Col"]]).isin(group_index))]
        matching = matching.drop_duplicates(subset=["Row", "Col", "COMID"])
        if matching.empty:
            LOG.warning(f"No matching BaseElev values found for COMID {comid}. Skipping...")
            continue

        aligned = group.set_index(["Row", "Col"]).join(matching.set_index(["Row", "Col"])["BaseElev"], how="inner")
        depth = np.concatenate([aligned[c].values - aligned["BaseElev"].values for c in wse_columns])
        q = np.concatenate([group[c].values for c in q_columns])
        t = np.concatenate([group[c].values for c in t_columns])
        v = np.concatenate([group[c].values for c in v_columns])
        try:
            tw_a, tw_b, _ = fit_power_law(q, t, TOP_WIDTH_GUESS)
            vel_a, vel_b, _ = fit_power_law(q, v, VELOCITY_GUESS)
            depth_a, depth_b, _ = fit_power_law(q, depth, DEPTH_GUESS)
        except Exception as e:
            LOG.warning(f"Regression failed for COMID {comid}: {e}")
            tw_a = tw_b = vel_a = vel_b = depth_a = depth_b = np.nan
            
        fits['COMID'].append(comid)
        for name, value in (('depth_a', depth_a), ('depth_b', depth_b), ('tw_a', tw_a), ('tw_b', tw_b),
                            ('vel_a', vel_a), ('vel_b', vel_b)):
            fits[name].append(np.round(value, 3) if not np.isnan(value) else np.nan)

    df = df.merge(pd.DataFrame(fits), on="COMID", how="left").dropna()
    return df.loc[(df['depth_a'] > 0) & (df['tw_a'] > 0) & (df['vel_a'] > 0)]


def write_curve_file(curves: RatingCurves, qmax: pd.Series, path: os.PathLike) -> pd.DataFrame:
    """Write each cell's power laws, as CSV or, for a .parquet path, Parquet, and return them."""
    df = curve_file_dataframe(curves, qmax)
    write_table(df, path)
    LOG.info('Finished writing ' + str(path))
    return df


def write_reach_average_curve_file(curves: RatingCurves, vdt_df: pd.DataFrame, qmax: pd.Series,
                                   path: os.PathLike) -> pd.DataFrame:
    """Write each reach's power laws with its cells' metadata, as CSV or Parquet, and return them."""
    df = reach_average_curve_file_dataframe(curves, vdt_df, qmax)
    write_table(df, path)
    LOG.info('Finished writing ' + str(path))
    return df
