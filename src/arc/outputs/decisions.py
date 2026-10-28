"""ARC's decision log and quality flags.

For each stream cell of the reaches in the flow file, the decision log (Print_Decision_Log) records what each step of
the run decided: where its slope came from, how its banks were found, how its channel was carved, whether it got a
rating curve (and if not, why), and what the standard step did with its water. QA_Flag sums the flags below that apply
to the cell, 0 for none, and QA names them. A flag marks a cell to look at, not an error: most runs have many.

The VDT database, the curve file and the AP database carry each cell's QA_Flag too, as their last column.

| Flag | Value | The cell |
| --- | --- | --- |
| SLOPE_FROM_NEIGHBOURS | 1 | Its reach had no slope of its own, and took its neighbours' in the stream network. |
| LOW_SPOT_MOVED | 2 | Its cross section is centred on lower ground beside the stream cell (Low_Spot_Range). |
| NO_BANKS_FOUND | 4 | Neither the water mask, the land cover nor the DEM showed its banks: the channel is as wide as the width power law says, or one cell. |
| NARROW_CHANNEL | 8 | Its channel is under arc.xsection.subcell.NARROW_CELLS cells across, where the DEM's cells blur it (see the docs' limitations), and Subcell_Channels didn't reconstruct it. |
| CARVED_TO_STREAM | 16 | Without banks of its own, its channel was carved to carry the baseflow at its stream cell's level (Bathy_Bankless_Max_Baseflow). |
| NO_SLOPE | 32 | It was skipped: neither the DEM nor the stream network gave its reach a slope. |
| NO_CROSS_SECTION | 64 | It has no cross section: its centre had no data. |
| REACH_DROPPED | 128 | Its reach was dropped from the bathymetry and the rating curves: no bank elevation anywhere (Bathy_Use_Banks). |
| NO_RATING_CURVE | 256 | Its cross section gave no rating curve: the discharge at the top is more than half the maximum flow away from it. |
| EDGE_DROPPED | 512 | Its rating curve was dropped, its water reaching the raster's edge (Drop_Edge_Sections). |
| CURVE_SPILLS | 1024 | Its rating curve's top discharge isn't the maximum flow: its water spills over a bank there. |
| STEP_HELD | 2048 | The standard step held its water at its baseflow level at some flow. |
| STEP_CRITICAL | 4096 | The standard step put its water at critical depth at some flow: no subcritical water surface was there. |
"""
from __future__ import annotations

import math
import os
from enum import IntFlag

import numpy as np
import pandas as pd

from arc import LOG
from arc.outputs.rating_curves import Q
from arc.outputs.tables import write_table
from arc.xsection.subcell import NARROW_CELLS

# Where a cell's slope came from (arc.pipeline.cell_slopes): its own, its reach's, or its neighbours' in the network
SLOPE_SOURCES = ("cell", "reach", "neighbours")
SLOPE_FROM_NEIGHBOURS = SLOPE_SOURCES.index("neighbours")
# Bank methods for which the banks weren't found (arc.bathymetry.banks.METHODS)
_NOT_FOUND = ("none", "single_cell", "target_width")
_SPILL_TOLERANCE = 1e-3  # a top discharge this share or more from the maximum flow isn't it

DECISION_LOG_COLUMNS = ["COMID", "Row", "Col", "Stream_Row", "Stream_Col", "Baseflow", "Max_Flow", "Slope",
                        "Slope_Source", "Bank_Method", "Channel_Width", "Bank_Elevation", "Carve", "Carve_Depth",
                        "Valley_Section", "Subcell", "Rating_Curve", "Step_Held", "Step_Critical", "QA_Flag", "QA"]


class QA(IntFlag):
    """The quality flags (see the notes above)."""
    SLOPE_FROM_NEIGHBOURS = 1
    LOW_SPOT_MOVED = 2
    NO_BANKS_FOUND = 4
    NARROW_CHANNEL = 8
    CARVED_TO_STREAM = 16
    NO_SLOPE = 32
    NO_CROSS_SECTION = 64
    REACH_DROPPED = 128
    NO_RATING_CURVE = 256
    EDGE_DROPPED = 512
    CURVE_SPILLS = 1024
    STEP_HELD = 2048
    STEP_CRITICAL = 4096


def qa_names(flag: int) -> str:
    """The names of the flags in a QA_Flag, joined by |, or "" for 0."""
    return "|".join(member.name for member in QA if flag & member)


def channel_width(section) -> float:
    """The width between the banks a section's hydraulics use, in metres, or NaN without them."""
    banks = section.hydraulic_banks if section.hydraulic_banks is not None else section.banks
    if banks is None:
        return math.nan
    return float(banks.left + banks.right)


def is_narrow(section) -> bool:
    """Whether a section's channel is under NARROW_CELLS of its ordinate spacings across."""
    width = channel_width(section)
    return bool(np.isfinite(width) and width < NARROW_CELLS * float(section.xs.ordinate_distance))


def rating_curve_status(results, k: int) -> str:
    """Whether cell k got a rating curve, or why not: rated, no_cross_section, reach_dropped, edge or no_curve ("" for
    a run that makes no rating curves)."""
    section = results.sections[k]
    if results.curves is None:
        return ""
    if section is None:
        return "no_cross_section"
    if not section.usable:
        return "reach_dropped"
    if np.isfinite(results.curves.increments[k, -1, Q]):
        return "rated"
    return "edge" if section.dropped_at_edge else "no_curve"


def qa_flags(results) -> np.ndarray:
    """Each cell's QA_Flag, in the order of results.cells (see the notes above)."""
    cells, sections, curves, step = results.cells, results.sections, results.curves, results.step
    flags = np.zeros(cells.count, dtype=np.int64)
    for k in range(cells.count):
        flag = QA(0)
        if results.slope_sources is not None and results.slope_sources[k] == SLOPE_FROM_NEIGHBOURS:
            flag |= QA.SLOPE_FROM_NEIGHBOURS
        section = sections[k]
        if section is None:
            flag |= QA.NO_CROSS_SECTION
        else:
            if (section.row, section.col) != (int(cells.rows[k]), int(cells.cols[k])) and not section.manual:
                flag |= QA.LOW_SPOT_MOVED
            if section.banks is not None and section.banks.method in _NOT_FOUND:
                flag |= QA.NO_BANKS_FOUND
            if is_narrow(section) and not section.xs.subcell:
                flag |= QA.NARROW_CHANNEL
            if section.carve == "stream_level":
                flag |= QA.CARVED_TO_STREAM
            if not section.usable:
                flag |= QA.REACH_DROPPED

        status = rating_curve_status(results, k)
        if status == "edge":
            flag |= QA.EDGE_DROPPED
        elif status == "no_curve":
            flag |= QA.NO_RATING_CURVE
        elif status == "rated":
            top = float(curves.increments[k, -1, Q])
            if abs(top / float(results.max_flow[k]) - 1.0) > _SPILL_TOLERANCE:
                flag |= QA.CURVE_SPILLS

        if step is not None and step.held_at is not None:
            if step.held_at[k]:
                flag |= QA.STEP_HELD
            if step.critical_at[k]:
                flag |= QA.STEP_CRITICAL
        flags[k] = int(flag)

    return flags


def decision_log_dataframe(results) -> pd.DataFrame:
    """The decision log: a row for each stream cell the run worked on, then one for each it skipped for want of a
    slope (see the notes above). Row and Col are the centre of its cross section, as in the VDT database, and
    Stream_Row and Stream_Col its stream cell."""
    cells, sections = results.cells, results.sections
    flags = results.qa_flags if results.qa_flags is not None else qa_flags(results)
    step = results.step
    rows = []
    for k in range(cells.count):
        section = sections[k]
        row, col = (section.row, section.col) if section is not None else (int(cells.rows[k]), int(cells.cols[k]))
        source = SLOPE_SOURCES[results.slope_sources[k]] if results.slope_sources is not None else ""
        rows.append({
            "COMID": int(cells.comids[k]), "Row": row, "Col": col,
            "Stream_Row": int(cells.rows[k]), "Stream_Col": int(cells.cols[k]),
            "Baseflow": float(results.baseflow[k]), "Max_Flow": float(results.max_flow[k]),
            "Slope": float(results.slopes[k]), "Slope_Source": source,
            "Bank_Method": "" if section is None or section.banks is None else section.banks.method,
            "Channel_Width": math.nan if section is None else channel_width(section),
            "Bank_Elevation": math.nan if section is None else section.bank_elevation,
            "Carve": "" if section is None else section.carve,
            "Carve_Depth": math.nan if section is None else section.carve_depth,
            "Valley_Section": section is not None and section.xs.valley is not None,
            "Subcell": section is not None and section.xs.subcell,
            "Rating_Curve": rating_curve_status(results, k),
            "Step_Held": int(step.held_at[k]) if step is not None and step.held_at is not None else 0,
            "Step_Critical": int(step.critical_at[k]) if step is not None and step.critical_at is not None else 0,
            "QA_Flag": int(flags[k]), "QA": qa_names(int(flags[k])),
        })

    skipped = results.skipped
    if skipped is not None:
        for k in range(skipped.count):
            rows.append({"COMID": int(skipped.comids[k]), "Row": int(skipped.rows[k]), "Col": int(skipped.cols[k]),
                         "Stream_Row": int(skipped.rows[k]), "Stream_Col": int(skipped.cols[k]),
                         "Baseflow": math.nan, "Max_Flow": math.nan, "Slope": math.nan, "Slope_Source": "",
                         "Bank_Method": "", "Channel_Width": math.nan, "Bank_Elevation": math.nan, "Carve": "",
                         "Carve_Depth": math.nan, "Valley_Section": False, "Subcell": False,
                         "Rating_Curve": "no_slope", "Step_Held": 0, "Step_Critical": 0,
                         "QA_Flag": int(QA.NO_SLOPE), "QA": QA.NO_SLOPE.name})

    df = pd.DataFrame(rows, columns=DECISION_LOG_COLUMNS)
    for col in ("Baseflow", "Max_Flow", "Channel_Width", "Bank_Elevation", "Carve_Depth"):
        df[col] = df[col].round(3)
    df["Slope"] = df["Slope"].round(8)
    return df


def summary(results) -> str:
    """How many cells each flag applies to, for the log."""
    flags = results.qa_flags if results.qa_flags is not None else qa_flags(results)
    counts = [(member.name, int(np.count_nonzero(flags & member))) for member in QA]
    skipped = 0 if results.skipped is None else results.skipped.count
    counts = [(name, count + (skipped if name == QA.NO_SLOPE.name else 0)) for name, count in counts]
    shown = ", ".join(f"{name} {count}" for name, count in counts if count)
    return f"QA flags over {results.cells.count + skipped} stream cells: " + (shown or "none")


def write_decision_log(results, path: os.PathLike, metadata: dict[str, str] | None = None) -> pd.DataFrame:
    """Write the decision log, as CSV or, for a .parquet path, Parquet (with metadata, see arc.outputs.tables), and
    return it."""
    df = decision_log_dataframe(results)
    write_table(df, path, metadata)
    LOG.info('Finished writing ' + str(path))
    return df
