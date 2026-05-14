"""
Automated Rating Curve (ARC).

This package exposes the public entry points for running ARC from Python or the
command line.

Primary entry points
--------------------
- :class:`arc.arc.Arc`: High-level runner for ARC simulations.
- :func:`arc.process_geospatial_data.Process_ARC_Geospatial_Data`: Utilities for preparing
  geospatial inputs.
- :func:`arc.streamflow_processing.Create_ARC_Streamflow_Input`: Utilities for preparing
  streamflow inputs.
- :func:`arc.Create_GeoJSON.Run_Main_Curve_to_GEOJSON_Program_Stream_Vector`: Convert ARC
  outputs to GeoJSON.

Notes
-----
Most of ARC's core computation lives in
:mod:`arc.Automated_Rating_Curve_Generator`.
"""

from ._log import LOG
from .arc import Arc
from .Create_GeoJSON import Run_Main_Curve_to_GEOJSON_Program_Stream_Vector
from .streamflow_processing import Create_ARC_Streamflow_Input
from .process_geospatial_data import Process_ARC_Geospatial_Data

__all__ = ['Arc', 'Run_Main_Curve_to_GEOJSON_Program_Stream_Vector', 'Create_ARC_Streamflow_Input', 'Process_ARC_Geospatial_Data']
__version__ = '0.3.0'
