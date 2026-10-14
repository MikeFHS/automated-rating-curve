"""
Automated Rating Curve (ARC).

ARC builds a rating curve (discharge, velocity, top width and water surface elevation) for every stream cell of a
DEM, from cross sections sampled across the stream.

Entry points
------------
- arc.Arc, or the ``arc`` command: run ARC from an input file or a dictionary (arc.pipeline.run).
- arc.streamflow_processing.Create_ARC_Streamflow_Input: make ARC's flow file from GEOGLOWS data.
- arc.process_geospatial_data.Process_ARC_Geospatial_Data: make ARC's input rasters and input file.
- arc.Create_GeoJSON.Run_Main_Curve_to_GEOJSON_Program_Stream_Vector: turn ARC's curve file into GeoJSON for flood
  mapping.
"""

from ._log import LOG
from .arc import Arc
from .Create_GeoJSON import Run_Main_Curve_to_GEOJSON_Program_Stream_Vector
from .streamflow_processing import Create_ARC_Streamflow_Input
from .process_geospatial_data import Process_ARC_Geospatial_Data

__all__ = ['Arc', 'Run_Main_Curve_to_GEOJSON_Program_Stream_Vector', 'Create_ARC_Streamflow_Input', 'Process_ARC_Geospatial_Data']
__version__ = '0.4.0'
