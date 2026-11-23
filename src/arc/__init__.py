"""
Automated Rating Curve (ARC).

ARC builds a rating curve (discharge, velocity, top width and water surface elevation) for each stream cell of the
reaches in its flow file, from cross sections sampled across the stream.

Entry points
------------
- arc.Arc, or the ``arc`` command: run ARC from an input file or a dictionary (arc.pipeline.run).
- arc.streamflow_processing.Create_ARC_Streamflow_Input: make ARC's flow file from GEOGLOWS data.
- arc.process_geospatial_data.Process_ARC_Geospatial_Data: make ARC's input rasters and input file.
- arc.create_geojson.Run_Main_Curve_to_GEOJSON_Program_Stream_Vector: turn ARC's curve file into GeoJSON for flood
  mapping.
- arc.release_freed_memory: hand the memory a run has freed back to the system, as a caller that goes on after it
  should once it has let go of the run's results (arc._memory).

The entry points are imported when first used, so that a run doesn't import the flow file tools' s3fs and xarray.
"""
import importlib

from ._log import LOG

_ENTRY_POINTS = {'Arc': '.arc', 'Run_Main_Curve_to_GEOJSON_Program_Stream_Vector': '.create_geojson',
                 'Create_ARC_Streamflow_Input': '.streamflow_processing',
                 'Process_ARC_Geospatial_Data': '.process_geospatial_data', 'release_freed_memory': '._memory'}

__all__ = ['LOG', 'Arc', 'Create_ARC_Streamflow_Input', 'Process_ARC_Geospatial_Data',
           'Run_Main_Curve_to_GEOJSON_Program_Stream_Vector', 'release_freed_memory']
__version__ = '2026.11.23'


def __getattr__(name):
    if name in _ENTRY_POINTS:
        value = getattr(importlib.import_module(_ENTRY_POINTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_ENTRY_POINTS))
