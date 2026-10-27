"""A vector file on disk (shapefile, GeoPackage, GeoJSON, GeoParquet...), read with GDAL through pyogrio and
geopandas.

A Vector reads its metadata from the file when it's made, and its features whenever they're asked for.
"""
from __future__ import annotations

import json
import os
import warnings
from functools import cached_property
from pathlib import Path

import geopandas as gpd
import pyogrio
from pyproj import CRS

_PARQUET = (".parquet", ".geoparquet", ".pq")


class Vector:
    """A layer of features in a vector file."""

    def __init__(self, filepath: os.PathLike, layer: str | int | None = None, use_threads: bool = True):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Cannot find the vector file {self.filepath}.")
        
        self.layer = layer
        self.use_threads = use_threads
        if self.filepath.suffix.lower() in _PARQUET:
            self._projection, self._bbox, self._columns, self._feature_count = self._parquet_info()
            return
        
        info = pyogrio.read_info(self.filepath, layer=layer, force_total_bounds=True)
        if info["crs"] is not None:
            self._projection = info["crs"]
        else:
            warnings.warn(f"Could not determine the CRS of {self.filepath}. Assuming EPSG:4326.", stacklevel=2)
            self._projection = "EPSG:4326"

        self._bbox = tuple(float(v) for v in info["total_bounds"])
        self._columns = tuple(str(name) for name in info["fields"])
        self._feature_count = int(info["features"])

    def __repr__(self) -> str:
        return f"Vector({str(self.filepath)!r})"

    def _parquet_info(self):
        """A GeoParquet file's CRS, bounds, columns and row count, from its own metadata with pyarrow, which doesn't
        need GDAL's Parquet driver."""
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(self.filepath)
        geo = json.loads((parquet.metadata.metadata or {}).get(b"geo", b"{}").decode("utf-8"))
        geometry_column = geo.get("primary_column", "geometry")
        column = geo.get("columns", {}).get(geometry_column, {})
        crs = column.get("crs", "missing")
        if crs is None or crs == "missing":
            # GeoParquet leaves the CRS out for OGC:CRS84, longitude and latitude
            projection = "EPSG:4326"
        elif isinstance(crs, dict) and "id" in crs:
            projection = ":".join(map(str, crs["id"].values()))
        else:
            projection = CRS.from_user_input(crs).to_wkt()

        bbox = column.get("bbox")
        if bbox is None:
            bbox = gpd.read_parquet(self.filepath, columns=[geometry_column]).total_bounds

        columns = tuple(name for name in parquet.schema_arrow.names if name != geometry_column)
        return projection, tuple(float(v) for v in bbox), columns, int(parquet.metadata.num_rows)

    # --- Metadata ----------------------------------------------------------------------------------------------

    @property
    def projection(self) -> str:
        return self._projection

    @cached_property
    def crs(self) -> CRS:
        return CRS.from_user_input(self._projection)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self._bbox

    @property
    def columns(self) -> tuple[str, ...]:
        """The attribute fields, without the geometry."""
        return self._columns

    def __len__(self) -> int:
        return self._feature_count

    # --- Features ----------------------------------------------------------------------------------------------

    def to_geopandas(self) -> gpd.GeoDataFrame:
        """The features, with their geometry."""
        if self.filepath.suffix.lower() in _PARQUET:
            return gpd.read_parquet(self.filepath, use_threads=self.use_threads)
        
        return gpd.read_file(self.filepath, layer=self.layer, use_arrow=True)
