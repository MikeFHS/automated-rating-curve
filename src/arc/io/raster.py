"""A raster file on disk, read and written with GDAL.

A Raster reads its grid and georeferencing from the file when it's made, and its values whenever they're asked for.
It doesn't keep the file open, so the file can be overwritten or deleted while the Raster is still around.
"""
from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

import numpy as np
from osgeo import gdal, gdal_array
from pyproj import CRS, Geod

gdal.UseExceptions()

_METRES = {"metre", "metres", "meter", "meters"}
_TILE = 512  # the side of a compressed raster's tiles, in cells
# The stored types that GDAL converts to each type exactly as numpy's astype does, every value kept
_EXACTLY_AS = {np.dtype(np.float64): {np.dtype(t) for t in (np.float32, np.int8, np.uint8, np.int16, np.uint16,
                                                             np.int32, np.uint32)},
               np.dtype(np.float32): {np.dtype(t) for t in (np.int8, np.uint8, np.int16, np.uint16)}}


def decode_threads() -> str:
    """The threads to decode a raster on: GDAL_NUM_THREADS as it's set, as a GDAL configuration option or an
    environment variable (which GDAL reads in that order), or else every CPU."""
    return gdal.GetConfigOption("GDAL_NUM_THREADS") or "ALL_CPUS"


class Raster:
    """A single-band raster file (the first band of a multi-band one)."""

    def __init__(self, filepath: os.PathLike):
        self.filepath = Path(filepath)
        if not self.filepath.is_file():
            raise FileNotFoundError(f"Cannot find the raster {self.filepath}.")
        ds = gdal.Open(str(self.filepath), gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        self._geotransform = tuple(float(v) for v in ds.GetGeoTransform())
        self._projection = ds.GetProjection()
        self._shape = (int(ds.RasterYSize), int(ds.RasterXSize))
        self._nodata_value = band.GetNoDataValue()
        self._dtype = np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType))

    def __repr__(self) -> str:
        return f"Raster({str(self.filepath)!r})"

    # --- The grid -----------------------------------------------------------------------------------------------

    @property
    def geotransform(self) -> tuple[float, float, float, float, float, float]:
        return self._geotransform

    @property
    def projection(self) -> str:
        """The coordinate reference system, as WKT."""
        return self._projection

    @property
    def shape(self) -> tuple[int, int]:
        """(nrows, ncols)"""
        return self._shape

    @property
    def nrows(self) -> int:
        return self._shape[0]

    @property
    def ncols(self) -> int:
        return self._shape[1]

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def nodata_value(self) -> float | None:
        return self._nodata_value

    @property
    def resolution(self) -> tuple[float, float]:
        """(x_resolution, y_resolution) in the raster's own units, both positive."""
        return abs(self._geotransform[1]), abs(self._geotransform[5])

    @property
    def is_rotated(self) -> bool:
        return self._geotransform[2] != 0.0 or self._geotransform[4] != 0.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) in the raster's own coordinates, ignoring any rotation."""
        gt = self._geotransform
        x0, x1 = gt[0], gt[0] + self.ncols * gt[1]
        y0, y1 = gt[3], gt[3] + self.nrows * gt[5]
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    @cached_property
    def crs(self) -> CRS:
        return CRS.from_wkt(self._projection)

    def axis_units(self) -> set[str]:
        """The units of the coordinate system's axes, in lower case."""
        return {(axis.unit_name or "").strip().lower() for axis in self.crs.axis_info if axis is not None} - {""}

    def cell_size_in_metres(self) -> tuple[float, float]:
        """The cells' width and height in metres.

        A geographic raster's cells are measured along the geodesics of its ellipsoid at the latitude half way
        between its top and bottom edges. A projected raster's are its resolution, which must be in metres.
        """
        size_x, size_y = self.resolution
        if self.crs.is_geographic:
            first_row_edge = self._geotransform[3]
            last_row_edge = first_row_edge + (-1.0 if self._geotransform[5] < 0.0 else 1.0) * self.nrows * size_y
            latitude = (last_row_edge + first_row_edge) / 2.0
            geod = self._geod()
            _, _, dy = geod.inv(0.0, latitude, 0.0, latitude + size_y)
            _, _, dx = geod.inv(0.0, latitude, size_x, latitude)
            return abs(float(dx)), abs(float(dy))
        if self.crs.is_projected:
            units = self.axis_units()
            if not units <= _METRES:
                raise ValueError(f"The raster {self.filepath} is projected in {', '.join(sorted(units))}, not metres.")
            return float(size_x), float(size_y)
        raise ValueError(f"The raster {self.filepath} is neither geographic nor projected.")

    def _geod(self) -> Geod:
        # The geodesic on the raster's ellipsoid, from its semi-major axis and inverse flattening, or else WGS 84
        ellipsoid = self.crs.ellipsoid
        if ellipsoid is not None and ellipsoid.semi_major_metre:
            if ellipsoid.inverse_flattening:
                return Geod(a=ellipsoid.semi_major_metre, rf=ellipsoid.inverse_flattening)
            return Geod(a=ellipsoid.semi_major_metre, b=ellipsoid.semi_minor_metre)  # a sphere
        return Geod(ellps="WGS84")

    def same_grid(self, other: Raster, tolerance: float = 1e-6) -> bool:
        """Whether another raster has this one's shape and cells, to within tolerance of a cell."""
        if self.shape != other.shape:
            return False
        size = max(self.resolution)
        return all(abs(a - b) <= tolerance * size for a, b in zip(self._geotransform, other._geotransform))

    # --- Values ------------------------------------------------------------------------------------------------

    def read_array(self, dtype: np.dtype | type | None = None) -> np.ndarray:
        """The first band's values, as stored or converted to dtype.

        The file is decoded on as many threads as GDAL_NUM_THREADS says, as a GDAL configuration option or an
        environment variable, and on every CPU if it's not set (decode_threads). Decoding on more than one also decodes
        it straight into the array rather than through GDAL's block cache. GDAL converts the values as it reads them
        where that's exact, so a float32 DEM read as float64 takes its float64 array and no more."""
        dtype = None if dtype is None else np.dtype(dtype)
        with gdal.config_option("GDAL_NUM_THREADS", decode_threads()):
            ds = gdal.Open(str(self.filepath), gdal.GA_ReadOnly)
            band = ds.GetRasterBand(1)
            if dtype is not None and dtype != self._dtype and self._dtype in _EXACTLY_AS.get(dtype, ()):
                return band.ReadAsArray(buf_type=gdal_array.NumericTypeCodeToGDALTypeCode(dtype))
            array = band.ReadAsArray()
        return array if dtype is None else array.astype(dtype, copy=False)

    @classmethod
    def write_array(cls, array: np.ndarray, reference: Raster, output_path: os.PathLike, *,
                    dtype: np.dtype | type | None = None, compression: str | None = "ZSTD",
                    nodata_value: float | None = None) -> Raster:
        """Write a single-band GeoTIFF on the reference raster's grid, and return it.

        A compressed raster is written in 512 x 512 tiles without a predictor, which suits ARC's bathymetry and flood
        rasters: they are almost all empty, and tiles hand the compressor whole empty blocks. A nodata value is only
        set if given.
        """
        array = np.asarray(array)
        if array.shape != reference.shape:
            raise ValueError(f"The array is {array.shape}, but the reference raster is {reference.shape}.")
        dtype = array.dtype if dtype is None else np.dtype(dtype)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        driver: gdal.Driver = gdal.GetDriverByName("GTiff")
        if output_path.exists():
            driver.Delete(str(output_path))
        options = []
        if compression and compression.upper() != "NONE":
            options += [f"COMPRESS={compression.upper()}", "TILED=YES", f"BLOCKXSIZE={_TILE}", f"BLOCKYSIZE={_TILE}"]
        ds = driver.Create(str(output_path), xsize=array.shape[1], ysize=array.shape[0], bands=1,
                           eType=gdal_array.NumericTypeCodeToGDALTypeCode(dtype), options=options)
        ds.SetGeoTransform(reference.geotransform)
        ds.SetProjection(reference.projection)
        band = ds.GetRasterBand(1)
        if nodata_value is not None:
            band.SetNoDataValue(float(nodata_value))
        band.WriteArray(array.astype(dtype, copy=False))
        band.FlushCache()
        band = None
        ds = None  # closes the file
        return cls(output_path)
