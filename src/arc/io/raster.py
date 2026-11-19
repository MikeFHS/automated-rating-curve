"""A raster file on disk, read and written with GDAL.

A Raster reads its grid and georeferencing from the file when it's made, and its values whenever they're asked for.
It doesn't keep the file open, so the file can be overwritten or deleted while the Raster is still around.
"""
from __future__ import annotations

import math
import os
from functools import cached_property
from pathlib import Path

import numpy as np
from osgeo import gdal, gdal_array
from pyproj import CRS, Geod

gdal.UseExceptions()

_METRES = {"metre", "metres", "meter", "meters"}
# The decimals, of a metre, a geographic raster's cell size is rounded to (see Raster.cell_size_in_metres)
CELL_SIZE_DECIMALS = 6
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

        A geographic raster's cells are measured on its ellipsoid at the latitude half way between its top and bottom
        edges (cell_size_on_ellipsoid), to the micrometre (CELL_SIZE_DECIMALS). A projected raster's are its
        resolution, which must be in metres.

        Every slope, direction and cross section is worked out from the cells' size, and a difference in its last
        digits moved the rating curves and the bathymetry by up to metres in a few cells: rounded, the size is the
        same on every system (see cell_size_on_ellipsoid).
        """
        size_x, size_y = self.resolution
        if self.crs.is_geographic:
            first_row_edge = self._geotransform[3]
            last_row_edge = first_row_edge + (-1.0 if self._geotransform[5] < 0.0 else 1.0) * self.nrows * size_y
            latitude = (last_row_edge + first_row_edge) / 2.0
            geod = self._geod()
            dx, dy = cell_size_on_ellipsoid(geod.a, geod.es, latitude, size_x, size_y)
            return round(dx, CELL_SIZE_DECIMALS), round(dy, CELL_SIZE_DECIMALS)
        
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
            ds: gdal.Dataset = gdal.Open(str(self.filepath), gdal.GA_ReadOnly)
            band: gdal.Band = ds.GetRasterBand(1)
            if dtype is not None and dtype != self._dtype and self._dtype in _EXACTLY_AS.get(dtype, ()):
                return band.ReadAsArray(buf_type=gdal_array.NumericTypeCodeToGDALTypeCode(dtype))
            
            array = band.ReadAsArray()

        return array if dtype is None else array.astype(dtype, copy=False)

    @classmethod
    def write_array(cls, array: np.ndarray, reference: Raster, output_path: os.PathLike, *,
                    dtype: np.dtype | type | None = None, compression: str | None = "ZSTD",
                    nodata_value: float | None = None, metadata: dict[str, str] | None = None) -> Raster:
        """Write a single-band GeoTIFF on the reference raster's grid, and return it.

        A compressed raster is written in 512 x 512 tiles without a predictor, which suits ARC's bathymetry raster: it
        is almost all empty, and tiles hand the compressor whole empty blocks. A nodata value is only set if given, and
        metadata, keys and values, go in the file's tags (as arc.outputs.tables.run_metadata gives them).
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
        if metadata:
            ds.SetMetadata(dict(metadata))
            
        band = ds.GetRasterBand(1)
        if nodata_value is not None:
            band.SetNoDataValue(float(nodata_value))
            
        band.WriteArray(array.astype(dtype, copy=False))
        band.FlushCache()
        band = None
        ds = None  # closes the file
        return cls(output_path)


def cell_size_on_ellipsoid(a: float, e2: float, latitude: float, size_x: float, size_y: float) -> tuple[float, float]:
    """The width and height in metres of a cell size_x by size_y degrees from a latitude to latitude + size_y, on an
    ellipsoid of semi-major axis a (m) and squared eccentricity e2.

    Its height is the meridian's arc across it, by Simpson's rule on the meridian's radius of curvature; its width the
    geodesic across it along the latitude: the parallel's arc, less its curvature's share, (size_x sin(latitude))^2 /
    24. For cells of up to 36 arcseconds, both are pyproj's geodesics (Geod.inv) to within 2e-9 m, the geodesics' own
    error. A geodesic is worked out from two distances from the equator, so a last-digit difference in sin or cos from
    one system's C math library to another's moved it by 6e-10 m between Windows and Linux: enough to round
    differently where a size falls that near a micrometre's half, about one latitude in a thousand (the 1-arcsecond
    tiles at 50 N and 49 S, for example). These move by 1e-14 m.
    """
    phi, d_phi, d_lambda = math.radians(latitude), math.radians(size_y), math.radians(size_x)

    def meridian_radius(at: float) -> float:
        return a * (1.0 - e2) / (1.0 - e2 * math.sin(at) ** 2) ** 1.5

    height = d_phi / 6.0 * (meridian_radius(phi) + 4.0 * meridian_radius(phi + d_phi / 2.0)
                            + meridian_radius(phi + d_phi))
    prime_vertical_radius = a / math.sqrt(1.0 - e2 * math.sin(phi) ** 2)
    width = prime_vertical_radius * math.cos(phi) * d_lambda * (1.0 - (d_lambda * math.sin(phi)) ** 2 / 24.0)
    return abs(width), abs(height)
