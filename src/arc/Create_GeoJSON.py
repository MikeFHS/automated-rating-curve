# -*- coding: utf-8 -*-
"""GeoJSON export utilities for ARC outputs.

This module contains helper functions that convert ARC outputs (VDT databases
and curve files) into GeoJSON/feature layers for visualization and QA.

The functions in this file are not required to run the core ARC computation,
but are useful for downstream workflows and diagnostics.
"""

# built-in imports
import argparse
import os, sys
import json
from io import StringIO
import time


# third-party imports
import numpy as np
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import interp1d
from geojson import Point
import pandas as pd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
from pyproj import CRS

# local imports
# from streamflow_processing import Get_Raster_Details, Read_Raster_GDAL



def Get_Raster_Details(DEM_File):
    """Read basic spatial metadata from a raster.

    Parameters
    ----------
    DEM_File : str
        Path to a raster readable by GDAL.

    Returns
    -------
    tuple
        ``(minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, projection)``.
    """
    print(DEM_File)
    gdal.Open(DEM_File, gdal.GA_ReadOnly)
    data = gdal.Open(DEM_File)
    geoTransform = data.GetGeoTransform()
    ncols = int(data.RasterXSize)
    nrows = int(data.RasterYSize)
    minx = geoTransform[0]
    dx = geoTransform[1]
    maxy = geoTransform[3]
    dy = geoTransform[5]
    maxx = minx + dx * ncols
    miny = maxy + dy * nrows
    Rast_Projection = data.GetProjectionRef()
    data = None
    return minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection

def Read_Raster_GDAL(InRAST_Name):
    """Read a raster band into a NumPy array and return basic metadata.

    Parameters
    ----------
    InRAST_Name : str
        Input raster path.

    Returns
    -------
    tuple
        ``(array, ncols, nrows, cellsize, yll, yur, xll, xur, lat, geotransform, projection)``.
    """
    try:
        dataset = gdal.Open(InRAST_Name, gdal.GA_ReadOnly)     
    except RuntimeError:
        sys.exit(" ERROR: Field Raster File cannot be read!")
    # Retrieve dimensions of cell size and cell count then close DEM dataset
    geotransform = dataset.GetGeoTransform()
    # Continue grabbing geospatial information for this use...
    band = dataset.GetRasterBand(1)
    RastArray = band.ReadAsArray()
    #global ncols, nrows, cellsize, yll, yur, xll, xur
    ncols=band.XSize
    nrows=band.YSize
    band = None
    cellsize = geotransform[1]
    yll = geotransform[3] - nrows * np.fabs(geotransform[5])
    yur = geotransform[3]
    xll = geotransform[0];
    xur = xll + (ncols)*geotransform[1]
    lat = np.fabs((yll+yur)/2.0)
    Rast_Projection = dataset.GetProjectionRef()
    dataset = None
    print('Spatial Data for Raster File:')
    print('   ncols = ' + str(ncols))
    print('   nrows = ' + str(nrows))
    print('   cellsize = ' + str(cellsize))
    print('   yll = ' + str(yll))
    print('   yur = ' + str(yur))
    print('   xll = ' + str(xll))
    print('   xur = ' + str(xur))
    return RastArray, ncols, nrows, cellsize, yll, yur, xll, xur, lat, geotransform, Rast_Projection

def Get_Raster_CRS(raster_file, rast_projection):
    """
    Parse a raster CRS so points derived from raster row/column indices always
    inherit the raster's true coordinate system.
    """
    if not rast_projection:
        raise ValueError(f"Raster '{raster_file}' does not define a coordinate system.")

    try:
        return CRS.from_user_input(rast_projection)
    except Exception as exc:
        raise ValueError(f"Unable to parse the coordinate system from '{raster_file}'.") from exc

def Extract_Scalar_Value(value):
    """Return a plain scalar when ARC interpolation stores a 0-d array."""
    return value.item() if isinstance(value, np.ndarray) else value

def Prepare_Interpolation_Pairs(flow_values, target_values):
    """
    Prepare interpolation inputs so scipy never receives NaN/Inf pairs or a
    degenerate x-axis that can trigger divide warnings during slope creation.
    """
    # Force both arrays to floating point before filtering so mixed/object CSV
    # values become numeric and non-numeric entries are handled as invalid.
    flow_values = np.asarray(flow_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)

    # Keep only finite x/y pairs because interp1d will emit invalid divide
    # warnings when NaN/Inf values survive into the interpolation setup.
    valid_mask = np.isfinite(flow_values) & np.isfinite(target_values)
    flow_values = flow_values[valid_mask]
    target_values = target_values[valid_mask]

    # De-duplicate the flow axis after filtering so each remaining x-value maps
    # to one y-value before interpolation is attempted.
    flow_values, unique_idx = np.unique(flow_values, return_index=True)
    target_values = target_values[unique_idx]

    return flow_values, target_values

def Get_Distance_CRS(stream_raster_crs):
    """
    Use the raster CRS directly when it is already projected. For geographic
    rasters, switch to a projected CRS so nearest-distance calculations are in
    linear units instead of degrees.
    """
    if stream_raster_crs.is_geographic:
        return CRS.from_epsg(6933)
    return stream_raster_crs

def Build_Stream_Point_GDF(source_df, x_column, y_column, stream_raster_crs):
    """
    Build point geometries from raster-centered x/y coordinates and attach the
    stream raster CRS at creation time so the coordinates are never relabeled.
    """
    geometry = [Point(xy) for xy in zip(source_df[x_column], source_df[y_column])]
    return gpd.GeoDataFrame(source_df, geometry=geometry, crs=stream_raster_crs)

def Reproject_GDF_If_Needed(gdf, target_crs):
    """
    Keep geodataframes in a common CRS for spatial joins, only reprojecting
    when the current CRS differs from the requested target CRS.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame is missing a coordinate system.")
    if CRS.from_user_input(gdf.crs) == CRS.from_user_input(target_crs):
        return gdf
    return gdf.to_crs(target_crs)

def Read_Vector_GDF(vector_file):
    """
    Read vector data with extension-aware handling so GeoParquet inputs can be
    consumed alongside file-based GDAL formats.
    """
    if vector_file.lower().endswith((".parquet", ".geoparquet")):
        return gpd.read_parquet(vector_file)
    return gpd.read_file(vector_file)

def Write_Vector_GDF(gdf, vector_file):
    """
    Write vector data with extension-aware handling so callers can target
    GeoParquet without changing the rest of the workflow.
    """
    if vector_file.lower().endswith((".parquet", ".geoparquet")):
        gdf.to_parquet(vector_file, index=False)
        return
    gdf.to_file(vector_file)

def Read_Raster_With_Metadata(raster_file):
    """
    Read a raster band into memory and return the metadata needed for point
    sampling.
    """
    dataset = gdal.Open(raster_file, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Raster '{raster_file}' could not be opened.")

    band = dataset.GetRasterBand(1)
    raster_array = band.ReadAsArray()
    nodata_value = band.GetNoDataValue()
    geo_transform = dataset.GetGeoTransform()
    rast_projection = dataset.GetProjectionRef()
    band = None
    dataset = None

    return raster_array, geo_transform, rast_projection, nodata_value

def Sample_Raster_Value_At_Point(raster_array, geo_transform, point, nodata_value=None, max_search_radius=2):
    """
    Sample the raster cell under a point, expanding outward a few cells when
    the point lands on NoData.
    """
    transform_matrix = np.array(
        [
            [geo_transform[1], geo_transform[2]],
            [geo_transform[4], geo_transform[5]],
        ],
        dtype=float,
    )
    point_offset = np.array(
        [point.x - geo_transform[0], point.y - geo_transform[3]],
        dtype=float,
    )

    try:
        pixel_x, pixel_y = np.linalg.solve(transform_matrix, point_offset)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Raster geotransform is not invertible for point sampling.") from exc

    base_col = int(np.floor(pixel_x))
    base_row = int(np.floor(pixel_y))

    def is_valid_value(value):
        if value is None:
            return False
        if np.isnan(value):
            return False
        if nodata_value is not None and value == nodata_value:
            return False
        return True

    nrows, ncols = raster_array.shape
    for radius in range(max_search_radius + 1):
        candidates = []
        row_min = max(0, base_row - radius)
        row_max = min(nrows, base_row + radius + 1)
        col_min = max(0, base_col - radius)
        col_max = min(ncols, base_col + radius + 1)

        for row_index in range(row_min, row_max):
            for col_index in range(col_min, col_max):
                value = raster_array[row_index, col_index]
                if not is_valid_value(value):
                    continue
                cell_offset = (row_index - base_row) ** 2 + (col_index - base_col) ** 2
                candidates.append((cell_offset, float(value)))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]

    return None

def point_key(point):
    """
    Return a stable coordinate tuple so point geometries can be used as graph
    nodes and dictionary keys.
    """
    return tuple(point.coords[0])

def get_line_boundary_points(line):
    """
    Extract the exposed terminal points from one line segment using the
    geometry boundary so no directional assumption is made about the stored
    coordinate sequence.
    """
    boundary = line.boundary
    if boundary.is_empty:
        return []
    if isinstance(boundary, Point):
        return [boundary]
    if boundary.geom_type == "MultiPoint":
        return list(boundary.geoms)
    raise TypeError("Line boundary must resolve to Point or MultiPoint")

def get_terminal_points(geometry):
    """
    Return the candidate terminal points for a reach without assuming the
    stored coordinate order identifies upstream versus downstream.

    For a single LineString, use the line boundary to get its two terminal
    endpoints. For a MultiLineString, evaluate each component line separately
    and collect the terminal boundary points from each segment.
    """
    if geometry is None or geometry.is_empty:
        return []

    if isinstance(geometry, LineString):
        candidate_points = get_line_boundary_points(geometry)
    elif isinstance(geometry, MultiLineString):
        # Collect the terminal boundary points from every component line so the
        # downstream comparison works from the original multipart geometry
        # instead of a merged surrogate.
        candidate_points = []
        for line in geometry.geoms:
            if line.is_empty:
                continue
            candidate_points.extend(get_line_boundary_points(line))
    else:
        raise TypeError("Geometry must be a LineString or MultiLineString")

    # Remove duplicate coordinates so shared junctions inside a multipart reach
    # are only considered once when scoring candidate endpoints.
    unique_points = []
    seen_coords = set()
    for point in candidate_points:
        coord_key = point_key(point)
        if coord_key in seen_coords:
            continue
        seen_coords.add(coord_key)
        unique_points.append(point)

    return unique_points

def prune_downstream_points_within_multiline(geometry, candidate_points, downstream_connection_points):
    """
    Remove candidate seed points that are downstream of other candidate seed
    points inside the same MultiLineString.

    The multipart reach is converted into an undirected endpoint graph using
    the original component lines. Distances to the downstream connection
    point(s) are then used to orient that graph in the downstream direction.
    Any candidate point reachable from another candidate point in that
    downstream-directed graph is removed because it sits farther downstream
    within the same multipart reach.
    """
    if not isinstance(geometry, MultiLineString):
        return candidate_points
    if len(candidate_points) <= 1 or not downstream_connection_points:
        return candidate_points

    # Build an undirected graph from the component line endpoints so the
    # multipart reach topology is evaluated without assuming any stored line
    # direction.
    multipart_graph = nx.Graph()
    for line in geometry.geoms:
        if line.is_empty:
            continue

        boundary_points = get_line_boundary_points(line)
        if len(boundary_points) < 2:
            continue

        start_key = point_key(boundary_points[0])
        end_key = point_key(boundary_points[1])
        multipart_graph.add_node(start_key)
        multipart_graph.add_node(end_key)
        multipart_graph.add_edge(start_key, end_key, weight=line.length)

    candidate_point_lookup = {point_key(point): point for point in candidate_points}
    downstream_keys = [
        point_key(point)
        for point in downstream_connection_points
        if point_key(point) in multipart_graph.nodes
    ]

    if multipart_graph.number_of_nodes() == 0 or not downstream_keys:
        return candidate_points

    # Measure graph distance from every node to the downstream connection.
    # Nodes with smaller values are farther downstream.
    distance_to_downstream = nx.multi_source_dijkstra_path_length(
        multipart_graph,
        downstream_keys,
        weight='weight'
    )

    # Orient the multipart graph toward the downstream connection using the
    # graph distances. Edges point from upstream nodes to downstream nodes.
    downstream_graph = nx.DiGraph()
    for start_key, end_key in multipart_graph.edges():
        if start_key not in distance_to_downstream or end_key not in distance_to_downstream:
            continue

        start_distance = distance_to_downstream[start_key]
        end_distance = distance_to_downstream[end_key]
        distance_tolerance = max(1e-9, max(start_distance, end_distance) * 1e-9)

        if start_distance > (end_distance + distance_tolerance):
            downstream_graph.add_edge(start_key, end_key)
        elif end_distance > (start_distance + distance_tolerance):
            downstream_graph.add_edge(end_key, start_key)

    # Keep only candidate points that are not reachable downstream from another
    # candidate point in the same multipart reach.
    candidate_keys = [
        key
        for key in candidate_point_lookup
        if key in downstream_graph.nodes or key in multipart_graph.nodes
    ]
    downstream_candidate_keys = set()
    for candidate_key in candidate_keys:
        for other_candidate_key in candidate_keys:
            if candidate_key == other_candidate_key:
                continue
            if nx.has_path(downstream_graph, other_candidate_key, candidate_key):
                downstream_candidate_keys.add(candidate_key)
                break

    retained_keys = [key for key in candidate_keys if key not in downstream_candidate_keys]
    if not retained_keys:
        return candidate_points

    return [candidate_point_lookup[key] for key in retained_keys]

def filter_terminal_points_adjacent_to_composite_ordinates(
    geometry,
    terminal_points,
    terminal_points_dem=None,
):
    """
    Remove candidate terminal points that are adjacent to two or more
    ordinates in the composite geometry.

    This filter is applied immediately before DEM-based outlet inference. It is
    intended to catch component endpoints that participate in an internal
    junction inside a composite LineString or MultiLineString. If a candidate
    terminal touches two or more adjacent ordinates across the composite
    geometry, it is acting like an internal branching node rather than an
    exposed terminus and should not be scored as a SEED candidate.

    Parameters
    ----------
    geometry: shapely LineString or MultiLineString
        The composite reach geometry being evaluated.
    terminal_points: list[Point]
        Candidate terminal points in the source geometry CRS.
    terminal_points_dem: list[Point] | None
        Optional paired candidate terminal points in the DEM CRS. When this
        list is supplied, the same indices removed from `terminal_points` are
        removed here as well so DEM sampling remains aligned.

    Returns
    -------
    list[Point] | tuple[list[Point], list[Point]]
        The filtered terminal points, plus the filtered DEM-space terminal
        points when `terminal_points_dem` is provided.
    """
    if geometry is None or geometry.is_empty or not terminal_points:
        if terminal_points_dem is None:
            return terminal_points
        return terminal_points, terminal_points_dem

    if terminal_points_dem is not None and len(terminal_points) != len(terminal_points_dem):
        return terminal_points, terminal_points_dem

    if isinstance(geometry, LineString):
        component_lines = [geometry]
    elif isinstance(geometry, MultiLineString):
        component_lines = [line for line in geometry.geoms if not line.is_empty]
    else:
        raise TypeError("Geometry must be a LineString or MultiLineString")

    retained_indices = []
    for index, point in enumerate(terminal_points):
        candidate_key = point_key(point)
        adjacent_ordinate_keys = set()

        for line in component_lines:
            line_coords = list(line.coords)
            if len(line_coords) < 2:
                continue

            if tuple(line_coords[0]) == candidate_key:
                adjacent_ordinate_keys.add(tuple(line_coords[1]))
            elif tuple(line_coords[-1]) == candidate_key:
                adjacent_ordinate_keys.add(tuple(line_coords[-2]))

            # Once a candidate is adjacent to two or more ordinates anywhere in
            # the composite geometry, it is functioning as an internal node and
            # should be removed from the DEM-based SEED inference step.
            if len(adjacent_ordinate_keys) >= 2:
                break

        if len(adjacent_ordinate_keys) < 2:
            retained_indices.append(index)

    filtered_terminal_points = [terminal_points[index] for index in retained_indices]
    if terminal_points_dem is None:
        return filtered_terminal_points

    filtered_terminal_points_dem = [terminal_points_dem[index] for index in retained_indices]
    return filtered_terminal_points, filtered_terminal_points_dem

def infer_upstream_points_from_dem(
    geometry,
    terminal_points,
    terminal_points_dem,
    dem_array,
    dem_geo_transform,
    dem_nodata_value,
):
    """
    Infer the outlet terminal from DEM elevations when the downstream reach
    geometry is absent from the local vector network.
    """
    if len(terminal_points) != len(terminal_points_dem):
        return terminal_points

    # Sample the DEM at each terminal so we can identify the outlet from
    # elevation alone when no downstream reach geometry is available.
    terminal_elevations = [
        Sample_Raster_Value_At_Point(
            dem_array,
            dem_geo_transform,
            point,
            nodata_value=dem_nodata_value,
        )
        for point in terminal_points_dem
    ]
    valid_elevations = [elevation for elevation in terminal_elevations if elevation is not None]
    if not valid_elevations:
        return terminal_points

    min_elevation = min(valid_elevations)
    elevation_tolerance = max(0.01, abs(min_elevation) * 1e-9)
    # Treat the lowest terminal point(s) as the downstream outlet and keep
    # only the higher terminal point(s) as candidate SEED locations.
    downstream_connection_points = [
        point
        for point, elevation in zip(terminal_points, terminal_elevations)
        if elevation is not None and elevation <= (min_elevation + elevation_tolerance)
    ]
    upstream_points = [
        point
        for point, elevation in zip(terminal_points, terminal_elevations)
        if elevation is None or elevation > (min_elevation + elevation_tolerance)
    ]

    if not upstream_points:
        # If every sampled terminal ties at the minimum elevation, fall back to
        # the highest sampled terminal so the reach still yields one upstream
        # seed point instead of none.
        valid_indices = [
            index for index, elevation in enumerate(terminal_elevations)
            if elevation is not None
        ]
        highest_index = max(valid_indices, key=lambda index: terminal_elevations[index])
        upstream_points = [terminal_points[highest_index]]

    # For multipart reaches, remove any retained terminal that still sits
    # downstream of another retained terminal inside the same geometry.
    return prune_downstream_points_within_multiline(
        geometry,
        upstream_points,
        downstream_connection_points
    )

def find_SEED_locations(StrmShp, DEM_Raster_File, SEED_Output_File, Stream_ID_Field, Downstream_ID_Field):
    """
    Finds the locations of SEED points, or the most upstream locations in our modeling domain, using the topology in the stream shapefile
    Parameters
    ----------
    StrmShp: str
        The file path and file name of the stream flowline vector network shapefile 
    DEM_Raster_File: str
        The file path and file name of the DEM raster aligned to the ARC grid.
        When a source reach points to a downstream identifier that is missing
        from the local vector network, terminal candidates that are adjacent to
        two or more ordinates in the composite geometry are removed before the
        DEM is sampled to infer which terminal is the outlet.
    SEED_Output_File: str
        The file path and file name of the output vector file that contains the
        SEED locations and the unique ID of the stream each represents
    OutProjection: str
        A EPSG formatted text descriptor of the output GeoJSON's coordinate system (e.g., "EPSG:4269")
    Stream_ID_Field: str
        The field in the StrmShp that is the streams unique identifier
    Downstream_ID_Field: str
        The field in the StrmShp that is used to identify the stream downstream of the stream
    
    Returns
    -------
    seed_gdf: geodataframe
        A geodataframe of all stream cell locations in the ARC model, now amended with SEED locations
    """
    # Load the hydrographic network data
    gdf = Read_Vector_GDF(StrmShp)
    dem_array, dem_geo_transform, dem_projection, dem_nodata_value = Read_Raster_With_Metadata(DEM_Raster_File)
    dem_raster_crs = Get_Raster_CRS(DEM_Raster_File, dem_projection)

    # Keep one geometry per stream identifier so downstream lookups are stable
    # even if the source file contains duplicate rows for the same reach.
    reach_gdf = gdf.drop_duplicates(subset=[Stream_ID_Field]).copy()
    reach_gdf_dem = Reproject_GDF_If_Needed(reach_gdf, dem_raster_crs)
    reach_geometry_lookup = reach_gdf.set_index(Stream_ID_Field).geometry.to_dict()
    reach_geometry_lookup_dem = reach_gdf_dem.set_index(Stream_ID_Field).geometry.to_dict()

    # Identify the uppermost reaches directly from the stream identifiers. A
    # reach is a SEED candidate only if its stream ID never appears as the
    # downstream target of another reach in the local network.
    downstream_reach_ids = set(reach_gdf[Downstream_ID_Field].dropna())
    source_reach_gdf = reach_gdf[~reach_gdf[Stream_ID_Field].isin(downstream_reach_ids)].copy()

    seed_points = []
    seed_linknos = []

    # Walk only the uppermost reaches and infer which terminal endpoint is
    # upstream by comparing the reach endpoints to the downstream-connected
    # geometry.
    for _, row in source_reach_gdf.iterrows():
        reach_id = row[Stream_ID_Field]
        terminal_points = get_terminal_points(row.geometry)
        dem_geometry = reach_geometry_lookup_dem.get(reach_id)
        terminal_points_dem = get_terminal_points(dem_geometry) if dem_geometry is not None else terminal_points

        if not terminal_points:
            continue

        downstream_id = row[Downstream_ID_Field]
        downstream_geometry = None
        if not pd.isna(downstream_id):
            downstream_geometry = reach_geometry_lookup.get(downstream_id)

        if downstream_geometry is not None and not downstream_geometry.is_empty:
            # Measure every terminal against the downstream-connected reach. Any
            # terminal with the minimum distance is treated as the downstream
            # connection point, and all other terminals are preserved as SEED
            # locations. This allows a MultiLineString source reach to produce
            # multiple SEED points when it contains multiple headwater branches.
            distances_to_downstream = [point.distance(downstream_geometry) for point in terminal_points]
            min_distance = min(distances_to_downstream)
            distance_tolerance = max(1e-9, min_distance * 1e-9)
            downstream_connection_points = [
                point for point, distance in zip(terminal_points, distances_to_downstream)
                if distance <= (min_distance + distance_tolerance)
            ]
            upstream_points = [
                point for point, distance in zip(terminal_points, distances_to_downstream)
                if distance > (min_distance + distance_tolerance)
            ]

            # If every terminal is effectively tied to the downstream geometry,
            # retain the farthest terminal so the reach still produces one SEED
            # point instead of none.
            if not upstream_points:
                upstream_points = [terminal_points[int(np.argmax(distances_to_downstream))]]

            # Remove any remaining candidate seed points that are downstream of
            # other candidates within the same multipart reach.
            upstream_points = prune_downstream_points_within_multiline(
                row.geometry,
                upstream_points,
                downstream_connection_points
            )
        else:
            # Remove component endpoints that are adjacent to two or more
            # ordinates in the composite reach. Those points behave like
            # internal junction nodes in the overall geometry, so they should
            # not enter DEM-based outlet inference.
            terminal_points, terminal_points_dem = filter_terminal_points_adjacent_to_composite_ordinates(
                row.geometry,
                terminal_points,
                terminal_points_dem,
            )
            if not terminal_points:
                continue

            # If the downstream reach is missing from the local network, infer
            # the outlet from the DEM and drop the lowest terminal point from
            # the SEED set.
            upstream_points = infer_upstream_points_from_dem(
                row.geometry,
                terminal_points,
                terminal_points_dem,
                dem_array,
                dem_geo_transform,
                dem_nodata_value,
            )

        # Write one SEED point for each upstream terminal that survived the
        # downstream-connection filter.
        for upstream_point in upstream_points:
            seed_points.append(upstream_point)
            seed_linknos.append(reach_id)

    # Create a new GeoDataFrame for the starting locations
    seed_gdf = gpd.GeoDataFrame({'LINKNO': seed_linknos, 'geometry': seed_points}, crs=gdf.crs)

    # Export the starting locations to a vector file. GeoParquet is supported
    # for lighter-weight single-file storage.
    Write_Vector_GDF(seed_gdf, SEED_Output_File)

    
    return (seed_gdf)

def FindClosestSEEDPoints(seed_gdf, curve_data_gdf, distance_crs):
    """
    Compares stream cell locations to SEED point locations (i.e., the uppermost headwater extents of the ARC model domain) and finds the closest stream cell for each SEED point 

    Parameters
    ----------
    seed_gdf: geodataframe
        A geodataframe of all SEED locations in your model domain, created using the find_SEED_locations function
    curve_data_gdf: geodataframe
        A geodataframe of all stream cell locations in the ARC model with depth, top-width, and velocity, all estimated using the ARC synthetic rating curves
    distance_crs: pyproj.CRS | str
        The projected coordinate system used when measuring distances between
        SEED points and stream-cell points

    Returns
    -------
    curve_data_gdf: geodataframe
        A geodataframe of all stream cell locations in the ARC model, now amended with SEED locations
    """
    # Reproject both layers into the same projected CRS before the nearest-
    # neighbor search so distances are evaluated in linear units.
    seed_gdf = Reproject_GDF_If_Needed(seed_gdf, distance_crs)
    curve_data_gdf = Reproject_GDF_If_Needed(curve_data_gdf, distance_crs)

    # Prefill the curve data file with a SEED value, this will be set to 1 if the column is designated as a SEED column
    curve_data_gdf['SEED'] = "0"

    # Spatial join to find the distance between each CP point and each SEED point
    nearest_cp = gpd.sjoin_nearest(seed_gdf, curve_data_gdf, how='left', distance_col='dist')

    # Find the minimum distance for each unique gdf1 row
    min_distance_idx = nearest_cp.groupby(nearest_cp.index)['dist'].idxmin()

    # Get the rows with the minimum distance
    nearest_cp = nearest_cp.loc[min_distance_idx]

    # Create a boolean mask for matching rows in the original dataframe
    mask = curve_data_gdf.set_index(['COMID', 'Row', 'Col']).index.isin(nearest_cp.set_index(['COMID', 'Row', 'Col']).index)

    # Update the 'SEED' value to 1 for matching rows
    curve_data_gdf.loc[mask, 'SEED'] = "1"

    return (curve_data_gdf)

def Run_Main_Curve_to_GEOJSON_Program_Stream_Vector(CurveParam_File, DEM_Raster_File, OutGeoJSON_File, OutProjection, StrmShp, Stream_ID_Field, Downstream_ID_Field, SEED_Output_File, Thin_Output=True, COMID_Q_File=None, comid_q_df=None):
    """
    Main program that generates GeoJSON file that contains stream cell locations and WSE estimates for a given domain and marks the appropriate stream cells as SEED locations

    Parameters
    ----------
    CurveParam_File: str
        The file path and file name of the ARC curve file you are using to estimate water surface elevation and water depth
    COMID_Q_File: str
        The file path and file name of the file that contains the streamflow estimates for the streams in your domain
    DEM_Raster_File: str
        The file path and file name of the DEM raster aligned to the ARC grid.
        The vector workflow uses this raster for row/column georeferencing and
        to infer outlet terminals when a source reach's downstream geometry is
        missing from the local vector network.
    OutGeoJSON_File: str
        The file path and file name of the output GeoJSON the program will be creating
    OutProjection: str
        A EPSG formatted text descriptor of the output GeoJSON's coordinate system (e.g., "EPSG:4269")
    StrmShp: str
        The file path and file name of the vector shapefile of flowlines
    Stream_ID_Field: str
        The field in the StrmShp that is the streams unique identifier
    Downstream_ID_Field: str
        The field in the StrmShp that is used to identify the stream downstream of the stream
    SEED_Output_File: str
        The file path and file name of the output vector file that contains the
        SEED locations and the unique ID of the stream each represents
    Thin_Output: bool
        True/False of whether or not to filter the output GeoJSON
    
    
    Returns
    -------
    None
    """

    if COMID_Q_File is not None:
        # Read the streamflow data into pandas
        comid_q_df = pd.read_csv(COMID_Q_File)
    else:
        pass
    
    # Assuming we want to rename the first two columns
    new_column_names = ['COMID', 'qout']

    # Create a mapping from the old column names to the new column names based on their positions
    column_mapping = {comid_q_df.columns[i]: new_column_names[i] for i in range(len(new_column_names))}

    # Rename the columns
    comid_q_df.rename(columns=column_mapping, inplace=True)    

    # Get the Extents and cellsizes of the Raster Data
    print('\nGetting the Spatial information from ' + DEM_Raster_File)
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection) = Get_Raster_Details(DEM_Raster_File)
    cellsize_x = abs(float(dx))
    cellsize_y = abs(float(dy))
    lat_base = float(maxy) - 0.5*cellsize_y
    lon_base = float(minx) + 0.5*cellsize_x
    print('Geographic Information:')
    print(lat_base)
    print(lon_base)
    print(cellsize_x)
    print(cellsize_y)

    # Determine the stream raster CRS once so every point created from raster
    # row and column indices is tagged with the raster's true source CRS.
    stream_raster_crs = Get_Raster_CRS(DEM_Raster_File, Rast_Projection)

    # When the raster is geographic, switch to a projected CRS for distance
    # calculations. When it is already projected, keep the native raster CRS.
    distance_crs = Get_Distance_CRS(stream_raster_crs)

    # Reading with pandas
    curve_data_df = pd.read_csv(CurveParam_File, 
                                dtype={'COMID': 'int64', 'Row': 'int64', 'Col': 'int64',
                                       'BaseElev': 'float64', 'DEM_Elev': 'float64', 'QMax': 'float64',
                                       'depth_a': 'float64', 'depth_b': 'float64', 
                                       'tw_a': 'float64', 'tw_b': 'float64',
                                       'vel_a': 'float64', 'vel_b': 'float64'})
    
    # Calculate Latitude and Longitude
    curve_data_df['CP_LAT'] = lat_base - curve_data_df['Row'] * cellsize_y
    curve_data_df['CP_LON'] = lon_base + curve_data_df['Col'] * cellsize_x

    # Build the stream-cell points in the raster CRS, then move them into the
    # projected distance CRS only when the raster started in geographic units.
    curve_data_gdf = Build_Stream_Point_GDF(curve_data_df, 'CP_LON', 'CP_LAT', stream_raster_crs)
    curve_data_gdf = Reproject_GDF_If_Needed(curve_data_gdf, distance_crs)

    # Merge using 'CP_COMID' from gdf and 'COMID_List' from df
    curve_data_gdf = curve_data_gdf.merge(comid_q_df, on="COMID")
    
    # estimate water depth and water surface elevation
    curve_data_gdf['CP_DEP'] = round(curve_data_gdf['depth_a']*curve_data_gdf['qout']**curve_data_gdf['depth_b'], 3)
    curve_data_gdf['WaterSurfaceElev_m'] = round(curve_data_gdf['CP_DEP']+curve_data_gdf['BaseElev'], 3)
    
    # estimate top-width
    curve_data_gdf['CP_TW'] = round(curve_data_gdf['tw_a']*curve_data_gdf['qout']**curve_data_gdf['tw_b'], 3)
    
    # estimate velocity
    curve_data_gdf['CP_VEL'] = round(curve_data_gdf['vel_a']*curve_data_gdf['qout']**curve_data_gdf['vel_b'], 3)

    # drop any stream cells where NaNs are present in the WaterSurfaceElev_m column
    curve_data_gdf = curve_data_gdf[~curve_data_gdf['WaterSurfaceElev_m'].isna()]

    # drop any stream cells where water surface elevation is <= 0
    curve_data_gdf = curve_data_gdf[curve_data_gdf['WaterSurfaceElev_m'] > 0]

    # find the median depth and WSE value for each COMID, these will be used for filtering
    COMID_MedDEP = curve_data_gdf.groupby('COMID')['CP_DEP'].median().reset_index()
    COMID_MedDEP.rename(columns={'CP_DEP': 'COMID_MedDEP'}, inplace=True)
    curve_data_gdf = curve_data_gdf.merge(COMID_MedDEP, on="COMID")
    COMID_MedWSE = curve_data_gdf.groupby('COMID')['WaterSurfaceElev_m'].median().reset_index()
    COMID_MedWSE.rename(columns={'WaterSurfaceElev_m': 'COMID_MedWSE'}, inplace=True)
    curve_data_gdf = curve_data_gdf.merge(COMID_MedWSE, on="COMID")

    # thin the data before we go looking for SEED locations
    if Thin_Output is True:
        curve_data_gdf = Thin_Curve_data(curve_data_gdf)

    # find the SEED locations
    if os.path.isfile(SEED_Output_File):
        print("SEED file exists, we're using it...")
        seed_gdf = Read_Vector_GDF(SEED_Output_File)
    else:
        seed_gdf = find_SEED_locations(StrmShp, DEM_Raster_File, SEED_Output_File, Stream_ID_Field, Downstream_ID_Field) 
    curve_data_gdf = FindClosestSEEDPoints(seed_gdf, curve_data_gdf, distance_crs)

    # output the GeoJSON file
    Write_GeoJSON_File(OutGeoJSON_File, OutProjection, curve_data_gdf)

    return

def Run_Main_VDT_to_GEOJSON_Program_Stream_Vector(VDTDatabaseFileName, DEM_Raster_File, OutGeoJSON_File, OutProjection, StrmShp, Stream_ID_Field, Downstream_ID_Field, SEED_Output_File, Thin_Output=True, COMID_Q_File=None, comid_q_df=None):
    """
    Main program that generates GeoJSON file that contains stream cell locations and WSE estimates for a given domain and marks the appropriate stream cells as SEED locations

    Parameters
    ----------
    VDTDatabaseFileName: str
        The file path and file name of the ARC VDT database file you are using to estimate water surface elevation and water depth
    COMID_Q_File: str
        The file path and file name of the file that contains the streamflow estimates for the streams in your domain
    DEM_Raster_File: str
        The file path and file name of the DEM raster aligned to the ARC grid.
        The vector workflow uses this raster for row/column georeferencing and
        to infer outlet terminals when a source reach's downstream geometry is
        missing from the local vector network.
    OutGeoJSON_File: str
        The file path and file name of the output GeoJSON the program will be creating
    OutProjection: str
        A EPSG formatted text descriptor of the output GeoJSON's coordinate system (e.g., "EPSG:4269")
    StrmShp: str
        The file path and file name of the vector shapefile of flowlines
    Stream_ID_Field: str
        The field in the StrmShp that is the streams unique identifier
    Downstream_ID_Field: str
        The field in the StrmShp that is used to identify the stream downstream of the stream
    SEED_Output_File: str
        The file path and file name of the output vector file that contains the
        SEED locations and the unique ID of the stream each represents
    Thin_Output: bool
        True/False of whether or not to filter the output GeoJSON
    
    
    Returns
    -------
    None
    """

    if COMID_Q_File is not None:
        # Read the streamflow data into pandas
        comid_q_df = pd.read_csv(COMID_Q_File)
    else:
        pass
    
    # Assuming we want to rename the first two columns
    new_column_names = ['COMID', 'qout']

    # Create a mapping from the old column names to the new column names based on their positions
    column_mapping = {comid_q_df.columns[i]: new_column_names[i] for i in range(len(new_column_names))}

    # Rename the columns
    comid_q_df.rename(columns=column_mapping, inplace=True)    

    # Get the Extents and cellsizes of the Raster Data
    print('\nGetting the Spatial information from ' + DEM_Raster_File)
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection) = Get_Raster_Details(DEM_Raster_File)
    cellsize_x = abs(float(dx))
    cellsize_y = abs(float(dy))
    lat_base = float(maxy) - 0.5*cellsize_y
    lon_base = float(minx) + 0.5*cellsize_x
    print('Geographic Information:')
    print(lat_base)
    print(lon_base)
    print(cellsize_x)
    print(cellsize_y)

    # Determine the stream raster CRS once so every point created from raster
    # row and column indices is tagged with the raster's true source CRS.
    stream_raster_crs = Get_Raster_CRS(DEM_Raster_File, Rast_Projection)

    # When the raster is geographic, switch to a projected CRS for distance
    # calculations. When it is already projected, keep the native raster CRS.
    distance_crs = Get_Distance_CRS(stream_raster_crs)

    # Reading with pandas
    vdt_df = pd.read_csv(VDTDatabaseFileName)
    
    # Add COMID flow information
    vdt_df = vdt_df.merge(comid_q_df, on='COMID', how='left')
    
    # Extract the columns for interpolation
    flow_cols = [col for col in vdt_df.columns if col.startswith('q_')]
    top_width_cols = [col for col in vdt_df.columns if col.startswith('t_')]
    wse_cols = [col for col in vdt_df.columns if col.startswith('wse_')]

    # Define the function to calculate TopWidth, Depth, and WSE for each row
    def calculate_values(row):
        flow = row['qout']
        if not np.isfinite(flow):
            # Skip interpolation entirely when the requested flow is invalid so
            # downstream cleanup receives a plain NaN instead of a warning.
            return pd.Series({'WSE': np.nan})

        # Extract flow, TopWidth, and WSE values for interpolation
        flow_values = row[flow_cols].values
        wse_values = row[wse_cols].values

        # Prepare the interpolation pairs first so interp1d never sees NaN/Inf
        # values or a duplicate/degenerate flow axis.
        flow_values, wse_values = Prepare_Interpolation_Pairs(flow_values, wse_values)

        if len(flow_values) == 0:
            # No finite rating-curve pairs exist for this row, so return NaN
            # and let the later dataframe filtering remove it cleanly.
            return pd.Series({'WSE': np.nan})
        if len(flow_values) == 1:
            # A single valid point cannot define a slope, so reuse that value
            # directly instead of calling interp1d and triggering divide logic.
            return pd.Series({'WSE': float(wse_values[0])})

        # Build the interpolator only after the inputs are known to contain at
        # least two distinct finite flow values.
        wse_interp = interp1d(flow_values, wse_values, kind='linear', bounds_error=False, fill_value='extrapolate')

        wse = wse_interp(flow)

        return pd.Series({'WSE': wse})

    # Apply the calculation function to each row
    vdt_df = vdt_df.join(vdt_df.apply(calculate_values, axis=1)).copy()

    # had some issues with values being stored as arrays, so convert them to floats here
    vdt_df = vdt_df.assign(WaterSurfaceElev_m=vdt_df['WSE'].map(Extract_Scalar_Value))

    # Remove outliers by COMID
    def remove_outliers(group):
        mean = group.mean()
        std = group.std()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        return group[(group >= lower_bound) & (group <= upper_bound)]

    for col in ['WaterSurfaceElev_m']:
        vdt_df[col] = vdt_df.groupby('COMID')[col].transform(remove_outliers)

    # Drop rows with NaN values introduced during outlier removal
    vdt_df = vdt_df.dropna(subset=['WaterSurfaceElev_m']).copy()
    
    # Calculate Latitude and Longitude
    vdt_df = vdt_df.assign(
        CP_LAT=lat_base - vdt_df['Row'] * cellsize_y,
        CP_LON=lon_base + vdt_df['Col'] * cellsize_x,
    )

    # Build the stream-cell points in the raster CRS, then move them into the
    # projected distance CRS only when the raster started in geographic units.
    vdt_gdf = Build_Stream_Point_GDF(vdt_df, 'CP_LON', 'CP_LAT', stream_raster_crs)

    # delete the VDT dataframe, we don't need it now
    del(vdt_df)

    # Keep the stream cells in a projected CRS for the nearest-distance work.
    vdt_gdf = Reproject_GDF_If_Needed(vdt_gdf, distance_crs)

    # Merge using 'CP_COMID' from gdf and 'COMID_List' from df
    vdt_gdf = vdt_gdf.merge(comid_q_df, on="COMID")

    # drop any stream cells where NaNs are present in the WaterSurfaceElev_m column
    vdt_gdf = vdt_gdf[~vdt_gdf['WaterSurfaceElev_m'].isna()]

    # drop any stream cells where water surface elevation is <= 0
    vdt_gdf = vdt_gdf[vdt_gdf['WaterSurfaceElev_m'] > 0]

    # find the median WSE value for each COMID, these will be used for filtering
    COMID_MedWSE = vdt_gdf.groupby('COMID')['WaterSurfaceElev_m'].median().reset_index()
    COMID_MedWSE.rename(columns={'WaterSurfaceElev_m': 'COMID_MedWSE'}, inplace=True)
    vdt_gdf = vdt_gdf.merge(COMID_MedWSE, on="COMID")


    # thin the data before we go looking for SEED locations
    if Thin_Output is True:
        vdt_gdf = Thin_Curve_data(vdt_gdf, False)


    # find the SEED locations
    if os.path.isfile(SEED_Output_File):
        print("SEED file exists, we're using it...")
        seed_gdf = Read_Vector_GDF(SEED_Output_File)
        
    else:
        seed_gdf = find_SEED_locations(StrmShp, DEM_Raster_File, SEED_Output_File, Stream_ID_Field, Downstream_ID_Field) 
    vdt_gdf = FindClosestSEEDPoints(seed_gdf, vdt_gdf, distance_crs)

    # output the GeoJSON file
    Write_GeoJSON_File(OutGeoJSON_File, OutProjection, vdt_gdf)

    return

def wse_diff_percentage(wse1, wse2):
    """
    Function to calculate WSE difference percentage

    Parameters
    ----------
    wse1: float
        A value representing the water surface elevation of our stream cell of interest
    wse2: float
        A value representing the water surface elevation of a stream cell within 50 meters of our stream cell of interest
    Returns
    -------
    The absolute percentage difference between the water surface elevation of our point of interest and the water surface elevation of a stream cell within 50 meters of it

    """
    return abs(wse1 - wse2) / ((wse1 + wse2) / 2) * 100

def Thin_Curve_data(gdf, use_depth=True):
    """
    Thins the stream cells out in two ways 
    
    1. Removes stream cells that have depths that are greater than 3 times the average depth for the stream reach
    2. Removes stream cells that are within 50 meters of other streams and have water surface elevations that are within 0.05% of one another.

    Parameters
    ----------
    curve_data_gdf: geodataframe
        A geodataframe of all stream cell locations in the ARC model

    Returns
    -------
    curve_data_gdf: geodataframe
        A geodataframe of all stream cell locations in the ARC model, now thinned 
    """
    
    # This is a filter Mike imposed to keep some of the outliers out
    # We use it if we're using the VDT curve file
    if use_depth is True:
        gdf = gdf[(gdf['COMID_MedDEP']>0) & (gdf['CP_DEP']<3.0*gdf['COMID_MedDEP'])]

    # Create spatial index
    sindex = gdf.sindex

    # List to keep track of indices to drop
    indices_to_drop = set()

    # Iterate through each stream cell
    for i, point in gdf.iterrows():
        if i in indices_to_drop:
            continue

        # Find potential neighbors within 50 meters
        buffer = point.geometry.buffer(50)
        possible_matches_index = list(sindex.intersection(buffer.bounds))
        possible_matches = gdf.iloc[possible_matches_index]

        # iterate and compare each stream cell to the other stream cells
        for j, other_point in possible_matches.iterrows():
            if i != j and j not in indices_to_drop:
                distance = point.geometry.distance(other_point.geometry)
                if distance <= 50:  # 50 meters
                    wse_diff = wse_diff_percentage(float(point['WaterSurfaceElev_m']), float(other_point['WaterSurfaceElev_m']))
                    if wse_diff <= 0.05:  # 0.05%
                        indices_to_drop.add(j)

    # Drop the points that meet the criteria
    gdf = gdf.drop(indices_to_drop)

    return (gdf)

def Write_SEED_Data_To_File_Using_Stream_Raster(STRM_Raster_File, DEM_Raster_File, SEED_Point_File):
    """
    Uses the stream raster and digital elevation model (DEM) raster to define the potential SEED points or the uppermost headwaters of an ARC domain.

    Parameters
    ----------
    STRM_Raster_File: str
        The file name and full path to the stream raster you are analyzing
    DEM_Raster_File: str
        The file name and full path to the digital elevation model (DEM) raster you are analyzing
    SEED_Point_File: str
        The file name and full path to the text file that will store the potential SEED locations the function defines

    Returns
    -------
    SEED_Lat: list
        A list of float values that represent the latitude of the SEED locations in the stream raster
    SEED_Lon: list
        A list of float values that represent the longitude of the SEED locations in the stream raster
    SEED_COMID: list
        A list of integer values that represent the unique identifier of the streams at each SEED location
    SEED_r: list
        A list of integer values that represent the row in stream and DEM raster where the SEED location is
    SEED_c: list
        A list of integer values that represent the column in stream and DEM raster where the SEED location is
    SEED_MinElev: list
        A list of float values that represent the minimum elevation of the stream cells on the stream reach of the SEED location
    SEED_MaxElev: list
        A list of float values that represent the maximum elevation of the stream cells on the stream reach of the SEED location
    """
    print('\nReading Data from Raster: ' + STRM_Raster_File)
    (S, ncols, nrows, cellsize, yll, yur, xll, xur, lat, geotransform, Rast_Projection) = Read_Raster_GDAL(STRM_Raster_File)
    dx = abs(xll-xur) / (ncols)
    dy = abs(yll-yur) / (nrows)
    print(str(dx) + '  ' + str(dy) + '  ' + str(cellsize))
    SN = np.asarray(S)
    SN_Flat = SN.flatten().astype(int)
    
    B = np.zeros((nrows+2,ncols+2))  #Create an array that is slightly larger than the STRM Raster Array
    B[1:(nrows+1), 1:(ncols+1)] = SN
    B = B.astype(int)
    B = np.where(B>0,1,0)   #Streams are identified with zeros
    
    COMID_Unique = np.unique(SN_Flat)
    COMID_Unique = np.delete(COMID_Unique, 0)  #We don't need the first entry of zero
    num_comid_unique = len(COMID_Unique)
    SN_Flat = None
    S = None
    
    print('\nReading Data from Raster: ' + DEM_Raster_File)
    (DEM, dncols, dnrows, dcellsize, dyll, dyur, dxll, dxur, dlat, geotransform, Rast_Projection) = Read_Raster_GDAL(DEM_Raster_File)
    #E = np.zeros((nrows+2,ncols+2))  #Create an array that is slightly larger than the STRM Raster Array
    #E[1:(nrows+1), 1:(ncols+1)] = DEM
    #E = E.astype(float)
    DEM = DEM.astype(float)
    
    COMID_Elev_Min = np.zeros(len(COMID_Unique))
    COMID_Elev_Max = np.zeros(len(COMID_Unique))
    COMID_Elev_Min = COMID_Elev_Min + 999999.9
    COMID_Elev_Max = COMID_Elev_Max - 9999.9
    
    #This is where we really try to find all the potential SEED Locations
    print('\n\nFind All the Potential SEED Locations...')
    SEED_r = []
    SEED_c = []
    SEED_Lat=[]
    SEED_Lon=[]
    SEED_COMID=[]
    SEED_MinElev=[]
    SEED_MaxElev=[]
    p_count = 0
    p_percent = int((num_comid_unique)/10.0)
    
    for i in range(num_comid_unique):
        p_count = p_count + 1
        if p_count >= p_percent:
            p_count = 0
            print('  Another Percent 10 Percent Complete')
        COMID = COMID_Unique[i]
        (RR,CC) = np.where(SN==COMID)
        num_comid = len(RR)
        for x in range(num_comid):
            r=RR[x]
            c=CC[x]
            if COMID_Elev_Min[i]>DEM[RR[x],CC[x]]:
                COMID_Elev_Min[i]=DEM[RR[x],CC[x]]
            if COMID_Elev_Max[i]<DEM[RR[x],CC[x]]:
                COMID_Elev_Max[i]=DEM[RR[x],CC[x]]
        for x in range(num_comid):
            r=RR[x] + 1  #Need the plus one to get to the larger boundaries of the B raster
            c=CC[x] + 1  #Need the plus one to get to the larger boundaries of the B raster
            
            n = np.count_nonzero(B[r-1:r+2,c-1:c+2])
            if n<=2 or (r==0 or c==0 or c==ncols-1 or r==nrows-1):
                lat_for_seed = float( yur - (0.5*dy) - ((r-1) * dy) )
                lon_for_seed = float( xll + (0.5*dx) + ((c-1) * dx) )
                SEED_Lat.append( lat_for_seed )
                SEED_Lon.append( lon_for_seed )
                SEED_COMID.append(COMID)
                SEED_r.append(r-1)
                SEED_c.append(c-1)
                SEED_MinElev.append(COMID_Elev_Min[i])
                SEED_MaxElev.append(COMID_Elev_Max[i])
    
    outfile = open(SEED_Point_File,'w')
    out_str = 'COMID,Lat,Long,Row,Col,MinElev,MaxElev'
    outfile.write(out_str)
    for i in range(len(SEED_COMID)):
        out_str = '\n' + str(SEED_COMID[i]) + ',' + str(SEED_Lat[i]) + ',' + str(SEED_Lon[i]) + ',' + str(SEED_r[i]) + ',' + str(SEED_c[i]) + ',' + str(SEED_MinElev[i]) + ',' + str(SEED_MaxElev[i])
        outfile.write(out_str)
    outfile.close()
    return SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, SEED_MinElev, SEED_MaxElev

def GetSEED_Data_From_File(SEED_Point_File):
    """
    Reads the pre-defined SEED points from a previously created SEED point file.

    Parameters
    ----------
    SEED_Point_File: str
        The file name and full path to the text file that stores the potential SEED locations the function defines as comma separated values

    Returns
    -------
    SEED_Lat: list
        A list of float values that represent the latitude of the SEED locations in the stream raster
    SEED_Lon: list
        A list of float values that represent the longitude of the SEED locations in the stream raster
    SEED_COMID: list
        A list of integer values that represent the unique identifier of the streams at each SEED location
    SEED_r: list
        A list of integer values that represent the row in stream and DEM raster where the SEED location is
    SEED_c: list
        A list of integer values that represent the column in stream and DEM raster where the SEED location is
    SEED_MinElev: list
        A list of float values that represent the minimum elevation of the stream cells on the stream reach of the SEED location
    SEED_MaxElev: list
        A list of float values that represent the maximum elevation of the stream cells on the stream reach of the SEED location
    """
    infile = open(SEED_Point_File,'r')
    lines = infile.readlines()
    n = len(lines)-1
    SEED_r = [0] * n
    SEED_c = [0] * n
    SEED_Lat=[0.0] * n
    SEED_Lon=[0.0] * n
    SEED_COMID=[0] * n
    SEED_MinElev=[0.0] * n
    SEED_MaxElev=[0.0] * n
    i=-1
    for line in lines[1:]:
        i=i+1
        (SEED_COMID[i], SEED_Lat[i], SEED_Lon[i], SEED_r[i], SEED_c[i], SEED_MinElev[i], SEED_MaxElev[i]) = line.strip().split(',')
    return SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, SEED_MinElev, SEED_MaxElev

def Write_GeoJSON_File(OutGeoJSON_File, OutProjection, curve_data_gdf):
    """
    Writes a GeoJSON file that describes the hydraulic characteristics of a region at a point level

    Parameters
    ----------
    OutGeoJSON_File: str
        A string that represents the file path and file name of the output GeoJSON the function creates
    OutProjection: str
        A EPSG formatted text descriptor of the output GeoJSON's coordinate system (e.g., "EPSG:4269")
    curve_data_gdf: geodataframe
        A geodataframe of all stream cell locations in the ARC model, now amended with SEED locations and water surface elevation estimates
    Thin_GeoJSON: bool
        True/False of whether or not to filter the output GeoJSON

    Returns
    -------
    None
    """
    # Select the desired columns
    selected_columns = ['WaterSurfaceElev_m', 'SEED', 'geometry']
    curve_data_gdf = curve_data_gdf[selected_columns]

    # Reproject the finished output into the caller's requested CRS before
    # writing the GeoJSON.
    curve_data_gdf = curve_data_gdf.to_crs(OutProjection)

    # Save the converted GeoDataFrame to a new file
    curve_data_gdf.to_file(OutGeoJSON_File, driver='GeoJSON')

    return

def FindClosestSEEDPoints_Based_On_LatLong(SEED_COMID, SEED_Lat, SEED_Lon, SEED_r, SEED_c, curve_data_gdf, stream_raster_crs, distance_crs, elev_column_name='BaseElev'):
    """
    Compares stream cell locations to SEED point locations (i.e., the uppermost headwater extents of the ARC model domain) and finds the closest stream cell for each SEED point 

    Parameters
    ----------
    SEED_COMID: list
        A list of integer values that represent the unique identifier of the streams at each SEED location
    SEED_Lat: list
        A list of float values that represent the latitude of the SEED locations in the stream raster
    SEED_Lon: list
        A list of float values that represent the longitude of the SEED locations in the stream raster
    SEED_r: list
        A list of integer values that represent the row in stream and DEM raster where the SEED location is
    SEED_c: list
        A list of integer values that represent the column in stream and DEM raster where the SEED location is
    curve_data_gdf: geodataframe
        A geodataframe of all stream cell locations in the ARC model with depth, top-width, and velocity, all estimated using the ARC synthetic rating curves
    stream_raster_crs: pyproj.CRS | str
        The source coordinate system of the stream raster used to create the
        stream-cell and SEED point coordinates
    distance_crs: pyproj.CRS | str
        The projected coordinate system used when measuring distances between
        SEED points and stream-cell points
    elev_column_name: str
        Represents the name of the column which represents the elevation of the stream cell
    
    Returns
    -------
    curve_data_gdf: geodataframe
        A geodataframe of all stream cell locations in the ARC model, now amended with SEED locations
    None
    """
    # Create a pandas DataFrame from the lists
    data = {
        'SEED_COMID': [int(x) for x in SEED_COMID],
        'SEED_Lat': [float(x) for x in SEED_Lat],
        'SEED_Lon': [float(x) for x in SEED_Lon],
        'SEED_r': [int(x) for x in SEED_r],
        'SEED_c': [int(x) for x in SEED_c],
    }
    df = pd.DataFrame(data)

    # Build the SEED points in the raster CRS because these coordinates were
    # derived from raster cells, not from the requested output projection.
    seed_gdf = Build_Stream_Point_GDF(df, 'SEED_Lon', 'SEED_Lat', stream_raster_crs)

    # Reproject both layers into the same projected CRS before measuring
    # distances so the nearest-neighbor search uses linear units.
    seed_gdf = Reproject_GDF_If_Needed(seed_gdf, distance_crs)
    curve_data_gdf = Reproject_GDF_If_Needed(curve_data_gdf, distance_crs)

    # Prefill the curve data file with a SEED value, this will be set to 1 if the column is designated as a SEED column
    curve_data_gdf['SEED'] = "0"

    # Spatial join to find the distance between each CP point and each SEED point
    nearest_cp = gpd.sjoin_nearest(seed_gdf, curve_data_gdf, how='inner', distance_col='dist')

    # Filter based on attributes
    # Make sure the COMID's match betweent the SEED and curves
    nearest_cp = nearest_cp[nearest_cp['SEED_COMID'] == nearest_cp['COMID']]

    # Find the minimum distance for each unique gdf1 row
    min_distance_idx = nearest_cp.groupby(nearest_cp.index)['dist'].idxmin()

    # Get the rows with the minimum distance
    nearest_cp = nearest_cp.loc[min_distance_idx]

    # Find the highest elevation for each COMID
    max_elev_idx = nearest_cp.groupby('SEED_COMID')[elev_column_name].idxmax()

    # Get the rows with the maximum elevation for each COMID
    nearest_cp = nearest_cp.loc[max_elev_idx]

    # Ensure the result contains only one value for each unique row in gdf1
    nearest_cp = nearest_cp.drop_duplicates(subset=['SEED_COMID','SEED_r','SEED_c'])

    # Create a boolean mask for matching rows in the original dataframe
    mask = curve_data_gdf.set_index(['COMID', 'Row', 'Col']).index.isin(nearest_cp.set_index(['COMID', 'Row', 'Col']).index)

    # Update the 'SEED' value to 1 for matching rows
    curve_data_gdf.loc[mask, 'SEED'] = "1"

    return (curve_data_gdf)

def Run_Main_Curve_to_GEOJSON_Program_Stream_Raster(CurveParam_File, COMID_Q_File, STRM_Raster_File, OutGeoJSON_File, OutProjection, SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, Thin_GeoJSON):
    """
    Main program that generates GeoJSON file that contains stream cell locations and WSE estimates for a given domain and marks the appropriate stream cells as SEED locations

    Parameters
    ----------
    CurveParam_File: str
        The file path and file name of the ARC curve file you are using to estimate water surface elevation and water depth
    COMID_Q_File: str
        The file path and file name of the file that contains the streamflow estimates for the streams in your domain
    STRM_Raster_File: str
        The file path and file name of the stream raster that contains the stream cells you used to run ARC 
    OutGeoJSON_File: str
        The file path and file name of the output GeoJSON the program will be creating
    OutProjection: str
        A EPSG formatted text descriptor of the output GeoJSON's coordinate system (e.g., "EPSG:4269")
    SEED_Lat: list
        A list of float values that represent the latitude of the SEED locations in the stream raster
    SEED_Lon: list
        A list of float values that represent the longitude of the SEED locations in the stream raster
    SEED_COMID: list
        A list of integer values that represent the unique identifier of the streams at each SEED location
    SEED_r: list
        A list of integer values that represent the row in stream and DEM raster where the SEED location is
    SEED_c: list
        A list of integer values that represent the column in stream and DEM raster where the SEED location is
    Thin_GeoJSON: bool
        True/False of whether or not to filter the output GeoJSON
    
    Returns
    -------
    None
    """

    # Read the streamflow data into pandas
    comid_q_df = pd.read_csv(COMID_Q_File)
    
    # Assuming we want to rename the first two columns
    new_column_names = ['COMID', 'qout']

    # Create a mapping from the old column names to the new column names based on their positions
    column_mapping = {comid_q_df.columns[i]: new_column_names[i] for i in range(len(new_column_names))}

    # Rename the columns
    comid_q_df.rename(columns=column_mapping, inplace=True)    

    # Get the Extents and cellsizes of the Raster Data
    print('\nGetting the Spatial information from ' + STRM_Raster_File)
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection) = Get_Raster_Details(STRM_Raster_File)
    cellsize_x = abs(float(dx))
    cellsize_y = abs(float(dy))
    lat_base = float(maxy) - 0.5*cellsize_y
    lon_base = float(minx) + 0.5*cellsize_x
    print('Geographic Information:')
    print(lat_base)
    print(lon_base)
    print(cellsize_x)
    print(cellsize_y)

    # Determine the stream raster CRS once so every point created from raster
    # row and column indices is tagged with the raster's true source CRS.
    stream_raster_crs = Get_Raster_CRS(STRM_Raster_File, Rast_Projection)

    # When the raster is geographic, switch to a projected CRS for distance
    # calculations. When it is already projected, keep the native raster CRS.
    distance_crs = Get_Distance_CRS(stream_raster_crs)

    # Reading with pandas
    curve_data_df = pd.read_csv(CurveParam_File, 
                                dtype={'COMID': 'int64', 'Row': 'int64', 'Col': 'int64',
                                       'BaseElev': 'float64', 'DEM_Elev': 'float64', 'QMax': 'float64',
                                       'depth_a': 'float64', 'depth_b': 'float64', 
                                       'tw_a': 'float64', 'tw_b': 'float64',
                                       'vel_a': 'float64', 'vel_b': 'float64'})
    
    # Calculate Latitude and Longitude
    curve_data_df['CP_LAT'] = lat_base - curve_data_df['Row'] * cellsize_y
    curve_data_df['CP_LON'] = lon_base + curve_data_df['Col'] * cellsize_x

    # Build the stream-cell points in the raster CRS, then move them into the
    # projected distance CRS only when the raster started in geographic units.
    curve_data_gdf = Build_Stream_Point_GDF(curve_data_df, 'CP_LON', 'CP_LAT', stream_raster_crs)
    curve_data_gdf = Reproject_GDF_If_Needed(curve_data_gdf, distance_crs)

    # Merge using 'CP_COMID' from gdf and 'COMID_List' from df
    curve_data_gdf = curve_data_gdf.merge(comid_q_df, on="COMID")
    
    # estimate water depth and water surface elevation
    curve_data_gdf['CP_DEP'] = round(curve_data_gdf['depth_a']*curve_data_gdf['qout']**curve_data_gdf['depth_b'], 3)
    curve_data_gdf['WaterSurfaceElev_m'] = round(curve_data_gdf['CP_DEP']+curve_data_gdf['BaseElev'], 3)
    
    # estimate top-width
    curve_data_gdf['CP_TW'] = round(curve_data_gdf['tw_a']*curve_data_gdf['qout']**curve_data_gdf['tw_b'], 3)
    
    # estimate velocity
    curve_data_gdf['CP_VEL'] = round(curve_data_gdf['vel_a']*curve_data_gdf['qout']**curve_data_gdf['vel_b'], 3)

    # drop any stream cells where NaNs are present in the WaterSurfaceElev_m column
    curve_data_gdf = curve_data_gdf[~curve_data_gdf['WaterSurfaceElev_m'].isna()]

    # drop any stream cells where water surface elevation is <= 0
    curve_data_gdf = curve_data_gdf[curve_data_gdf['WaterSurfaceElev_m'] > 0]

    # find the median depth and WSE value for each COMID, these will be used for filtering
    COMID_MedDEP = curve_data_gdf.groupby('COMID')['CP_DEP'].median().reset_index()
    COMID_MedDEP.rename(columns={'CP_DEP': 'COMID_MedDEP'}, inplace=True)
    curve_data_gdf = curve_data_gdf.merge(COMID_MedDEP, on="COMID")
    COMID_MedWSE = curve_data_gdf.groupby('COMID')['WaterSurfaceElev_m'].median().reset_index()
    COMID_MedWSE.rename(columns={'WaterSurfaceElev_m': 'COMID_MedWSE'}, inplace=True)
    curve_data_gdf = curve_data_gdf.merge(COMID_MedWSE, on="COMID")

    # thin the data before we go looking for SEED locations
    if Thin_GeoJSON is True:
        curve_data_gdf = Thin_Curve_data(curve_data_gdf)

    # find the SEED locations 
    curve_data_gdf = FindClosestSEEDPoints_Based_On_LatLong(SEED_COMID, SEED_Lat, SEED_Lon, SEED_r, SEED_c, curve_data_gdf, stream_raster_crs, distance_crs)

    # output the GeoJSON file
    Write_GeoJSON_File(OutGeoJSON_File, OutProjection, curve_data_gdf)

    return

def Run_Main_VDT_to_GEOJSON_Program_Stream_Raster(VDTDatabaseFileName, COMID_Q_File, STRM_Raster_File, OutGeoJSON_File, OutProjection, SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, Thin_GeoJSON):
    """
    Main program that generates GeoJSON file that contains stream cell locations and WSE estimates for a given domain and marks the appropriate stream cells as SEED locations

    Parameters
    ----------
    VDTDatabaseFileName: str
        The file path and file name of the ARC velocity-depth-top-width (VDT) database file you are using to estimate water surface elevation and water depth
    COMID_Q_File: str
        The file path and file name of the file that contains the streamflow estimates for the streams in your domain
    STRM_Raster_File: str
        The file path and file name of the stream raster that contains the stream cells you used to run ARC 
    OutGeoJSON_File: str
        The file path and file name of the output GeoJSON the program will be creating
    OutProjection: str
        A EPSG formatted text descriptor of the output GeoJSON's coordinate system (e.g., "EPSG:4269")
    SEED_Lat: list
        A list of float values that represent the latitude of the SEED locations in the stream raster
    SEED_Lon: list
        A list of float values that represent the longitude of the SEED locations in the stream raster
    SEED_COMID: list
        A list of integer values that represent the unique identifier of the streams at each SEED location
    SEED_r: list
        A list of integer values that represent the row in stream and DEM raster where the SEED location is
    SEED_c: list
        A list of integer values that represent the column in stream and DEM raster where the SEED location is
    Thin_GeoJSON: bool
        True/False of whether or not to filter the output GeoJSON
    
    Returns
    -------
    None
    """

    # Read the streamflow data into pandas
    comid_q_df = pd.read_csv(COMID_Q_File)
    
    # Assuming we want to rename the first two columns
    new_column_names = ['COMID', 'qout']

    # Create a mapping from the old column names to the new column names based on their positions
    column_mapping = {comid_q_df.columns[i]: new_column_names[i] for i in range(len(new_column_names))}

    # Rename the columns
    comid_q_df.rename(columns=column_mapping, inplace=True)    

    # Get the Extents and cellsizes of the Raster Data
    print('\nGetting the Spatial information from ' + STRM_Raster_File)
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection) = Get_Raster_Details(STRM_Raster_File)
    cellsize_x = abs(float(dx))
    cellsize_y = abs(float(dy))
    lat_base = float(maxy) - 0.5*cellsize_y
    lon_base = float(minx) + 0.5*cellsize_x
    print('Geographic Information:')
    print(lat_base)
    print(lon_base)
    print(cellsize_x)
    print(cellsize_y)

    # Determine the stream raster CRS once so every point created from raster
    # row and column indices is tagged with the raster's true source CRS.
    stream_raster_crs = Get_Raster_CRS(STRM_Raster_File, Rast_Projection)

    # When the raster is geographic, switch to a projected CRS for distance
    # calculations. When it is already projected, keep the native raster CRS.
    distance_crs = Get_Distance_CRS(stream_raster_crs)

    # Reading with pandas
    vdt_df = pd.read_csv(VDTDatabaseFileName)
    
    # Add COMID flow information
    vdt_df = vdt_df.merge(comid_q_df, on='COMID', how='left')
    
    # Extract the columns for interpolation
    flow_cols = [col for col in vdt_df.columns if col.startswith('q_')]
    top_width_cols = [col for col in vdt_df.columns if col.startswith('t_')]
    wse_cols = [col for col in vdt_df.columns if col.startswith('wse_')]

    # Define the function to calculate TopWidth, Depth, and WSE for each row
    def calculate_values(row):
        flow = row['qout']
        if not np.isfinite(flow):
            # Skip interpolation entirely when the requested flow is invalid so
            # downstream cleanup receives a plain NaN instead of a warning.
            return pd.Series({'WSE': np.nan})

        # Extract flow, TopWidth, and WSE values for interpolation
        flow_values = row[flow_cols].values
        wse_values = row[wse_cols].values

        # Prepare the interpolation pairs first so interp1d never sees NaN/Inf
        # values or a duplicate/degenerate flow axis.
        flow_values, wse_values = Prepare_Interpolation_Pairs(flow_values, wse_values)

        if len(flow_values) == 0:
            # Zero flows exist for this row, so return NaN
            # and let the later dataframe filtering remove it cleanly.
            return pd.Series({'WSE': np.nan})
        if len(flow_values) == 1:
            # A single valid point cannot define a slope, so reuse that value
            # directly instead of calling interp1d and triggering divide logic.
            return pd.Series({'WSE': round(float(wse_values[0]), 3)})

        # Build the interpolator only after the inputs are known to contain at
        # least two distinct finite flow values.
        wse_interp = interp1d(flow_values, wse_values, kind='linear', bounds_error=False, fill_value='extrapolate')

        wse = round(wse_interp(flow), 3)

        return pd.Series({'WSE': wse})

    # Apply the calculation function to each row
    vdt_df = vdt_df.join(vdt_df.apply(calculate_values, axis=1)).copy()

    # had some issues with values being stored as arrays, so convert them to floats here
    vdt_df = vdt_df.assign(WaterSurfaceElev_m=vdt_df['WSE'].map(Extract_Scalar_Value))

    # Remove outliers by COMID
    def remove_outliers(group):
        mean = group.mean()
        std = group.std()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        return group[(group >= lower_bound) & (group <= upper_bound)]

    for col in ['WaterSurfaceElev_m']:
        vdt_df[col] = vdt_df.groupby('COMID')[col].transform(remove_outliers)

    # Drop rows with NaN values introduced during outlier removal
    vdt_df = vdt_df.dropna(subset=['WaterSurfaceElev_m']).copy()
    
    # Calculate Latitude and Longitude
    vdt_df = vdt_df.assign(
        CP_LAT=lat_base - vdt_df['Row'] * cellsize_y,
        CP_LON=lon_base + vdt_df['Col'] * cellsize_x,
    )

    # Build the stream-cell points in the raster CRS, then move them into the
    # projected distance CRS only when the raster started in geographic units.
    vdt_gdf = Build_Stream_Point_GDF(vdt_df, 'CP_LON', 'CP_LAT', stream_raster_crs)

    # delete the VDT dataframe, we don't need it now
    del(vdt_df)

    # Keep the stream cells in a projected CRS for the nearest-distance work.
    vdt_gdf = Reproject_GDF_If_Needed(vdt_gdf, distance_crs)

    # Merge using 'CP_COMID' from gdf and 'COMID_List' from df
    vdt_gdf = vdt_gdf.merge(comid_q_df, on="COMID")

    # drop any stream cells where NaNs are present in the WaterSurfaceElev_m column
    vdt_gdf = vdt_gdf[~vdt_gdf['WaterSurfaceElev_m'].isna()]

    # drop any stream cells where water surface elevation is <= 0
    vdt_gdf = vdt_gdf[vdt_gdf['WaterSurfaceElev_m'] > 0]

    # find the median depth and WSE value for each COMID, these will be used for filtering
    COMID_MedWSE = vdt_gdf.groupby('COMID')['WaterSurfaceElev_m'].median().reset_index()
    COMID_MedWSE.rename(columns={'WaterSurfaceElev_m': 'COMID_MedWSE'}, inplace=True)
    vdt_gdf = vdt_gdf.merge(COMID_MedWSE, on="COMID")

    # thin the data before we go looking for SEED locations
    if Thin_GeoJSON is True:
        vdt_gdf = Thin_Curve_data(vdt_gdf, False)

    # find the SEED locations 
    vdt_gdf = FindClosestSEEDPoints_Based_On_LatLong(SEED_COMID, SEED_Lat, SEED_Lon, SEED_r, SEED_c, vdt_gdf, stream_raster_crs, distance_crs, elev_column_name="Elev")

    # output the GeoJSON file
    Write_GeoJSON_File(OutGeoJSON_File, OutProjection, vdt_gdf)

    return


def main_write_seed():
    parser = argparse.ArgumentParser(description="Write SEED data to a file using stream and DEM raster files.")
    parser.add_argument("--strm_raster_file", type=str, required=True, help="Path to the stream raster file (TIFF).")
    parser.add_argument("--dem_raster_file", type=str, required=True, help="Path to the DEM raster file (TIFF).")
    parser.add_argument("--seed_point_file", type=str, required=True, help="Path to the output SEED points file.")

    args = parser.parse_args()

    # Call the Write_SEED_Data_To_File function with parsed arguments
    Write_SEED_Data_To_File_Using_Stream_Raster(
        STRM_Raster_File=args.strm_raster_file,
        DEM_Raster_File=args.dem_raster_file,
        SEED_Point_File=args.seed_point_file
    )

def main_Run_Main_Curve_to_GEOJSON_Program_Stream_Vector():
    parser = argparse.ArgumentParser(description="Generate GeoJSON from rating curves and stream vector data.")
    parser.add_argument("--curve_param_file", type=str, required=True, help="Path to ARC curve parameter file (CSV).")
    parser.add_argument(
        "--dem_raster_file",
        "--strm_raster_file",
        dest="dem_raster_file",
        type=str,
        required=True,
        help="Path to DEM raster file (TIFF) used for vector-grid coordinates and SEED outlet inference.",
    )
    parser.add_argument("--out_geojson_file", type=str, required=True, help="Output GeoJSON file path.")
    parser.add_argument("--out_projection", type=str, required=True, help="Output projection (e.g., EPSG:4269).")
    parser.add_argument("--thin_geojson", type=bool, default=True, help="Whether to thin the output GeoJSON (default: True).")
    parser.add_argument("--strm_shp", type=str, required=True, help="Path to stream shapefile.")
    parser.add_argument("--stream_id_field", type=str, required=True, help="Field in shapefile with unique stream identifiers.")
    parser.add_argument("--downstream_id_field", type=str, required=True, help="Field in shapefile with downstream identifiers.")
    parser.add_argument("--seed_output_file", type=str, required=True, help="Output file path for SEED locations vector file.")
    parser.add_argument("--thin_output", type=bool, default=True, help="Whether to thin the output GeoJSON (default: True).")
    parser.add_argument("--comid_q_file", type=str, required=True, help="Path to file with streamflow estimates (CSV).")

    args = parser.parse_args()

    # Call the main function with parsed arguments
    Run_Main_Curve_to_GEOJSON_Program_Stream_Vector(
        CurveParam_File=args.curve_param_file,
        DEM_Raster_File=args.dem_raster_file,
        OutGeoJSON_File=args.out_geojson_file,
        OutProjection=args.out_projection,
        StrmShp=args.strm_shp,
        Stream_ID_Field=args.stream_id_field,
        Downstream_ID_Field=args.downstream_id_field,
        SEED_Output_File=args.seed_output_file,
        Thin_Output=args.thin_output,
        COMID_Q_File=args.comid_q_file,
    )

# Add this to make both functions callable
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for Run_Main_Curve_to_GEOJSON_Program_Stream_Vector
    curve_to_geojson_parser = subparsers.add_parser("curve-to-geojson-stream-vector", help="Run curve to GeoJSON program")
    curve_to_geojson_parser.add_argument("--curve_param_file", type=str, required=True, help="Path to ARC curve parameter file (CSV).")
    curve_to_geojson_parser.add_argument(
        "--dem_raster_file",
        "--strm_raster_file",
        dest="dem_raster_file",
        type=str,
        required=True,
        help="Path to DEM raster file (TIFF) used for vector-grid coordinates and SEED outlet inference.",
    )
    curve_to_geojson_parser.add_argument("--out_geojson_file", type=str, required=True, help="Output GeoJSON file path.")
    curve_to_geojson_parser.add_argument("--out_projection", type=str, required=True, help="Output projection (e.g., EPSG:4269).")
    curve_to_geojson_parser.add_argument("--strm_shp", type=str, required=True, help="Path to stream shapefile.")
    curve_to_geojson_parser.add_argument("--stream_id_field", type=str, required=True, help="Field in shapefile with unique stream identifiers.")
    curve_to_geojson_parser.add_argument("--downstream_id_field", type=str, required=True, help="Field in shapefile with downstream identifiers.")
    curve_to_geojson_parser.add_argument("--seed_output_file", type=str, required=True, help="Output file path for SEED locations vector file.")
    curve_to_geojson_parser.add_argument("--thin_output", type=bool, default=True, help="Whether to thin the output GeoJSON (default: True).")
    curve_to_geojson_parser.add_argument("--comid_q_file", type=str, help="Path to file with streamflow estimates (optional).")

    # Subparser for Write_SEED_Data_To_File
    write_seed_parser = subparsers.add_parser("write-seed-stream-raster", help="Write SEED data to a file")
    write_seed_parser.add_argument("--strm_raster_file", type=str, required=True, help="Path to the stream raster file (TIFF).")
    write_seed_parser.add_argument("--dem_raster_file", type=str, required=True, help="Path to the DEM raster file (TIFF).")
    write_seed_parser.add_argument("--seed_point_file", type=str, required=True, help="Path to the output SEED points file.")

    args = parser.parse_args()

    if args.command == "curve-to-geojson-stream-vector":
        Run_Main_Curve_to_GEOJSON_Program_Stream_Vector(
            CurveParam_File=args.curve_param_file,
            DEM_Raster_File=args.dem_raster_file,
            OutGeoJSON_File=args.out_geojson_file,
            OutProjection=args.out_projection,
            StrmShp=args.strm_shp,
            Stream_ID_Field=args.stream_id_field,
            Downstream_ID_Field=args.downstream_id_field,
            SEED_Output_File=args.seed_output_file,
            Thin_Output=args.thin_output,
            COMID_Q_File=args.comid_q_file,
        )
    elif args.command == "write-seed-stream-raster":
        Write_SEED_Data_To_File_Using_Stream_Raster(
            STRM_Raster_File=args.strm_raster_file,
            DEM_Raster_File=args.dem_raster_file,
            SEED_Point_File=args.seed_point_file
        )
