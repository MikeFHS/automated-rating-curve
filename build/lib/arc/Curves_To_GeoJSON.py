# built-in imports
import argparse
import os, sys
import json
from io import StringIO
import time


# third-party imports
import numpy as np
from scipy.sparse import csr_matrix
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import geojson
from geojson import Point, Feature, FeatureCollection
import pandas as pd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString

# local imports
# from streamflow_processing import Get_Raster_Details, Read_Raster_GDAL



def Get_Raster_Details(DEM_File):
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

def find_SEED_locations(StrmShp, SEED_Output_File, Stream_ID_Field, Downstream_ID_Field):
    """
    Finds the locations of SEED points, or the most upstream locations in our modeling domain, using the topology in the stream shapefile
    Parameters
    ----------
    StrmShp: str
        The file path and file name of the stream flowline vector network shapefile 
    SEED_Output_File: str
        The file path and file name of the output shapefile that contains the SEED locations and the unique ID of the stream each represents
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
    gdf = gpd.read_file(StrmShp)

    # Build the graph using LINKNO and DSLINKNO
    G = nx.DiGraph()

    for idx, row in gdf.iterrows():
        if not pd.isna(row[Downstream_ID_Field]):
            G.add_edge(row[Stream_ID_Field], row[Downstream_ID_Field])

    # Find all source nodes (uppermost locations)
    sources = [node for node in G.nodes() if G.in_degree(node) == 0]

    # Filter the source geometries
    source_geometries = gdf[gdf[Stream_ID_Field].isin(sources)].geometry
    source_linknos = gdf[gdf[Stream_ID_Field].isin(sources)][Stream_ID_Field]

    # Function to extract start and end coordinates, excluding confluence points
    def get_coords(geometry):
        if isinstance(geometry, LineString):
            return [geometry.coords[-1]], [geometry.coords[0]]
        elif isinstance(geometry, MultiLineString):
            start_coords = [line.coords[-1] for line in geometry.geoms]
            end_coords = [line.coords[0] for line in geometry.geoms]
            return start_coords, end_coords
        else:
            raise TypeError("Geometry must be a LineString or MultiLineString")

    # Extract the start and end coordinates
    start_coords_list = []
    end_coords_list = []
    linkno_list = []

    for geometry, linkno in zip(source_geometries, source_linknos):
        start_coords, end_coords = get_coords(geometry)
        start_coords_list.extend(start_coords)
        end_coords_list.extend(end_coords)
        linkno_list.extend([linkno] * len(start_coords))

    # Convert end_coords to a set for efficient lookup
    end_coords_set = set(end_coords_list)

    # Filter start_coords to exclude any coordinates present in end_coords
    filtered_seed_coords = [(coord, linkno) for coord, linkno in zip(start_coords_list, linkno_list) if coord not in end_coords_set]

    # Convert the filtered start coordinates to Point geometries and keep the LINKNO field
    seed_points = [Point(coords) for coords, _ in filtered_seed_coords]
    seed_linknos = [linkno for _, linkno in filtered_seed_coords]

    # Create a new GeoDataFrame for the starting locations
    seed_gdf = gpd.GeoDataFrame({'LINKNO': seed_linknos, 'geometry': seed_points}, crs=gdf.crs)

    # Export the starting locations to a point shapefile
    seed_gdf.to_file(SEED_Output_File)

    print("SEED locations have been exported as a separate point shapefile.")
    
    return (seed_gdf)

def FindClosestSEEDPoints(seed_gdf, curve_data_gdf):
    """
    Compares stream cell locations to SEED point locations (i.e., the uppermost headwater extents of the ARC model domain) and finds the closest stream cell for each SEED point 

    Parameters
    ----------
    seed_gdf: geodataframe
        A geodataframe of all SEED locations in your model domain, created using the find_SEED_locations function
    curve_data_gdf: geodataframe
        A geodataframe of all stream cell locations in the ARC model with depth, top-width, and velocity, all estimated using the ARC synthetic rating curves

    Returns
    -------
    curve_data_gdf: geodataframe
        A geodataframe of all stream cell locations in the ARC model, now amended with SEED locations
    """
    # Reproject to a projected coordinate system so that we can accurately measure distances
    seed_gdf = seed_gdf.to_crs(epsg=6933) 

    # Prefill the curve data file with a SEED value, this will be set to 1 if the column is designated as a SEED column
    curve_data_gdf['SEED'] = "0"

    # Spatial join to find the distance between each CP point and each SEED point
    nearest_cp = gpd.sjoin_nearest(seed_gdf, curve_data_gdf, how='left', distance_col='dist')

    # Filter based on attributes
    # Make sure the COMID's match betweent the SEED and curves
    nearest_cp = nearest_cp[nearest_cp['LINKNO'] == nearest_cp['COMID']]

    # Find the minimum distance for each unique gdf1 row
    min_distance_idx = nearest_cp.groupby(nearest_cp.index)['dist'].idxmin()

    # Get the rows with the minimum distance
    nearest_cp = nearest_cp.loc[min_distance_idx]

    # Create a boolean mask for matching rows in the original dataframe
    mask = curve_data_gdf.set_index(['COMID', 'Row', 'Col']).index.isin(nearest_cp.set_index(['COMID', 'Row', 'Col']).index)

    # Update the 'SEED' value to 1 for matching rows
    curve_data_gdf.loc[mask, 'SEED'] = "1"

    return (curve_data_gdf)

def Run_Main_Curve_to_GEOJSON_Program_Stream_Vector(CurveParam_File, STRM_Raster_File, OutGeoJSON_File, OutProjection, StrmShp, Stream_ID_Field, Downstream_ID_Field, SEED_Output_File, Thin_Output=True, COMID_Q_File=None, comid_q_df=None):
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
    StrmShp: str
        The file path and file name of the vector shapefile of flowlines
    Stream_ID_Field: str
        The field in the StrmShp that is the streams unique identifier
    Downstream_ID_Field: str
        The field in the StrmShp that is used to identify the stream downstream of the stream
    SEED_Output_File: str
        The file path and file name of the output shapefile that contains the SEED locations and the unique ID of the stream each represents
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

    # Create a GeoSeries from the latitude and longitude values
    geometry = [Point(xy) for xy in zip(curve_data_df['CP_LON'], curve_data_df['CP_LAT'])]

    # Create a GeoDataFrame
    curve_data_gdf = gpd.GeoDataFrame(curve_data_df, geometry=geometry)

    # set the coordinate system for the geodataframe
    curve_data_gdf = curve_data_gdf.set_crs(OutProjection, inplace=True)

    # Reproject to a projected coordinate system so that we can accurately measure distances
    curve_data_gdf = curve_data_gdf.to_crs(epsg=6933)

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
        seed_gdf = gpd.read_file(SEED_Output_File)
    else:
        seed_gdf = find_SEED_locations(StrmShp, SEED_Output_File, Stream_ID_Field, Downstream_ID_Field) 
    curve_data_gdf = FindClosestSEEDPoints(seed_gdf, curve_data_gdf)

    # output the GeoJSON file
    Write_GeoJSON_File(OutGeoJSON_File, OutProjection, curve_data_gdf)

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

def Thin_Curve_data(curve_data_gdf):
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
    curve_data_gdf = curve_data_gdf[(curve_data_gdf['COMID_MedDEP']>0) & (curve_data_gdf['CP_DEP']<3.0*curve_data_gdf['COMID_MedDEP'])]

    # Create spatial index
    sindex = curve_data_gdf.sindex

    # List to keep track of indices to drop
    indices_to_drop = set()

    # Iterate through each stream cell
    for i, point in curve_data_gdf.iterrows():
        if i in indices_to_drop:
            continue

        # Find potential neighbors within 50 meters
        buffer = point.geometry.buffer(50)
        possible_matches_index = list(sindex.intersection(buffer.bounds))
        possible_matches = curve_data_gdf.iloc[possible_matches_index]

        # iterate and compare each stream cell to the other stream cells
        for j, other_point in possible_matches.iterrows():
            if i != j and j not in indices_to_drop:
                distance = point.geometry.distance(other_point.geometry)
                if distance <= 50:  # 50 meters
                    wse_diff = wse_diff_percentage(float(point['WaterSurfaceElev_m']), float(other_point['WaterSurfaceElev_m']))
                    if wse_diff <= 0.05:  # 0.05%
                        indices_to_drop.add(j)

    # Drop the points that meet the criteria
    curve_data_gdf = curve_data_gdf.drop(indices_to_drop)

    return (curve_data_gdf)

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

    # Reproject to a projected coordinate system so that we can accurately measure distances
    curve_data_gdf = curve_data_gdf.to_crs(OutProjection)

    # Save the converted GeoDataFrame to a new file
    curve_data_gdf.to_file(OutGeoJSON_File, driver='GeoJSON')

    return

def FindClosestSEEDPoints_Based_On_LatLong(SEED_COMID, SEED_Lat, SEED_Lon, SEED_r, SEED_c, curve_data_gdf, OutProjection):
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
    OutProjection: str
        A EPSG formatted text descriptor of the output GeoJSON's coordinate system (e.g., "EPSG:4269")
    
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

    # Create a GeoSeries from the latitude and longitude values
    geometry = [Point(xy) for xy in zip(df['SEED_Lon'], df['SEED_Lat'])]

    # Create a GeoDataFrame
    seed_gdf = gpd.GeoDataFrame(df, geometry=geometry)

    # Set the coordinate reference system (CRS)
    seed_gdf.set_crs(OutProjection, inplace=True)  

    # Reproject to a projected coordinate system so that we can accurately measure distances
    seed_gdf = seed_gdf.to_crs(epsg=6933) 

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
    max_elev_idx = nearest_cp.groupby('SEED_COMID')['BaseElev'].idxmax()

    # Get the rows with the maximum elevatio for each COMID
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

    # Create a GeoSeries from the latitude and longitude values
    geometry = [Point(xy) for xy in zip(curve_data_df['CP_LON'], curve_data_df['CP_LAT'])]

    # Create a GeoDataFrame
    curve_data_gdf = gpd.GeoDataFrame(curve_data_df, geometry=geometry)

    # set the coordinate system for the geodataframe
    curve_data_gdf = curve_data_gdf.set_crs(OutProjection, inplace=True)

    # Reproject to a projected coordinate system so that we can accurately measure distances
    curve_data_gdf = curve_data_gdf.to_crs(epsg=6933)

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
    curve_data_gdf = FindClosestSEEDPoints_Based_On_LatLong(SEED_COMID, SEED_Lat, SEED_Lon, SEED_r, SEED_c, curve_data_gdf, OutProjection)

    # output the GeoJSON file
    Write_GeoJSON_File(OutGeoJSON_File, OutProjection, curve_data_gdf)

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
    parser.add_argument("--strm_raster_file", type=str, required=True, help="Path to stream raster file (TIFF).")
    parser.add_argument("--out_geojson_file", type=str, required=True, help="Output GeoJSON file path.")
    parser.add_argument("--out_projection", type=str, required=True, help="Output projection (e.g., EPSG:4269).")
    parser.add_argument("--thin_geojson", type=bool, default=True, help="Whether to thin the output GeoJSON (default: True).")
    parser.add_argument("--strm_shp", type=str, required=True, help="Path to stream shapefile.")
    parser.add_argument("--stream_id_field", type=str, required=True, help="Field in shapefile with unique stream identifiers.")
    parser.add_argument("--downstream_id_field", type=str, required=True, help="Field in shapefile with downstream identifiers.")
    parser.add_argument("--seed_output_file", type=str, required=True, help="Output file path for SEED locations shapefile.")
    parser.add_argument("--thin_output", type=bool, default=True, help="Whether to thin the output GeoJSON (default: True).")
    parser.add_argument("--comid_q_file", type=str, required=True, help="Path to file with streamflow estimates (CSV).")

    args = parser.parse_args()

    # Call the main function with parsed arguments
    Run_Main_Curve_to_GEOJSON_Program_Stream_Vector(
        CurveParam_File=args.curve_param_file,
        STRM_Raster_File=args.strm_raster_file,
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
    curve_to_geojson_parser.add_argument("--strm_raster_file", type=str, required=True, help="Path to stream raster file (TIFF).")
    curve_to_geojson_parser.add_argument("--out_geojson_file", type=str, required=True, help="Output GeoJSON file path.")
    curve_to_geojson_parser.add_argument("--out_projection", type=str, required=True, help="Output projection (e.g., EPSG:4269).")
    curve_to_geojson_parser.add_argument("--strm_shp", type=str, required=True, help="Path to stream shapefile.")
    curve_to_geojson_parser.add_argument("--stream_id_field", type=str, required=True, help="Field in shapefile with unique stream identifiers.")
    curve_to_geojson_parser.add_argument("--downstream_id_field", type=str, required=True, help="Field in shapefile with downstream identifiers.")
    curve_to_geojson_parser.add_argument("--seed_output_file", type=str, required=True, help="Output file path for SEED locations shapefile.")
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
            STRM_Raster_File=args.strm_raster_file,
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