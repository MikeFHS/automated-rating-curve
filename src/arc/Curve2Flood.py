"""
Code from https://github.com/MikeFHS/curve2flood
"""


#This code looks at a DEM raster to find the dimensions, then writes a script to create a STRM raster.
# built-in imports
import sys
import os
from datetime import datetime
import json


from osgeo import gdal
from osgeo import osr
from osgeo import ogr

import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from numba import njit

from arc import LOG

gdal.UseExceptions()


def convert_cell_size(dem_cell_size, dem_lower_left, dem_upper_right):
    """
    Determines the x and y cell sizes based on the geographic location

    Parameters
    ----------
    None. All input data is available in the parent object

    Returns
    -------
    None. All output data is set into the object

    """

    ### Get the cell size ###
    d_lat = np.fabs((dem_lower_left + dem_upper_right) / 2)

    ### Determine if conversion is needed
    if dem_cell_size > 0.5:
        # This indicates that the DEM is projected, so no need to convert from geographic into projected.
        x_cell_size = dem_cell_size
        y_cell_size = dem_cell_size
        projection_conversion_factor = 1

    else:
        # Reprojection from geographic coordinates is needed
        assert d_lat > 1e-16, "Please use lat and long values greater than or equal to 0."

        # Determine the latitude range for the model
        if d_lat >= 0 and d_lat <= 10:
            d_lat_up = 110.61
            d_lat_down = 110.57
            d_lon_up = 109.64
            d_lon_down = 111.32
            d_lat_base = 0.0

        elif d_lat > 10 and d_lat <= 20:
            d_lat_up = 110.7
            d_lat_down = 110.61
            d_lon_up = 104.64
            d_lon_down = 109.64
            d_lat_base = 10.0

        elif d_lat > 20 and d_lat <= 30:
            d_lat_up = 110.85
            d_lat_down = 110.7
            d_lon_up = 96.49
            d_lon_down = 104.65
            d_lat_base = 20.0

        elif d_lat > 30 and d_lat <= 40:
            d_lat_up = 111.03
            d_lat_down = 110.85
            d_lon_up = 85.39
            d_lon_down = 96.49
            d_lat_base = 30.0

        elif d_lat > 40 and d_lat <= 50:
            d_lat_up = 111.23
            d_lat_down = 111.03
            d_lon_up = 71.70
            d_lon_down = 85.39
            d_lat_base = 40.0

        elif d_lat > 50 and d_lat <= 60:
            d_lat_up = 111.41
            d_lat_down = 111.23
            d_lon_up = 55.80
            d_lon_down = 71.70
            d_lat_base = 50.0

        elif d_lat > 60 and d_lat <= 70:
            d_lat_up = 111.56
            d_lat_down = 111.41
            d_lon_up = 38.19
            d_lon_down = 55.80
            d_lat_base = 60.0

        elif d_lat > 70 and d_lat <= 80:
            d_lat_up = 111.66
            d_lat_down = 111.56
            d_lon_up = 19.39
            d_lon_down = 38.19
            d_lat_base = 70.0

        elif d_lat > 80 and d_lat <= 90:
            d_lat_up = 111.69
            d_lat_down = 111.66
            d_lon_up = 0.0
            d_lon_down = 19.39
            d_lat_base = 80.0

        else:
            raise AttributeError('Please use legitimate (0-90) lat and long values.')

        ## Convert the latitude ##
        d_lat_conv = d_lat_down + (d_lat_up - d_lat_down) * (d_lat - d_lat_base) / 10
        y_cell_size = dem_cell_size * d_lat_conv * 1000.0  # Converts from degrees to m

        ## Longitude Conversion ##
        d_lon_conv = d_lon_down + (d_lon_up - d_lon_down) * (d_lat - d_lat_base) / 10
        x_cell_size = dem_cell_size * d_lon_conv * 1000.0  # Converts from degrees to m

        ## Make sure the values are in bounds ##
        if d_lat_conv < d_lat_down or d_lat_conv > d_lat_up or d_lon_conv < d_lon_up or d_lon_conv > d_lon_down:
            raise ArithmeticError("Problem in conversion from geographic to projected coordinates")

        ## Calculate the conversion factor ##
        projection_conversion_factor = 1000.0 * (d_lat_conv + d_lon_conv) / 2.0
    return x_cell_size, y_cell_size, projection_conversion_factor

def FindFlowRateForEachCOMID_Ensemble(comid_file_lines, flow_event_num, COMID_to_ID, MinCOMID, COMID_Unique_Flow):    
    num_lines = len(comid_file_lines)
    for n in range(1,num_lines):
        splitlines = comid_file_lines[n].strip().split(',')
        COMID = splitlines[0]
        Q = splitlines[1:]
        try:
            i = COMID_to_ID[int(COMID)-MinCOMID]
            COMID_Unique_Flow[i] = float(Q[flow_event_num])
        except:
            COMID_Unique_Flow[i] = 0.0
    return

def Calculate_TW_D_ForEachCOMID(CurveParamFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact):
    num_unique = len(COMID_Unique)
    COMID_Unique_TW = np.zeros(num_unique)
    COMID_Unique_Depth = np.zeros(num_unique)
    COMID_NumRecord = np.zeros(num_unique)
    LOG.info('\nOpening and Reading ' + CurveParamFileName)

    # read the curve data in as a Pandas dataframe
    curve_df = pd.read_csv(CurveParamFileName)

    # Reading the COMID and flow data in as Pandas dataframes
    streamflow_df = pd.DataFrame()
    # Adding the arrays as columns to the dataframe
    streamflow_df['COMID'] = COMID_Unique
    streamflow_df['Q'] = COMID_Unique_Flow

    # merging the curve and streamflow data together
    curve_df = pd.merge(curve_df, streamflow_df, how="left", on="COMID")

    # calculating depth and top-width with the COMID's discharge and the curve parameters
    curve_df['Depth'] = curve_df['depth_a']*curve_df['Q']**curve_df['depth_b']
    curve_df['TopWidth'] = curve_df['tw_a']*curve_df['Q']**curve_df['tw_b']
    curve_df = curve_df[curve_df['Depth']>0]
    curve_df = curve_df[curve_df['TopWidth']>0]
    curve_df['WSE'] = curve_df['Depth'] + curve_df['BaseElev']

    # Calculate mean and standard deviation
    mean = curve_df['TopWidth'].mean()
    std = curve_df['TopWidth'].std()

    # Define the lower and upper bounds (e.g., 2 standard deviations)
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std

    # Filter the DataFrame to remove outliers
    curve_df = curve_df[(curve_df['TopWidth'] >= lower_bound) & (curve_df['TopWidth'] <= upper_bound)]

    # Calculate mean and standard deviation
    mean = curve_df['WSE'].mean()
    std = curve_df['WSE'].std()

    # Define the lower and upper bounds (e.g., 2 standard deviations)
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std

    # Filter the DataFrame to remove outliers
    curve_df = curve_df[(curve_df['WSE'] >= lower_bound) & (curve_df['WSE'] <= upper_bound)]

    # Define a function to replace outliers with median for each group
    def replace_outliers_with_median(group):
        mean = group['TopWidth'].mean()
        std = group['TopWidth'].std()
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        median_value = group['TopWidth'].median()
        group.loc[(group['TopWidth'] < lower_bound) | (group['TopWidth'] > upper_bound), 'TopWidth'] = median_value
        return group

    # Apply the function to each group
    curve_df = curve_df.groupby('COMID', group_keys=False).apply(replace_outliers_with_median)



    # fill in the T_Rast and W_Rast
    for index, row in curve_df.iterrows():
        T_Rast[int(row['Row']),int(row['Col'])] = row['TopWidth'] * TW_MultFact
        W_Rast[int(row['Row']),int(row['Col'])] = row['Depth'] + row['BaseElev']
    

    # Grouping by 'COMID' and calculating the median of 'TopWidth' and 'Depth' for each 'COMID'
    median_top_width_by_comid = curve_df.groupby('COMID')['TopWidth'].median()
    median_depth_by_comid = curve_df.groupby('COMID')['Depth'].median()

    # iterate through the pandas data series and create the output matrices
    for index, value in median_top_width_by_comid.items():
        i = COMID_to_ID[int(index)-MinCOMID]
        COMID_Unique_TW[i] = value
    
    for index, value in median_depth_by_comid.items():
        i = COMID_to_ID[int(index)-MinCOMID]
        COMID_Unique_Depth[i] = value

    TopWidthMax = COMID_Unique_TW.max()

    # see if this helps with memory 
    del(curve_df)

    return (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax)

def Calculate_TW_D_ForEachCOMID_VDTDatabase(VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact):
    num_unique = len(COMID_Unique)
    COMID_Unique_TW = np.zeros(num_unique)
    COMID_Unique_Depth = np.zeros(num_unique)
    COMID_NumRecord = np.zeros(num_unique)
    LOG.info('\nOpening and Reading ' + VDTDatabaseFileName)
    infile = open(VDTDatabaseFileName,'r')
    lines = infile.readlines()
    infile.close()
    
    num_lines = len(lines)
    for n in range(1,num_lines):
        linesplit = lines[n].strip().split(',')
        (COMID, R, C, E, QB) = linesplit[0:5]
        QVTW = linesplit[5:]
        num_q = int(len(QVTW)/4)
        i = COMID_to_ID[int(COMID)-MinCOMID]
        
        #LOG.info(str(COMID_Unique_Flow[i]) + '  ' + '  ' + str(QB) + '  ' + str(QVTW[0]) + '   ' + str(QVTW[(num_q-1)*4]))
        
        Depth = -1.0 
        TopWidth = -1.0 
        WSE = -1.0
        
        #Convert from String to floating point
        QVTW = np.array(QVTW, dtype=np.float32)
        R = int(R)
        C = int(C)
        E = float(E)
        QB = float(QB)
        
        
        if COMID_Unique_Flow[i] <= QB:   #Flow is below baseflow, so ignore
            TopWidth = 1.0
            Depth = 0.001
            WSE = E
        elif COMID_Unique_Flow[i] >=QVTW[0]:
            TopWidth = QVTW[2]
            WSE = QVTW[3]
            Depth = WSE - E
        elif COMID_Unique_Flow[i] <= QVTW[(num_q-1)*4]:
            TopWidth = QVTW[(num_q-1)*4+2]
            WSE = QVTW[(num_q-1)*4+3]
            Depth = WSE - E
        else:
            for x in range(1,num_q):
                if COMID_Unique_Flow[i] >= QVTW[x*4]:
                    denom_val = (QVTW[(x-1)*4] - QVTW[x*4])
                    if abs(denom_val)>0.0001:
                        fractval = (COMID_Unique_Flow[i] - QVTW[x*4]) / denom_val
                        WSE = QVTW[x*4+3] + fractval * (QVTW[(x-1)*4+3] - QVTW[(x-1)*4+3])
                        if WSE < E:
                            WSE = E
                        Depth = WSE - E
                        TopWidth = QVTW[x*4+2] + fractval * (QVTW[(x-1)*4+2] - QVTW[(x-1)*4+2])
                    else:
                        WSE = QVTW[x*4+3]
                        if WSE < E:
                            WSE = E
                        Depth = WSE - E
                        TopWidth = QVTW[x*4+2]
                    #TopWidth = TopWidth * TW_MultFact
                    break
        TopWidth = TopWidth * TW_MultFact
        #LOG.info(str(Depth) + '  ' + str(TopWidth))
        if TopWidth>0.0001 and WSE>0.0001:
            T_Rast[R,C] = TopWidth
            W_Rast[R,C] = WSE
        
        
        #Calculate the Average Depth and TopWidth
        if Depth > 0.00001 and TopWidth > 0.0001:
            COMID_NumRecord[i] = COMID_NumRecord[i] + 1
            COMID_Unique_TW[i] = ( COMID_Unique_TW[i]*(COMID_NumRecord[i]-1) + TopWidth ) / COMID_NumRecord[i]
            COMID_Unique_Depth[i] = ( COMID_Unique_Depth[i]*(COMID_NumRecord[i]-1) + Depth ) / COMID_NumRecord[i]
        
        
        #if int(COMID) == 750189551:
        #    LOG.info('Q =    ' + str(COMID_Unique_Flow[i]))
        #    LOG.info('Depth = ' + str(Depth))
        #    LOG.info('WSE = ' + str(WSE)) 
        #    LOG.info('TW = ' + str(TopWidth))
        
        #T_Rast[int(R),int(C)] = 100.1
        #W_Rast[int(R),int(C)] = float(E) + 1.1
        
        #LOG.info(COMID)
        #LOG.info(R)
        #LOG.info(C)
        #LOG.info(COMID_Unique_Flow[i])
        #LOG.info(Depth)
        #LOG.info(TopWidth)
    
    TopWidthMax = COMID_Unique_TW.max()

    # remove these to save memory
    del(lines)

    return (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax)

    

def Get_Raster_Details(DEM_File):
    LOG.info(DEM_File)
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
    LOG.info('Spatial Data for Raster File:')
    LOG.info('   ncols = ' + str(ncols))
    LOG.info('   nrows = ' + str(nrows))
    LOG.info('   cellsize = ' + str(cellsize))
    LOG.info('   yll = ' + str(yll))
    LOG.info('   yur = ' + str(yur))
    LOG.info('   xll = ' + str(xll))
    LOG.info('   xur = ' + str(xur))
    return RastArray, ncols, nrows, cellsize, yll, yur, xll, xur, lat, geotransform, Rast_Projection

def GetListOfDEMs(inputfolder):
    DEM_Files = []
    for file in os.listdir(inputfolder):
        #if file.startswith('return_') and file.endswith('.geojson'):
        if file.endswith('.tif') or file.endswith('.img'):
            DEM_Files.append(file)
    return DEM_Files

def Write_Output_Raster(s_output_filename, raster_data, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type):   
    o_driver = gdal.GetDriverByName(s_file_format)  #Typically will be a GeoTIFF "GTiff"
    #o_metadata = o_driver.GetMetadata()
    
    # Construct the file with the appropriate data shape
    o_output_file = o_driver.Create(s_output_filename, xsize=ncols, ysize=nrows, bands=1, eType=s_output_type)
    
    # Set the geotransform
    o_output_file.SetGeoTransform(dem_geotransform)
    
    # Set the spatial reference
    o_output_file.SetProjection(dem_projection)
    
    # Write the data to the file
    o_output_file.GetRasterBand(1).WriteArray(raster_data)
    
    # Once we're done, close properly the dataset
    o_output_file = None

def Write_Output_Raster_As_GeoDataFrame(raster_data, ncols, nrows, dem_geotransform, dem_projection, s_output_type):
    # Create an in-memory raster dataset
    driver = gdal.GetDriverByName('MEM')
    raster_ds = driver.Create('', xsize=ncols, ysize=nrows, bands=1, eType=s_output_type)

    # Set the geotransform and projection
    raster_ds.SetGeoTransform(dem_geotransform)
    raster_ds.SetProjection(dem_projection)

    # Write the data to the in-memory raster dataset
    raster_ds.GetRasterBand(1).WriteArray(raster_data)

    # Create an in-memory vector layer for the polygonized data
    memory_driver = ogr.GetDriverByName('Memory')
    vector_ds = memory_driver.CreateDataSource('')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dem_projection)
    layer = vector_ds.CreateLayer('polygons', srs=srs)

    # Add a field to the layer
    field = ogr.FieldDefn("Value", ogr.OFTInteger)
    layer.CreateField(field)

    # Polygonize the raster and write to the vector layer
    gdal.Polygonize(raster_ds.GetRasterBand(1), None, layer, 0, [], callback=None)

    # Convert the OGR layer to GeoPandas GeoDataFrame
    polygons = []
    values = []
    
    for feature in layer:
        geom = feature.GetGeometryRef()
        # Parse the JSON string to a dictionary
        geom_dict = json.loads(geom.ExportToJson())
        polygons.append(shape(geom_dict))        
        values.append(feature.GetField("Value"))

    # Create a GeoDataFrame
    flood_gdf = gpd.GeoDataFrame({'Value': values, 'geometry': polygons})

    # Set the CRS
    flood_gdf.set_crs(dem_projection, inplace=True)

    # filter to only the flooded area
    flood_gdf = flood_gdf[flood_gdf['Value']>0]

    # Clean up
    raster_ds = None
    vector_ds = None

    return flood_gdf

def Remove_Crop_Circles(flood_gdf, StrmShp_File, s_output_filename):

    strm_gdf = gpd.read_file(StrmShp_File)

    #flood_gdf = gpd.sjoin(flood_gdf, strm_gdf, how="inner", op="intersects")
    flood_gdf = gpd.sjoin(flood_gdf, strm_gdf, how="inner", predicate="intersects")

    shp_output_filename = f"{s_output_filename[:-4]}.gpkg"

    flood_gdf.to_file(shp_output_filename)

    return flood_gdf

@njit(cache=True)
def FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use):
    FourMatrix = np.zeros((3,3)) + 4
    
    nrows_local = r_max-r_min+2
    ncols_local = c_max-c_min+2
    FloodLocal = np.zeros((nrows_local,ncols_local))
    
    FloodLocal[1:nrows_local-1,1:ncols_local-1] = np.where(E_Box<=WSE,1,0)
    
    #This is the Stream Cell.  Mark it with a 4
    FloodLocal[(r_use-r_min+1),(c_use-c_min+1)] = 4 
    
    #Go through and mark all the cells that 
    for r in range((r_use-r_min+1),nrows_local-1):
        for c in range((c_use-c_min+1),ncols_local-1):
            #LOG.info(FloodLocal[r-1:r+2,c-1:c+2].shape)
            #LOG.info(FourMatrix.shape)
            #LOG.info(FloodLocal[r-1:r+2,c-1:c+2])
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix
    for r in range((r_use-r_min+1), 0, -1):
        for c in range((c_use-c_min+1), 0, -1):
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix
    
    for r in range(1, nrows_local-1):
        for c in range(1, ncols_local-1):
            if FloodLocal[r,c]>=3:
                FloodLocal[r-1:r+2,c-1:c+2] = FloodLocal[r-1:r+2,c-1:c+2] * FourMatrix
    
    #LOG.info(FloodLocal)
    #FloodReturn = np.where(FloodLocal[1:nrows_local-1,1:ncols_local-1]>0.0,1.0,0.0)
    #LOG.info(np.where(FloodLocal[1:nrows_local-1,1:ncols_local-1]>3.0,1.0,0.0))
    return np.where(FloodLocal[1:nrows_local-1,1:ncols_local-1]>3.0,1.0,0.0)

@njit(cache=True)
def CreateWeightAndElipseMask(TW_temp, dx, dy, TW_MultFact):
    TW = int(TW_temp)  #This is the number of cells in the top-width
    ElipseMask = np.zeros((TW+1,int(TW*2+1),int(TW*2+1)))  #3D Array
    WeightBox = np.zeros((int(TW*2+1),int(TW*2+1)))  #2D Array
    for i in range(1,TW+1):
        TWDX = i*dx*i*dx
        TWDY = i*dy*i*dy
        for r in range(0,i+1):
            for c in range(0,i+1):
                is_elipse = (c*dx*c*dx/(TWDX)) + (r*dy*r*dy/(TWDY))   #https://www.mathopenref.com/coordgeneralellipse.html
                if is_elipse<=1.0:
                    ElipseMask[i,TW+r,TW+c] = 1.0
                    ElipseMask[i,TW-r,TW+c] = 1.0
                    ElipseMask[i,TW+r,TW-c] = 1.0
                    ElipseMask[i,TW-r,TW-c] = 1.0
    #LOG.info(ElipseMask[2,TW-4:TW+4+1,TW-4:TW+4+1].astype(int))
    #LOG.info(ElipseMask[10,TW-14:TW+14+1,TW-14:TW+14+1].astype(int))
    #LOG.info(ElipseMask[40,TW-44:TW+44+1,TW-44:TW+44+1].astype(int))
    
    for r in range(0,TW+1):
        for c in range(0,TW+1):
            z = pow((c*dx*c*dx + r*dy*r*dy), 0.5)
            if z<0.0001:
                z=0.001
            WeightBox[TW+r,TW+c] = 1 / (z*z)
            WeightBox[TW-r,TW+c] = 1 / (z*z)
            WeightBox[TW+r,TW-c] = 1 / (z*z)
            WeightBox[TW-r,TW-c] = 1 / (z*z)
    
    return WeightBox, ElipseMask

def CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, E, B, nrows, ncols, sd, TW_m, dx, dy, LocalFloodOption, COMID_Unique, COMID_to_ID, MinCOMID, COMID_Unique_TW, COMID_Unique_Depth, WeightBox, ElipseMask, TW_for_WeightBox_ElipseMask, TW, TW_MultFact, TopWidthPlausibleLimit, Set_Depth, disable_pbar: bool = False):
       
    COMID_Averaging_Method = 0
    
    WSE_Times_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float64)
    Total_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float64)
    
    WSE_Times_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float64)
    Total_Weight = np.zeros((nrows+2,ncols+2), dtype=np.float64)
    
    #Now go through each cell
    num_nonzero = len(RR)
    for i in tqdm.tqdm(range(num_nonzero), total=num_nonzero, disable=disable_pbar):
        r = RR[i]
        c = CC[i]
        r_use = r
        c_use = c
        E_Min = E[r,c]
        
        #Now start with rows and start flooding everything in site
        if Set_Depth>0.0:
            WSE = float(E[r_use,c_use] + Set_Depth)
            COMID_TW_m = TopWidthPlausibleLimit
        elif COMID_Averaging_Method!=0 or W_Rast[r-1,c-1]<0.001 or T_Rast[r-1,c-1]<0.00001:
            #Get COMID, TopWidth, and Depth Information for this cell
            COMID_Value = int(B[r,c])
            iii = COMID_to_ID[COMID_Value - MinCOMID]
            COMID_TW_m = COMID_Unique_TW[iii]
            COMID_D = COMID_Unique_Depth[iii]
            WSE = float(E[r_use,c_use] + COMID_D)
        else:
            #These are Based on the AutoRoute Results, not averaged for COMID
            WSE = W_Rast[r-1,c-1]  #Have to have the '-1' because of the Row and Col being inset on the B raster.
            COMID_TW_m = T_Rast[r-1,c-1]
            #LOG.info(str(WSE) + '  ' + str(COMID_TW_m))
        
        if WSE<0.001 or COMID_TW_m<0.00001:
            continue
        
        if COMID_TW_m > TW_m:
            COMID_TW_m = TW_m
        COMID_TW = int(max(round(COMID_TW_m/dx,0),round(COMID_TW_m/dy,0)))  #This is how many cells we will be looking at surrounding our stream cell
        
        if COMID_TW<=1:
            COMID_TW=2
        
        #Find minimum elevation within the search box
        if sd<1:
            r_use = r
            c_use = c
        else:
            for rr in range(r-sd,r+sd+1):
                for cc in range(c-sd,c+sd+1):
                    if rr>0 and rr<(nrows-1) and cc>0 and cc<=(ncols-1) and E[rr,cc]>0.1 and E[rr,cc] < E_Min:
                        E_Min = E[rr,cc]
                        r_use = rr
                        c_use = cc
        
        
        r_min = r_use-COMID_TW
        r_max = r_use+COMID_TW+1
        if r_min<1:
            r_min = 1 
        if r_max>(nrows+1):
            r_max=nrows+1
        c_min = c_use-COMID_TW
        c_max = c_use+COMID_TW+1
        if c_min<1:
            c_min = 1 
        if c_max>(ncols+1):
            c_max=ncols+1
        
        #Find what would flood local
        if LocalFloodOption==True:
            E_Box = E[r_min:r_max,c_min:c_max]
            FloodLocalMask = FloodAllLocalAreas(WSE, E_Box, r_min, r_max, c_min, c_max, r_use, c_use)
        
        #This uses the weighting method from FloodSpreader to create a flood map
        #   Here we use TW instead of COMID_TW.  This is because we are trying to find the center of the weight raster, which was set based on TW (not COMID_TW).  COMID_TW mainly applies to the r_min, r_max, c_min, c_max
        w_r_min = TW_for_WeightBox_ElipseMask-(r_use-r_min)
        w_r_max = TW_for_WeightBox_ElipseMask+r_max-r_use
        w_c_min = TW_for_WeightBox_ElipseMask-(c_use-c_min)
        w_c_max = TW_for_WeightBox_ElipseMask+c_max-c_use
        
       
        if LocalFloodOption==True:
            WSE_Times_Weight[r_min:r_max,c_min:c_max] = WSE_Times_Weight[r_min:r_max,c_min:c_max] + WSE * WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW, w_r_min:w_r_max,w_c_min:w_c_max] * FloodLocalMask
            Total_Weight[r_min:r_max,c_min:c_max] = Total_Weight[r_min:r_max,c_min:c_max] + WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW, w_r_min:w_r_max,w_c_min:w_c_max] * FloodLocalMask
        else:
            WSE_Times_Weight[r_min:r_max,c_min:c_max] = WSE_Times_Weight[r_min:r_max,c_min:c_max] + WSE * WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW, w_r_min:w_r_max,w_c_min:w_c_max]
            Total_Weight[r_min:r_max,c_min:c_max] = Total_Weight[r_min:r_max,c_min:c_max] + WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW, w_r_min:w_r_max,w_c_min:w_c_max] 
        
        
        ###INSTEAD OF ENFORCING BATHY VALUES ON EVERYWHERE ELSE, Just build the bathymetry in another separate function!!!!
        '''
        #Bathymetry
        if Bathy_Yes==1:
            #Find all the bathymetry points within the box that are rivers (ARBathyMask) and have a bathymetry value from ARC (ARBathy)
            (bathy_r_list, bathy_c_list) = np.where(ARBathy[r_min:r_max,c_min:c_max] > 0.1)
            num_bathy_pnts = len(bathy_r_list)
            if num_bathy_pnts>0:
                for bbb in range(num_bathy_pnts):
                    bathy_r = bathy_r_list[bbb] + r_min
                    bathy_c = bathy_c_list[bbb] + c_min
                    
                    b_r_min = bathy_r-COMID_TW_Bathy
                    b_r_max = bathy_r+COMID_TW_Bathy+1
                    if b_r_min<1:
                        b_r_min = 1 
                    if b_r_max>(nrows+1):
                        b_r_max=nrows+1
                    b_c_min = bathy_c-COMID_TW_Bathy
                    b_c_max = bathy_c+COMID_TW_Bathy+1
                    if b_c_min<1:
                        b_c_min = 1 
                    if b_c_max>(ncols+1):
                        b_c_max=ncols+1
                    
                    w_r_min = TW_for_WeightBox_ElipseMask-(bathy_r-b_r_min)
                    w_r_max = TW_for_WeightBox_ElipseMask+b_r_max-bathy_r
                    w_c_min = TW_for_WeightBox_ElipseMask-(bathy_c-b_c_min)
                    w_c_max = TW_for_WeightBox_ElipseMask+b_c_max-bathy_c
                    
                    
                    Bathy_Times_Weight[b_r_min:b_r_max,b_c_min:b_c_max] = Bathy_Times_Weight[b_r_min:b_r_max,b_c_min:b_c_max] + ARBathy[bathy_r,bathy_c] * WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW_Bathy, w_r_min:w_r_max,w_c_min:w_c_max] ###* ARBathyMask[b_r_min:b_r_max,b_c_min:b_c_max]
                    Bathy_Total_Weight[b_r_min:b_r_max,b_c_min:b_c_max] = Bathy_Total_Weight[b_r_min:b_r_max,b_c_min:b_c_max] + WeightBox[w_r_min:w_r_max,w_c_min:w_c_max] * ElipseMask[COMID_TW_Bathy, w_r_min:w_r_max,w_c_min:w_c_max] ###* ARBathyMask[b_r_min:b_r_max,b_c_min:b_c_max]
                    ARBathy[bathy_r,bathy_c]=0  #Since we evaluated this one, no need to evaluate again
        '''


    #LOG.info('Calculating the Weighted Spatial Average for Water Surface Elevation')
    #with np.errstate(divide='ignore',invalid='ignore'):
    Total_Weight = np.where(Total_Weight == 0, 0.0001, Total_Weight)
    WSE_divided_by_weight = WSE_Times_Weight / Total_Weight
    
    #Also make sure all the Cells that have Stream are counted as flooded.
    Flooded = np.where(WSE_divided_by_weight>E,1,0)
    
    for i in range(num_nonzero):
        Flooded[RR[i],CC[i]] = 1
    
    return Flooded

@njit(cache=True)
def Create_Topobathy_Dataset(RR, CC, E, B, nrows, ncols, WeightBox, ElipseMask, TW_for_WeightBox_ElipseMask, Bathy_Yes, ARBathy, ARBathyMask):
    #Bathymetry
    Bathy = np.copy(ARBathy)
    
    Max_TW_to_Search_for_Bathy_Point = 20
    
    (r_cells_to_evaluate, c_cells_to_evaluate) = np.where(ARBathy==0)
    num_cells_to_evaluate = len(r_cells_to_evaluate)
    if num_cells_to_evaluate>0:
        for bbb in range(num_cells_to_evaluate):
            bathy_r = r_cells_to_evaluate[bbb]
            bathy_c = c_cells_to_evaluate[bbb]
            
            for COMID_TW_Bathy in range(1,Max_TW_to_Search_for_Bathy_Point):
                b_r_min = bathy_r-COMID_TW_Bathy
                b_r_max = bathy_r+COMID_TW_Bathy+1
                if b_r_min<1:
                    b_r_min = 1 
                if b_r_max>(nrows+1):
                    b_r_max=nrows+1
                b_c_min = bathy_c-COMID_TW_Bathy
                b_c_max = bathy_c+COMID_TW_Bathy+1
                if b_c_min<1:
                    b_c_min = 1 
                if b_c_max>(ncols+1):
                    b_c_max=ncols+1
                
                w_r_min = TW_for_WeightBox_ElipseMask-(bathy_r-b_r_min)
                w_r_max = TW_for_WeightBox_ElipseMask+b_r_max-bathy_r
                w_c_min = TW_for_WeightBox_ElipseMask-(bathy_c-b_c_min)
                w_c_max = TW_for_WeightBox_ElipseMask+b_c_max-bathy_c
                
                # COMID_TW_Bathy = min(COMID_TW_Bathy, ElipseMask.shape[0] -1) ricky
                (r_has_bathy, c_has_bathy) = np.where( (ARBathy[b_r_min:b_r_max,b_c_min:b_c_max]*ARBathyMask[b_r_min:b_r_max,b_c_min:b_c_max]*ElipseMask[COMID_TW_Bathy, w_r_min:w_r_max,w_c_min:w_c_max]) > 0   )
                if len(r_has_bathy)>0:
                    
                    #This should be a list of rows and columns within the ARBathy that actually have bathymetry values
                    r_has_bathy = r_has_bathy + b_r_min
                    c_has_bathy = c_has_bathy + b_c_min
                    
                    #This should be a list of rows and columns for the Weight and Elipse Rasters.
                    w_r = TW_for_WeightBox_ElipseMask-(r_has_bathy-b_r_min)
                    w_c = TW_for_WeightBox_ElipseMask-(c_has_bathy-b_c_min)
                    
                    Bathy[bathy_r,bathy_c] = (ARBathy[r_has_bathy,c_has_bathy] * WeightBox[w_r,w_c]).sum() / WeightBox[w_r,w_c].sum()
                    #LOG.info(Bathy[bathy_r,bathy_c])

    Bathy = np.where(Bathy>0, Bathy, E)
    return Bathy[1:nrows+1,1:ncols+1]

def Calculate_Depth_TopWidth_TWMax(CurveParamFileName, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact, TopWidthPlausibleLimit, dx, dy, Set_Depth):
    if Set_Depth>0.0:
        num_unique = len(COMID_Unique)
        COMID_Unique_TW = np.ones(num_unique, dtype=float) * TopWidthPlausibleLimit
        COMID_Unique_Depth = np.ones(num_unique, dtype=float) * Set_Depth
        TopWidthMax = TopWidthPlausibleLimit
    elif len(CurveParamFileName)>1:
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax) = Calculate_TW_D_ForEachCOMID(CurveParamFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact)
    elif len(VDTDatabaseFileName)>1:
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax) = Calculate_TW_D_ForEachCOMID_VDTDatabase(VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact)
    LOG.info('Maximum Top Width = ' + str(TopWidthMax))
    
    for x in range(len(COMID_Unique)):
        if COMID_Unique_TW[x]>TopWidthPlausibleLimit:
            LOG.info('Ignoring ' + str(COMID_Unique[x]) + '  ' + str(COMID_Unique_Flow[x])  + '  ' + str(COMID_Unique_Flow[x]*Q_Fraction) + '  ' + str(COMID_Unique_Depth[x]) + '  ' + str(COMID_Unique_TW[x]))             
    
    if TopWidthPlausibleLimit < TopWidthMax:
        TopWidthMax = TopWidthPlausibleLimit
    
    #Create a Weight Box and an Elipse Mask that can be used for all of the cells
    X_cells = round(TopWidthMax/dx,0)
    Y_cells = round(TopWidthMax/dy,0)
    TW = int(max(Y_cells,X_cells))  #This is how many cells we will be looking at surrounding our stream cell
    
    return COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, TW

def Curve2Flood(E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, num_comids, MinCOMID, MaxCOMID, COMID_to_ID, COMID_Unique_Flow, CurveParamFileName, VDTDatabaseFileName, Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, WeightBox, ElipseMask, TW_for_WeightBox_ElipseMask, LocalFloodOption, Set_Depth, disable_pbar: bool = False):
    
    #These are gridded data from Curve Parameter or VDT Database File
    T_Rast = np.zeros((nrows,ncols))
    W_Rast = np.zeros((nrows,ncols))
    T_Rast = T_Rast - 1.0
    W_Rast = W_Rast - 1.0
    
    
    #Calculate an Average Top Width and Depth for each stream reach.
    (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, TW) = Calculate_Depth_TopWidth_TWMax(CurveParamFileName, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact, TopWidthPlausibleLimit, dx, dy, Set_Depth)
    
    #(WeightBox, ElipseMask) = CreateWeightAndElipseMask(TW, dx, dy, TW_MultFact)  #3D Array with the same row/col dimensions as the WeightBox
    
    
    #Create a simple Flood Map Data
    search_dist_for_min_elev = 0
    LOG.info('Creating Rough Flood Map Data...')
    
    Flood = CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, E, B, nrows, ncols, search_dist_for_min_elev, TopWidthMax, dx, dy, LocalFloodOption, COMID_Unique, COMID_to_ID, MinCOMID, COMID_Unique_TW, COMID_Unique_Depth, WeightBox, ElipseMask, TW_for_WeightBox_ElipseMask, TW, TW_MultFact, TopWidthPlausibleLimit, Set_Depth, disable_pbar)
    return Flood[1:nrows+1,1:ncols+1]


def Set_Stream_Locations(nrows, ncols, infilename):
    S = np.zeros((nrows,ncols))  #Create an array
    LOG.info('Opening ' + infilename)
    infile = open(infilename, 'r')
    lines = infile.readlines()
    infile.close()
    for i in range(1,len(lines)):
        ls = lines[i].strip().split(',')
        S[int(ls[1]),int(ls[2])] = int(ls[0])
    # remove these to save memory
    del(lines)
    return S

def Curve2Flood_MainFunction(DEM_File, STRM_File, StrmShp_File, FlowFileName, CurveParamFileName, VDTDatabaseFileName, Flood_File, FloodImpact_File, Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, LocalFloodOption, Set_Depth, BathyWaterMaskFileName, BathyFromARFileName, BathyOutputFileName, disable_pbar: bool = False):
    
    LOG.info('Get the Raster Dimensions for ' + DEM_File)
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows, dem_geoTransform, dem_projection) = Get_Raster_Details(DEM_File)
    cellsize_x = abs(float(dx))
    cellsize_y = abs(float(dy))
    lat_base = float(maxy) - 0.5*cellsize_y
    lon_base = float(minx) + 0.5*cellsize_x
    
    #Get the Stream Locations from the Curve or VDT File
    if Set_Depth>0.0:
        (S, ncols, nrows, cellsize, yll, yur, xll, xur, lat, dem_geotransform, dem_projection) = Read_Raster_GDAL(STRM_File)
    elif len(CurveParamFileName)>1:
        S = Set_Stream_Locations(nrows, ncols, CurveParamFileName)
    elif len(VDTDatabaseFileName)>1:
        S = Set_Stream_Locations(nrows, ncols, VDTDatabaseFileName)
    else:
        LOG.error('NEED EITHER A CURVE PARAMATER FILE OR A VDT DATABASE FILE')
        
    LOG.info('Opening ' + DEM_File)
    (DEM, ncols, nrows, cellsize, yll, yur, xll, xur, lat, dem_geotransform, dem_projection) = Read_Raster_GDAL(DEM_File)

    
    cellsize_x = abs(float(dx))
    cellsize_y = abs(float(dy))
    lat_base = float(maxy) - 0.5*cellsize_y
    lon_base = float(minx) + 0.5*cellsize_x
        
        
    DEM = np.pad(DEM, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    # E = np.zeros((nrows+2,ncols+2))  #Create an array that is slightly larger than the STRM Raster Array
    # E[1:(nrows+1), 1:(ncols+1)] = DEM
    # E = E.astype(float)

    
    #Get Cellsize Information
    (dx, dy, dm) = convert_cell_size(cellsize, yll, yur)
    dz = pow(dx*dx+dy*dy,0.5)
    
    #Get list of Unique Stream IDs.  Also find where all the cell values are.
    B = np.zeros((nrows+2,ncols+2))  #Create an array that is slightly larger than the STRM Raster Array
    B[1:(nrows+1), 1:(ncols+1)] = S
    B = B.astype(int)
    (RR,CC) = np.where(B > 0)
    COMID_Unique = np.unique(B)
    COMID_Unique = np.delete(COMID_Unique, 0)  #We don't need the first entry of zero
    
    # Sort from Smallest to highest values
    # Ensure it's treated as integers
    COMID_Unique = COMID_Unique.astype(int)
    COMID_Unique = np.sort(COMID_Unique)
   
    # we dont need the S array anymore
    del(S)
    
    # find the number of unique COMID values
    num_comids = len(COMID_Unique)    
    
    # Compute necessary values
    MinCOMID = int(COMID_Unique[0])
    MaxCOMID = int(COMID_Unique[-1])

    LOG.info('\nCOMID Ranges from ' + str(MinCOMID) + ' to ' + str(MaxCOMID))
    
    # Initialize COMID_to_ID array
    COMID_to_ID = np.full(MaxCOMID - MinCOMID + 1, -1, dtype=int)
    
    # Get the unique identifier set
    indices = (COMID_Unique - MinCOMID).astype(int)
    COMID_to_ID[indices] = np.arange(len(COMID_Unique))
    
    #COMID Flow File Read-in
    num_unique = len(COMID_Unique)
    COMID_Unique_Flow = np.zeros(num_unique)
    LOG.info('\nOpening and Reading ' + FlowFileName)
    infile = open(FlowFileName,'r')
    comid_file_lines = infile.readlines()
    infile.close()
    
    #Order from highest to lowest flow
    ls = comid_file_lines[1].split(',')
    num_flows = len(ls)-1
    LOG.info('Evaluating ' + str(num_flows) + ' Flow Events')
    
    #Creating the Weight and Eclipse Boxes
    LOG.info('Creating the Weight and Eclipse Boxes')
    TW = int( max( round(TopWidthPlausibleLimit/dx,0), round(TopWidthPlausibleLimit/dy,0) ) )  #This is how many cells we will be looking at surrounding our stream cell
    TW_for_WeightBox_ElipseMask = TW
    (WeightBox, ElipseMask) = CreateWeightAndElipseMask(TW_for_WeightBox_ElipseMask, dx, dy, TW_MultFact)  #3D Array with the same row/col dimensions as the WeightBox
    
    
    Flood_Ensemble = np.zeros((nrows,ncols))
    
    #If you're setting a set-depth value for all streams, just need to simulate one flood event
    if Set_Depth>=0.0:
        num_flows = 1
    
    #Go through all the Flow Events
    for flow_event_num in range(num_flows):
        LOG.info('Working on Flow Event ' + str(flow_event_num))
        #Get an Average Flow rate associated with each stream reach.
        if Set_Depth<=0.000000001:
            FindFlowRateForEachCOMID_Ensemble(comid_file_lines, flow_event_num, COMID_to_ID, MinCOMID, COMID_Unique_Flow)
        Flood = Curve2Flood(DEM, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, num_comids, MinCOMID, MaxCOMID, COMID_to_ID, COMID_Unique_Flow, CurveParamFileName, VDTDatabaseFileName, Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, WeightBox, ElipseMask, TW_for_WeightBox_ElipseMask, LocalFloodOption, Set_Depth, disable_pbar)
        
        Bathy_Yes = 0  #This keeps the Bathymetry only running on the first flow rate (no need to run it on all flow rates)
        Flood_Ensemble = Flood_Ensemble + Flood
    
    #Turn into a percentage
    Flood_Ensemble = (100 * Flood_Ensemble / num_flows).astype(int)
    
    LOG.info('Creating Ensemble Flood Map...' + str(Flood_File))
    if StrmShp_File:
        # convert the raster to a geodataframe
        flood_gdf = Write_Output_Raster_As_GeoDataFrame(Flood_Ensemble, ncols, nrows, dem_geotransform, dem_projection, gdal.GDT_Int32)
    
        flood_gdf = Remove_Crop_Circles(flood_gdf, StrmShp_File, Flood_File)

    # write the final output raster
    Write_Output_Raster(Flood_File, Flood_Ensemble, ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Int32)
    
    # remove these to conserve memory
    del(comid_file_lines)
    
    
    
    #Bathymetry
    LOG.info('Working on Bathymetry')
    Bathy_Yes = 0
    if BathyFromARFileName is not None:
        Bathy_Yes = 1
        LOG.info('Attempting to open these files to do the Bathymetry work:')
        LOG.info('   ' + BathyFromARFileName)
        try:
            (ARBath, ncolsar, nrowsar, cellsizear, yllar, yurar, xllar, xurar, latar, dem_geotransformar, dem_projectionar) = Read_Raster_GDAL(BathyFromARFileName)
            ARBathy = np.zeros((nrows+2,ncols+2))  #Create an array that is slightly larger than the Bathy Raster Array
            ARBathy[1:(nrows+1), 1:(ncols+1)] = ARBath
            del(ARBath)
        except:
            Bathy_Yes = 0
            LOG.error('Could not open AutoRoute Bathymetry File: ' + BathyFromARFileName)
        
        LOG.info('   ' + BathyWaterMaskFileName)
        try:
            (ARBathyMas, ncolsar, nrowsar, cellsizear, yllar, yurar, xllar, xurar, latar, dem_geotransformar, dem_projectionar) = Read_Raster_GDAL(BathyWaterMaskFileName)
            ARBathyMask = np.zeros((nrows+2,ncols+2))  #Create an array that is slightly larger than the Bathy Raster Array
            ARBathyMask[1:(nrows+1), 1:(ncols+1)] = np.where(ARBathyMas>0,1,0)
            del(ARBathyMas)
        except:
            LOG.warning('Could not open Bathymetry Water Mask File: ' + BathyWaterMaskFileName)
            LOG.warning('Going to use the Flood Map')
            ARBathyMask = np.pad(np.where(Flood_Ensemble>0,1,0), ((1, 1), (1, 1)), mode='constant', constant_values=0)
    
    if Bathy_Yes == 0:
        LOG.info('Not doing Bathymetry.  If you want to do bathymetry add these input cards and files:')
        LOG.info('   BathyWaterMask')
        LOG.info('   BATHY_Out_Filek')
        LOG.info('   FSOutBATHY')
    
    if Bathy_Yes == 1 and False: # NOTE this is disabled for now
        ARBathy = ARBathy * ARBathyMask
        ARBathy = np.where(ARBathyMask==1, ARBathy, -99)
        Bathy = Create_Topobathy_Dataset(RR, CC, DEM, B, nrows, ncols, WeightBox, ElipseMask, TW_for_WeightBox_ElipseMask, Bathy_Yes, ARBathy, ARBathyMask)
        # write the Bathy output raster
        Write_Output_Raster(BathyOutputFileName, Bathy, ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32)

    return


def ReadInputFile(lines,P):
    num_lines = len(lines)
    for i in range(num_lines):
        ls = lines[i].strip().split()
        if len(ls)>1 and ls[0]==P:
            if P=='LocalFloodOption':
                return True
            return ls[1]
    if P=='Q_Fraction':
        return 1.0
    if P=='TopWidthPlausibleLimit':
        return 2500.0
    if P=='TW_MultFact':
        return 3.0
    if P=='Set_Depth':
        return -1.1
    if P=='LocalFloodOption':
        return False
    
    return ''


if __name__ == "__main__":
    
    #If a Main Input File is given, read in the input file
    if len(sys.argv) > 1 or True:
        # MIF_Name = sys.argv[1]
        MIF_Name = '/Users/ricky/Documents/data_dir/mifns/USGS_1_n40w111_20240130__mifn.txt'
        LOG.info('Main Input File Given: ' + MIF_Name)
        
        #Open the Input File
        infile = open(MIF_Name,'r')
        lines = infile.readlines()
        infile.close()
        
        DEM_File = ReadInputFile(lines,'DEM_File')
        STRM_File = ReadInputFile(lines,'Stream_File')
        StrmShp_File = ReadInputFile(lines,'StrmShp_File')
        Flood_File = ReadInputFile(lines,'OutFLD')
        FloodImpact_File = ReadInputFile(lines,'FloodImpact_File')
        FlowFileName = ReadInputFile(lines,'COMID_Flow_File') or ReadInputFile(lines,'Comid_Flow_File')
        VDTDatabaseFileName = ReadInputFile(lines,'Print_VDT_Database')
        CurveParamFileName = ReadInputFile(lines,'Print_Curve_File')
        Q_Fraction = ReadInputFile(lines,'Q_Fraction')
        TopWidthPlausibleLimit = ReadInputFile(lines,'TopWidthPlausibleLimit')
        TW_MultFact = ReadInputFile(lines,'TW_MultFact')
        Set_Depth = ReadInputFile(lines,'Set_Depth')
        LocalFloodOption = ReadInputFile(lines,'LocalFloodOption')
        BathyWaterMaskFileName = ReadInputFile(lines,'BathyWaterMask')
        BathyFromARFileName = ReadInputFile(lines,'BATHY_Out_File')
        BathyOutputFileName = ReadInputFile(lines,'FSOutBATHY')
    else:
        LOG.info('Moving forward with Default File Names')
        #These are the main inputs to the model
        Q_Fraction = 1.0
        TopWidthPlausibleLimit = 200.0
        TW_MultFact = 3.0
        Set_Depth = 0.1
        LocalFloodOption = False
        MainFolder = 'C:/Projects/2023_MultiModelFloodMapping/Yellowstone_GeoGLOWS_FABDEM/'
        DEM_File = MainFolder + 'DEM_FABDEM/Yellowstone_FABDEM.tif' 
        STRM_File = MainFolder + 'STRM/Yellowstone_FABDEM_STRM_Raster_Clean.tif' 
        StrmShp_File = MainFolder + 'StrmShp/Streams_714_Flow_4326_Yellowstone.shp'
        Flood_File = MainFolder + 'FloodMap/Yellowstone_Flood_PY.tif' 
        FloodImpact_File = '' 
        FlowFileName = MainFolder + 'FLOW/Yellowstone_FABDEM_Flow_COMID_Q.txt'
        
        #Option to input a Curve Paramater File, or the VDT_Database File
        CurveParamFileName = MainFolder + 'VDT/Yellowstone_FABDEM_CurveFile_Initial.csv'
        VDTDatabaseFileName = MainFolder + 'VDT/Yellowstone_FABDEM_VDT_Database_Initial.txt'
        VDTDatabaseFileName = ''
        
        #Bathymetry Datasets
        BathyWaterMaskFileName = ''
        BathyFromARFileName = ''
        BathyOutputFileName = ''
    Q_Fraction = float(Q_Fraction)
    TopWidthPlausibleLimit = float(TopWidthPlausibleLimit)
    TW_MultFact = float(TW_MultFact)
    
    Model_Start_Time = datetime.now()
    Curve2Flood_MainFunction(DEM_File, STRM_File, StrmShp_File, FlowFileName, CurveParamFileName, VDTDatabaseFileName, Flood_File, FloodImpact_File, Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, LocalFloodOption, Set_Depth, BathyWaterMaskFileName, BathyFromARFileName, BathyOutputFileName)
    Model_Simulation_Time = datetime.now() - Model_Start_Time
    LOG.info('\n' + 'Simulation time (sec)= ' + str(Model_Simulation_Time.seconds))
