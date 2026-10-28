"""Geospatial preprocessing utilities for ARC.

The functions in this module help convert common ARC geospatial inputs
(DEM, land cover raster, and stream flowline vectors) into the aligned rasters
expected by the ARC core model:

- DEM-aligned land cover raster
- DEM-aligned stream ID raster
- Starter Model Input File (MIF) and related folders

The upstream wiki documents a recommended workflow using
``Process_ARC_Geospatial_Data``:
https://github.com/MikeFHS/automated-rating-curve/wiki/Processing-ARC-Geospatial-Inputs
"""

# built-in imports
import os
import shutil
import sys

import numpy as np

# third-party imports
from osgeo import gdal


def Create_Folder(F):
    """
    Creates an empty directory

    Parameters
    ----------
    F: str
        The path to the directory you want to create
    
    Returns
    -------
    None

    """
    if os.path.exists(F):
        shutil.rmtree(F)
    if not os.path.exists(F): 
        os.makedirs(F)

def Create_ARC_Model_Input_File(
    ARC_Input_File,
    DEM_File,
    COMID_Param,
    Q_Param,
    Q_BF_Param,
    STRM_File_Clean,
    LAND_File,
    FLOW_File,
    VDT_File,
    Curve_File,
    ManningN,
    FloodMapFile,
    DepthMapFile,
    ARC_BathyFile,
    XS_Out_File,
    DEM_Cleaner_File,
    bathy_use_banks,
    use_land_cover_to_find_banks,
    STRM_SHP_File='',
    drainage_area_field='',
    coefficient_depth=None,
    exponent_depth=None,
    coefficient_width=None,
    exponent_width=None,
    reach_id_field='',
    downstream_reach_id_field='',
    lc_water_value=None,
):
    """
    Creates an input text file for the Automated Rating Curve (ARC) tool

    Parameters
    ----------
    ARC_Input_File: str
        The path and file name of the text file that contains the ARC execution instructions
    DEM_File: str
        The path and file name of the digital elevation model (DEM) raster file that will be used in the ARC simulation
    COMID_Param: str
        The header of the column in the FLOW_File that contains the unique identifiers for all streams and is identicial to the stream cell values in the stream raster
    Q_Param: str
        The header of the column in the FLOW_File that contains the maximum streamflow that will be used int he ARC simulation
    Q_BF_Param: str
        The header of the column in the FLOW_File that contains the streamflow value that will be used by ARC to estimate bathymetry
    STRM_File_Clean: str
        The path and file name of the stream raster that will be used in the ARC simulation
    STRM_SHP_File: str, optional
        The path and file name of the stream vector dataset used for stream
        slopes and optional drainage-area bathymetry relationships.
    LAND_File: str
        The path and file name of the land-use/land-cover raster that will be used in the ARC simulation
    FLOW_File: str
        The path and file name of the comma-delimited text file that contains the COMID_Param, Q_Param, and Q_BF_Param values that will be used in the ARC simulation
    VDT_File: str
        The path and file name of the velocity, depth, top-width (VDT) text file database file that will be output by ARC
    Curve_File: str
        The path and file name of the file that contains power function curves for each stream cell that will be output by ARC
    ManningN: str
        The path and file name of the file that contains the Manning's n values for each land-cover in the LAND_File
    FloodMapFile: str
        The path and file name of the output raster that provides a flood inundation map that will be output by FloodSpreader
    DepthMapFile: str
        Not used: the input file names no depth map (add Curve2Flood's ``OutDEP`` to it for one). Kept so that existing
        calls still work.
    ARC_BathyFile: str
        The path and file name of the output raster containing the bathymetry estimates output by ARC
    XS_Out_File: str
        The path and file name of the output comma-delimited text file that contains data on the cross-section output by ARC.
    DEM_Cleaner_File:
        The path and file name of the text file containing streamflows that will be passed to the DEM cleaner program to clean the DEM 
    bathy_use_banks: bool
        True/False argument on whether to run ARC bathymetry estimation using the bank elevations (True) or water surface elevation (False)
    use_land_cover_to_find_banks: bool
        True/False argument on whether to look for the banks first where the land cover's water ends (True), before
        the DEM's methods (the width-to-depth ratio, then the flat water), or to take them from the DEM alone (False)
    drainage_area_field: str, optional
        Name of the drainage-area field in ``STRM_SHP_File`` to use when
        estimating bathymetry depth/width from power laws.
    coefficient_depth, exponent_depth, coefficient_width, exponent_width: float, optional
        Power-law coefficients and exponents for the bathymetry. A reach's
        depth from the depth law takes the place of the depth from
        ``Flow_File_BF`` wherever it is valid, so with it ``Q_BF_Param`` may be
        empty; the width law sets the width of channels whose banks aren't
        found.
    reach_id_field, downstream_reach_id_field: str, optional
        The fields in ``STRM_SHP_File`` holding each reach's ID and the ID of the
        reach downstream of it (``reach_id`` and ``downstream_reach_id``). The
        bathymetry needs both, and the input file always asks for it
        (``AROutBATHY``): ARC only leaves it out, with a warning, when there is
        neither a baseflow column nor a depth power law.
    lc_water_value: int, optional
        The water class in ``LAND_File`` and ``ManningN`` (``LC_Water_Value``).
        ARC defaults to 80, ESA WorldCover's water class.

    Returns
    -------
    None

    """
    out_file = open(ARC_Input_File,'w')
    out_file.write('#ARC Inputs')
    out_file.write('\n' + 'DEM_File	' + DEM_File)
    out_file.write('\n' + 'Stream_File	' + STRM_File_Clean)
    if STRM_SHP_File:
        out_file.write('\n' + 'StrmShp_File\t' + STRM_SHP_File)
        if reach_id_field:
            out_file.write('\n' + 'reach_id\t' + reach_id_field)
        if downstream_reach_id_field:
            out_file.write('\n' + 'downstream_reach_id\t' + downstream_reach_id_field)
    out_file.write('\n' + 'LU_Raster_SameRes	' + LAND_File)
    out_file.write('\n' + 'LU_Manning_n	' + ManningN)
    if lc_water_value is not None:
        out_file.write('\n' + 'LC_Water_Value\t' + str(lc_water_value))
    out_file.write('\n' + 'Flow_File	' + FLOW_File)
    out_file.write('\n' + 'Flow_File_ID	' + COMID_Param)
    if Q_BF_Param:
        out_file.write('\n' + 'Flow_File_BF	' + Q_BF_Param)
    out_file.write('\n' + 'Flow_File_QMax	' + Q_Param)
    out_file.write('\n' + 'Spatial_Units	deg')
    out_file.write('\n' + 'Degree_Manip	6.1')
    out_file.write('\n' + 'Degree_Interval	1.5')
    out_file.write('\n' + 'Low_Spot_Range	10')
    out_file.write('\n' + 'Str_Limit_Val	1')
    out_file.write('\n' + 'Gen_Dir_Dist	10')
    out_file.write('\n' + 'Gen_Slope_Dist	10')
    
    out_file.write('\n\n#Output CurveFile')
    out_file.write('\n' + 'Print_Curve_File	' + Curve_File)
    
    out_file.write('\n\n#Print_VDT is an old key ARC ignores: ARC writes the VDT database to Print_VDT_Database, below')
    out_file.write('\n' + 'Print_VDT	' + VDT_File)
    
    
    out_file.write('\n\n#Bathymetry Information')
    out_file.write('\n' + 'Bathy_Trap_H	0.20')
    out_file.write('\n' + 'Bathy_Use_Banks' + '\t' + str(bathy_use_banks))
    out_file.write('\n' + 'FindBanksBasedOnLandCover' + '\t' + str(use_land_cover_to_find_banks))
    if drainage_area_field and (
        (coefficient_depth is not None and exponent_depth is not None)
        or (coefficient_width is not None and exponent_width is not None)
    ):
        out_file.write('\n' + 'drainage_area_field\t' + str(drainage_area_field))
        if coefficient_depth is not None and exponent_depth is not None:
            out_file.write('\n' + 'coefficient_depth\t' + str(coefficient_depth))
            out_file.write('\n' + 'exponent_depth\t' + str(exponent_depth))
        if coefficient_width is not None and exponent_width is not None:
            out_file.write('\n' + 'coefficient_width\t' + str(coefficient_width))
            out_file.write('\n' + 'exponent_width\t' + str(exponent_width))
    out_file.write('\n' + 'AROutBATHY	' + ARC_BathyFile)

    out_file.write('\n\n#Cross Section Information')
    out_file.write('\n' + 'XS_Out_File	' + XS_Out_File)

    out_file.write('\n\n#The VDT database ARC writes, which Curve2Flood reads, and Curve2Flood inputs ARC ignores')
    out_file.write('\n' + 'Print_VDT_Database	' + VDT_File)
    if DEM_Cleaner_File is not False:
        out_file.write('\n' + 'COMID_Flow_File	' + DEM_Cleaner_File)
    else:
        out_file.write('\n' + 'COMID_Flow_File	' + FLOW_File)
    out_file.write('\n' + 'OutFLD	' + FloodMapFile)

    out_file.close()


def Create_BaseLine_Manning_n_File(ManningN):
    """
    Creates an text file that associates Manning's n values with National Land Cover Database (NLCD) land cover classes
    
    Values were used in https://doi.org/10.5194/nhess-20-625-2020 and https://doi.org/10.1111/1752-1688.12476
    
    Parameters
    ----------
    ManningN:
        The path and file name of the file that contains the Manning's n values for each land-cover in the LAND_File
    
    Returns
    -------
    None

    """
    out_file = open(ManningN,'w')
    out_file.write('LC_ID	Description	Manning_n')
    out_file.write('\n' + '11	Water	0.030')
    out_file.write('\n' + '21	Dev_Open_Space	0.013')
    out_file.write('\n' + '22	Dev_Low_Intesity	0.050')
    out_file.write('\n' + '23	Dev_Med_Intensity	0.075')
    out_file.write('\n' + '24	Dev_High_Intensity	0.100')
    out_file.write('\n' + '31	Barren_Land	0.030')
    out_file.write('\n' + '41	Decid_Forest	0.120')
    out_file.write('\n' + '42	Evergreen_Forest	0.120')
    out_file.write('\n' + '43	Mixed_Forest	0.120')
    out_file.write('\n' + '52	Shrub	0.050')
    out_file.write('\n' + '71	Grass_Herb	0.030')
    out_file.write('\n' + '81	Pasture_Hay	0.040')
    out_file.write('\n' + '82	Cultivated_Crops	0.035')
    out_file.write('\n' + '90	Woody_Wetlands	0.100')
    out_file.write('\n' + '95	Emergent_Herb_Wet	0.100')
    out_file.close()


def Create_ARC_LandRaster(LandCoverFile, LAND_File, projWin_extents, ncols, nrows):
    """
    Creates an land cover raster that is cloped to a specified extent and cell size
    
   
    Parameters
    ----------
    LandCoverFile: str
        The path and file name of the source National Land Cover Database land-use/land-cover raster
    LAND_File: str
        The path and file name of the output land-use/land-cover dataset 
    projWin_extents: list
        A list of the minimum and maximum extents to which the LAND_File will be clipped, specified as [minimum longitude, maximum latitude, maximum longitude, minimum latitude]
    ncols: int
        The number of columns in the output LAND_File raster
    nrows: int
        The number of rows in the output LAND_File raster
    
    Returns
    -------
    None

    """
    ds = gdal.Open(LandCoverFile)
    ds = gdal.Translate(LAND_File, ds, projWin = projWin_extents, width=ncols, height = nrows)
    ds = None

def Create_ARC_StrmRaster(StrmSHP, STRM_File, outputBounds, ncols, nrows, Param):
    """
    Creates an stream raster from an input stream shapefile that is cloped to a specified extent and cell size
       
    Parameters
    ----------
    StrmSHP: str
        The path and filename of the input stream shapefile that will be used in the ARC analysis
    STRM_File
        The path and filename of the output stream raster that will be used in the ARC analysis 
    outputBounds: list
        A list of the minimum and maximum extents to which the STRM_File will be clipped, specified as [minimum longitude, mininum latitude, maximum longitude, maximum latitude] 
    ncols: int
        The number of columns in the output STRM_File raster
    nrows: int
        The number of rows in the output STRM_File raster
    Param:
        The field in the StrmSHP that will be used to populate the stream cells in the STRM_File
    
    Returns
    -------
    None

    """
    source_ds = gdal.OpenEx(StrmSHP)
    gdal.Rasterize(STRM_File, source_ds, format='GTiff', outputType=gdal.GDT_Int64, outputBounds = outputBounds, width = ncols, height = nrows, noData = -9999, attribute = Param)
    source_ds = None

def Write_Output_Raster(s_output_filename, raster_data, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type):   
    """
    Creates a raster from the specified inputs using GDAL
       
    Parameters
    ----------
    s_output_filename: str
        The path and file name of the output raster
    raster_data: arr
        An array of data values that will be written to the output raster
    ncols: int
        The number of columns in the output raster
    nrows: int
        The number of rows in the output raster
    dem_geotransform: list
        A GDAL GetGeoTransform list that is passed to the output raster
    dem_projection: str
        A GDAL GetProjectionRef() string that contains the projection reference that is passed to the output raster
    s_file_format
        The string that specifies the type of raster that will be output (e.g., GeoTIFF = "GTiff")
    s_output_type
        The type of value that the output raster will be stored as (e.g., gdal.GDT_Int32)
    Returns
    -------
    None

    """
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


def Get_Raster_Details(DEM_File):
    """
    Retrieves the geograhic details of a raster using GDAL in a slightly different way than Read_Raster_GDAL()

    Parameters
    ----------
    DEM_File: str
        The file name and full path to the raster you are analyzing

    Returns
    -------
    minx: float
        The longitude of the top left corner of the top pixel of the raster
    miny: 
        The lowest latitude of the the raster
    maxx: 
        The highest latitude of the the raster
    maxy:
        The latitude of the top left corner of the top pixel of the raster
    dx: float
        The pixel size of the raster longitudinally
    dy: float
        The pixel size of the raster latitudinally 
    ncols: int
        The raster width in pixels
    nrows: int
        The raster height in pixels
    geoTransform: list
        A list of geotranform characteristics for the raster
    Rast_Projection:str
        The projection system reference for the raster
    """
    print(DEM_File)
    gdal.Open(DEM_File, gdal.GA_ReadOnly)
    data = gdal.Open(DEM_File)
    geoTransform = data.GetGeoTransform()
    ncols = int(data.RasterXSize)
    nrows = int(data.RasterYSize)
    dx = geoTransform[1]
    dy = geoTransform[5]
    # The raster's edges, whichever way up it is (dy is negative for a north-up raster, positive for a south-up one)
    minx, maxx = sorted((geoTransform[0], geoTransform[0] + dx * ncols))
    miny, maxy = sorted((geoTransform[3], geoTransform[3] + dy * nrows))
    Rast_Projection = data.GetProjectionRef()
    data = None
    return minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection

def Read_Raster_GDAL(InRAST_Name):
    """
    Retrieves the geograhic details of a raster using GDAL in a slightly different way than Get_Raster_Details()

    Parameters
    ----------
    InRAST_Name: str
        The file name and full path to the raster you are analyzing

    Returns
    -------
    RastArray: arr
        A numpy array of the values in the first band of the raster you are analyzing
    ncols: int
        The raster width in pixels
    nrows: int
        The raster height in pixels
    cellsize: float
        The pixel size of the raster longitudinally
    yll: float
        The lowest latitude of the the raster
    yur: float
        The latitude of the top left corner of the top pixel of the raster
    xll: float
        The longitude of the top left corner of the top pixel of the raster
    xur: float
        The highest longitude of the the raster
    lat: float
        The average of the yur and yll latitude values
    geoTransform: list
        A list of geotranform characteristics for the raster
    Rast_Projection:str
        The projection system reference for the raster
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
    # The bottom and top edges, whichever way up the raster is
    yll, yur = sorted((geotransform[3], geotransform[3] + nrows * geotransform[5]))
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

def Match_DEM_Orientation(raster_file, dem_geotransform):
    """
    GDAL writes the land cover and stream rasters north-up. For a south-up DEM, turn a raster's rows round and give it
    the DEM's geotransform, so that it is on the DEM's grid, which ARC needs.

    Parameters
    ----------
    raster_file: str
        The path and file name of a raster covering the DEM's extent with the DEM's number of rows and columns
    dem_geotransform: list
        The DEM's GDAL geotransform

    Returns
    -------
    None
    """
    if not dem_geotransform[5] > 0:
        return
    dataset = gdal.Open(raster_file, gdal.GA_Update)
    band = dataset.GetRasterBand(1)
    band.WriteArray(band.ReadAsArray()[::-1].copy())
    dataset.SetGeoTransform(tuple(dem_geotransform))
    band = None
    dataset = None
    return

def Clean_STRM_Raster(STRM_File, STRM_File_Clean):
    """
    Removes redundant stream cells from the stream raster

    Parameters
    ----------
    STRM_File: str
        The path and file name of the input stream raster
    STRM_File_Clean: str
        The path and file name of the ouput stream raster

    Returns
    -------
    None

    """
    print('\nCleaning up the Stream File.')
    (SN, ncols, nrows, cellsize, yll, yur, xll, xur, lat, dem_geotransform, dem_projection) = Read_Raster_GDAL(STRM_File)
    
    #Create an array that is slightly larger than the STRM Raster Array
    B = np.zeros((nrows+2,ncols+2))
    
    #Imbed the STRM Raster within the Larger Zero Array
    B[1:(nrows+1), 1:(ncols+1)] = SN
    (RR,CC) = B.nonzero()
    num_nonzero = len(RR)
    
    for filterpass in range(2):
        #First pass is just to get rid of single cells hanging out not doing anything
        p_count = 0
        p_percent = (num_nonzero+1)/100.0
        n=0
        for x in range(num_nonzero):
            if x>=p_count*p_percent:
                p_count = p_count + 1
                print(' ' + str(p_count), end =" ")
            r=RR[x]
            c=CC[x]
            V = B[r,c]
            if V>0:
                #Left and Right cells are zeros
                if B[r,c+1]==0 and B[r,c-1]==0:
                    #The bottom cells are all zeros as well, but there is a cell directly above that is legit
                    if (B[r+1,c-1]+B[r+1,c]+B[r+1,c+1])==0 and B[r-1,c]>0 or (B[r-1,c-1]+B[r-1,c]+B[r-1,c+1])==0 and B[r+1,c]>0:
                        B[r,c] = 0
                        n=n+1
                #top and bottom cells are zeros
                if B[r,c]>0 and B[r+1,c]==0 and B[r-1,c]==0:
                    #All cells on the right are zero, but there is a cell to the left that is legit
                    if (B[r+1,c+1]+B[r,c+1]+B[r-1,c+1])==0 and B[r,c-1]>0 or (B[r+1,c-1]+B[r,c-1]+B[r-1,c-1])==0 and B[r,c+1]>0:
                        B[r,c] = 0
                        n=n+1
        print('\nFirst pass removed ' + str(n) + ' cells')
        
        
        #This pass is to remove all the redundant cells
        n=0
        p_count = 0
        p_percent = (num_nonzero+1)/100.0
        for x in range(num_nonzero):
            if x>=p_count*p_percent:
                p_count = p_count + 1
                print(' ' + str(p_count), end =" ")
            r=RR[x]
            c=CC[x]
            V = B[r,c]
            if V>0:
                if B[r+1,c]==V and (B[r+1,c+1]==V or B[r+1,c-1]==V):
                    if sum(B[r+1,c-1:c+2])==0:
                        B[r+1,c] = 0
                        n=n+1
                elif B[r-1,c]==V and (B[r-1,c+1]==V or B[r-1,c-1]==V):
                    if sum(B[r-1,c-1:c+2])==0:
                        B[r-1,c] = 0
                        n=n+1
                elif B[r,c+1]==V and (B[r+1,c+1]==V or B[r-1,c+1]==V):
                    if sum(B[r-1:r+1,c+2])==0:
                        B[r,c+1] = 0
                        n=n+1
                elif B[r,c-1]==V and (B[r+1,c-1]==V or B[r-1,c-1]==V):
                    if sum(B[r-1:r+1,c-2])==0:
                            B[r,c-1] = 0
                            n=n+1
        print('\nSecond pass removed ' + str(n) + ' redundant cells')
    
    print('Writing Output File ' + STRM_File_Clean)
    Write_Output_Raster(STRM_File_Clean, B[1:nrows+1,1:ncols+1], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Int32)
    #return B[1:nrows+1,1:ncols+1], ncols, nrows, cellsize, yll, yur, xll, xur

def Process_ARC_Geospatial_Data(
    Main_Directory,
    id_field,
    max_flow_field,
    baseflow_field,
    flow_file_path,
    bathy_use_banks,
    use_land_cover_to_find_banks,
    drainage_area_field='',
    coefficient_depth=None,
    exponent_depth=None,
    coefficient_width=None,
    exponent_width=None,
    downstream_id_field='',
):
    """
    Create DEM-aligned ARC geospatial inputs and a starter MIF. Assumes that you've stored your inputs in a specific way. In the same directory where you're planning to store your inputs should be:

    1. A "DEM" directory with a GeoTIFF named "DEM.tif" representing your DEM input.
    2. A "LandCover" directory with a GeoTIFF named "LandCover.tif" representing your LU/LC input.
    3. A "StrmShp" directory with a shapefile named "StreamShapefile.shp" representing your streamline input. 

    Parameters
    ----------
    Main_Directory : str
        Path to the directory where ARC inputs are stored and outputs will be written.
    id_field : str
        Name of the unique identifier field in the stream shapefile (e.g., LINKNO
        for the GEOGLOWS streams). It is burned into the stream raster and
        written to the input file as both ``reach_id`` and ``Flow_File_ID``, so
        the flow file needs a column of the same name holding the same IDs.
    max_flow_field : str
        Name of the maximum-flow field/column (must exist in the flow file).
    baseflow_field : str
        Name of the baseflow/bankfull field/column (must exist in the flow
        file), or ``""`` to leave ``Flow_File_BF`` out: the bathymetry's depth
        then comes from the depth power law, and without that there is no
        bathymetry.
    flow_file_path : str
        Path to the flow file (CSV) used to populate the ARC run.
    bathy_use_banks : bool
        Whether bathymetry estimation should use bank elevations (``True``) vs.
        a WSE-derived approach (``False``).
    use_land_cover_to_find_banks : bool
        Whether bank finding should look first where the land cover's water
        ends (``True``), or use the DEM alone (``False``): its width-to-depth
        ratio, then its flat water.
    drainage_area_field : str, optional
        Name of the drainage-area field in the stream shapefile to use when the
        optional bathymetry power-law mode is desired.
    coefficient_depth, exponent_depth, coefficient_width, exponent_width : float, optional
        Power-law parameters used to estimate bathymetry depth and fallback
        width from the drainage area field. The depth law's depth takes the
        place of the baseflow's wherever it is valid.
    downstream_id_field : str, optional
        Name of the field in the stream shapefile holding the ID of the reach
        downstream of each reach (e.g., DSLINKNO). The input file always asks
        for bathymetry, which ARC makes whenever there is a ``baseflow_field``
        or a depth power law, and the bathymetry needs this field: without it ARC
        stops with an error.
    """
    # we won't worry about using DEM cleaner files in this version
    DEM_Cleaner_File = False
    
    #Input Dataset
    ARC_Folder = os.path.join(Main_Directory, 'ARC_InputFiles')
    ARC_FileName = os.path.join(Main_Directory, ARC_Folder, 'ARC_Input_File.txt')
    DEM_File = os.path.join(Main_Directory,'DEM', 'DEM.tif')
    LandCoverFile = os.path.join(Main_Directory,'LandCover', 'LandCover.tif')
    ManningN = os.path.join(Main_Directory, 'LAND', 'AR_Manning_n_for_NLCD_MED.txt')
    StrmSHP = os.path.join(Main_Directory,'StrmShp', 'StreamShapefile.shp')
    FlowNC = os.path.join(Main_Directory,'FlowData', 'returnperiods_714.nc')
    VDT_Test_File = os.path.join(Main_Directory, 'VDT', 'VDT_FS.csv')
    
    #Datasets to be Created
    STRM_File = os.path.join(Main_Directory, 'STRM', 'STRM_Raster.tif')
    STRM_File_Clean = STRM_File.replace('.tif','_Clean.tif')
    LAND_File = os.path.join(Main_Directory, 'LAND', 'LAND_Raster.tif')
    BathyFileFolder = os.path.join(Main_Directory, 'Bathymetry')
    FloodFolder = os.path.join(Main_Directory, 'FloodMap')
    STRMFolder = os.path.join(Main_Directory, 'STRM') 
    ARC_Folder = os.path.join(Main_Directory, 'ARC_InputFiles')
    XSFileFolder = os.path.join(Main_Directory, 'XS')
    LandFolder = os.path.join(Main_Directory, 'LAND')
    VDTFolder = os.path.join(Main_Directory, 'VDT')
    FISTFolder = os.path.join(Main_Directory, 'FIST')
    VDT_File = os.path.join(Main_Directory, 'VDT', 'VDT_Database.txt')
    Curve_File = os.path.join(Main_Directory, 'VDT', 'CurveFile.csv')
    FloodMapFile = os.path.join(FloodFolder,'ARC_Flood.tif')
    DepthMapFile = os.path.join(FloodFolder, 'ARC_Depth.tif')
    ARC_BathyFile = os.path.join(BathyFileFolder,'ARC_Bathy.tif')
    XS_Out_File = os.path.join(XSFileFolder, 'XS_File.txt')
    
    #Create Folders
    Create_Folder(STRMFolder)
    Create_Folder(LandFolder)
    Create_Folder(VDTFolder)
    Create_Folder(FloodFolder)
    Create_Folder(ARC_Folder)
    Create_Folder(BathyFileFolder)
    Create_Folder(XSFileFolder)
    Create_Folder(FISTFolder)
    
    
    #Get the Spatial Information from the DEM Raster
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows, dem_geoTransform, dem_projection) = Get_Raster_Details(DEM_File)
    projWin_extents = [minx, maxy, maxx, miny]
    outputBounds = [minx, miny, maxx, maxy]  #https://gdal.org/api/python/osgeo.gdal.html
    
    
    #Create Land Dataset
    if os.path.isfile(LAND_File):
        print(LAND_File + ' Already Exists')
    else: 
        print('Creating ' + LAND_File) 
        Create_ARC_LandRaster(LandCoverFile, LAND_File, projWin_extents, ncols, nrows)
        Match_DEM_Orientation(LAND_File, dem_geoTransform)
    
    #Create Stream Raster
    if os.path.isfile(STRM_File):
        print(STRM_File + ' Already Exists')
    else:
        print('Creating ' + STRM_File)
        Create_ARC_StrmRaster(StrmSHP, STRM_File, outputBounds, ncols, nrows, id_field)
        Match_DEM_Orientation(STRM_File, dem_geoTransform)
    
    #Clean Stream Raster
    if os.path.isfile(STRM_File_Clean):
        print(STRM_File_Clean + ' Already Exists')
    else:
        print('Creating ' + STRM_File_Clean)
        Clean_STRM_Raster(STRM_File, STRM_File_Clean)
     
    #Create a Baseline Manning N File
    print('Creating Manning n file: ' + ManningN)
    Create_BaseLine_Manning_n_File(ManningN)
    
    #Create a Starting AutoRoute Input File
    ARC_FileName = os.path.join(ARC_Folder,'ARC_Input_File.txt')
    print('Creating ARC Input File: ' + ARC_FileName)
    Create_ARC_Model_Input_File(
        ARC_FileName,
        DEM_File,
        id_field,
        max_flow_field,
        baseflow_field,
        STRM_File_Clean,
        LAND_File,
        flow_file_path,
        VDT_File,
        Curve_File,
        ManningN,
        FloodMapFile,
        DepthMapFile,
        ARC_BathyFile,
        XS_Out_File,
        DEM_Cleaner_File,
        bathy_use_banks,
        use_land_cover_to_find_banks,
        STRM_SHP_File=StrmSHP,
        drainage_area_field=drainage_area_field,
        coefficient_depth=coefficient_depth,
        exponent_depth=exponent_depth,
        coefficient_width=coefficient_width,
        exponent_width=exponent_width,
        reach_id_field=id_field,
        downstream_reach_id_field=downstream_id_field,
        lc_water_value=11,  # the water class of the NLCD Manning's n table written above
    )
    
    
    print('\n\n')
    print('Inputs created. You are now ready to run ARC!')    
