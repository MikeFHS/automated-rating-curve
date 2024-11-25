# built-in imports
import sys
import os

# third-party imports
import shutil
try:
    import gdal 
except: 
    from osgeo import gdal
import numpy as np
import netCDF4
import geopandas as gpd

def Process_AutoRoute_Geospatial_Data_for_testing(test_case, id_field, flow_field, baseflow_field, medium_flow_field, low_flow_field, dem_cleaner, use_clean_dem):
    #Input Dataset
    if use_clean_dem is False:
        Main_Directory = ''
        ARC_Folder = os.path.join(test_case, 'ARC_InputFiles')
        ARC_FileName = os.path.join(test_case, ARC_Folder, 'ARC_Input_File.txt')
        DEM_File = os.path.join(test_case,'DEM', 'DEM.tif')
        LandCoverFile = os.path.join(test_case,'LandCover', 'LandCover.tif')
        ManningN = os.path.join(test_case, 'LAND', 'AR_Manning_n_for_NLCD_MED.txt')
        StrmSHP = os.path.join(test_case,'StrmShp', 'StreamShapefile.shp')
        FlowNC = os.path.join(test_case,'FlowData', 'returnperiods_714.nc')
        VDT_Test_File = os.path.join(test_case, 'VDT', 'VDT_FS.csv')
        
        #Datasets to be Created
        STRM_File = os.path.join(test_case, 'STRM', 'STRM_Raster.tif')
        STRM_File_Clean = STRM_File.replace('.tif','_Clean.tif')
        LAND_File = os.path.join(test_case, 'LAND', 'LAND_Raster.tif')
        FLOW_File = os.path.join(test_case, 'FLOW', 'FlowFile.txt')
        FlowFileFolder = os.path.join(test_case, 'FlowFile')
        BathyFileFolder = os.path.join(test_case, 'Bathymetry')
        FloodFolder = os.path.join(test_case, 'FloodMap')
        STRMFolder = os.path.join(test_case, 'STRM') 
        ARC_Folder = os.path.join(test_case, 'ARC_InputFiles')
        XSFileFolder = os.path.join(test_case, 'XS')
        LandFolder = os.path.join(test_case, 'LAND')
        FlowFolder = os.path.join(test_case, 'FLOW')
        VDTFolder = os.path.join(test_case, 'VDT')
        VDT_File = os.path.join(test_case, 'VDT', 'VDT_Database.txt')
        Curve_File = os.path.join(test_case, 'VDT', 'CurveFile.csv')
        FloodMapFile = os.path.join(FloodFolder,'ARC_Flood.tif')
        DepthMapFile = os.path.join(FloodFolder, 'ARC_Depth.tif')
        ARC_BathyFile = os.path.join(BathyFileFolder,'ARC_Bathy.tif')
        XS_Out_File = os.path.join(XSFileFolder, 'XS_File.txt')
    else:
        Main_Directory = ''
        dem_cleaned_text = "Modified_DEM"
        ARC_Folder = os.path.join(test_case, f'ARC_InputFiles_{dem_cleaned_text}')
        ARC_FileName = os.path.join(test_case, ARC_Folder, f'ARC_Input_File.txt')
        DEM_File = os.path.join(test_case,'DEM_Updated', f'{dem_cleaned_text}.tif')
        LandCoverFile = os.path.join(test_case,'LandCover', 'LandCover.tif')
        ManningN = os.path.join(test_case, 'LAND', 'AR_Manning_n_for_NLCD_MED.txt')
        StrmSHP = os.path.join(test_case,'StrmShp', 'StreamShapefile.shp')
        FlowNC = os.path.join(test_case,'FlowData', 'returnperiods_714.nc')
        VDT_Test_File = os.path.join(test_case, f'VDT_{dem_cleaned_text}', 'VDT_FS.csv')
        
        #Datasets to be Created
        STRM_File = os.path.join(test_case, f'STRM_{dem_cleaned_text}', 'STRM_Raster.tif')
        STRM_File_Clean = STRM_File.replace('.tif',f'_Clean_{dem_cleaned_text}.tif')
        LAND_File = os.path.join(test_case, 'LAND', 'LAND_Raster.tif')
        FLOW_File = os.path.join(test_case, 'FLOW', 'FlowFile.txt')
        FlowFileFolder = os.path.join(test_case, 'FlowFile')
        BathyFileFolder = os.path.join(test_case, f'Bathymetry_{dem_cleaned_text}')
        FloodFolder = os.path.join(test_case, f'FloodMap_{dem_cleaned_text}')
        STRMFolder = os.path.join(test_case, f'STRM_{dem_cleaned_text}') 
        ARC_Folder = os.path.join(test_case, f'ARC_InputFiles_{dem_cleaned_text}')
        XSFileFolder = os.path.join(test_case, f'XS_{dem_cleaned_text}')
        LandFolder = os.path.join(test_case, 'LAND')
        FlowFolder = os.path.join(test_case, 'FLOW')
        VDTFolder = os.path.join(test_case, f'VDT_{dem_cleaned_text}')
        VDT_File = os.path.join(VDTFolder, 'VDT_Database.txt')
        Curve_File = os.path.join(VDTFolder, 'CurveFile.csv')
        FloodMapFile = os.path.join(FloodFolder,'ARC_Flood.tif')
        DepthMapFile = os.path.join(FloodFolder, 'ARC_Depth.tif')
        ARC_BathyFile = os.path.join(BathyFileFolder,'ARC_Bathy.tif')
        XS_Out_File = os.path.join(XSFileFolder, 'XS_File.txt')
    
    #Create Folders
    Create_Folder(STRMFolder)
    Create_Folder(LandFolder)
    Create_Folder(FlowFolder)
    Create_Folder(VDTFolder)
    Create_Folder(FloodFolder)
    Create_Folder(FlowFileFolder)
    Create_Folder(ARC_Folder)
    Create_Folder(BathyFileFolder)
    Create_Folder(XSFileFolder)
    
    
    
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
    
    #Create Stream Raster
    if os.path.isfile(STRM_File):
        print(STRM_File + ' Already Exists')
    else:
        print('Creating ' + STRM_File)
        Create_ARC_StrmRaster(StrmSHP, STRM_File, outputBounds, ncols, nrows, id_field)
    
    #Clean Stream Raster
    if os.path.isfile(STRM_File_Clean):
        print(STRM_File_Clean + ' Already Exists')
    else:
        print('Creating ' + STRM_File_Clean)
        Clean_STRM_Raster(STRM_File, STRM_File_Clean)
    
    #Read all of the flow information for each stream reach
    stream_gdf = gpd.read_file(StrmSHP)
    ID = stream_gdf[id_field].array
    QMax = stream_gdf[flow_field].array
    QBaseflow = stream_gdf[baseflow_field].array
    
    #Get the unique values for all the stream ids
    (S, ncols, nrows, cellsize, yll, yur, xll, xur, lat, dem_geotransform, dem_projection) = Read_Raster_GDAL(STRM_File_Clean)
    (RR,CC) = S.nonzero()
    # (RR, CC) = np.where(S > 0)
    num_strm_cells = len(RR)
    COMID_Unique = np.unique(S)
    COMID_Unique = COMID_Unique[COMID_Unique > 0]
    print(COMID_Unique)
    # COMID_Unique = np.delete(COMID_Unique, 0)  #We don't need the first entry of zero
    COMID_Unique = np.sort(COMID_Unique).astype(int)
    num_comids = len(COMID_Unique)
    
    #Organize the recurrence interval flow data so it easily links with stream raster data
    print('Linking data from ' + STRM_File_Clean + '  with  ' + FlowNC)
    print('\n\nCreating COMID Flow Files in the folder ' + FlowFileFolder)
    MinCOMID = int(COMID_Unique.min())
    MaxCOMID = int(COMID_Unique.max())
    Create_COMID_Flow_Files(stream_gdf, COMID_Unique, num_comids, MinCOMID, MaxCOMID, FlowFileFolder, id_field, baseflow_field, flow_field, medium_flow_field, low_flow_field)
    
    #Write the Flow File for ARC
    num_unique_comid = len(COMID_Unique)
    print('Writing ' + FLOW_File + ' for ' + str(num_unique_comid) + ' unique ID values')
    out_file = open(FLOW_File,'w')
    out_str = f'COMID,{baseflow_field},{flow_field}'
    out_file.write(out_str)
    for i, row in stream_gdf.iterrows():
        out_str = '\n' + str(row[id_field]) + ',' + str(row[baseflow_field]) + ',' + str(row[flow_field])
        out_file.write(out_str)
    out_file.close()

    if dem_cleaner is True:
        DEM_Cleaner_File = os.path.join(FlowFileFolder, "dem_cleaner_flow.txt")
        out_file = open(DEM_Cleaner_File,'w')
        out_str = f'COMID,{baseflow_field},{flow_field}'
        out_file.write(out_str)
        for i, row in stream_gdf.iterrows():
            out_str = f'COMID,{baseflow_field},{flow_field}'
            out_str = '\n' + str(row[id_field]) + ',' + str(row[baseflow_field]) + ',' + str(row[baseflow_field])
            out_file.write(out_str)
    else:
        DEM_Cleaner_File = False
    
    #Create a Baseline Manning N File
    print('Creating Manning n file: ' + ManningN)
    Create_BaseLine_Manning_n_File(ManningN)
    
    #Create a Starting AutoRoute Input File
    ARC_FileName = os.path.join(ARC_Folder,'ARC_Input_File.txt')
    print('Creating ARC Input File: ' + ARC_FileName)
    COMID_Q_File = FlowFileFolder + '/' + 'COMID_Q_qout_max.txt'
    Create_ARC_Model_Input_File(ARC_FileName, DEM_File, id_field, flow_field, baseflow_field, STRM_File_Clean, LAND_File, FLOW_File, VDT_File, Curve_File, ManningN, FloodMapFile, DepthMapFile, ARC_BathyFile, XS_Out_File, DEM_Cleaner_File)
    
    
    print('\n\n')
    print('Next Step is to Run Automated_Rating_Curve_Generator.py by copying the following into the Command Prompt:')
    print('python Automated_Rating_Curve_Generator.py ' + Main_Directory + ARC_Folder + '/' + 'ARC_Input_File.txt')
    
    return

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
    return

def Create_ARC_Model_Input_File(ARC_Input_File, DEM_File, COMID_Param, Q_Param, Q_BF_Param, STRM_File_Clean, LAND_File, FLOW_File, VDT_File, Curve_File, ManningN, FloodMapFile, DepthMapFile, ARC_BathyFile, XS_Out_File, DEM_Cleaner_File, bathy_use_banks):
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
        The path and file name of the output raster that provides a depth-grid flood inundation map that will be output by FloodSpreader
    ARC_BathyFile: str
        The path and file name of the output raster containing the bathymetry estimates output by ARC
    XS_Out_File: str
        The path and file name of the output comma-delimited text file that contains data on the cross-section output by ARC.
    DEM_Cleaner_File:
        The path and file name of the text file containing streamflows that will be passed to the DEM cleaner program to clean the DEM 
    bathy_use_banks: bool
        True/False argument on whether to run ARC bathymetry estimation using the bank elevations (True) or water surface elevation (False)

    Returns
    -------
    None

    """
    out_file = open(ARC_Input_File,'w')
    out_file.write('#ARC Inputs')
    out_file.write('\n' + 'DEM_File	' + DEM_File)
    out_file.write('\n' + 'Stream_File	' + STRM_File_Clean)
    out_file.write('\n' + 'LU_Raster_SameRes	' + LAND_File)
    out_file.write('\n' + 'LU_Manning_n	' + ManningN)
    out_file.write('\n' + 'Flow_File	' + FLOW_File)
    out_file.write('\n' + 'Flow_File_ID	' + COMID_Param)
    out_file.write('\n' + 'Flow_File_BF	' + Q_BF_Param)
    out_file.write('\n' + 'Flow_File_QMax	' + Q_Param)
    out_file.write('\n' + 'Spatial_Units	deg')
    out_file.write('\n' + 'X_Section_Dist	5000.0')
    out_file.write('\n' + 'Degree_Manip	6.1')
    out_file.write('\n' + 'Degree_Interval	1.5')
    out_file.write('\n' + 'Low_Spot_Range	10')
    out_file.write('\n' + 'Str_Limit_Val	1')
    out_file.write('\n' + 'Gen_Dir_Dist	10')
    out_file.write('\n' + 'Gen_Slope_Dist	10')
    
    out_file.write('\n\n#Output CurveFile')
    out_file.write('\n' + 'Print_Curve_File	' + Curve_File)
    
    out_file.write('\n\n#VDT File For Testing Purposes Only')
    out_file.write('\n' + 'Print_VDT	' + VDT_File)
    
    
    out_file.write('\n\n#Bathymetry Information')
    out_file.write('\n' + 'Bathy_Trap_H	0.20')
    out_file.write('\n' + 'Bathy_Use_Banks' + '\t' + str(bathy_use_banks))
    out_file.write('\n' + 'AROutBATHY	' + ARC_BathyFile)

    out_file.write('\n\n#Cross Section Information')
    out_file.write('\n' + 'XS_Out_File	' + XS_Out_File)

    out_file.write('\n\n#Current FloodSpreader Inputs')
    out_file.write('\n' + 'Print_VDT_Database	' + VDT_File)
    if DEM_Cleaner_File is not False:
        out_file.write('\n' + 'COMID_Flow_File	' + DEM_Cleaner_File)
    else:
        out_file.write('\n' + 'COMID_Flow_File	' + FLOW_File)
    out_file.write('\n' + 'OutFLD	' + FloodMapFile)

    out_file.close()

    return

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

    return

def Create_COMID_Flow_Files(stream_gdf, COMID_Unique, num_comids, MinCOMID, MaxCOMID, FlowFileFolder, id_field, baseflow_field, flow_field, medium_flow_field, low_flow_field):
    fmax = open(str(FlowFileFolder) + '/COMID_Q_qout_max.txt', 'w')
    fmax.write('COMID,qout')
    fmed = open(str(FlowFileFolder) + '/COMID_Q_qout_med.txt', 'w')
    fmed.write('COMID,qout')
    flow = open(str(FlowFileFolder) + '/COMID_Q_qout_low.txt', 'w')
    flow.write('COMID,qout')
    fbaseflow = open(str(FlowFileFolder) + '/COMID_Q_baseflow.txt', 'w')
    fbaseflow.write('COMID,qout')
    for i, row in stream_gdf.iterrows():
        out_str = '\n' + str(row[id_field]) + ',' + str(row[flow_field])
        fmax.write(out_str)
        out_str = '\n' + str(row[id_field]) + ',' + str(row[medium_flow_field])
        fmed.write(out_str)
        out_str = '\n' + str(row[id_field]) + ',' + str(row[low_flow_field])
        flow.write(out_str)
        out_str = '\n' + str(row[id_field]) + ',' + str(row[baseflow_field])
        fbaseflow.write(out_str)
    fmax.close()
    fmed.close()
    flow.close()
    fbaseflow.close()
    return

def PullNetCDFInfo(infilename, id_index, q_max, q_2, q_5, q_10, q_25, q_50, q_100):
    print('Opening ' + infilename)
    
    #For NetCDF4
    file2read = netCDF4.Dataset(infilename) 
    temp = file2read.variables[id_index]
    ID = temp[:]*1 
    
    temp = file2read.variables[q_max]
    QMax = temp[:]*1 
    
    temp = file2read.variables[q_2]
    Q2 = temp[:]*1 
    
    temp = file2read.variables[q_5]
    Q5 = temp[:]*1 
    
    temp = file2read.variables[q_10]
    Q10 = temp[:]*1 
    
    temp = file2read.variables[q_25]
    Q25 = temp[:]*1 
    
    temp = file2read.variables[q_50]
    Q50 = temp[:]*1 
    
    temp = file2read.variables[q_100]
    Q100 = temp[:]*1 
    
    file2read.close()
    print('Closed ' + infilename)
    
    #This is for NetCDF3
    '''
    file2read = netcdf.NetCDFFile(infilename,'r') 
    
    ID = []
    Q = []
    rivid = file2read.variables[id_index] # var can be 'Theta', 'S', 'V', 'U' etc..
    q = file2read.variables[q_index] # var can be 'Theta', 'S', 'V', 'U' etc..
    n=-1
    for i in rivid:
        n=n+1
        #min_val = min(q[n])
        max_val = max(q[n])
        ID.append(i)
        Q.append(max_val)
    file2read.close()
    '''
    return ID, QMax, Q2, Q5, Q10, Q25, Q50, Q100

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
    return

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
    return

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

    return

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
                    if (B[r+1,c-1]+B[r+1,c]+B[r+1,c+1])==0 and B[r-1,c]>0:
                        B[r,c] = 0
                        n=n+1
                    #The top cells are all zeros as well, but there is a cell directly below that is legit
                    elif (B[r-1,c-1]+B[r-1,c]+B[r-1,c+1])==0 and B[r+1,c]>0:
                        B[r,c] = 0
                        n=n+1
                #top and bottom cells are zeros
                if B[r,c]>0 and B[r+1,c]==0 and B[r-1,c]==0:
                    #All cells on the right are zero, but there is a cell to the left that is legit
                    if (B[r+1,c+1]+B[r,c+1]+B[r-1,c+1])==0 and B[r,c-1]>0:
                        B[r,c] = 0
                        n=n+1
                    elif (B[r+1,c-1]+B[r,c-1]+B[r-1,c-1])==0 and B[r,c+1]>0:
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
    return

def Process_ARC_Geospatial_Data(Main_Directory, id_field, max_flow_field, baseflow_field, flow_file_path, bathy_use_banks):
    """
    The main function that orchestrates the creation of ARC inputs

    Parameters
    ----------
    Main_Directory: str
        The path to the directory where ARC inputs are stored and where outputs will also be stored
    id_field: str
        The name of the field containing the unique identifier for your stream shapefile
    max_flow_field: str
        The name of the field containing the maximum streamflow input into ARC that is in the flow_file_path file
    baseflow_field: str
        The name of the field containing the baseflow streamflow input into ARC that is in the flow_file_path file
    flow_file_path: str
        The path and file name of the csv file containing the maximum flow and baseflow that will be used to create ARC outputs
    bathy_use_banks: bool
        True/False argument on whether to run ARC bathymetry estimation using the bank elevations (True) or water surface elevation (False)
    Returns
    -------
    None
    
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
    
    #Create Stream Raster
    if os.path.isfile(STRM_File):
        print(STRM_File + ' Already Exists')
    else:
        print('Creating ' + STRM_File)
        Create_ARC_StrmRaster(StrmSHP, STRM_File, outputBounds, ncols, nrows, id_field)
    
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
    Create_ARC_Model_Input_File(ARC_FileName, DEM_File, id_field, max_flow_field, baseflow_field, STRM_File_Clean, LAND_File, flow_file_path, VDT_File, Curve_File, ManningN, FloodMapFile, DepthMapFile, ARC_BathyFile, XS_Out_File, DEM_Cleaner_File, bathy_use_banks)
    
    
    print('\n\n')
    print('Next Step is to Run Automated_Rating_Curve_Generator.py by copying the following into the Command Prompt:')
    print('python Automated_Rating_Curve_Generator.py ' + str(os.path.join(ARC_Folder, 'ARC_Input_File.txt')))
    
    return

if __name__ == "__main__":

    dem_cleaner = False
    use_clean_dem = True
    # test_cases = ['Shields_TestCase', 'Gardiner_TestCase']
    test_cases = ['SC_TestCase','OH_TestCase','TX_TestCase','IN_TestCase','PA_TestCase']
    # test_cases = ['SC_TestCase','OH_TestCase','IN_TestCase','PA_TestCase']
    # test_cases = ['TX_TestCase']


    test_cases_dict = {'SC_TestCase':
                        {'id_field':'COMID',
                         'flow_field': 'HighFlow',
                         'baseflow_field': 'BaseFlow',
                         'medium_flow_field': 'MedFlow',
                         'low_flow_field': 'LowFlow',
                         'flow_file': 'FlowFile.txt'
                        },
                      'OH_TestCase':
                        {'id_field':'COMID',
                         'flow_field': 'HighFlow',
                         'baseflow_field': 'BaseFlow',
                         'medium_flow_field': 'MedFlow',
                         'low_flow_field': 'LowFlow',
                         'flow_file': 'FlowFile.txt'
                        },
                      'TX_TestCase':
                        {'id_field':'permanent_',
                         'flow_field': 'HighFlow',
                         'baseflow_field': 'BaseFlow',
                         'medium_flow_field': 'MedFlow',
                         'low_flow_field': 'LowFlow',
                         'flow_file': 'FlowFile.txt'
                        },
                      'IN_TestCase':
                        {'id_field':'COMID',
                         'flow_field': 'HighFlow',
                         'baseflow_field': 'MedFlow',
                         'medium_flow_field': 'MedFlow',
                         'low_flow_field': 'LowFlow',
                         'flow_file': 'FlowFile.txt'
                        },
                      'PA_TestCase':
                        {'id_field':'permanent_',
                         'flow_field': 'HighFlow',
                         'baseflow_field': 'BaseFlow',
                         'medium_flow_field': 'MedFlow',
                         'low_flow_field': 'LowFlow',
                         'flow_file': 'FlowFile.txt'
                        },
                     }

    for test_case in test_cases:
        test_case_dict = test_cases_dict[test_case]

        FLOW_File = os.path.join(test_case, "FLOW", test_case_dict['flow_file'])

        # Process_AutoRoute_Geospatial_Data_for_testing(test_case, test_case_dict['id_field'], test_case_dict['flow_field'], 
        #                                   test_case_dict['baseflow_field'], test_case_dict['medium_flow_field'], 
        #                                   test_case_dict['low_flow_field'], dem_cleaner, use_clean_dem)
        Process_ARC_Geospatial_Data(test_case, test_case_dict['id_field'], 'HighFlow', 'BaseFlow', FLOW_File)
    