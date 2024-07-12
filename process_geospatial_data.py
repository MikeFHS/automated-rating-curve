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

def Create_Folder(F):
    if os.path.exists(F):
        shutil.rmtree(F)
    if not os.path.exists(F): 
        os.makedirs(F)
    return

def Create_ARC_Model_Input_File(MD, ARC_Input_File, DEM_File, COMID_Param, Q_Param, Q_BF_Param, STRM_File_Clean, LAND_File, FLOW_File, VDT_File, Curve_File, ManningN, FloodMapFile, DepthMapFile, ARC_BathyFile, VDT_Test_File,  XS_Out_File, DEM_Cleaner_File):
    out_file = open(ARC_Input_File,'w')
    out_file.write('#ARC_Inputs')
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
    
    out_file.write('\n\n#VDT_Output_File_and_CurveFile')
    out_file.write('\n' + 'Print_VDT_Database	' + VDT_File)
    out_file.write('\n' + 'Print_Curve_File	' + Curve_File)
    
    out_file.write('\n\n#VDT_File_For_TestingPurposes_Only')
    out_file.write('\n' + 'Print_VDT	' + VDT_Test_File)
    out_file.write('\n' + 'OutFLD	' + FloodMapFile)
    
    out_file.write('\n\n#Bathymetry_Information')
    out_file.write('\n' + 'Bathy_Trap_H	0.20')
    out_file.write('\n' + 'AROutBATHY	' + ARC_BathyFile)

    out_file.write('\n\n#Cross Section Information')
    out_file.write('\n' + 'XS_Out_File	' + XS_Out_File)

    out_file.write('\n\n#Current FloodSpreader Inputs')
    out_file.write('\n' + 'Print_VDT_Database	' + VDT_File)
    
    if DEM_Cleaner_File is not False:
        out_file.write('\n' + 'COMID_Flow_File	' + DEM_Cleaner_File)
    else:
        out_file.write('\n' + 'COMID_Flow_File	' + FLOW_File)

    out_file.close()

    return
    

def Create_BaseLine_Manning_n_File(ManningN):
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

def Create_AR_LandRaster(LandCoverFile, LAND_File, projWin_extents, ncols, nrows):
    ds = gdal.Open(LandCoverFile)
    ds = gdal.Translate(LAND_File, ds, projWin = projWin_extents, width=ncols, height = nrows)
    ds = None
    return

def Create_AR_StrmRaster(StrmSHP, STRM_File, outputBounds, minx, miny, maxx, maxy, dx, dy, ncols, nrows, Param):
    source_ds = gdal.OpenEx(StrmSHP)
    # gdal.Rasterize(STRM_File, source_ds, format='GTiff', outputType=gdal.GDT_Int32, outputBounds = outputBounds, width = ncols, height = nrows, noData = 0, attribute = Param)
    gdal.Rasterize(STRM_File, source_ds, format='GTiff', outputType=gdal.GDT_Int64, outputBounds = outputBounds, width = ncols, height = nrows, noData = -9999, attribute = Param)
    source_ds = None
    return

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

def Clean_STRM_Raster(STRM_File, STRM_File_Clean):
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


def Process_AutoRoute_Geospatial_Data(Main_Directory, id_field, max_flow_field, baseflow_field, flow_file_path):
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
    
    
    #Get the Spatial Information from the DEM Raster
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows, dem_geoTransform, dem_projection) = Get_Raster_Details(DEM_File)
    projWin_extents = [minx, maxy, maxx, miny]
    outputBounds = [minx, miny, maxx, maxy]  #https://gdal.org/api/python/osgeo.gdal.html
    
    
    #Create Land Dataset
    if os.path.isfile(LAND_File):
        print(LAND_File + ' Already Exists')
    else: 
        print('Creating ' + LAND_File) 
        Create_AR_LandRaster(LandCoverFile, LAND_File, projWin_extents, ncols, nrows)
    
    #Create Stream Raster
    if os.path.isfile(STRM_File):
        print(STRM_File + ' Already Exists')
    else:
        print('Creating ' + STRM_File)
        Create_AR_StrmRaster(StrmSHP, STRM_File, outputBounds, minx, miny, maxx, maxy, dx, dy, ncols, nrows, id_field)
    
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
    Create_ARC_Model_Input_File(Main_Directory, ARC_FileName, DEM_File, id_field, max_flow_field, baseflow_field, STRM_File_Clean, LAND_File, flow_file_path, VDT_File, Curve_File, ManningN, FloodMapFile, DepthMapFile, ARC_BathyFile, VDT_Test_File,  XS_Out_File, DEM_Cleaner_File)
    
    
    print('\n\n')
    print('Next Step is to Run Automated_Rating_Curve_Generator.py by copying the following into the Command Prompt:')
    print('python Automated_Rating_Curve_Generator.py ' + str(os.path.join(ARC_Folder, 'ARC_Input_File.txt')))
    
    return

if __name__ == "__main__":

    test_cases = ['SC_TestCase','OH_TestCase','TX_TestCase','IN_TestCase','PA_TestCase']

    test_cases_dict = {'SC_TestCase':
                        {'id_field':'COMID',
                         'flow_field': 'HighFlow',
                         'baseflow_field': 'BaseFlow',
                         'medium_flow_field': 'MedFlow',
                         'low_flow_field': 'LowFlow',
                         'flow_file': 'COMID_Q_qout_max.txt'
                        },
                      'OH_TestCase':
                        {'id_field':'COMID',
                         'flow_field': 'HighFlow',
                         'baseflow_field': 'BaseFlow',
                         'medium_flow_field': 'MedFlow',
                         'low_flow_field': 'LowFlow',
                         'flow_file': 'COMID_Q_qout_max.txt'
                        },
                      'TX_TestCase':
                        {'id_field':'permanent_',
                         'flow_field': 'HighFlow',
                         'baseflow_field': 'BaseFlow',
                         'medium_flow_field': 'MedFlow',
                         'low_flow_field': 'LowFlow',
                         'flow_file': 'COMID_Q_qout_max.txt'
                        },
                      'IN_TestCase':
                        {'id_field':'COMID',
                         'flow_field': 'HighFlow',
                         'baseflow_field': 'MedFlow',
                         'medium_flow_field': 'MedFlow',
                         'low_flow_field': 'LowFlow',
                         'flow_file': 'COMID_Q_qout_max.txt'
                        },
                      'PA_TestCase':
                        {'id_field':'permanent_',
                         'flow_field': 'HighFlow',
                         'baseflow_field': 'BaseFlow',
                         'medium_flow_field': 'MedFlow',
                         'low_flow_field': 'LowFlow',
                         'flow_file': 'COMID_Q_qout_max.txt'
                        },
                     }

    for test_case in test_cases:
        test_case_dict = test_cases_dict[test_case]

        FLOW_File = os.path.join(test_case, test_case_dict['flow_file'])

        Process_AutoRoute_Geospatial_Data(test_case, test_case_dict['id_field'], 'qout_max', 'qout_median', FLOW_File)
    