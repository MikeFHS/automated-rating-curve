# Mike Follum
# This code uses regression curves to define depth and top-width based on regression curves.  This information is then exported to a GeoJSON format for FIST
import numpy as np

from scipy.sparse import csr_matrix

import os, sys

try:
    from osgeo import gdal
except:
    import gdal
    from gdalconst import GA_ReadOnly


def Get_Raster_Details(DEM_File):
    """
    Retrieves the geograhic details of a raster using GDAL in a slightly different way than Read_Raster_GDAL().

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
    data = None
    return minx, miny, maxx, maxy, dx, dy, ncols, nrows

def Read_Raster_GDAL(InRAST_Name):
    """
    Retrieves the geograhic details of a raster using GDAL in a slightly different way than Get_Raster_Details().

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
    lat
        The average of the yur and yll latitude values
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
    ncols=band.XSize
    nrows=band.YSize
    band = None
    cellsize = geotransform[1]
    yll = geotransform[3] - nrows * np.fabs(geotransform[5])
    yur = geotransform[3]
    xll = geotransform[0];
    xur = xll + (ncols)*geotransform[1]
    lat = np.fabs((yll+yur)/2.0)
    dataset = None
    print('Spatial Data for Raster File:')
    print('   ncols = ' + str(ncols))
    print('   nrows = ' + str(nrows))
    print('   cellsize = ' + str(cellsize))
    print('   yll = ' + str(yll))
    print('   yur = ' + str(yur))
    print('   xll = ' + str(xll))
    print('   xur = ' + str(xur))
    return RastArray, ncols, nrows, cellsize, yll, yur, xll, xur, lat

def Write_SEED_Data_To_File(STRM_Raster_File, DEM_Raster_File, SEED_Point_File ):
    """
    Uses the stream raster and digital elevation model (DEM) raster to define the potential SEED points or the locations of outlets for streams in the stream raster.

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
    (S, ncols, nrows, cellsize, yll, yur, xll, xur, lat) = Read_Raster_GDAL(STRM_Raster_File)
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
    (DEM, dncols, dnrows, dcellsize, dyll, dyur, dxll, dxur, dlat) = Read_Raster_GDAL(DEM_Raster_File)
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

def Write_GeoJSON_File(OutGeoJSON_File, num_records, OutProjection, CP_COMID, CP_WSE, CP_DEP, CP_VEL, CP_TW, CP_ELEV, CP_LON, CP_LAT, CP_SEED, CP_Q, COMID_List, COMID_MedDEP):
    """
    Writes a GeoJSON file that describes the hydraulic characteristics of a region at a point level

    Parameters
    ----------
    OutGeoJSON_File: str
        A string that represents the file path and file name of the output GeoJSON the function creates
    num_records: int
        An integer representing the number of stream cells in the modeled area
    OutProjection: str
        A EPSG formatted text descriptor of the output GeoJSON's coordinate system (e.g., "EPSG:4269")
    CP_COMID: list
        List of unique identifiers for the stream of each stream cell that read from the ARC curve file
    CP_WSE: list
        List of water surface elevation estimates for each stream cell estimated using the ARC curves
    CP_DEP: list
        List of water depths at each stream cell estimated using the ARC curves
    CP_VEL: list
        List of cross-sectional average streamflow velocities for each stream cell estimated using the ARC curves
    CP_TW: list
        List of top-width estimates of streamflow for each stream cell estimated using the ARC curves
    CP_ELEV: list
        List of channel bottom elevation estimates for each stream cell estimated using the ARC curves
    CP_LON: list
        List of longitude values for each stream cell
    CP_LAT: list
        List of latitude values for each stream cell
    CP_SEED: list
        A list of binary 0 and 1 values with 1 representing a stream cell designated as a SEED location and 0 representing a stream cell location that is not a SEED location.
    CP_Q: list
        A list of float values representing streamflow estimates that are used to estimate CP_WSE, CP_DEP, CP_VEL, and CP_TW values using the ARC curves
    COMID_List: list
        A list of integers that represent unique identifiers for each stream in the modeled area
    COMID_MedDEP: list
        A list of float values that represent the median water depth values for each stream in the modeled area

    Returns
    -------
    None
    """
    # Write out the Initial Header
    print('\nWriting the GeoJSON File ' + OutGeoJSON_File)
    outfile = open(OutGeoJSON_File,'w')
    out_str = '{ "type": "FeatureCollection",'
    
    # Give the projection information
    out_str = out_str + '\n' + '  "crs": {' + '\n' + '    "type": "name",' + '\n' + '    "properties":' + '\n' + '    {'
    out_str = out_str + '\n' + '      "name": "' + OutProjection + '"' #+ ','
    out_str = out_str + '\n' + '    }' #+ ',' 
    out_str = out_str + '\n' + '  },'
    
    out_str = out_str + '\n' + '  "features": ['
    outfile.write(out_str)
    
    p=-1
    for x in range(len(COMID_List)):
        c_index = np.where(CP_COMID==int(COMID_List[x]))
        if np.any(c_index)==True:
            for i in c_index[0]:
                # If the depth values are whack, don't report them
                if COMID_MedDEP[x]>0 and CP_DEP[i]<3.0*COMID_MedDEP[x]:
                    p=p+1
                    if p==0:
                        out_str = '\n'
                    else:
                        out_str = ',\n'
                    out_str = out_str + '    { "type": "Feature",' + '\n' + '      "geometry": {"type": "Point", "coordinates": ['
                    out_str = out_str + str(CP_LON[i]) + ', ' + str(CP_LAT[i]) + ']},' + '\n'
                    out_str = out_str + '      "properties": {' + '\n'
                    
                    out_str = out_str + '           	"WaterSurfaceElev_m": "' + str(CP_WSE[i]) + '",' + '\n'
                    
                    out_str = out_str + '           	"Elev_m": "' + str(CP_ELEV[i]) + '",' + '\n'
                    
                    out_str = out_str + '           	"COMID": "' + str(int(CP_COMID[i])) + '",' + '\n'
                    
                    out_str = out_str + '           	"Flow_cms": "' + str(CP_Q[i]) + '",' + '\n'
                    
                    out_str = out_str + '           	"Velocity": "' + str(CP_VEL[i]) + '",' + '\n'
                    
                    out_str = out_str + '           	"TopWidth": "' + str(CP_TW[i]) + '",' + '\n'
                    
                    out_str = out_str + '           	"SEED": "'
                    out_str = out_str + str(int(CP_SEED[i]))
                    #    out_str = out_str + str(int(CP_SEED[i]))
                    #else:
                    #    out_str = out_str + ''
                    out_str = out_str + '"'
                    
                    out_str = out_str + '\n' + '      	}' + '\n' + '      }'
                    outfile.write(out_str)
    out_str = '\n' + '    ]' + '\n' + '  }'
    outfile.write(out_str)
    outfile.close()
    return

def DetermineMedianDEP_MedianWSE_for_Each_COMID(COMID_List, CP_COMID, CP_DEP, CP_WSE):
    """
    Estimates the median water depth and water surface elevation for all streams in a modeled area

    Parameters
    ----------
    COMID_List: list
        A list of integers that represent unique identifiers for each stream in the modeled area
    CP_COMID: list
        List of unique identifiers for the stream of each stream cell that read from the ARC curve file
    CP_DEP: list
        List of water depths at each stream cell estimated using the ARC curves
    CP_WSE: list
        List of water surface elevation estimates for each stream cell estimated using the ARC curves       

    Returns
    -------
    COMID_MedDEP: list
        A list of float values that represent the median water depth values for each stream in the modeled area
    COMID_MedWSE: list
        A list of float values that represent the median water surface elevation values for each stream in the modeled area   
    """
    print('Determining the Median DEP / WSE for each COMID.')
    COMID_MedDEP = [0.0]*len(COMID_List)
    COMID_MedWSE = [0.0]*len(COMID_List)
    for i in range(len(COMID_List)):
        CID = int(COMID_List[i])
        try:
            index_array = np.where(CP_COMID==CID)
            depth_array = np.array(CP_DEP)[index_array]
            dep_med = np.median(depth_array[0])
            COMID_MedDEP[i]=float(dep_med)
            
            #Although the below says depth, these are WSE
            depth_array = np.array(CP_WSE)[index_array]
            dep_med = np.median(depth_array[0])
            COMID_MedWSE[i]=float(dep_med)
        except:
            COMID_MedDEP[i]=-999.9
            COMID_MedWSE[i]=-999.9
    return COMID_MedDEP, COMID_MedWSE


def FindClosestSEEDPoints_Based_On_LatLong(SEED_COMID, SEED_Lat, SEED_Lon, SEED_r, SEED_c, dx, dy, CP_SEED, CP_ROW, CP_COL, CP_LAT, CP_LON, CP_COMID, CP_WSE, CP_DEP, COMID_List, COMID_MedDEP, COMID_MedWSE):
    """
    Compares stream cell locations to SEED point locations (i.e., stream outlets) and finds the closest stream cell for each SEED point 

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
    dx: float
        The pixel size of the raster longitudinally
    dy: float
        The pixel size of the raster latitudinally 
    CP_SEED: list
        A list of binary 0 and 1 values with 1 representing a stream cell designated as a SEED location and 0 representing a stream cell location that is not a SEED location.
    CP_ROW: list
        List of the row in the stream raster where each stream cell is located
    CP_COL: list
        List of the column in the stream raster where each stream cell is located
    CP_LAT: list
        List of latitude values for each stream cell
    CP_LON: list
        List of longitude values for each stream cell
    CP_COMID: list
        List of unique identifiers for the stream of each stream cell that read from the ARC curve file
    CP_WSE: list
        List of water surface elevation estimates for each stream cell estimated using the ARC curves
    CP_DEP: list
        List of water depths at each stream cell estimated using the ARC curves
    COMID_List: list
        A list of integers that represent unique identifiers for each stream in the modeled area
    COMID_MedDEP: list
        A list of float values that represent the median water depth values for each stream in the modeled area
    COMID_MedWSE
        A list of float values that represent the median water surface elevation values for each stream in the modeled area
    
    Returns
    -------
    None
    """
    Lat_np = np.array(SEED_Lat).astype(float)
    Lon_np = np.array(SEED_Lon).astype(float)
    
    for i in range(len(SEED_COMID)):
        index_vals = np.where(CP_COMID==int(SEED_COMID[i]))
        
        if np.any(index_vals)==True:
            print('Finding SEED location for ' + str(SEED_COMID[i]) + ' with Lat=' + str(Lat_np[i]) + '  and Lon=' + str(Lon_np[i]))
            D_Lat2 = CP_LAT[index_vals] - Lat_np[i]
            D_Lon2 = CP_LON[index_vals] - Lon_np[i]
            D_Lat2 = D_Lat2 * D_Lat2
            D_Lon2 = D_Lon2 * D_Lon2
            
            D = np.sqrt(D_Lat2 + D_Lon2)
            min_index = np.where(D==min(D))
            min_index = min_index[0] # There are a lot of min_index[0] because there could be multiple places that have similar distance, so just go with the first one.
            print(D[min_index[0]])
            first_index = int(index_vals[0][min_index[0]])
            print('         Found Closest cell with Lat/Lon of ' + str(CP_LAT[first_index]) + ' / '  + str(CP_LON[first_index]))
            c_index = np.where(COMID_List==CP_COMID[first_index])
            try:
                c_index = c_index[0][0]
                print('           ' + str(CP_WSE[first_index]) + '  vs  ' + str(COMID_MedWSE[c_index]))
                if CP_WSE[first_index]>=(COMID_MedWSE[c_index]-0.01):
                    CP_SEED[first_index] = 1 # This just marks that this location within the Curve Paramter file should be a SEED Value
                else:
                    print('             NOT USING DUE TO BEING AT OUTLET, NOT START')
            except:
                print('             NOT USING DUE TO COMPLICATIONS')
        else:
            print('Skipping SEED location ' + str(SEED_COMID[i]) + ' with Lat=' + str(Lat_np[i]) + '  and Lon=' + str(Lon_np[i]))
    return


def Run_Main_REDUCED_Curve_to_GEOJSON_Program(WatershedName, CurveParam_File, COMID_Q_File, STRM_Raster_File, DEM_Raster_File, OutGeoJSON_File, OutProjection, SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, SEED_MinElev, SEED_MaxElev):
    """
    Main program that generates GeoJSON file that contains stream cell locations and WSE estimates for a given domain

    Parameters
    ----------
    WatershedName: str
        The name of the watershed or domain you are modeling
    CurveParam_File: str
        The file path and file name of the ARC curve file you are using to estimate water surface elevation and water depth
    COMID_Q_File: str
        The file path and file name of the file that contains the streamflow estimates for the streams in your domain
    STRM_Raster_File: str
        The file path and file name of the stream raster that contains the stream cells you used to run ARC 
    DEM_Raster_File: str
        The file path and file name of the digital elevation model (DEM) that you used to run ARC
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
    SEED_MinElev: list
        A list of float values that represent the minimum elevation of the stream cells on the stream reach of the SEED location
    SEED_MaxElev: list
        A list of float values that represent the maximum elevation of the stream cells on the stream reach of the SEED location
    
    Returns
    -------
    None
    """

    # Read in the COMID_Q file to get all the COMID values and the associated Flow Rates
    print('\nOpening and Reading ' + COMID_Q_File)
    comid_q_data = np.genfromtxt(COMID_Q_File, delimiter=',')
    COMID_List = comid_q_data[1:,0].astype(int)
    Q_List = comid_q_data[1:,1].astype(float)
    comid_q_data = None
    
    
    # Get the Extents and cellsizes of the Raster Data
    print('\nGetting the Spatial information from ' + STRM_Raster_File)
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows) = Get_Raster_Details(STRM_Raster_File)
    cellsize_x = abs(float(dx))
    cellsize_y = abs(float(dy))
    lat_base = float(maxy) - 0.5*cellsize_y
    lon_base = float(minx) + 0.5*cellsize_x
    
    print('Geographic Information:')
    print(lat_base)
    print(lon_base)
    print(cellsize_x)
    print(cellsize_y)
    
    
    print('\nOpening and Reading ' + CurveParam_File)
    curve_data = np.genfromtxt(CurveParam_File, delimiter=',')
    
    # Go through the Curve Params File and Create A list of all the COMID, Row, and Col
    CP_COMID = curve_data[1:,0].astype(int)
    CP_ROW = curve_data[1:,1].astype(int)
    CP_COL = curve_data[1:,2].astype(int)
    CP_ELEV = curve_data[1:,3].astype(float)
    CP_BF = curve_data[1:,4].astype(float)
    CP_MF = curve_data[1:,5].astype(float)
    
    num_records = len(CP_COMID)
    CP_SEED = np.zeros(num_records)
    CP_LAT = np.zeros(num_records)
    CP_LON = np.zeros(num_records)
    CP_Q = np.zeros(num_records).astype(float)
    
    
    # Set the Flow Rates
    for i in range(len(COMID_List)):
        c_index = np.where(CP_COMID==int(COMID_List[i]))
        if np.any(c_index)==True:
            CP_Q[c_index] = Q_List[i]
    
    # Calculate Depth Values
    a = curve_data[1:,6].astype(float)
    b = curve_data[1:,7].astype(float)
    CP_DEP = a*np.power(CP_Q, b, dtype=float)
    CP_DEP = np.maximum(CP_DEP,float(0.05)) # This is so you always have some sort of value to map
    CP_WSE = CP_ELEV + CP_DEP

    # Calculate TopWidth Values
    a = curve_data[1:,8].astype(float)
    b = curve_data[1:,9].astype(float)
    CP_TW = a*np.power(CP_Q, b, dtype=float)
    
    #Calculate Velocity Values
    a = curve_data[1:,10].astype(float)
    b = curve_data[1:,11].astype(float)
    CP_VEL = a*np.power(CP_Q, b, dtype=float)
    
    # Calculate Latitude and Longitude
    CP_LAT = lat_base - CP_ROW * cellsize_y
    CP_LON = lon_base + CP_COL * cellsize_x
    
    
    # For Each COMID, find the Median Depth Value that you would expect to see
    (COMID_MedDEP, COMID_MedWSE) = DetermineMedianDEP_MedianWSE_for_Each_COMID(COMID_List, CP_COMID, CP_DEP, CP_WSE)
    
    # Now look for the Curve points that are closest to the SEED Points that we previously discovered
    FindClosestSEEDPoints_Based_On_LatLong(SEED_COMID, SEED_Lat, SEED_Lon, SEED_r, SEED_c, dx, dy, CP_SEED, CP_ROW, CP_COL, CP_LAT, CP_LON, CP_COMID, CP_WSE, CP_DEP, COMID_List, COMID_MedDEP, COMID_MedWSE)
    
    # Write the GeoJSON File
    Write_GeoJSON_File(OutGeoJSON_File, num_records, OutProjection, CP_COMID, CP_WSE, CP_DEP, CP_VEL, CP_TW, CP_ELEV, CP_LON, CP_LAT, CP_SEED, CP_Q, COMID_List, COMID_MedDEP)

    return


if __name__ == "__main__":
       
    #This forces the program to redo the SEED values
    Redo_Seed_Point_File = False
    Updated_DEM = False
    Filtered_Results = True
       
    Watershed_List = ['SC_TestCase','OH_TestCase','TX_TestCase','IN_TestCase','PA_TestCase']

    for WatershedName in Watershed_List:

        if Updated_DEM is False and Filtered_Results is False:

            
            OutFolder = os.path.join(WatershedName, "FIST_2")

            if os.path.exists(OutFolder):
                pass
            else:
                os.mkdir(OutFolder)

            STRM_Raster_File = os.path.join(WatershedName,"STRM","STRM_Raster_Clean.tif")
            DEM_Raster_File = os.path.join(WatershedName,"DEM","DEM.tif")
            SEED_Point_File = os.path.join(OutFolder,"SEED_Points.txt")
            OutProjection = os.path.join("EPSG:4269")
            CurveParam_File = os.path.join(WatershedName,"VDT", "CurveFile.csv")
        
        if Updated_DEM is True and Filtered_Results is False:

            updated_dem_text = "Modified_DEM"

            OutFolder = os.path.join(WatershedName, f"FIST_{updated_dem_text}")

            if os.path.exists(OutFolder):
                pass
            else:
                os.mkdir(OutFolder)

            STRM_Raster_File = os.path.join(WatershedName,"STRM","STRM_Raster_Clean.tif")
            DEM_Raster_File = os.path.join(WatershedName,"DEM_Updated",f"{updated_dem_text}.tif")
            SEED_Point_File = os.path.join(OutFolder,"SEED_Points.txt")
            OutProjection = os.path.join("EPSG:4269")
            CurveParam_File = os.path.join(WatershedName,f"VDT_{updated_dem_text}", "CurveFile.csv")

        if Filtered_Results is True and Updated_DEM is False:
            updated_filter_text = "filtered"

            OutFolder = os.path.join(WatershedName, f"FIST_{updated_filter_text}")

            if os.path.exists(OutFolder):
                pass
            else:
                os.mkdir(OutFolder)

            STRM_Raster_File = os.path.join(WatershedName,"STRM","STRM_Raster_Clean.tif")
            DEM_Raster_File = os.path.join(WatershedName,"DEM",f"DEM.tif")
            SEED_Point_File = os.path.join(OutFolder,f"SEED_Points_{updated_filter_text}.txt")
            OutProjection = os.path.join("EPSG:4269")
            CurveParam_File = os.path.join(WatershedName,f"VDT", f"CurveFile_{updated_filter_text}.csv")



        scenarios = ['max', 'med', 'low']

        print('\n\nStarting to Work on ' + WatershedName)

        for scenario in scenarios:
            COMID_Q_File = os.path.join(WatershedName, "FlowFile",f"COMID_Q_qout_{scenario}.txt")
            OutGeoJSON_File = os.path.join(OutFolder,f"FIST_Input_{scenario}.geojson")
            # Get the SEED Values
            if os.path.isfile(SEED_Point_File)==False or Redo_Seed_Point_File==True:
                (SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, SEED_MinElev, SEED_MaxElev) = Write_SEED_Data_To_File(STRM_Raster_File, DEM_Raster_File, SEED_Point_File )  #The Write_SEED_Data_To_File_FAST_UPDATED version uses less RAM
            else:
                (SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, SEED_MinElev, SEED_MaxElev) = GetSEED_Data_From_File(SEED_Point_File)
            
            #Run the Main Program to Create a GeoJSON output file
            Run_Main_REDUCED_Curve_to_GEOJSON_Program(WatershedName, CurveParam_File, COMID_Q_File, STRM_Raster_File, DEM_Raster_File, OutGeoJSON_File, OutProjection, SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, SEED_MinElev, SEED_MaxElev)

    
    

    
    