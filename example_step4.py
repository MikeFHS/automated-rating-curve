if __name__ == "__main__":

    #### STEP 4 Example ####
    # Remember to run STEP 3 via the command line before running this example!
    # built-in imports
    import os

    # local imports
    import Curves_To_GeoJSON

    # The path to the stream raster that you created as input for ARC in Step 2.
    STRM_Raster_File = ""
    # The path to the DEM raster that you used as input for ARC in Step 2.
    DEM_Raster_File = "" 
    # The path to the file where you will store you SEED output.
    SEED_Point_File =""

    # By-pass creating the SEED file, if it already exists
    if os.path.isfile(SEED_Point_File) is False:
        (SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, SEED_MinElev, SEED_MaxElev) = Curves_To_GeoJSON.Write_SEED_Data_To_File_FAST_UPDATED(STRM_Raster_File, DEM_Raster_File, SEED_Point_File)
    else:
        (SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, SEED_MinElev, SEED_MaxElev) = Curves_To_GeoJSON.GetSEED_Data_From_File(SEED_Point_File)
    
    # The name of the watershed or area of interest you're simulating.
    WatershedName = ""
    # The CurveFile.csv you created in Step 3.
    CurveParam_File = ""
    # The file containing the unique ID and streamflow for your area of interest that you intend to simulate flood inundation for using FIST.
    COMID_Q_File = ""
    # The path to the stream raster that you created as input for ARC in Step 2.
    STRM_Raster_File = ""
    # The path to the DEM raster that you used as input for ARC in Step 2.
    DEM_Raster_File = ""
    # The path to the GeoJSON file that will be used as input into the FIST model.
    OutGeoJSON_File = ""
    
    Curves_To_GeoJSON.Run_Main_REDUCED_Curve_to_GEOJSON_Program(WatershedName, CurveParam_File, COMID_Q_File, STRM_Raster_File, DEM_Raster_File, OutGeoJSON_File, SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, SEED_MinElev, SEED_MaxElev)