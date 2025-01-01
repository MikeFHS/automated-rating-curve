# built-in imports
import os

# third-party imports
from arc.Curves_To_GeoJSON import Run_Main_Curve_to_GEOJSON_Program_Stream_Vector

if __name__ == "__main__":

    #### STEP 4 Example ####
    # The CurveFile.csv you created in Step 3.
    CurveParam_File = r"C:\Users\jlgut\OneDrive\Desktop\FHS_OperationalFloodMapping\Bathy_Test_Curve2Flood_FABDEM_BanksBathy_Clean\VDT\MO_FABDEM_CurveFile_Bathy.csv"
    # The file containing the unique ID and streamflow for your area of interest that you intend to simulate flood inundation for using FIST.
    COMID_Q_File = r"C:\Users\jlgut\OneDrive\Desktop\FHS_OperationalFloodMapping\Bathy_Test_Curve2Flood_FABDEM_BanksBathy_Clean\FLOW\MO_FABDEM_20241027_GeoGLOWS_forecast.csv"
    # The path to the stream raster that you created as input for ARC in Step 2.
    STRM_Raster_File = r"C:\Users\jlgut\OneDrive\Desktop\FHS_OperationalFloodMapping\Bathy_Test_Curve2Flood_FABDEM_BanksBathy_Clean\STRM\MO_FABDEM_STRM_Raster_Clean.tif"
    # The path to the GeoJSON file that will be used as input into the FIST model.
    OutGeoJSON_File = r"C:\Users\jlgut\OneDrive\Desktop\FHS_OperationalFloodMapping\Bathy_Test_Curve2Flood_FABDEM_BanksBathy_Clean\FIST\MO_FABDEM_test.geojson"
    # The ESPG description of the coordinate system of the OutGeoJSON_File.
    OutProjection = "EPSG:4269"
    # The file path and file name of the vector shapefile of flowlines
    StrmShp = r"C:\Users\jlgut\OneDrive\Desktop\FHS_OperationalFloodMapping\Bathy_Test_Curve2Flood_FABDEM_BanksBathy_Clean\STRM\MO_FABDEM_StrmShp.shp"
    # The field in the StrmShp that is the streams unique identifier
    Stream_ID_Field = "LINKNO"
    # The field in the StrmShp that is used to identify the stream downstream of the stream
    Downstream_ID_Field = "DSLINKNO"
    # The file path and file name of the output shapefile that contains the SEED locations and the unique ID of the stream each represents
    SEED_Output_File = r"C:\Users\jlgut\OneDrive\Desktop\FHS_OperationalFloodMapping\Bathy_Test_Curve2Flood_FABDEM_BanksBathy_Clean\FIST\MO_FABDEM_Seed.shp"
    # True/False of whether or not to filter the output GeoJSON
    Thin_Output = True
        
    Run_Main_Curve_to_GEOJSON_Program_Stream_Vector(CurveParam_File, STRM_Raster_File, OutGeoJSON_File, OutProjection, StrmShp, Stream_ID_Field, Downstream_ID_Field, SEED_Output_File, Thin_Output, COMID_Q_File)
