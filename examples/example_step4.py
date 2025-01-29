# built-in imports
import os

# local imports
from arc.Curves_To_GeoJSON import Run_Main_Curve_to_GEOJSON_Program_Stream_Vector
from arc.Curves_To_GeoJSON import Write_SEED_Data_To_File_Using_Stream_Raster
from arc.Curves_To_GeoJSON import GetSEED_Data_From_File
from arc.Curves_To_GeoJSON import Run_Main_Curve_to_GEOJSON_Program_Stream_Raster

if __name__ == "__main__":

    #### STEP 4 Example ####
    # The CurveFile.csv you created in Step 3.
    CurveParam_File = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\VDT\CurveFile.csv"
    # The file containing the unique ID and streamflow for your area of interest that you intend to simulate flood inundation for using FIST.
    COMID_Q_File = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\714_ExampleForecast\Shields_Flow_COMID_Q_qout_max.txt"
    # The path to the stream raster that you created as input for ARC in Step 2.
    STRM_Raster_File = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\STRM\STRM_Raster_Clean.tif"
    # The path to the GeoJSON file that will be used as input into the FIST model.
    OutGeoJSON_File = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\FIST\Shields_Flow_COMID_Q_qout_max.geojson"
    # The ESPG description of the coordinate system of the OutGeoJSON_File.
    OutProjection = "EPSG:4269"
    # The file path and file name of the vector shapefile of flowlines
    StrmShp = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\StrmShp\StreamShapefile.shp"
    # The field in the StrmShp that is the streams unique identifier
    Stream_ID_Field = "LINKNO"
    # The field in the StrmShp that is used to identify the stream downstream of the stream
    Downstream_ID_Field = "DSLINKNO"
    # The file path and file name of the output text document that contains the SEED locations and the unique ID of the stream each represents
    SEED_Output_Text_File = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\FIST\FIST_Seed.txt"
    # The file path and file name of the output shapefile that contains the SEED locations and the unique ID of the stream each represents
    SEED_Output_Shapefile = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\FIST\FIST_Seed.shp"
    # True/False of whether or not to filter the output GeoJSON
    Thin_Output = True
    # Path to the DEM raster file (TIFF).
    DEM_Raster_File = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\DEM\DEM.tif"

        
    # # here is what you need to run if you're using the stream raster to find your SEED locations
    # # here we're passing the creation of the SEED file to the function if that file exists
    # if os.path.exists(SEED_Output_Text_File):
    #     pass
    # else:
    #     Write_SEED_Data_To_File_Using_Stream_Raster(
    #                                                 STRM_Raster_File,
    #                                                 DEM_Raster_File,
    #                                                 SEED_Output_Text_File)
    # # read in the SEED file
    # SEED_Lat, SEED_Lon, SEED_COMID, SEED_r, SEED_c, SEED_MinElev, SEED_MaxElev = GetSEED_Data_From_File(SEED_Output_Text_File)
    # Run_Main_Curve_to_GEOJSON_Program_Stream_Raster(CurveParam_File, COMID_Q_File, STRM_Raster_File, OutGeoJSON_File, OutProjection, SEED_Lat, SEED_Lon,
    #                                                 SEED_COMID, SEED_r, SEED_c, Thin_Output)    

    

    # here is what you need to run if you're using the stream vector to find your SEED locations
    Run_Main_Curve_to_GEOJSON_Program_Stream_Vector(CurveParam_File, STRM_Raster_File, OutGeoJSON_File, 
                                                    OutProjection, StrmShp, Stream_ID_Field, Downstream_ID_Field, 
                                                    SEED_Output_Shapefile, Thin_Output, COMID_Q_File)