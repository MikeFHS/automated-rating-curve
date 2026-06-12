# built-in imports
import os

# local imports
from arc.Create_GeoJSON import Run_Main_Curve_to_GEOJSON_Program_Stream_Vector
from arc.Create_GeoJSON import Write_SEED_Data_To_File_Using_Stream_Raster
from arc.Create_GeoJSON import GetSEED_Data_From_File
from arc.Create_GeoJSON import Run_Main_Curve_to_GEOJSON_Program_Stream_Raster

if __name__ == "__main__":

    #### STEP 4 Example ####
    # The CurveFile.csv you created in Step 3.
    CurveParam_File = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\VDT\CurveFile.csv"
    # The file containing the unique ID and streamflow for your area of interest that you intend to simulate flood inundation for using FIST.
    COMID_Q_File = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\714_ExampleForecast\Shields_Flow_COMID_Q_qout_max.txt"
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
    # Path to the DEM raster file (TIFF) used to georeference the ARC grid and
    # to drop the lowest outlet terminal when a downstream reach geometry is
    # missing from the local vector network.
    DEM_Raster_File = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\DEM\DEM.tif"    

    # here is what you need to run if you're using the stream vector to find your SEED locations
    Run_Main_Curve_to_GEOJSON_Program_Stream_Vector(CurveParam_File, DEM_Raster_File, OutGeoJSON_File, 
                                                    OutProjection, StrmShp, Stream_ID_Field, Downstream_ID_Field, 
                                                    SEED_Output_Shapefile, Thin_Output, COMID_Q_File)
