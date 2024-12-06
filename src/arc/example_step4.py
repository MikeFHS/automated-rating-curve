# built-in imports
import os

if __name__ == "__main__":

    #### STEP 4 Example ####
    # Remember to run STEP 3 via the command line before running this example!
    # built-in imports
    import os

    # local imports
    import Curves_To_GeoJSON

    # The CurveFile.csv you created in Step 3.
    CurveParam_File = ""
    # The file containing the unique ID and streamflow for your area of interest that you intend to simulate flood inundation for using FIST.
    COMID_Q_File = ""
    # The path to the stream raster that you created as input for ARC in Step 2.
    STRM_Raster_File = ""
    # The path to the GeoJSON file that will be used as input into the FIST model.
    OutGeoJSON_File = ""
    # The ESPG description of the coordinate system of the OutGeoJSON_File.
    OutProjection = ""
    # True/False of whether or not to filter the output GeoJSON
    Thin_GeoJSON = True
    # The file path and file name of the vector shapefile of flowlines
    StrmShp = ""
    # The field in the StrmShp that is the streams unique identifier
    Stream_ID_Field = ""
    # The field in the StrmShp that is used to identify the stream downstream of the stream
    Downstream_ID_Field = ""
    # The file path and file name of the output shapefile that contains the SEED locations and the unique ID of the stream each represents
    SEED_Output_File = ""
    # True/False of whether or not to filter the output GeoJSON
    Thin_Output = True
        
    Curves_To_GeoJSON.Run_Main_Curve_to_GEOJSON_Program_Stream_Vector(CurveParam_File, COMID_Q_File, STRM_Raster_File, OutGeoJSON_File, OutProjection, StrmShp, Stream_ID_Field, Downstream_ID_Field, SEED_Output_File, Thin_Output)
