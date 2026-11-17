if __name__ == "__main__":

    #### STEP 2 Example ####
    # local imports
    from arc import Process_ARC_Geospatial_Data

    # Path to the directory where ARC inputs are stored and where outputs will also be stored.    
    Main_Directory = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase"
    # Name of the ID field containing the unique identifier for your stream shapefile (LINKNO in the GEOGLOWS streams).
    # It is burned into the stream raster and written to the input file as reach_id.
    id_field = "LINKNO"
    # Name of the flow file's column holding the same IDs, written to the input file as Flow_File_ID. Step 1 writes
    # them in a column named COMID, which is the default.
    flow_id_field = "COMID"
    # Name of the field in your stream shapefile holding the ID of the reach downstream of each reach (DSLINKNO in the
    # GEOGLOWS streams). The input file asks for bathymetry, which follows the stream network, so ARC needs this field
    # whenever it has a baseflow field or the drainage-area depth power law below.
    downstream_id_field = "DSLINKNO"
    # Name of the field containing the maximum streamflow input into ARC that is within the flow file you generated in Step 1.
    max_flow_field = "qout_max"
    # Name of the field containing the baseflow or bankfull/channel forming discharge
    # input into ARC that is within the flow file you generated in Step 1.
    # Leave this as an empty string to take the channel depth from the
    # drainage-area depth power law below instead (without either, ARC makes no bathymetry).
    baseflow_bankfull_field = "qout_median"
    # Path to the flow file you generated in Step 1.
    flow_file_path = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\Streamflow_Output\streamflow_for_ARC.csv"
    # Do you want to use the estimates of bank elevations to estimate bathymetry?
    bathy_use_banks = False
    # Do you want to look for the banks first where the land cover's water ends (True), or find them from the DEM
    # alone (False), by its width-to-depth ratio and then its flat water?
    use_land_cover_to_find_banks = True
    # Optional drainage-area bathymetry parameters. With drainage_area_field, the depth law's depth takes the place
    # of the baseflow's wherever it is valid, and the width law sets the width of channels whose banks aren't found.
    drainage_area_field = ""
    coefficient_depth = None
    exponent_depth = None
    coefficient_width = None
    exponent_width = None

    Process_ARC_Geospatial_Data(
        Main_Directory,
        id_field,
        max_flow_field,
        baseflow_bankfull_field,
        flow_file_path,
        bathy_use_banks,
        use_land_cover_to_find_banks,
        drainage_area_field=drainage_area_field,
        coefficient_depth=coefficient_depth,
        exponent_depth=exponent_depth,
        coefficient_width=coefficient_width,
        exponent_width=exponent_width,
        downstream_id_field=downstream_id_field,
        flow_id_field=flow_id_field,
    )
