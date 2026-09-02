if __name__ == "__main__":

    #### STEP 2 Example ####
    # local imports
    from arc import Process_ARC_Geospatial_Data

    # Path to the directory where ARC inputs are stored and where outputs will also be stored.    
    Main_Directory = r"C:\Users\jlgut\OneDrive\Desktop\Montana_Test_Case"
    # Name of the ID field containing the unique identifier for your stream shapefile.
    id_field = "COMID"
    # Name of the field containing the maximum streamflow input into ARC that is within the flow file you generated in Step 1.
    max_flow_field = "base"
    # Name of the field containing the baseflow or bankfull/channel forming discharge
    # input into ARC that is within the flow file you generated in Step 1.
    # Leave this as an empty string only if you plan to use the optional
    # drainage-area bathymetry power-law parameters below instead.
    baseflow_bankfull_field = "max"
    # Path to the flow file you generated in Step 1.
    flow_file_path = r"C:\Users\jlgut\OneDrive\Desktop\Montana_Test_Case\Flow_Files\RFS1_Base_Max.csv"
    # Do you want to use the estimates of bank elevations to estimate bathymetry?
    bathy_use_banks = False
    # Do you want to use land cover to find banks or use the flat surface approach?
    use_land_cover_to_find_banks = True
    # Optional drainage-area bathymetry parameters. When all five are supplied
    # together, ARC can estimate bathymetry without Flow_File_BF.
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
    )
