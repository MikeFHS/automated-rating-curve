if __name__ == "__main__":

    #### STEP 2 Example ####
    # local imports
    from arc import Process_ARC_Geospatial_Data

    # Path to the directory where ARC inputs are stored and where outputs will also be stored.    
    Main_Directory = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase"
    # Name of the ID field containing the unique identifier for your stream shapefile.
    id_field = "LINKNO"
    # Name of the field containing the maximum streamflow input into ARC that is within the flow file you generated in Step 1.
    max_flow_field = "qout_max"
    # Name of the field containing the baseflow input into ARC that is within the flow file you generated in Step 1.
    baseflow_field = "qout_median"
    # Path to the flow file you generated in Step 1.
    flow_file_path = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\Streamflow_Output\streamflow_for_ARC.csv"
    # Do you want to use the estimates of bank elevations to estimate bathymetry?
    bathy_use_banks = False

    Process_ARC_Geospatial_Data(Main_Directory, id_field, max_flow_field, baseflow_field, flow_file_path, bathy_use_banks)