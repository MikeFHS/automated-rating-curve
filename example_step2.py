if __name__ == "__main__":

    #### STEP 2 Example ####
    # local imports
    import process_geospatial_data

    # Path to the directory where ARC inputs are stored and where outputs will also be stored.    
    Main_Directory = ""
    # Name of the ID field containing the unique identifier for your stream shapefile.
    id_field = ""
    # Name of the field containing the maximum streamflow input into ARC that is within the flow file you generated in Step 1.
    max_flow_field = ""
    # Name of the field containing the baseflow input into ARC that is within the flow file you generated in Step 1.
    baseflow_field = ""
    # Path to the flow file you generated in Step 1.
    flow_file_path = ""

    process_geospatial_data.Process_AutoRoute_Geospatial_Data()

    # Remember to run STEP 3 via the command line!
