if __name__ == "__main__":

    #### STEP 1 Example ####
    # local imports
    import streamflow_processing

    # The path to the a NetCDF of GEOGLOWS recurrence intervals for the stream reaches in your domain of interest.
    NetCDF_RecurrenceInterval_File_Path = ""
    # The path to the directory containing NetCDF's of GEOGLOWS retrospective streamflow estimates for the stream reaches in your domain of interest.
    NetCDF_Historical_Folder = "" 
    # The path and filename of the file containing the ARC formatted streamflow data.
    Outfile_file_path = ""

    streamflow_processing.Create_ARC_Streamflow_Input(NetCDF_RecurrenceInterval_File_Path, NetCDF_Historical_Folder, Outfile_file_path)

