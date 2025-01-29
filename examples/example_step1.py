if __name__ == "__main__":

    #### STEP 1 Example ####
    # local imports
    from arc import Create_ARC_Streamflow_Input

    # The path to the a NetCDF of GEOGLOWS recurrence intervals for the stream reaches in your domain of interest.
    NetCDF_RecurrenceInterval_File_Path = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\714_ReturnPeriods\returnperiods_714.nc"
    # The path to the directory containing NetCDF's of GEOGLOWS retrospective streamflow estimates for the stream reaches in your domain of interest.
    NetCDF_Historical_Folder = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\714_HistoricFlows"
    # The path and filename of the file containing the ARC formatted streamflow data.
    Outfile_file_path = r"C:\Users\jlgut\OneDrive\Desktop\Shields_TestCase\Streamflow_Output\streamflow_for_ARC.csv"

    Create_ARC_Streamflow_Input(NetCDF_RecurrenceInterval_File_Path, NetCDF_Historical_Folder, Outfile_file_path)

