if __name__ == "__main__":

    #### Representative Cross-Section Example ####
    # local imports
    import os
    from arc import Arc

    # This example is modeled on the South Africa flood test case inputs in:
    Main_Dir = r"C:\Users\jlgut\OneDrive\Desktop\nencarta_libraries\Test_Cases\Africa_Test_Case_Representative_Cross_Section"

    # Here are the inputs for the representative cross-section example that are in the Main_Dir folder:
    DEM_File = os.path.join(Main_Dir, "FABDEM_projected", "FABDEM.tif")
    Stream_File = os.path.join(Main_Dir, "Results", "Test1", "STRM", "GEOGLOWS_FABDEM_STRM_Raster_Clean.tif")
    reach_id = "LINKNO"
    downstream_reach_id = "DSLINKNO"
    LU_Raster_SameRes = os.path.join(Main_Dir, "Results", "Test1", "LAND", "FABDEM_LAND_Raster.tif")
    LU_Manning_n = os.path.join(Main_Dir, "Results", "Test1", "LAND", "AR_Manning_n_MED.txt")
    drainage_area_field = "DSContArea_km2"
    coefficient_depth = 0.27
    exponent_depth = 0.21
    coefficient_width = 2.44
    exponent_width = 0.34
    X_Section_Dist = 5000.0
    Degree_Manip = 90
    Degree_Interval = 1.5
    Low_Spot_Range = 2
    Str_Limit_Val = 1
    Gen_Dir_Dist = 1
    Gen_Slope_Dist = 10
    Stream_Slope_Method = "local_average_corrected"
    Reach_Average_Curve_File = True
    StrmShp_File = os.path.join(Main_Dir, "Results", "Test1", "STRM", "GEOGLOWS_FABDEM_StrmShp.gpkg")
    Bathy_Trap_H = 0.2
    Bathy_Use_Banks = True
    LAND_WaterValue = 80
    FindBanksBasedOnLandCover = True
    BATHY_Out_File = os.path.join(Main_Dir, "Results", "Test1", "Bathymetry", "GEOGLOWS_FABDEM_ARC_Bathy.tif")
    Build_Representative_Cross_Section = True
    Representative_Cross_Section_File = os.path.join(Main_Dir, "Results", "Test1", "XS", "GEOGLOWS_FABDEM_Representative_Cross_Section.csv")

    # now we can call and run ARC
    Arc(
        args={
            "DEM_File": DEM_File,
            "Stream_File": Stream_File,
            "reach_id": reach_id,
            "downstream_reach_id": downstream_reach_id,
            "LU_Raster_SameRes": LU_Raster_SameRes,
            "LU_Manning_n": LU_Manning_n,
            "drainage_area_field": drainage_area_field,
            "coefficient_depth": coefficient_depth,
            "exponent_depth": exponent_depth,
            "coefficient_width": coefficient_width,
            "exponent_width": exponent_width,
            "X_Section_Dist": X_Section_Dist,
            "Degree_Manip": Degree_Manip,
            "Degree_Interval": Degree_Interval,
            "Low_Spot_Range": Low_Spot_Range,
            "Str_Limit_Val": Str_Limit_Val,
            "Gen_Dir_Dist": Gen_Dir_Dist,
            "Gen_Slope_Dist": Gen_Slope_Dist,
            "Stream_Slope_Method": Stream_Slope_Method,
            "Reach_Average_Curve_File": Reach_Average_Curve_File,
            "StrmShp_File": StrmShp_File,
            "Bathy_Trap_H": Bathy_Trap_H,
            "Bathy_Use_Banks": Bathy_Use_Banks,
            "LAND_WaterValue": LAND_WaterValue,
            "FindBanksBasedOnLandCover": FindBanksBasedOnLandCover,
            "BATHY_Out_File": BATHY_Out_File,
            "Build_Representative_Cross_Section": Build_Representative_Cross_Section,
            "Representative_Cross_Section_File": Representative_Cross_Section_File,
        }
    ).run()
