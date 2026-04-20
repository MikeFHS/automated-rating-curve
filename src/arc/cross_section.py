import math

import numpy as np
from numba import njit
from scipy.signal import savgol_filter

class CrossSection:
    def __init__(self, 
                 dx: float, dy: float,
                 i_precompute_angles: int, d_precompute_angles: float,
                 dm_elevation: np.ndarray, dm_land_use: np.ndarray,
                 d_x_section_distance: float, b_FindBanksBasedOnLandCover: bool, i_lc_water_value: int, d_bathymetry_trapzoid_height: float, b_bathy_use_banks: bool):
        self.d_x_section_distance = d_x_section_distance
        self.i_center_point = int((self.d_x_section_distance / (sum([dx, dy]) * 0.5)) / 2.0) + 1

        self.da_xs_profile1 = np.zeros(self.i_center_point + 1, dtype=np.float64)
        self.da_xs_profile2 = np.zeros(self.i_center_point + 1, dtype=np.float64)
        self.ia_lc_xs1 = np.zeros(self.i_center_point + 1, dtype=np.int64)
        self.ia_lc_xs2 = np.zeros(self.i_center_point + 1, dtype=np.int64)
        self.mannings_n1 = np.zeros(self.i_center_point + 1, dtype=np.float64)
        self.mannings_n2 = np.zeros(self.i_center_point + 1, dtype=np.float64)

        self.dm_elevation = dm_elevation
        self.dm_land_use = dm_land_use

        self.b_FindBanksBasedOnLandCover = b_FindBanksBasedOnLandCover
        self.i_lc_water_value = i_lc_water_value
        self.d_bathymetry_trapzoid_height = d_bathymetry_trapzoid_height
        self.b_bathy_use_banks = b_bathy_use_banks

        self.create_cross_section_ordinates(dx, dy, i_precompute_angles, d_precompute_angles)

    def is_valid(self) -> bool:
        return self.xs1_n > 0 or self.xs2_n > 0

    def create_cross_section_ordinates(self, dx: float, dy: float, i_precompute_angles: int, d_precompute_angles: float):
        i_center_point = self.i_center_point
        self.ia_xc_dr_index_main = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=np.int64)  # Only need to go to center point, because the other side of xs we can just use *-1
        self.ia_xc_dc_index_main = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=np.int64)  # Only need to go to center point, because the other side of xs we can just use *-1
        self.ia_xc_dr_index_second = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=np.int64)  # Only need to go to center point, because the other side of xs we can just use *-1
        self.ia_xc_dc_index_second = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=np.int64)  # Only need to go to center point, because the other side of xs we can just use *-1
        self.d_distance_z = np.zeros(i_precompute_angles + 1, dtype=np.float64)
        self.da_xc_main_fract = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=np.float64)
        self.da_xc_second_fract = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=np.float64)

        for i in range(i_precompute_angles+1):
            d_xs_direction = d_precompute_angles * i
            # Get the Cross-Section Ordinates
            self.d_distance_z[i]= get_xs_index_values_precalculated(self.ia_xc_dr_index_main[i], self.ia_xc_dc_index_main[i], self.ia_xc_dr_index_second[i], self.ia_xc_dc_index_second[i], self.da_xc_main_fract[i], self.da_xc_second_fract[i], d_xs_direction,
                                                                                           i_center_point, dx, dy)

    def set_boundary_extents(self, i_boundary_number: int, nrows: int, ncols: int):
        """
        Get the max and min row and col that we can go for later search functions (based on max of slope and direction distance parameters.)
        """
        self.i_boundary_number = i_boundary_number
        self.i_row_bottom = i_boundary_number
        self.i_row_top = nrows + i_boundary_number - 1
        self.i_column_bottom = i_boundary_number
        self.i_column_top = ncols + i_boundary_number - 1
    
    def set_cross_section(self, row: int, col: int, i_precompute_angle_closest: int, d_xs_direction: float):
        """
        Parameters
        """
        self.row = row
        self.col = col
        self.i_precompute_angle_closest = i_precompute_angle_closest
        self.d_xs_direction = d_xs_direction
        self.xs1_n = 0
        self.xs2_n = 0
        self.da_xs_profile1[:] = 0.0
        self.da_xs_profile2[:] = 0.0
        self.d_ordinate_dist = self.d_distance_z[self.i_precompute_angle_closest] # space between ordinates in the cross-section

        self.ia_xc_row1_index_main = self.row + self.ia_xc_dr_index_main[self.i_precompute_angle_closest]
        self.ia_xc_row2_index_main = self.row - self.ia_xc_dr_index_main[self.i_precompute_angle_closest]
        self.ia_xc_column1_index_main = self.col + self.ia_xc_dc_index_main[self.i_precompute_angle_closest]
        self.ia_xc_column2_index_main = self.col - self.ia_xc_dc_index_main[self.i_precompute_angle_closest]
        
        self.ia_xc_row1_index_second = self.row + self.ia_xc_dr_index_second[self.i_precompute_angle_closest]
        self.ia_xc_row2_index_second = self.row - self.ia_xc_dr_index_second[self.i_precompute_angle_closest]
        self.ia_xc_column1_index_second = self.col + self.ia_xc_dc_index_second[self.i_precompute_angle_closest]
        self.ia_xc_column2_index_second = self.col - self.ia_xc_dc_index_second[self.i_precompute_angle_closest]

        self.xs1_n = _sample_side(
            self.da_xs_profile1,
            self.ia_lc_xs1,
            self.ia_xc_row1_index_main,
            self.ia_xc_column1_index_main,
            self.ia_xc_row1_index_second,
            self.ia_xc_column1_index_second,
            self.da_xc_main_fract[self.i_precompute_angle_closest],
            self.da_xc_second_fract[self.i_precompute_angle_closest],
            self.i_center_point,
            self.i_row_bottom,
            self.i_row_top,
            self.i_column_bottom,
            self.i_column_top,
            self.dm_elevation,
            self.dm_land_use
        )

        self.xs2_n = _sample_side(
            self.da_xs_profile2,
            self.ia_lc_xs2,
            self.ia_xc_row2_index_main,
            self.ia_xc_column2_index_main,
            self.ia_xc_row2_index_second,
            self.ia_xc_column2_index_second,
            self.da_xc_main_fract[self.i_precompute_angle_closest],
            self.da_xc_second_fract[self.i_precompute_angle_closest],
            self.i_center_point,
            self.i_row_bottom,
            self.i_row_top,
            self.i_column_bottom,
            self.i_column_top,
            self.dm_elevation,
            self.dm_land_use
        )
        
    
    
    def adjust_cross_section_to_lowest_point(self, i_low_spot_range: int):
        self.row, self.col = _adjust_cross_section_to_lowest_point(
            i_low_spot_range,
            self.da_xs_profile1,
            self.da_xs_profile2,
            self.ia_xc_row1_index_main,
            self.ia_xc_row2_index_main,
            self.ia_xc_column1_index_main,
            self.ia_xc_column2_index_main,
            self.i_center_point,
            self.xs1_n,
            self.xs2_n
        )

        # re-sample the cross-section to make sure all of the low-spot data has the same values through interpolation
        self.set_cross_section(self.row, self.col, self.i_precompute_angle_closest, self.d_xs_direction)

    def get_row_col(self):
        return self.ia_xc_row1_index_main[0], self.ia_xc_column1_index_main[0]
    
    def get_thalweg(self):
        return self.da_xs_profile1[0]
    
    def get_best_xsection_angle(self, d_precompute_angles: float, l_angles_to_test: list[float]):
        d_test_depth = 0.5
        d_shortest_tw_angle = 0.0
        d_t_test = np.inf

        # Loop through the angles to test
        for d_entry_angle_adjustment in l_angles_to_test:
            # Ensure angle is between 0 and pi
            d_xs_angle_use = (self.d_xs_direction + d_entry_angle_adjustment) % np.pi
        
            #We now precompute the cross-section ordinates
            i_precompute_angle_closest = int(round(d_xs_angle_use / d_precompute_angles))

            # Pull the cross-section again
            self.set_cross_section(self.row, self.col, i_precompute_angle_closest, self.d_xs_direction)
            d_wse = self.get_thalweg() + d_test_depth
            top_width = self.calculate_top_width_of_wse(d_wse)

            if top_width < d_t_test:
                d_t_test = top_width
                d_shortest_tw_angle = d_xs_angle_use

        return d_shortest_tw_angle
    
    
    def calculate_top_width_of_wse(self, d_wse: float):
        return (
            _calculate_side_top_width(d_wse, self.da_xs_profile1, self.xs1_n, self.d_ordinate_dist) +
            _calculate_side_top_width(d_wse, self.da_xs_profile2, self.xs2_n, self.d_ordinate_dist)
        )

    def _find_wse_and_banks_by_lc(self):
        #Initially set the bank info to zeros
        i_bank_1_index = 0
        i_bank_2_index = 0
        
        bank_elev_1 = self.da_xs_profile1[0]
        bank_elev_2 = self.da_xs_profile2[0]
        for i in range(1, self.xs1_n):
            if self.ia_lc_xs1[i] == self.i_lc_water_value:
                if self.da_xs_profile1[i] < bank_elev_1:
                    bank_elev_1 = self.da_xs_profile1[i]
            else:
                i_bank_1_index = i
                break

        for i in range(1, self.xs2_n):
            if self.ia_lc_xs2[i] == self.i_lc_water_value:
                if self.da_xs_profile2[i] < bank_elev_2:
                    bank_elev_2 = self.da_xs_profile2[i]
            else:
                i_bank_2_index = i
                break
        
        if bank_elev_1>self.da_xs_profile1[0]:
            if bank_elev_2>self.da_xs_profile1[0]:
                d_wse_from_dem = min(bank_elev_1, bank_elev_2)
            else:
                d_wse_from_dem = bank_elev_1
        elif bank_elev_2>self.da_xs_profile1[0]:
            d_wse_from_dem = bank_elev_2
        else:
            d_wse_from_dem = self.get_thalweg() + 0.1
        
        return d_wse_from_dem, i_bank_1_index, i_bank_2_index
    
    def _find_bank(self, profile: np.ndarray, i_cross_section_number: int, wse: bool = False):
        """
        Finds the cell containing the bank of the cross section. Subtract 1 to get WSE elevation

        Parameters
        ----------
        da_xs_profile: ndarray
            Elevations of the stream cross section
        i_cross_section_number: int
            Index of the cross section cell
        d_z_target: float
            Target elevation that defines the bank
        elevation_wanter: str
            Determines if the elevation is the bank elevation or the water surface elevation                


        Returns
        -------
        i_cross_section_number: int
            Updated cell index that defines the bank

        """

        # Loop on the cells of the cross section
        for entry in range(1, i_cross_section_number):
            # Check if the profile elevation matches the target elevation
            if profile[entry] >= self.get_thalweg() + 0.1:
                return entry - 1 if wse else entry

        # Return to the calling function
        return i_cross_section_number
    
    def _find_bank_using_width_to_depth_ratio(self):
        """
        da_xs_profile1: ndarray
            Elevations of the stream cross section on one side
        da_xs_profile2: ndarray
            Elevations of the stream cross section on the other side
        xs1_n: int
            Index of the cross section cells on one of the cross section
        xs2_n: int
            Index of the cross section cells on the other side of the cross section
        d_distance_z: float
            Incremental distance per cell parallel to the orientation of the cross section

        """

        # We don't use mannings n in this func, so these are just dummys (they are generated really quickly)
        d_bottom_elevation = self.get_thalweg()
        d_depth = 0
        d_new_width_to_depth_ratio = 0
        d_width_to_depth_ratio = np.inf  # Start with a large value

        prev_t1 = 0.
        prev_t2 = 0.

        # we will assume that if we get to a depth of 25 meters, something has gone wrong
        while d_new_width_to_depth_ratio <= d_width_to_depth_ratio and d_depth <= 25:
            d_depth += 0.01
            d_wse = d_bottom_elevation + d_depth
            
            # Calculate stream geometry for both sides
            # T1 = calculate_top_width(da_xs_profile1_sliced, d_wse, d_distance_z)
            # T2 = calculate_top_width(da_xs_profile2_sliced, d_wse, d_distance_z)
            
            # TW = T1 + T2
            T1 = _calculate_side_top_width(d_wse, self.da_xs_profile1, self.xs1_n, self.d_ordinate_dist)
            T2 = _calculate_side_top_width(d_wse, self.da_xs_profile2, self.xs2_n, self.d_ordinate_dist)
            TW = T1 + T2
            d_new_width_to_depth_ratio = TW / d_depth

            if d_new_width_to_depth_ratio > d_width_to_depth_ratio:
                # Recalculate the last valid depth
                d_depth -= 0.01
                T1 = prev_t1
                T2 = prev_t2            
                break

            d_width_to_depth_ratio = d_new_width_to_depth_ratio
            prev_t1 = T1
            prev_t2 = T2

        if d_depth < 25:
            i_bank_1_index = int(T1 / self.d_ordinate_dist)
            i_bank_2_index = int(T2 / self.d_ordinate_dist)
        # if we have made it to 25 on d_depth, something is wrong and the banks will be set at the stream cell
        elif d_depth >= 25:
            i_bank_1_index = 0
            i_bank_2_index = 0

        return (i_bank_1_index, i_bank_2_index)
    
    def _find_bank_inflection_point(self, da_xs_profile: np.ndarray, i_cross_section_number: int, window_length: int = 11, polyorder: int = 3):
        """
        Finds the cell containing the bank of the cross section, with smoothing applied.

        Parameters
        ----------
        da_xs_profile: ndarray
            Elevations of the stream cross section
        i_cross_section_number: int
            Index of the cross section cell
        d_distance_z: float
            Incremental distance per cell parallel to the orientation of the cross section
        window_length: int, optional
            The length of the filter window for smoothing (must be an odd number, default is 11)
        polyorder: int, optional
            The order of the polynomial used to fit the samples for smoothing (default is 3)

        Returns
        -------
        i_cross_section_number: int
            Updated cell index that defines the bank
        """
        # Apply smoothing to the cross-section data
        # da_xs_smooth = da_xs_profile
        # If our window is bigger than the number of wet cells, than we need to adjust the window size and polyorder.
        # Otherwise, the smoothing will go wild because of 9999 next to 0, not erroring but producing a bad result.
        window_length = min(window_length, i_cross_section_number)
        polyorder = min(polyorder, window_length - 1)
        try:
            da_xs_smooth = savgol_filter(da_xs_profile[:i_cross_section_number], window_length=window_length, polyorder=polyorder)
        except np.linalg.LinAlgError:
            # If the rare case smoothing fails, just use original profile
            da_xs_smooth = da_xs_profile
            
        return self._find_bank_inflection_point_helper(da_xs_smooth, i_cross_section_number)

    def _find_bank_inflection_point_helper(self, da_xs_smooth: np.ndarray, i_cross_section_number: int) -> int:
        # Loop on the smoothed cross-section cells
        entry = 0
        previous_delta_elevation = 0.0
        total_width = 0.0
        while entry < min(i_cross_section_number, len(da_xs_smooth) - 1):
            elevation_0 = da_xs_smooth[entry]
            elevation_1 = da_xs_smooth[entry + 1]

            current_delta_elevation = elevation_1 - elevation_0

            if current_delta_elevation >= previous_delta_elevation:
                previous_delta_elevation = current_delta_elevation
                total_width += self.d_ordinate_dist
                entry += 1  # move forward
            else:
                # Found the bank – go back one if needed
                return entry  # or return entry - 1 if you want the previous one

        # Return to the calling function
        return 0
    
    
    
    def Calculate_Bathymetry_Based_on_WSE_or_LC(self, d_q_baseflow: float, d_slope_use: float, output_bathymetry: np.ndarray):
        """
        Calculate bathymetry based on water surface elevations.
        """


        # set the function used to none before we start running things
        function_used = None
        
        # First find the bank information
        if self.b_FindBanksBasedOnLandCover:   
            (d_wse_from_dem, i_bank_1_index, i_bank_2_index) = self._find_wse_and_banks_by_lc()
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells > 1:
                function_used = "find_wse_and_banks_by_lc"
        else:
            i_bank_1_index = self._find_bank(self.da_xs_profile1, self.xs1_n, wse=True)
            i_bank_2_index = self._find_bank(self.da_xs_profile2, self.xs2_n, wse=True)
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells > 1:
                function_used = "find_wse_and_banks_by_flat_water"

        if i_total_bank_cells <= 1:
            (i_bank_1_index, i_bank_2_index) = self._find_bank_using_width_to_depth_ratio()
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells > 1:
                function_used = "find_bank_using_width_to_depth_ratio"

        if i_total_bank_cells <= 1:
            i_bank_1_index = self._find_bank_inflection_point(self.da_xs_profile1, self.xs1_n)
            i_bank_2_index = self._find_bank_inflection_point(self.da_xs_profile2, self.xs2_n)
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells > 1:
                function_used = "find_bank_inflection_point"

        if i_total_bank_cells < 1:
            i_total_bank_cells = 1

        #Trapezoid Shape
        #      d_total_bank_dist 
        #   -----------------------
        #    -                   -
        #     -                 -
        #      -               -
        #       ---------------
        #         d_trap_base
        #  |    | <-d_h_dist->|    |
        #                     |    |<--d_h_dist = d_bathymetry_trapzoid_height * d_total_bank_dist
        # d_bathymetry_trapzoid_height is the fraction of d_total_bank_dist that is for the sloped part (see Follum et al., 2023).
        #        Basically, it assumes ~40% of the total top-width of the trapezoid is part of the sloping part
        #        Typically, d_bathymetry_trapzoid_height is set to 0.2
        
        d_total_bank_dist = i_total_bank_cells * self.d_ordinate_dist
        d_h_dist = self.d_bathymetry_trapzoid_height * d_total_bank_dist
        d_trap_base = d_total_bank_dist - 2.0 * d_h_dist

        d_y_bathy = 0.0  # Initialize d_y_bathy to avoid UnboundLocalError

        if d_q_baseflow > 0.0 and function_used != None:
            d_y_depth = find_depth_of_bathymetry(d_q_baseflow, d_trap_base, d_total_bank_dist, d_slope_use, 0.03)
            if d_y_depth >= 25:
                if i_total_bank_cells <= 1:
                    (i_bank_1_index, i_bank_2_index) = self._find_bank_using_width_to_depth_ratio()
                    i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
                    d_total_bank_dist = i_total_bank_cells * self.d_ordinate_dist
                    d_h_dist = self.d_bathymetry_trapzoid_height * d_total_bank_dist
                    d_trap_base = d_total_bank_dist - 2.0 * d_h_dist
                    d_y_depth = find_depth_of_bathymetry(d_q_baseflow, d_trap_base, d_total_bank_dist, d_slope_use, 0.03)
                    function_used = "find_bank_using_width_to_depth_ratio"

                if d_y_depth >= 25 and function_used == "find_bank_using_width_to_depth_ratio":
                    i_bank_1_index = self._find_bank_inflection_point(self.da_xs_profile1, self.xs1_n)
                    i_bank_2_index = self._find_bank_inflection_point(self.da_xs_profile2, self.xs2_n)
                    i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
                    d_total_bank_dist = i_total_bank_cells * self.d_ordinate_dist
                    d_h_dist = self.d_bathymetry_trapzoid_height * d_total_bank_dist
                    d_trap_base = d_total_bank_dist - 2.0 * d_h_dist
                    d_y_depth = find_depth_of_bathymetry(d_q_baseflow, d_trap_base, d_total_bank_dist, d_slope_use, 0.03)
                    function_used = "find_bank_inflection_point"

                if d_y_depth >= 25:
                    d_y_depth = 0.0
                    d_y_bathy = self.get_thalweg() - d_y_depth
                    i_bank_1_index = 0
                    i_bank_2_index = 0
                    i_total_bank_cells = 1
            if i_total_bank_cells > 1:
                d_y_bathy = self.get_thalweg() - d_y_depth
                _adjust_one_side_for_bathymetry(i_bank_1_index, d_total_bank_dist, d_trap_base, d_h_dist, self.ia_xc_row1_index_main, self.ia_xc_column1_index_main, self.da_xs_profile1, output_bathymetry, 0.0, d_y_bathy, d_y_depth, self.d_ordinate_dist, self.dm_elevation, self.b_bathy_use_banks)
                _adjust_one_side_for_bathymetry(i_bank_2_index, d_total_bank_dist, d_trap_base, d_h_dist, self.ia_xc_row2_index_main, self.ia_xc_column2_index_main, self.da_xs_profile2, output_bathymetry, 0.0, d_y_bathy, d_y_depth, self.d_ordinate_dist, self.dm_elevation, self.b_bathy_use_banks)

        else:
            d_y_depth = 0.0

        return i_bank_1_index, i_bank_2_index, i_total_bank_cells, d_y_depth, d_y_bathy
    
    def set_mannings_n_values(self, dm_manning_n_raster: np.ndarray):
        self.mannings_n1 = dm_manning_n_raster[self.ia_xc_row1_index_main[:self.xs1_n], self.ia_xc_column1_index_main[:self.xs1_n]]
        self.mannings_n2 = dm_manning_n_raster[self.ia_xc_row2_index_main[:self.xs2_n], self.ia_xc_column2_index_main[:self.xs2_n]]
    
    def calculate_stream_geometry_side_1(self, wse: float):
        return _calculate_stream_geometry(self.da_xs_profile1, wse, self.xs1_n, self.d_ordinate_dist, self.mannings_n1)
    
    def calculate_stream_geometry_side_2(self, wse: float):
        return _calculate_stream_geometry(self.da_xs_profile2, wse, self.xs2_n, self.d_ordinate_dist, self.mannings_n2)
    
    def calculate_stream_geometry_and_topwidth_side_1(self, wse: float):
        return _calculate_stream_geometry_and_topwidth(self.da_xs_profile1, wse, self.xs1_n, self.d_ordinate_dist, self.mannings_n1)
    
    def calculate_stream_geometry_and_topwidth_side_2(self, wse: float):
        return _calculate_stream_geometry_and_topwidth(self.da_xs_profile2, wse, self.xs2_n, self.d_ordinate_dist, self.mannings_n2)
    
    def _calc_side_distance(self, profile, bank_index, bankfull_elev):
            """Compute the horizontal distance along a side based on elevation difference."""
            try:
                d_d_elev = profile[bank_index + 1] - profile[bank_index]
                if d_d_elev > 0:
                    side_dist = self.d_ordinate_dist * (bankfull_elev - profile[bank_index]) / d_d_elev
                    if side_dist < 0.0 or side_dist > self.d_ordinate_dist:
                        return 0.5 * self.d_ordinate_dist
                    return side_dist
                else:
                    return 0.0
            except Exception:
                return 0.5 * self.d_ordinate_dist
            
    def get_calculate_discharge_from_wse_args(self):
        return self.da_xs_profile1, self.xs1_n, self.mannings_n1, self.da_xs_profile2, self.xs2_n, self.mannings_n2, self.d_ordinate_dist
    
    def _compute_depth(self, i_total_bank_cells, i_bank_1_index, i_bank_2_index, d_bankfull_elevation, d_q_baseflow, d_slope_use):
        
        """Compute trapezoid dimensions and the corresponding water depth."""
        d_side1_dist = self._calc_side_distance(self.da_xs_profile1, i_bank_1_index, d_bankfull_elevation)
        d_side2_dist = self._calc_side_distance(self.da_xs_profile2, i_bank_2_index, d_bankfull_elevation)
        d_total_bank_dist = i_total_bank_cells * self.d_ordinate_dist + d_side1_dist + d_side2_dist
        d_h_dist = self.d_bathymetry_trapzoid_height * d_total_bank_dist
        d_trap_base = d_total_bank_dist - 2.0 * d_h_dist
        d_y_depth = find_depth_of_bathymetry(d_q_baseflow, d_trap_base, d_total_bank_dist, d_slope_use, 0.03)
        return d_side1_dist, d_side2_dist, d_total_bank_dist, d_h_dist, d_trap_base, d_y_depth
    
    def Calculate_Bathymetry_Based_on_RiverBank_Elevations(self, d_q_baseflow: float, d_slope_use: float, dm_output_bathymetry: np.ndarray):
        """
        Calculate the bathymetry (water depth and thalweg elevation) based on river bank elevations.
        """
        # Initialize variables
        function_used = None
        i_landcover_for_bathy = self.ia_lc_xs1[0]
        
        # Initially set the bank info to zeros and bank elevations to the current water surface elevation
        i_bank_1_index = 0
        i_bank_2_index = 0
        bank_elev_1 = self.da_xs_profile1[0]
        bank_elev_2 = self.da_xs_profile2[0]
        d_y_depth = 0.0

        # === First: find the bank information === #
        if self.b_FindBanksBasedOnLandCover:
            # Use land cover data to find the banks of the stream
            if self.xs1_n >= 1 and i_landcover_for_bathy == self.i_lc_water_value:
                bank_elev_1 = self.da_xs_profile1[0]
                for i in range(1, self.xs1_n):
                    if self.ia_lc_xs1[i] != self.i_lc_water_value:
                        bank_elev_1 = self.da_xs_profile1[i]
                        i_bank_1_index = i - 1
                        break
            if self.xs2_n >= 1 and i_landcover_for_bathy == self.i_lc_water_value:
                bank_elev_2 = self.da_xs_profile2[0]
                for i in range(1, self.xs2_n):
                    if self.ia_lc_xs2[i] != self.i_lc_water_value:
                        bank_elev_2 = self.da_xs_profile2[i]
                        i_bank_2_index = i - 1
                        break
            i_total_bank_cells = i_bank_1_index + i_bank_2_index  - 1
            if i_total_bank_cells <= 1:
                i_total_bank_cells = 1
            else:
                function_used = "find_wse_and_banks_by_lc"
        else:
            #Default is to determine bank locations based on the flat water within the DEM
            i_bank_1_index = self._find_bank(self.da_xs_profile1, self.xs1_n)
            i_bank_2_index = self._find_bank(self.da_xs_profile2, self.xs2_n)
            # set the bank elevations
            bank_elev_1 = self.da_xs_profile1[i_bank_1_index]
            bank_elev_2 = self.da_xs_profile2[i_bank_2_index]
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells > 1:
                function_used = "find_wse_and_banks_by_flat_water"

        # Try the width-to-depth ratio method if the banks are not found
        if i_total_bank_cells <= 1:
            (i_bank_1_index, i_bank_2_index) = self._find_bank_using_width_to_depth_ratio()
            bank_elev_1 = self.da_xs_profile1[i_bank_1_index]
            bank_elev_2 = self.da_xs_profile2[i_bank_2_index]
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells <= 1:
                i_total_bank_cells = 1
            else:
                function_used = "find_bank_using_width_to_depth_ratio"

        # If still not found, try the inflection point method
        if i_total_bank_cells <= 1:
            i_bank_1_index = self._find_bank_inflection_point(self.da_xs_profile1, self.xs1_n)
            bank_elev_1 = self.da_xs_profile1[i_bank_1_index]
            i_bank_2_index = self._find_bank_inflection_point(self.da_xs_profile2, self.xs2_n)
            bank_elev_2 = self.da_xs_profile2[i_bank_2_index]
            i_total_bank_cells = i_bank_1_index + i_bank_2_index
            if i_total_bank_cells <= 1:
                i_total_bank_cells = 1
            else:
                function_used = "find_bank_inflection_point"

        # Calculate bankfull elevation using the base water surface elevation (first point of profile1)
        base_elev = self.da_xs_profile1[0]
        d_bankfull_elevation = calc_bankfull_elevation(base_elev, bank_elev_1, bank_elev_2)

        # === Estimate bathymetry depth === #
        if d_q_baseflow > 0.0 and function_used is not None:
            # Calculate trapezoid dimensions and initial depth estimate
            (d_side1_dist, d_side2_dist, d_total_bank_dist, d_h_dist,
            d_trap_base, d_y_depth) = self._compute_depth(
                                                    i_total_bank_cells, 
                                                    i_bank_1_index, i_bank_2_index, d_bankfull_elevation,
                                                    d_q_baseflow, d_slope_use
                                                    )
            # calculate the elevation of the bathy depth and re-calculate if higher than the bankfull elevation
            d_y_bathy = d_bankfull_elevation - d_y_depth
            # If the estimated depth is an outlier, try alternate approaches
            if d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation and (function_used == "find_wse_and_banks_by_lc" or
                                    function_used == "find_wse_and_banks_by_flat_water"):
                # Recalculate using width-to-depth ratio
                (i_bank_1_index, i_bank_2_index) = self._find_bank_using_width_to_depth_ratio()
                i_total_bank_cells = i_bank_1_index + i_bank_2_index -1
                if i_total_bank_cells <= 1:
                    i_total_bank_cells = 1
                else:
                    function_used = "find_bank_using_width_to_depth_ratio"
                
                # find the elevation of the banks
                bank_elev_1 = self.da_xs_profile1[i_bank_1_index]
                bank_elev_2 = self.da_xs_profile2[i_bank_2_index]
                d_bankfull_elevation = calc_bankfull_elevation(base_elev, bank_elev_1, bank_elev_2)
                (d_side1_dist, d_side2_dist, d_total_bank_dist, d_h_dist,
                d_trap_base, d_y_depth) = self._compute_depth(
                    i_total_bank_cells,
                    i_bank_1_index, i_bank_2_index, d_bankfull_elevation,
                    d_q_baseflow, d_slope_use
                )
                # calculate the elevation of the bathy depth and re-calculate if higher than the bankfull elevation
                d_y_bathy = d_bankfull_elevation - d_y_depth
                if d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation:
                    # Try using the inflection point method
                    i_bank_1_index = self._find_bank_inflection_point(self.da_xs_profile1, self.xs1_n)
                    bank_elev_1 = self.da_xs_profile1[i_bank_1_index]
                    i_bank_2_index = self._find_bank_inflection_point(self.da_xs_profile2, self.xs2_n)
                    bank_elev_2 = self.da_xs_profile2[i_bank_2_index]
                    i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
                    if i_total_bank_cells <= 1:
                        i_total_bank_cells = 1
                    else:
                        function_used = "find_bank_inflection_point"
                    d_bankfull_elevation = calc_bankfull_elevation(base_elev, bank_elev_1, bank_elev_2)
                    (d_side1_dist, d_side2_dist, d_total_bank_dist, d_h_dist,
                    d_trap_base, d_y_depth) = self._compute_depth(
                        i_total_bank_cells,
                        i_bank_1_index, i_bank_2_index, d_bankfull_elevation,
                        d_q_baseflow, d_slope_use
                    )
                    # calculate the elevation of the bathy depth and re-calculate if higher than the bankfull elevation
                    d_y_bathy = d_bankfull_elevation - d_y_depth
                    if d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation or i_total_bank_cells <= 1:
                        d_y_depth = 0
                        d_y_bathy = self.da_xs_profile1[0]
                        i_bank_1_index = 0
                        i_bank_2_index = 0
                        i_total_bank_cells = 1

            elif d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation and function_used == "find_bank_using_width_to_depth_ratio":
                # Use the inflection point method directly
                i_bank_1_index = self._find_bank_inflection_point(self.da_xs_profile1, self.xs1_n)
                bank_elev_1 = self.da_xs_profile1[i_bank_1_index]
                i_bank_2_index = self._find_bank_inflection_point(self.da_xs_profile2, self.xs2_n)
                bank_elev_2 = self.da_xs_profile2[i_bank_2_index]
                i_total_bank_cells = i_bank_1_index + i_bank_2_index -1
                if i_total_bank_cells <= 1:
                    i_total_bank_cells = 1
                else:
                    function_used = "find_bank_inflection_point"
                d_bankfull_elevation = calc_bankfull_elevation(base_elev, bank_elev_1, bank_elev_2)
                (d_side1_dist, d_side2_dist, d_total_bank_dist, d_h_dist,
                d_trap_base, d_y_depth) = self._compute_depth(
                    i_total_bank_cells,
                    i_bank_1_index, i_bank_2_index, d_bankfull_elevation,
                    d_q_baseflow, d_slope_use
                )
                # calculate the elevation of the bathy depth and re-calculate if higher than the bankfull elevation
                d_y_bathy = d_bankfull_elevation - d_y_depth
                if d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation or i_total_bank_cells <= 1:
                    d_y_depth = 0
                    d_y_bathy = self.da_xs_profile1[0]
                    i_bank_1_index = 0
                    i_bank_2_index = 0
                    i_total_bank_cells = 1

            elif d_y_depth >= 25 or d_y_bathy > d_bankfull_elevation and function_used == "find_bank_inflection_point":
                d_y_depth = 0
                d_y_bathy = self.da_xs_profile1[0]
                i_bank_1_index = 0
                i_bank_2_index = 0
                i_total_bank_cells = 1
                function_used = None

        else:
            # No valid baseflow or method; set defaults.
            d_y_depth = 0.0
            d_y_bathy = self.da_xs_profile1[0]
            i_bank_1_index = 0
            i_bank_2_index = 0
            i_total_bank_cells = 1
        
        # if function_used == "find_wse_and_banks_by_flat_water":
        #     i_bank_1_index = 0
        #     i_bank_2_index = 0
        #     i_total_bank_cells = 0
        #     d_y_depth = 0
        #     d_y_bathy = 0


        # --- Adjust bathymetry on both profiles if valid banks were found --- #
        if i_total_bank_cells > 1:
            # Add 1 to the bank index to get to the actual bank cell
            _adjust_one_side_for_bathymetry(
                i_bank_1_index + 1, d_total_bank_dist,
                d_trap_base, d_h_dist, self.ia_xc_row1_index_main, self.ia_xc_column1_index_main,
                self.da_xs_profile1, dm_output_bathymetry, d_side1_dist, d_y_bathy, d_y_depth, self.d_ordinate_dist, self.dm_elevation, self.b_bathy_use_banks
            )
            _adjust_one_side_for_bathymetry(
                i_bank_2_index + 1, d_total_bank_dist,
                d_trap_base, d_h_dist, self.ia_xc_row2_index_main, self.ia_xc_column2_index_main,
                self.da_xs_profile2, dm_output_bathymetry, d_side2_dist, d_y_bathy, d_y_depth, self.d_ordinate_dist, self.dm_elevation, self.b_bathy_use_banks
            )
            # adjust_profile_for_bathymetry(
            #     da_xs_profile1, i_bank_1_index + 1, d_total_bank_dist,
            #     d_trap_base, d_distance_z, d_h_dist, d_y_bathy, d_y_depth,
            #     dm_output_bathymetry, ia_xc_r1_index_main, ia_xc_c1_index_main,
            #     d_side1_dist, dm_elevation,
            #     b_bathy_use_banks
            # )
            # adjust_profile_for_bathymetry(
            #     da_xs_profile2, i_bank_2_index + 1, d_total_bank_dist,
            #     d_trap_base, d_distance_z, d_h_dist, d_y_bathy, d_y_depth,
            #     dm_output_bathymetry, ia_xc_r2_index_main, ia_xc_c2_index_main,
            #     d_side2_dist, dm_elevation,
            #     b_bathy_use_banks
            # )

        return i_bank_1_index, i_bank_2_index, i_total_bank_cells, d_y_depth, d_y_bathy

@njit(cache=True)
def _adjust_cross_section_to_lowest_point(i_low_spot_range: int,
                                          da_xs_profile1: np.ndarray,
                                          da_xs_profile2: np.ndarray,
                                          ia_xc_row1_index_main: np.ndarray,
                                          ia_xc_row2_index_main: np.ndarray,
                                          ia_xc_column1_index_main: np.ndarray,
                                          ia_xc_column2_index_main: np.ndarray,
                                          i_center_point: int,
                                          xs1_n: int,
                                          xs2_n: int
                                          ):
    """
    Reorients the cross section through the lowest point of the stream. Cross-section needs to be re-sampled if the low spot in the cross-section changes location.

    Parameters
    ----------
    i_low_point_index: int
        Offset index along the cross section of the lowest point
    d_dem_low_point_elev: float
        Elevation of the lowest point
    da_xs_profile_one: ndarray
        Cross section elevations of the first cross section
    da_xs_profile_two: ndarray
        Cross section elevations of the second cross section
    ia_xc_r1_index_main: ndarray
        Row indices of the first cross section
    ia_xc_r2_index_main: ndarray
        Row indices of the second cross section
    ia_xc_c1_index_main: ndarray
        Column indices of the first cross section
    ia_xc_c2_index_main: ndarray
        Column indicies of the second cross section
    da_xs1_mannings: ndarray
        Manning's roughness of the first cross section
    da_xs2_mannings: ndarray
        Manning's roughness of the second cross section
    i_center_point: int
        Center point index
    i_low_spot_range: int
        The number of cells on each side of the cross-section we're looking at moving to. 
    """
    d_dem_low_point_elev = da_xs_profile1[0]
    i_low_point_index = 0

    # Loop on the search range for the low point
    for i_entry in range(i_low_spot_range):
        if i_entry >= da_xs_profile1.shape[0] or i_entry >= da_xs_profile2.shape[0]:
            break
        # Look in the first profile
        if da_xs_profile1[i_entry] > 0.0 and da_xs_profile1[i_entry] < d_dem_low_point_elev:
            # New low point was found. Update the index.
            d_dem_low_point_elev = da_xs_profile1[i_entry]
            i_low_point_index = i_entry

        # Look in the second profile
        if da_xs_profile2[i_entry] > 0.0 and da_xs_profile2[i_entry] < d_dem_low_point_elev:
            # New low point was found. Update the index.
            d_dem_low_point_elev = da_xs_profile2[i_entry]
            i_low_point_index = i_entry * -1

    # Process based on if the low point is in the first or second profile
    if i_low_point_index > 0:
        # Low point is in the first profile. Update the cross section and mannings.
        da_xs_profile2[i_low_point_index:i_center_point] = da_xs_profile2[0:i_center_point - i_low_point_index]
        da_xs_profile2[0:i_low_point_index + 1] = np.flip(da_xs_profile1[0:i_low_point_index + 1])
        da_xs_profile1[0:i_center_point - i_low_point_index] = da_xs_profile1[i_low_point_index:i_center_point]
        da_xs_profile1[xs1_n - i_low_point_index] = 99999.9

        # Update the row indices
        ia_xc_row2_index_main[i_low_point_index:i_center_point] = ia_xc_row2_index_main[0:i_center_point - i_low_point_index]
        ia_xc_row2_index_main[0:i_low_point_index + 1] = np.flip(ia_xc_row1_index_main[0:i_low_point_index + 1])
        ia_xc_row1_index_main[0:i_center_point - i_low_point_index] = ia_xc_row1_index_main[i_low_point_index:i_center_point]

        # Update the column indices
        ia_xc_column2_index_main[i_low_point_index:i_center_point] = ia_xc_column2_index_main[0:i_center_point - i_low_point_index]
        ia_xc_column2_index_main[0:i_low_point_index + 1] = np.flip(ia_xc_column1_index_main[0:i_low_point_index + 1])
        ia_xc_column1_index_main[0:i_center_point - i_low_point_index] = ia_xc_column1_index_main[i_low_point_index:i_center_point]

    elif i_low_point_index < 0:
        # Low point is in the second profile Update the cross section and mannings.
        i_low_point_index = i_low_point_index * -1
        da_xs_profile1[i_low_point_index:i_center_point] = da_xs_profile1[0:i_center_point - i_low_point_index]
        da_xs_profile1[0:i_low_point_index + 1] = np.flip(da_xs_profile2[0:i_low_point_index + 1])
        da_xs_profile2[0:i_center_point - i_low_point_index] = da_xs_profile2[i_low_point_index:i_center_point]
        da_xs_profile2[xs2_n - i_low_point_index] = 99999.9

        # Update the row indices
        ia_xc_row1_index_main[i_low_point_index:i_center_point] = ia_xc_row1_index_main[0:i_center_point - i_low_point_index]
        ia_xc_row1_index_main[0:i_low_point_index + 1] = np.flip(ia_xc_row2_index_main[0:i_low_point_index + 1])
        ia_xc_row2_index_main[0:i_center_point - i_low_point_index] = ia_xc_row2_index_main[i_low_point_index:i_center_point]

        # Update the column indices
        ia_xc_column1_index_main[i_low_point_index:i_center_point] = ia_xc_column1_index_main[0:i_center_point - i_low_point_index]
        ia_xc_column1_index_main[0:i_low_point_index + 1] = np.flip(ia_xc_column2_index_main[0:i_low_point_index + 1])
        ia_xc_column2_index_main[0:i_center_point - i_low_point_index] = ia_xc_column2_index_main[i_low_point_index:i_center_point]
    else:
        return ia_xc_row1_index_main[i_center_point], ia_xc_column1_index_main[i_center_point]  
    
    # The r and c for the stream cell is adjusted because it may have moved
    row, col = ia_xc_row1_index_main[0], ia_xc_column1_index_main[0]
    return row, col
    

@njit(cache=True)
def _sample_side(
        profile: np.ndarray,
        lc_profile: np.ndarray,
        ia_xc_row_index_main: np.ndarray,
        ia_xc_column_index_main: np.ndarray,
        ia_xc_row_index_second: np.ndarray,
        ia_xc_column_index_second: np.ndarray,
        da_xc_main_fract: np.ndarray,
        da_xc_second_fract: np.ndarray,
        i_center_point: int,
        i_row_bottom: int,
        i_row_top: int,
        i_column_bottom: int,
        i_column_top: int,
        dm_elevation: np.ndarray,
        dm_land_use: np.ndarray
    ):
        i_xs_length_indice = i_center_point

        for i in range(i_xs_length_indice):
            if (
                ia_xc_row_index_main[i] <= i_row_bottom or
                ia_xc_row_index_second[i] <= i_row_bottom or
                ia_xc_row_index_main[i] >= i_row_top or
                ia_xc_row_index_second[i] >= i_row_top or
                ia_xc_column_index_main[i] <= i_column_bottom or
                ia_xc_column_index_second[i] <= i_column_bottom or
                ia_xc_column_index_main[i] >= i_column_top or
                ia_xc_column_index_second[i] >= i_column_top
            ):
                i_xs_length_indice = i
                break

        profile[i_xs_length_indice] = 99999.9

        # profile[:i_xs_length_indice] = (
        #     self.dm_elevation[ia_xc_row_index_main[:i_xs_length_indice], ia_xc_column_index_main[:i_xs_length_indice]] * self.da_xc_main_fract[self.i_precompute_angle_closest, :i_xs_length_indice] +
        #     self.dm_elevation[ia_xc_row_index_second[:i_xs_length_indice], ia_xc_column_index_second[:i_xs_length_indice]] * self.da_xc_second_fract[self.i_precompute_angle_closest, :i_xs_length_indice]
        # )

        for i in range(i_xs_length_indice):
            row_main = ia_xc_row_index_main[i]
            col_main = ia_xc_column_index_main[i]
            row_second = ia_xc_row_index_second[i]
            col_second = ia_xc_column_index_second[i]

            profile[i] = (
                dm_elevation[row_main, col_main] * da_xc_main_fract[i] +
                dm_elevation[row_second, col_second] * da_xc_second_fract[i]
            )

        # lc_profile[:i_xs_length_indice] = self.dm_land_use[ia_xc_row_index_main[:i_xs_length_indice], ia_xc_column_index_main[:i_xs_length_indice]]
        for i in range(i_xs_length_indice):
            row_main = ia_xc_row_index_main[i]
            col_main = ia_xc_column_index_main[i]
            lc_profile[i] = int(dm_land_use[row_main, col_main])

        return i_xs_length_indice

@njit(cache=True)
def _check_for_negative_depths(da_y_depth: np.ndarray):
    # Take action if there are values < 0
    lt_0_in_depths = False
    i_target_index = 0
    for i_target_index, value in enumerate(da_y_depth[1:]):
        if value <= 0:
            lt_0_in_depths = True
            break

    return lt_0_in_depths, i_target_index

@njit(cache=True)
def _get_distance_to_use(da_y_depth: np.ndarray, i_target_index: int, d_ordinate_dist: float):
    return d_ordinate_dist * da_y_depth[i_target_index - 1] / (np.abs(da_y_depth[i_target_index - 1]) + np.abs(da_y_depth[i_target_index]))

@njit(cache=True)
def _calculate_top_width_up_to_point(i_target_index: int, d_dist_use: float, d_ordinate_dist: float):
    return d_ordinate_dist * (i_target_index - 1) + d_dist_use

@njit(cache=True)
def _calculate_top_width_from_all(da_y_depth: np.ndarray, d_ordinate_dist: float):
    return d_ordinate_dist * (da_y_depth.shape[0] - 1)

@njit(cache=True)
def _get_stream_depths(d_wse: float, profile: np.ndarray, n: int):
    da_y_depth = d_wse - profile[:n]

    if da_y_depth.shape[0] <= 0 or da_y_depth[0] <= 1e-16:
        return None
    
    return da_y_depth

@njit(cache=True)
def _calculate_side_top_width(d_wse: float, profile: np.ndarray, n: int, d_ordinate_dist: float):
    da_y_depth = _get_stream_depths(d_wse, profile, n)

    if da_y_depth is None:
        return 0

    lt_0_in_depths, i_target_index = _check_for_negative_depths(da_y_depth)

    if lt_0_in_depths:
        i_target_index += 1
        d_dist_use = _get_distance_to_use(da_y_depth, i_target_index, d_ordinate_dist)
        return _calculate_top_width_up_to_point(i_target_index, d_dist_use, d_ordinate_dist)
    else:
        return _calculate_top_width_from_all(da_y_depth, d_ordinate_dist)


@njit(cache=True)
def _calculate_stream_geometry(da_xs_profile: np.ndarray,
                                d_wse: float,
                                n: int,
                                d_ordinate_dist: float,
                                da_n_profile: np.ndarray = None,) -> tuple[float, ...]:
    # Initial output
    d_area, d_perimeter, d_composite_n = 0.0, 0.0, 0.0

    # Estimate the depth of the stream
    da_y_depth = _get_stream_depths(d_wse, da_xs_profile, n)

    # Return if the depth is not valid.
    if da_y_depth is None:
        return 0, 0, 0

    # Take action if there are values < 0
    lt_0_in_depths, i_target_index = _check_for_negative_depths(da_y_depth)
    
    if lt_0_in_depths:
        # A value < 0 exists. Calculate up to that value then break for the rest of hte values.
        # Get the index of the first bad vadlue
        i_target_index += 1

        # Calculate the distance to use
        d_dist_use = _get_distance_to_use(da_y_depth, i_target_index, d_ordinate_dist)

        # Calculate the geometric variables
        d_area = np.sum(d_ordinate_dist * 0.5 * (da_y_depth[1:i_target_index] + da_y_depth[:i_target_index-1])) + 0.5 * d_dist_use * da_y_depth[i_target_index-1]

        d_perimeter_i = calculate_hypotnuse(d_dist_use, da_y_depth[i_target_index - 1])
        perim_array = calculate_hypotnuse(d_ordinate_dist, (da_y_depth[1:i_target_index] - da_y_depth[:i_target_index-1]))

        d_perimeter = np.sum(perim_array) + d_perimeter_i
        
        # Calculate the composite n
        d_composite_n = np.sum(perim_array[:i_target_index-1] * da_n_profile[1:i_target_index]**1.5) + d_perimeter_i * da_n_profile[i_target_index - 1]**1.5
    else:
        # All values are positive, so include them all.

        # Calculate the geometric values
        d_area = np.sum(d_ordinate_dist * 0.5 * (da_y_depth[2:] + da_y_depth[1:-1]))

        perim_array = calculate_hypotnuse(d_ordinate_dist, da_y_depth[1:] - da_y_depth[:-1])

        d_perimeter = np.sum(perim_array[1:])

        d_composite_n = np.sum(perim_array * da_n_profile[1:]**1.5)

    # Return to the calling function
    return d_area, d_perimeter, d_composite_n

@njit(cache=True)
def _calculate_stream_geometry_and_topwidth(da_xs_profile: np.ndarray, 
                            d_wse: float, 
                            n: int, 
                            d_ordinate_dist: float,
                            da_n_profile: np.ndarray,) -> tuple[float, ...]:
    """
    Estimates the stream geometry

    Uses a composite Manning's n as given by:
    Composite Manning N based on https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.5/theoretical-basis-for-one-dimensional-and-two-dimensional-hydrodynamic-calculations/1d-steady-flow-water-surface-profiles/composite-manning-s-n-for-the-main-channel

    Parameters
    ----------
    da_xs_profile: ndarray
        Elevations of the stream cross section
    d_wse: float
        Water surface elevation
    d_distance_z: float
        Incremental distance per cell parallel to the orientation of the cross section
    da_n_profile: float
        Input initial Manning's n for the stream

    Returns
    -------
    d_area, d_perimeter, d_composite_n, d_top_width

    """
    # Initial output
    d_area, d_perimeter, d_composite_n, d_top_width = 0.0, 0.0, 0.0, 0.0

    # Estimate the depth of the stream
    da_y_depth = _get_stream_depths(d_wse, da_xs_profile, n)

    # Return if the depth is not valid.
    if da_y_depth is None:
        return 0, 0, 0, 0

    # Take action if there are values < 0
    lt_0_in_depths, i_target_index = _check_for_negative_depths(da_y_depth)
    
    if lt_0_in_depths:
        # A value < 0 exists. Calculate up to that value then break for the rest of hte values.
        # Get the index of the first bad vadlue
        i_target_index += 1

        # Calculate the distance to use
        d_dist_use = _get_distance_to_use(da_y_depth, i_target_index, d_ordinate_dist)

        # Calculate the geometric variables
        d_area = np.sum(d_ordinate_dist * 0.5 * (da_y_depth[1:i_target_index] + da_y_depth[:i_target_index-1])) + 0.5 * d_dist_use * da_y_depth[i_target_index-1]

        d_perimeter_i = calculate_hypotnuse(d_dist_use, da_y_depth[i_target_index - 1])
        perim_array = calculate_hypotnuse(d_ordinate_dist, (da_y_depth[1:i_target_index] - da_y_depth[:i_target_index-1]))

        d_perimeter = np.sum(perim_array) + d_perimeter_i
        
        # Calculate the composite n
        d_composite_n = np.sum(perim_array[:i_target_index-1] * da_n_profile[1:i_target_index]**1.5) + d_perimeter_i * da_n_profile[i_target_index - 1]**1.5

        # Update the top width
        d_top_width = _calculate_top_width_up_to_point(i_target_index, d_dist_use, d_ordinate_dist)

    else:
        # All values are positive, so include them all.

        # Calculate the geometric values
        d_area = np.sum(d_ordinate_dist * 0.5 * (da_y_depth[2:] + da_y_depth[1:-1]))

        perim_array = calculate_hypotnuse(d_ordinate_dist, da_y_depth[1:] - da_y_depth[:-1])

        d_perimeter = np.sum(perim_array[1:])

        d_composite_n = np.sum(perim_array * da_n_profile[1:]**1.5)

        d_top_width = _calculate_top_width_from_all(da_y_depth, d_ordinate_dist)

    # Return to the calling function
    return d_area, d_perimeter, d_composite_n, d_top_width

@njit(cache=True)
def calculate_discharge_from_wse(wse: float, sqrt_slope: float, profile1: np.ndarray, xs1_n: float, mannings_n1: float,
                                profile2: np.ndarray, xs2_n: float, mannings_n2: float, d_ordinate_dist: float):
        # Calculate the geometry
    A1, P1, np1 = _calculate_stream_geometry(profile1, wse, xs1_n, d_ordinate_dist, mannings_n1)
    A2, P2, np2 = _calculate_stream_geometry(profile2, wse, xs2_n, d_ordinate_dist, mannings_n2)

    # Aggregate the geometric properties
    d_a_sum = A1 + A2
    d_p_sum = max(P1 + P2, 1e-6)  # Avoid division by zero

    d_composite_n = np.round(((np1 + np2) / d_p_sum)**(2 / 3), 4)

    # Check that the mannings n is physically realistic
    if d_composite_n < 0.0001:
        d_composite_n = 0.035

    discharge = (1 / d_composite_n) * d_a_sum * (d_a_sum / d_p_sum)**(2 / 3) * sqrt_slope
    return discharge

@njit(cache=True)
def _adjust_one_side_for_bathymetry(i_bank_index: int, d_total_bank_dist: float,
                                    d_trap_base: float, d_distance_h: float, ia_xc_r_index_main: np.ndarray, 
                                    ia_xc_c_index_main: np.ndarray, da_xs_profile: np.ndarray, dm_output_bathymetry: np.ndarray,
                                    d_side_dist: float, d_y_bathy: float, d_y_depth: float, d_ordinate_dist: float,
                                    dm_elevation: np.ndarray, b_bathy_use_banks: bool):
    """
    Adjusts the profile for the estimated bathymetry

    Parameters
    ----------
    da_xs_profile: ndarray
        Elevations of the stream cross section
    i_bank_index: int
        Distance in index space from the stream to the bank
    d_total_bank_dist: float
        Distance to the bank estimated in unit space
    d_trap_base: float
        Bottom distance of the stream cross section
    d_distance_z: float
        Incremental distance per cell parallel to the orientation of the cross section
    d_distance_h: float
        Distance of the slope section of the trapezoidal channel.  Typically d_distance_h = 0.2* TW of Trapezoid
    d_y_bathy: float
        Bathymetry elevation of the bottom
    d_y_depth: float
        Depth.  Basically water surface elevation (WSE) minus d_y_bathy
    dm_output_bathymetry: ndarray
        Output bathymetry matrix
    ia_xc_r_index_main: ndarray
        Row indices for the stream cross section
    ia_xc_c_index_main: ndarray
        Column indices for the stream cross section

    Returns
    -------
    None. Values are updated in the output bathymetry matrix

    """

    # If banks are calculated, make an adjustment to the trapezoidal bathymetry
    if i_bank_index <= 0:
        return
    
    # Loop over the bank width offset indices
    for x in range(min(i_bank_index + 1, len(ia_xc_r_index_main))):
        # Calculate the distance to the bank
        d_dist_cell_to_bank = (i_bank_index - x) * d_ordinate_dist + d_side_dist   #d_side_dist should be zero if using Flat WSE or LC method.
        # lc_grid_val = int(dm_land_use[ia_xc_r_index_main[x], ia_xc_c_index_main[x]])

        # if lc_grid_val<0 or (i_lc_water_value>0 and lc_grid_val!=i_lc_water_value):
        #     return

        # # Joseph added this because it looks like we aren't getting a bathymetry output for the first cell in the cross-section
        # if x == 0:
        #     # If the cell is the first cell, then set it to the bottom elevation of the trapezoid.
        #     da_xs_profile[x] = d_y_bathy
        #     dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

        # If the cell is in the flat part of the trapezoidal cross-section, set it to the bottom elevation of the trapezoid.
        if d_dist_cell_to_bank > d_distance_h:
            if b_bathy_use_banks == False and d_y_bathy < dm_elevation[ia_xc_r_index_main[x], ia_xc_c_index_main[x]]:
                da_xs_profile[x] = d_y_bathy
                dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]
            elif b_bathy_use_banks == True:
                da_xs_profile[x] = d_y_bathy
                dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

        # If the cell is in the slope part of the trapezoid you need to find the elevation based on the slope of the trapezoid side.
        elif d_dist_cell_to_bank <= d_distance_h and d_dist_cell_to_bank < d_trap_base + d_distance_h:
            if b_bathy_use_banks == False and (d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank / d_distance_h))) < dm_elevation[ia_xc_r_index_main[x], ia_xc_c_index_main[x]]:
                da_xs_profile[x] = d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank / d_distance_h))
                dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]
            elif b_bathy_use_banks == True:
                da_xs_profile[x] = d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank / d_distance_h))
                dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

        # Similar to above, but on the far-side slope of the trapezoid.  You need to find the elevation based on the slope of the trapezoid side.
        elif d_dist_cell_to_bank >= d_trap_base + d_distance_h:
            d_dist_cell_to_bank_other_side = d_total_bank_dist - d_dist_cell_to_bank
            if b_bathy_use_banks == False and d_dist_cell_to_bank_other_side>0.0 and (d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank_other_side / d_distance_h))) < dm_elevation[ia_xc_r_index_main[x], ia_xc_c_index_main[x]]:
                da_xs_profile[x] = d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank_other_side / d_distance_h))
                dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]
            elif b_bathy_use_banks == True:
                da_xs_profile[x] = d_y_bathy + d_y_depth * (1.0 - (d_dist_cell_to_bank_other_side / d_distance_h))
                dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]
            #if (d_y_bathy + d_y_depth * (d_dist_cell_to_bank - (d_trap_base + d_distance_h)) / d_distance_h) < dm_elevation[ia_xc_r_index_main[x], ia_xc_c_index_main[x]]:
            #    da_xs_profile[x] = d_y_bathy + d_y_depth * (d_dist_cell_to_bank - (d_trap_base + d_distance_h)) / d_distance_h
            #    dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

        # If the cell is outside of the banks, then just ignore this cell (set it to it's same elevation).  No need to update the output bathymetry raster.
        elif d_dist_cell_to_bank <= 0 or d_dist_cell_to_bank >= d_total_bank_dist:
            return


        
        #JUST FOR TESTING
        #da_xs_profile[x] = d_y_bathy
        #dm_output_bathymetry[ia_xc_r_index_main[x], ia_xc_c_index_main[x]] = da_xs_profile[x]

    return

@njit(cache=True)
def get_xs_index_values_precalculated(ia_xc_dr_index_main: np.ndarray, ia_xc_dc_index_main: np.ndarray, ia_xc_dr_index_second: np.ndarray, ia_xc_dc_index_second: np.ndarray, da_xc_main_fract: np.ndarray,
                        da_xc_second_fract: np.ndarray, d_xs_direction: np.ndarray, i_centerpoint: int, d_dx: float, d_dy: float):
    """
    Calculates the distance of the stream cross section

    Parameters
    ----------
    ia_xc_dr_index_main: ndarray
        Indices of the first cross section index
    ia_xc_dc_index_main: ndarray
        Index offsets of the first cross section index
    ia_xc_dr_index_second: ndarray
        Indices of the second cross section index
    ia_xc_dc_index_second: ndarray
        Index offsets of the second cross section index
    da_xc_main_fract: ndarray: ndarray
        # todo: add
    da_xc_second_fract: ndarray
        # todo: add
    d_xs_direction: float
        Orientation of the cross section
    i_centerpoint: int
        Distance from the cell to search
    d_dx: float
        Cell resolution in the x direction
    d_dy: float
        Cell resolution in the y direction

    Returns
    -------
    d_distance_z: float
        Distance along the cross section direction

    """
    
    
    '''
    Assume there are 4 quadrants:
            Q3 | Q4      r<0 c<0  |  r<0 c>0
            Q2 | Q1      r>0 c<0  |  r>0 c>0
    
    These quadrants are inversed about the x-axis due to rows being positive in the downward direction
    '''
    
    
    # Determine the best direction to perform calcualtions
    #  Row-Dominated
    if d_xs_direction >= (math.pi / 4) and d_xs_direction <= (3 * math.pi / 4):
        # Calculate the distance in the x direction
        da_distance_x = np.arange(i_centerpoint) * d_dy * math.cos(d_xs_direction)

        # Convert the distance to a number of indices
        ia_x_index_offset: int = da_distance_x // d_dx

        ia_xc_dr_index_main[0:i_centerpoint] = np.arange(i_centerpoint)
        ia_xc_dc_index_main[0:i_centerpoint] = ia_x_index_offset

        # Calculate the sign of the angle
        ia_sign = np.ones(i_centerpoint)
        ia_sign[da_distance_x < 0] = -1

        # Round using the angle direction
        ia_x_index_offset = np.round((da_distance_x / d_dx) + 0.5 * ia_sign, 0)

        # Set the values in as index locations
        ia_xc_dr_index_second[0:i_centerpoint] = np.arange(i_centerpoint)
        ia_xc_dc_index_second[0:i_centerpoint] = ia_x_index_offset

        # ddx is the distance from the main cell to the location where the line passes through.  Do 1-ddx to get the weight
        da_ddx = np.fabs((da_distance_x / d_dx) - ia_x_index_offset)
        da_xc_main_fract[0:i_centerpoint] = 1.0 - da_ddx
        da_xc_second_fract[0:i_centerpoint] = da_ddx

        # Distance between each increment
        d_distance_z = math.sqrt((d_dy * math.cos(d_xs_direction)) * (d_dy * math.cos(d_xs_direction)) + d_dy * d_dy)

    # Col-Dominated
    else:
        # Calculate based on the column being the dominate direction
        # Calculate the distance in the y direction
        da_distance_y = np.arange(i_centerpoint) * d_dx * math.sin(d_xs_direction)

        # Convert the distance to a number of indices
        ia_y_index_offset: int = da_distance_y // d_dy
        
        column_pos_or_neg = 1 
        if d_xs_direction >= (math.pi / 2): 
            column_pos_or_neg = -1

        ia_xc_dr_index_main[0:i_centerpoint] = ia_y_index_offset
        ia_xc_dc_index_main[0:i_centerpoint] = np.arange(i_centerpoint) * column_pos_or_neg

        # Calculate the sign of the angle
        ia_sign = np.ones(i_centerpoint)   #I think this can always just be positive one
        #ia_sign[da_distance_y < 0] = -1
        #ia_sign[da_distance_y > 0] = -1
        #ia_sign = ia_sign * -1

        # Round using the angle direction
        ia_y_index_offset = np.round((da_distance_y / d_dy) + 0.5 * ia_sign, 0)

        # Set the values in as index locations
        ia_xc_dr_index_second[0:i_centerpoint] = ia_y_index_offset
        ia_xc_dc_index_second[0:i_centerpoint] = np.arange(i_centerpoint) * column_pos_or_neg

        # ddy is the distance from the main cell to the location where the line passes through.  Do 1-ddx to get the weight
        da_ddy = np.fabs((da_distance_y / d_dy) - ia_y_index_offset)
        da_xc_main_fract[0:i_centerpoint] = 1.0 - da_ddy
        da_xc_second_fract[0:i_centerpoint] = da_ddy

        # Distance between each increment
        d_distance_z = math.sqrt((d_dx * math.sin(d_xs_direction)) * (d_dx * math.sin(d_xs_direction)) + d_dx * d_dx)

    # Return to the calling function
    return d_distance_z

@njit(cache=True)
def find_depth_of_bathymetry(d_baseflow: float, d_bottom_width: float, d_top_width: float, d_slope: float, d_mannings_n: float):
    """
    Estimates the depth iteratively by comparing the calculated flow to the baseflow

    Parameters
    ----------
    d_baseflow: float
        Baseflow input for flow convergence calculation
    d_bottom_width: float
        Bottom width of the stream
    d_top_width: float
        Top width of the stream
    d_slope: float
        Slope of the stream
    d_mannings_n: float
        Manning's roughness of the stream

    Returns
    -------
    d_working_depth: float
        Estimated depth of the stream

    """

    # Calculate the average width of the stream
    d_average_width = (d_top_width - d_bottom_width) * 0.5

    # Assign a starting depth
    d_depth_start = 0.0

    # Set the incremental convergence targets
    l_dy_list = [1.0, 0.5, 0.1, 0.01]
    
    # Loop over each convergence target
    for d_dy in l_dy_list:
        # Set the initial value
        d_flow_calculated = 0.0
        d_working_depth = d_depth_start

        # This will prevent infinite loops
        d_max_depth = d_depth_start + 25

        # Converge until the calculate flow is above the baseflow
        while d_flow_calculated <= d_baseflow and d_working_depth < d_max_depth:
            d_working_depth = d_working_depth + d_dy
            d_area = d_working_depth * (d_bottom_width + d_top_width) / 2.0
            d_perimeter = d_bottom_width + 2.0 * math.sqrt(d_average_width * d_average_width + d_working_depth * d_working_depth)
            d_hydraulic_radius = d_area / d_perimeter
            d_flow_calculated = (1.0 / d_mannings_n) * d_area * d_hydraulic_radius**(2 / 3) * d_slope**0.5

        # Update the starting depth
        d_depth_start = d_working_depth - d_dy

    # Update the calculated depth
    d_working_depth = d_working_depth - d_dy

    # Debugging variables
    # A = y * (B + TW) / 2.0
    # P = B + 2.0*math.sqrt(H*H + y*y)
    # R = A / P
    # Qcalc = (1.0/n)*A*math.pow(R,(2/3)) * pow(slope,0.5)
    # print(str(d_top_width) + ' ' + str(d_working_depth) + '  ' + str(d_flow_calculated) + ' vs ' + str(d_baseflow))

    return d_working_depth

@njit(cache=True)
def calculate_hypotnuse(d_side_one: float, d_side_two: float):
    """
    Calculates the hypotenuse distance of a right triangle

    Parameters
    ----------
    d_side_one: float
        Length of the first right triangle side
    d_side_two: float
        Length of the second right triangle side

    Returns
    -------
    d_distance: float
        Length of the hypotenuse

    """

    # Calculate the distance
    d_distance = (d_side_one ** 2 + d_side_two ** 2)**(1/2)

    # Return to the calling function
    return d_distance

def is_valid_number(elev):
    """Check if elev is a valid number (not None, NaN, or non-numeric)."""
    return isinstance(elev, (int, float)) and not np.isnan(elev)

def calc_bankfull_elevation(base_elev, bank_elev_1, bank_elev_2): 
    """
    Determine the bankfull elevation based on the two bank elevation values.
    It collects all valid bank elevations that are at least base_elev.
    If both are valid, it picks the minimum one.
    If neither is valid, it defaults to base_elev.
    """
    valid_banks = [elev for elev in (bank_elev_1, bank_elev_2) if is_valid_number(elev) and elev >= base_elev]
    return min(valid_banks, default=base_elev)
