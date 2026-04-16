import math

import numpy as np
from numba import njit

class CrossSection:
    def __init__(self, 
                 i_center_point: int, 
                 dx: float, dy: float, i_precompute_angles: int, d_precompute_angles: float):
        self.i_center_point = i_center_point

        self.da_xs_profile1 = np.zeros(self.i_center_point + 1)
        self.da_xs_profile2 = np.zeros(self.i_center_point + 1)
        self.ia_lc_xs1 = np.zeros(self.i_center_point + 1)
        self.ia_lc_xs2 = np.zeros(self.i_center_point + 1)

        self.create_cross_section_ordinates(i_center_point, dx, dy, i_precompute_angles, d_precompute_angles)

    def create_cross_section_ordinates(self, i_center_point: int, dx: float, dy: float, i_precompute_angles: int, d_precompute_angles: float):
        self.i_center_point = i_center_point
        self.ia_xc_dr_index_main = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
        self.ia_xc_dc_index_main = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
        self.ia_xc_dr_index_second = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
        self.ia_xc_dc_index_second = np.zeros((i_precompute_angles + 1, i_center_point + 1), dtype=int)  # Only need to go to center point, because the other side of xs we can just use *-1
        self.d_distance_z = np.zeros(i_precompute_angles + 1, dtype=float)
        self.da_xc_main_fract = np.zeros((i_precompute_angles + 1, i_center_point + 1))
        self.da_xc_second_fract = np.zeros((i_precompute_angles + 1, i_center_point + 1))

        for i in range(i_precompute_angles+1):
            d_xs_direction = d_precompute_angles * i
            # Get the Cross-Section Ordinates
            self.d_distance_z[i]= get_xs_index_values_precalculated(self.ia_xc_dr_index_main[i], self.ia_xc_dc_index_main[i], self.ia_xc_dr_index_second[i], self.ia_xc_dc_index_second[i], self.da_xc_main_fract[i], self.da_xc_second_fract[i], d_xs_direction,
                                                                                           i_center_point, dx, dy)

    def set_boundary_extents(self, i_boundary_number: int, nrows: int, ncols: int):
        """
        Get the max and min row and col that we can go for later search functions (based on max of slope and direction distance parameters.)
        """
        self.i_row_bottom = i_boundary_number
        self.i_row_top = nrows + i_boundary_number - 1
        self.i_column_bottom = i_boundary_number
        self.i_column_top = ncols + i_boundary_number - 1
    
    def reset(self, row: int, col: int, i_precompute_angle_closest: int):
        """
        Parameters
        """
        self.row = row
        self.col = col
        self.i_precompute_angle_closest = i_precompute_angle_closest
        self.xs1_n = 0
        self.xs2_n = 0
        self.da_xs_profile1[:] = 0.0
        self.da_xs_profile2[:] = 0.0

        self.ia_xc_row1_index_main = self.row + self.ia_xc_dr_index_main[self.i_precompute_angle_closest]
        self.ia_xc_column1_index_main = self.col + self.ia_xc_dc_index_main[self.i_precompute_angle_closest]
        self.ia_xc_row2_index_main = self.row + self.ia_xc_dr_index_main[self.i_precompute_angle_closest] * -1
        self.ia_xc_column2_index_main = self.col + self.ia_xc_dc_index_main[self.i_precompute_angle_closest] * -1
        
        self.ia_xc_row1_index_second = self.row + self.ia_xc_dr_index_second[self.i_precompute_angle_closest]
        self.ia_xc_column1_index_second = self.col + self.ia_xc_dc_index_second[self.i_precompute_angle_closest]
        self.ia_xc_row2_index_second = self.row + self.ia_xc_dr_index_second[self.i_precompute_angle_closest] * -1
        self.ia_xc_column2_index_second = self.col + self.ia_xc_dc_index_second[self.i_precompute_angle_closest] * -1

    def sample_from_dem(self, dm_elevation, dm_land_use=None):
        self.xs1_n = self._sample_side(
            dm_elevation,
            dm_land_use,
            profile=self.da_xs_profile1,
            lc_profile=self.ia_lc_xs1,
            ia_xc_row_index_main=self.ia_xc_row1_index_main,
            ia_xc_column_index_main=self.ia_xc_column1_index_main,
            ia_xc_row_index_second=self.ia_xc_row1_index_second,
            ia_xc_column_index_second=self.ia_xc_column1_index_second
        )

        self.xs2_n = self._sample_side(
            dm_elevation,
            dm_land_use,
            profile=self.da_xs_profile2,
            lc_profile=self.ia_lc_xs2,
            ia_xc_row_index_main=self.ia_xc_row2_index_main,
            ia_xc_column_index_main=self.ia_xc_column2_index_main,
            ia_xc_row_index_second=self.ia_xc_row2_index_second,
            ia_xc_column_index_second=self.ia_xc_column2_index_second
        )
        
    def _sample_side(
        self,
        dm_elevation: np.ndarray,
        dm_land_use: np.ndarray,
        profile: np.ndarray,
        lc_profile: np.ndarray,
        ia_xc_row_index_main: np.ndarray,
        ia_xc_column_index_main: np.ndarray,
        ia_xc_row_index_second: np.ndarray,
        ia_xc_column_index_second: np.ndarray
    ):
        i_xs_length_indice = self.i_center_point

        for i in range(i_xs_length_indice):
            if (
                ia_xc_row_index_main[i] <= self.i_row_bottom or
                ia_xc_row_index_second[i] <= self.i_row_bottom or
                ia_xc_row_index_main[i] >= self.i_row_top or
                ia_xc_row_index_second[i] >= self.i_row_top or
                ia_xc_column_index_main[i] <= self.i_column_bottom or
                ia_xc_column_index_second[i] <= self.i_column_bottom or
                ia_xc_column_index_main[i] >= self.i_column_top or
                ia_xc_column_index_second[i] >= self.i_column_top
            ):
                i_xs_length_indice = i
                break

        profile[i_xs_length_indice] = 99999.9

        profile[:i_xs_length_indice] = (
            dm_elevation[ia_xc_row_index_main[:i_xs_length_indice], ia_xc_column_index_main[:i_xs_length_indice]] * self.da_xc_main_fract[self.i_precompute_angle_closest, :i_xs_length_indice] +
            dm_elevation[ia_xc_row_index_second[:i_xs_length_indice], ia_xc_column_index_second[:i_xs_length_indice]] * self.da_xc_second_fract[self.i_precompute_angle_closest, :i_xs_length_indice]
        )

        # for i in range(i_xs_length_indice):
        #     row_main = ia_xc_row_index_main[i]
        #     col_main = ia_xc_column_index_main[i]
        #     row_second = ia_xc_row_index_second[i]
        #     col_second = ia_xc_column_index_second[i]

        #     profile[i] = (
        #         dm_elevation[row_main, col_main] * self.da_xc_main_fract[i] +
        #         dm_elevation[row_second, col_second] * self.da_xc_second_fract[i]
        #     )

        if dm_land_use is not None:
            lc_profile[:i_xs_length_indice] = dm_land_use[ia_xc_row_index_main[:i_xs_length_indice], ia_xc_column_index_main[:i_xs_length_indice]]
            # for i in range(i_xs_length_indice):
            #     row_main = ia_xc_row_index_main[i]
            #     col_main = ia_xc_column_index_main[i]
            #     lc_profile[i] = int(dm_land_use[row_main, col_main])

        return i_xs_length_indice
    
    def adjust_cross_section_to_lowest_point(self, i_low_spot_range: int, dm_elevation: np.ndarray, dm_land_use: np.ndarray):
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
        d_dem_low_point_elev = self.da_xs_profile1[0]
        i_low_point_index = 0

        # Loop on the search range for the low point
        for i_entry in range(i_low_spot_range):
            if i_entry >= self.da_xs_profile1.shape[0] or i_entry >= self.da_xs_profile2.shape[0]:
                break
            # Look in the first profile
            if self.da_xs_profile1[i_entry] > 0.0 and self.da_xs_profile1[i_entry] < d_dem_low_point_elev:
                # New low point was found. Update the index.
                d_dem_low_point_elev = self.da_xs_profile1[i_entry]
                i_low_point_index = i_entry

            # Look in the second profile
            if self.da_xs_profile2[i_entry] > 0.0 and self.da_xs_profile2[i_entry] < d_dem_low_point_elev:
                # New low point was found. Update the index.
                d_dem_low_point_elev = self.da_xs_profile2[i_entry]
                i_low_point_index = i_entry * -1

        # Process based on if the low point is in the first or second profile
        if i_low_point_index > 0:
            # Low point is in the first profile. Update the cross section and mannings.
            self.da_xs_profile2[i_low_point_index:self.i_center_point] = self.da_xs_profile2[0:self.i_center_point - i_low_point_index]
            self.da_xs_profile2[0:i_low_point_index + 1] = np.flip(self.da_xs_profile1[0:i_low_point_index + 1])
            self.da_xs_profile1[0:self.i_center_point - i_low_point_index] = self.da_xs_profile1[i_low_point_index:self.i_center_point]
            self.da_xs_profile1[self.xs1_n - i_low_point_index] = 99999.9

            # Update the row indices
            self.ia_xc_row2_index_main[i_low_point_index:self.i_center_point] = self.ia_xc_row2_index_main[0:self.i_center_point - i_low_point_index]
            self.ia_xc_row2_index_main[0:i_low_point_index + 1] = np.flip(self.ia_xc_row1_index_main[0:i_low_point_index + 1])
            self.ia_xc_row1_index_main[0:self.i_center_point - i_low_point_index] = self.ia_xc_row1_index_main[i_low_point_index:self.i_center_point]

            # Update the column indices
            self.ia_xc_column2_index_main[i_low_point_index:self.i_center_point] = self.ia_xc_column2_index_main[0:self.i_center_point - i_low_point_index]
            self.ia_xc_column2_index_main[0:i_low_point_index + 1] = np.flip(self.ia_xc_column1_index_main[0:i_low_point_index + 1])
            self.ia_xc_column1_index_main[0:self.i_center_point - i_low_point_index] = self.ia_xc_column1_index_main[i_low_point_index:self.i_center_point]

        elif i_low_point_index < 0:
            # Low point is in the second profile Update the cross section and mannings.
            i_low_point_index = i_low_point_index * -1
            self.da_xs_profile1[i_low_point_index:self.i_center_point] = self.da_xs_profile1[0:self.i_center_point - i_low_point_index]
            self.da_xs_profile1[0:i_low_point_index + 1] = np.flip(self.da_xs_profile2[0:i_low_point_index + 1])
            self.da_xs_profile2[0:self.i_center_point - i_low_point_index] = self.da_xs_profile2[i_low_point_index:self.i_center_point]
            self.da_xs_profile2[self.xs2_n - i_low_point_index] = 99999.9

            # Update the row indices
            self.ia_xc_row1_index_main[i_low_point_index:self.i_center_point] = self.ia_xc_row1_index_main[0:self.i_center_point - i_low_point_index]
            self.ia_xc_row1_index_main[0:i_low_point_index + 1] = np.flip(self.ia_xc_row2_index_main[0:i_low_point_index + 1])
            self.ia_xc_row2_index_main[0:self.i_center_point - i_low_point_index] = self.ia_xc_row2_index_main[i_low_point_index:self.i_center_point]

            # Update the column indices
            self.ia_xc_column1_index_main[i_low_point_index:self.i_center_point] = self.ia_xc_column1_index_main[0:self.i_center_point - i_low_point_index]
            self.ia_xc_column1_index_main[0:i_low_point_index + 1] = np.flip(self.ia_xc_column2_index_main[0:i_low_point_index + 1])
            self.ia_xc_column2_index_main[0:self.i_center_point - i_low_point_index] = self.ia_xc_column2_index_main[i_low_point_index:self.i_center_point]
        else:
            return    
        
        # The r and c for the stream cell is adjusted because it may have moved
        self.row, self.col = self.get_row_col()

        
        # re-sample the cross-section to make sure all of the low-spot data has the same values through interpolation
        self.sample_from_dem(dm_elevation, dm_land_use)

    
    def get_row_col(self):
        return self.ia_xc_row1_index_main[0], self.ia_xc_column1_index_main[0]
    
    def get_thalweg(self):
        return self.da_xs_profile1[0]

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