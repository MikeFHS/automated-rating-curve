"""Cross-section sampling and hydraulic geometry helpers.

This module contains the :class:`~arc.cross_section.CrossSection` class, which
samples two-sided cross-sections from a DEM/land-cover raster neighborhood
around a stream cell. It also contains a set of Numba-accelerated helper
functions for:

- Sampling cross-section ordinates with interpolation
- Finding banks and estimating bathymetry
- Computing geometric properties (area/perimeter/top width) at a given WSE
- Computing discharge from WSE using Manning's equation

Most functions here are called from
``src/arc/Automated_Rating_Curve_Generator.py`` during the per-cell compute
loop.
"""

import math

import numpy as np
from numba import njit
from scipy.signal import savgol_filter
from scipy.stats import linregress


from arc import LOG

INFLECT_REGRESSION_WINDOW = 10

class CrossSection:
    """Reusable cross-section sampler for stream cells.

    A ``CrossSection`` instance holds preallocated arrays for the sampled
    profile and land-cover values on both sides of the stream cell. The instance
    can be reused for many cells by calling :meth:`set_cross_section` to update
    indices/profile values in-place.

    Parameters
    ----------
    dx, dy : float
        Raster cell size in the x and y direction. These values are used to
        convert index offsets to physical distances along the cross-section.
    dm_elevation : numpy.ndarray
        Elevation raster (typically a padded DEM).
    dm_land_use : numpy.ndarray
        Land cover raster (aligned to ``dm_elevation``).
    params : dict
        Parsed ARC parameters. The cross-section code expects (at minimum)
        values such as ``d_x_section_distance``, ``d_degree_manipulation``,
        ``d_degree_interval``, ``i_boundary_number``, ``nrows``, and ``ncols``.
    """
    i_precompute_angles = 30
    d_precompute_angles = np.pi / i_precompute_angles

    def __init__(self, 
                 dx: float, dy: float,
                 dm_elevation: np.ndarray, dm_land_use: np.ndarray,
                 dm_stream: np.ndarray | None,
                 params: dict):
        """Initialize a reusable sampler and allocate working arrays."""
        self.d_x_section_distance = params["d_x_section_distance"]
        self.i_center_point = int((self.d_x_section_distance / (sum([dx, dy]) * 0.5)) / 2.0) + 1
        self.dx = dx
        self.dy = dy

        self.da_xs_profile1 = np.zeros(self.i_center_point + 1, dtype=np.float64)
        self.da_xs_profile2 = np.zeros(self.i_center_point + 1, dtype=np.float64)
        self.ia_lc_xs1 = np.zeros(self.i_center_point + 1, dtype=np.uint8)
        self.ia_lc_xs2 = np.zeros(self.i_center_point + 1, dtype=np.uint8)
        self.mannings_n1 = np.zeros(self.i_center_point + 1, dtype=np.float64)
        self.mannings_n2 = np.zeros(self.i_center_point + 1, dtype=np.float64)

        self.dm_elevation = dm_elevation
        self.dm_land_use = dm_land_use
        self.dm_stream = dm_stream

        self.b_FindBanksBasedOnLandCover = params["b_FindBanksBasedOnLandCover"]
        self.i_lc_water_value = params["i_lc_water_value"]
        self.d_bathymetry_trapzoid_height = params["d_bathymetry_trapzoid_height"]
        self.b_bathy_use_banks = params["b_bathy_use_banks"]
        self.reach_scale_inflect_bank_depth = -1.0

        # self.create_cross_section_ordinates()

        # Find all the different angle increments to test
        self.set_angles_to_test(params)

        # Get the extents of the boundaries
        self.set_boundary_extents(params["i_boundary_number"], params["nrows"], params["ncols"])

    def is_valid(self) -> bool:
        """Return ``True`` if either side of the profile has sampled values."""
        return self.xs1_n > 0 or self.xs2_n > 0

    def set_reach_scale_inflect_bank_index(self, bank_index: float | None) -> None:
        """Attach a reach-average INFLECT bank depth to the current section.

        The method name is preserved for compatibility with the existing ARC
        workflow, but the stored value is now the actual representative depth
        above the thalweg rather than a synthetic curve index. ARC computes
        that depth by averaging INFLECT ``d2W/dy^2`` curves and their aligned
        depth axes across each reach, then selecting the bank location from
        the reach-mean curve.
        """
        if bank_index is None:
            self.reach_scale_inflect_bank_depth = -1.0
        else:
            self.reach_scale_inflect_bank_depth = float(bank_index)

    @classmethod
    def create_cross_section_ordinates(cls, params: dict):
        """Precompute cross-section index offsets for a set of discrete angles.

        This precomputation amortizes the cost of converting an angle into
        integer index offsets + interpolation fractions. At runtime, a cell's
        direction is snapped to the closest precomputed angle, and the sampled
        values can be gathered efficiently.

        Parameters
        ----------
        params : dict
            ARC parameters. Must include ``d_x_section_distance``, ``dx``, and
            ``dy``.

        Returns
        -------
        index_arrays : numpy.ndarray
            Integer index offsets for both the "main" and "secondary"
            interpolation indices. Shape is
            ``(n_angles, n_points, 4)``.
        z_distance_array : numpy.ndarray
            Physical spacing between successive ordinates for each precomputed
            angle.
        index_fract_arrays : numpy.ndarray
            Fractional interpolation weights associated with the main/secondary
            indices. Shape is ``(n_angles, n_points, 2)``.
        """
        # Only need to go to center point, because the other side of xs we can just use *-1
        i_center_point = int((params["d_x_section_distance"] / (sum([params["dx"], params["dy"]]) * 0.5)) / 2.0) + 1
        index_arrays = np.zeros((cls.i_precompute_angles + 1, i_center_point + 1, 4), dtype=np.int64)  
        z_distance_array = np.zeros(cls.i_precompute_angles + 1, dtype=np.float64)
        index_fract_arrays = np.zeros((cls.i_precompute_angles + 1, i_center_point + 1, 2), dtype=np.float64)

        ia_xc_dr_index_main = index_arrays[:, :, 0]
        ia_xc_dc_index_main = index_arrays[:, :, 1]
        ia_xc_dr_index_second = index_arrays[:, :, 2]
        ia_xc_dc_index_second = index_arrays[:, :, 3]

        for i in range(cls.i_precompute_angles+1):
            d_xs_direction = cls.d_precompute_angles * i
            # Get the Cross-Section Ordinates
            z_distance_array[i] = get_xs_index_values_precalculated(ia_xc_dr_index_main[i], ia_xc_dc_index_main[i], ia_xc_dr_index_second[i], ia_xc_dc_index_second[i], index_fract_arrays[i, :, 0], index_fract_arrays[i, :, 1], d_xs_direction,
                                                                                           i_center_point, params["dx"], params["dy"])
            
        return index_arrays, z_distance_array, index_fract_arrays
    
    def associate_with_precomputed_index_arrays(self, index_arrays: np.ndarray, z_distance_array: np.ndarray, index_fract_arrays: np.ndarray):
        """Attach precomputed index/fraction arrays produced by
        :meth:`create_cross_section_ordinates`.

        Parameters
        ----------
        index_arrays, z_distance_array, index_fract_arrays : numpy.ndarray
            Arrays returned by :meth:`create_cross_section_ordinates`.
        """
        self.ia_xc_dr_index_main = index_arrays[:, :, 0]
        self.ia_xc_dc_index_main = index_arrays[:, :, 1]
        self.ia_xc_dr_index_second = index_arrays[:, :, 2]
        self.ia_xc_dc_index_second = index_arrays[:, :, 3]
        self.d_distance_z = z_distance_array
        self.da_xc_main_fract = index_fract_arrays[:, :, 0]
        self.da_xc_second_fract = index_fract_arrays[:, :, 1]

    def set_angles_to_test(self, params: dict):
        """Build the list of angle perturbations used during angle search.

        ARC can search around an initial cross-section direction by testing a
        symmetric set of angle offsets (``Degree_Manip`` / ``Degree_Interval``).
        The offsets are stored in radians in ``self.l_angles_to_test``.
        """
        self.l_angles_to_test = [0.0]
        self.d_increments = 0
        d_degree_manipulation = params['d_degree_manipulation']
        d_degree_interval = params['d_degree_interval']
        if d_degree_manipulation > 0.0 and d_degree_interval > 0.0:
            # Calculate the increment
            self.d_increments = int(d_degree_manipulation / (2.0 * d_degree_interval))

            # Test if the increment should be considered
            if self.d_increments > 0:
                for d in range(1, self.d_increments + 1):
                    for s in range(-1, 2, 2):
                        self.l_angles_to_test.append(s * d * d_degree_interval)

        LOG.info('With Degree_Manip=' + str(d_degree_manipulation) + '  and  Degree_Interval=' + str(d_degree_interval) + '\n  Angles to evaluate= ' + str(self.l_angles_to_test))
        self.l_angles_to_test = np.multiply(self.l_angles_to_test, math.pi / 180.0)
        LOG.info('  Angles (radians) to evaluate= ' + str(self.l_angles_to_test))

    def has_angles_to_test(self) -> bool:
        """Return ``True`` if angle-search will test more than one direction."""
        return len(self.l_angles_to_test) > 0

    def test_angles_and_reset_cross_section(self, i_row_cell, i_column_cell):
        """Search candidate angles and re-sample the cross-section at the best one.

        Parameters
        ----------
        i_row_cell, i_column_cell : int
            Row/column indices of the stream cell (in the padded raster grid).
        """
        d_precompute_angles = np.pi / self.i_precompute_angles
        d_xs_direction = self.get_best_xsection_angle(d_precompute_angles)

        # Now Pull the Cross-Section again with the new angle
        if d_xs_direction > np.pi:
            i_precompute_angle_closest = int(round((d_xs_direction-np.pi) / d_precompute_angles))
        else:
            i_precompute_angle_closest = int(round(d_xs_direction / d_precompute_angles))

        self.set_cross_section(i_row_cell, i_column_cell, i_precompute_angle_closest, d_xs_direction)

    def set_boundary_extents(self, i_boundary_number: int, nrows: int, ncols: int):
        """Set index bounds used by subsequent neighborhood searches.

        ARC pads rasters by ``i_boundary_number``. This method records the
        inclusive bounds for valid sampling indices within the padded arrays so
        later routines can quickly reject samples that fall outside the usable
        domain.

        Parameters
        ----------
        i_boundary_number : int
            Number of padded cells around the original raster.
        nrows, ncols : int
            Dimensions of the *unpadded* raster.
        """
        self.i_boundary_number = i_boundary_number
        self.i_row_bottom = i_boundary_number
        self.i_row_top = nrows + i_boundary_number - 1
        self.i_column_bottom = i_boundary_number
        self.i_column_top = ncols + i_boundary_number - 1
    
    def set_cross_section(self, row: int, col: int, i_precompute_angle_closest: int, d_xs_direction: float):
        """Sample a two-sided cross-section centered on ``(row, col)``.

        The sampled profiles and land-cover arrays are written in-place to the
        preallocated ``da_xs_profile*``/``ia_lc_xs*`` arrays, and the number of
        valid sampled ordinates for each side is recorded in ``xs1_n`` and
        ``xs2_n``.

        Parameters
        ----------
        row, col : int
            Center cell location in the padded raster arrays.
        i_precompute_angle_closest : int
            Index of the nearest precomputed angle (see
            :meth:`create_cross_section_ordinates`).
        d_xs_direction : float
            Cross-section orientation in radians.
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
        """Shift the cross-section center to the local thalweg candidate.

        The "low spot" adjustment searches within ``i_low_spot_range`` cells
        along the sampled profiles for a lower elevation point and recenters the
        cross-section on that location. This helps avoid sampling cross-sections
        that are slightly offset from the true channel thalweg due to raster
        discretization.

        Parameters
        ----------
        i_low_spot_range : int
            Number of ordinates on each side to consider when searching for the
            local low point.
        """
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
        """Return the (row, col) indices of the cross-section center cell."""
        return self.ia_xc_row1_index_main[0], self.ia_xc_column1_index_main[0]
    
    def get_thalweg(self):
        """Return the thalweg elevation at the cross-section center cell."""
        return self.da_xs_profile1[0]

    def get_best_xsection_angle(self, d_precompute_angles: float):
        """Choose the candidate angle producing the narrowest test top width.

        The current implementation evaluates candidate angle offsets from
        ``self.l_angles_to_test`` and selects the angle that minimizes top width
        at a small test depth above the thalweg. Candidate angles that pass
        through stream cells away from the thalweg center are deprioritized so
        ARC prefers cross sections that cut across the channel rather than
        follow the stream network.

        Parameters
        ----------
        d_precompute_angles : float
            Angular spacing (radians) of the precomputed angle table.

        Returns
        -------
        float
            Selected cross-section direction in radians.
        """
        d_test_depth = 0.5
        d_shortest_tw_angle = 0.0
        d_t_test = np.inf
        best_stream_hit_count = np.iinfo(np.int64).max
        tested_indices: set[int] = set()

        def _evaluate_angle(d_xs_angle_use: float, i_precompute_angle_closest: int) -> None:
            nonlocal d_shortest_tw_angle
            nonlocal d_t_test
            nonlocal best_stream_hit_count

            tested_indices.add(int(i_precompute_angle_closest))
            self.set_cross_section(self.row, self.col, i_precompute_angle_closest, d_xs_angle_use)
            if not self.is_valid():
                return

            stream_hit_count = self._count_offcenter_stream_cells()
            d_wse = self.get_thalweg() + d_test_depth
            top_width = self.calculate_top_width_of_wse(d_wse)
            if not np.isfinite(top_width) or top_width <= 0.0:
                top_width = np.inf

            if (
                stream_hit_count < best_stream_hit_count
                or (
                    stream_hit_count == best_stream_hit_count
                    and top_width < d_t_test
                )
            ):
                best_stream_hit_count = int(stream_hit_count)
                d_t_test = top_width
                d_shortest_tw_angle = d_xs_angle_use

        # Loop through the angles to test
        for d_entry_angle_adjustment in self.l_angles_to_test:
            # Ensure angle is between 0 and pi
            d_xs_angle_use = (self.d_xs_direction + d_entry_angle_adjustment) % np.pi
        
            #We now precompute the cross-section ordinates
            i_precompute_angle_closest = round(d_xs_angle_use / d_precompute_angles)
            _evaluate_angle(d_xs_angle_use, i_precompute_angle_closest)
            if best_stream_hit_count == 0:
                break

        if best_stream_hit_count > 0:
            for i_precompute_angle_closest in range(self.i_precompute_angles + 1):
                if i_precompute_angle_closest in tested_indices:
                    continue
                d_xs_angle_use = float(i_precompute_angle_closest) * float(d_precompute_angles)
                _evaluate_angle(d_xs_angle_use, i_precompute_angle_closest)
                if best_stream_hit_count == 0:
                    break

        return d_shortest_tw_angle

    def _count_offcenter_stream_cells(self) -> int:
        """Count sampled stream cells beyond the thalweg for the current angle."""
        if self.dm_stream is None:
            return 0

        stream_hit_count = 0
        if self.xs1_n > 1:
            side_1_stream_values = self.dm_stream[
                self.ia_xc_row1_index_main[1:self.xs1_n],
                self.ia_xc_column1_index_main[1:self.xs1_n],
            ]
            stream_hit_count += int(np.count_nonzero(side_1_stream_values > 0))

        if self.xs2_n > 1:
            side_2_stream_values = self.dm_stream[
                self.ia_xc_row2_index_main[1:self.xs2_n],
                self.ia_xc_column2_index_main[1:self.xs2_n],
            ]
            stream_hit_count += int(np.count_nonzero(side_2_stream_values > 0))

        return stream_hit_count
    
    
    def calculate_top_width_of_wse(self, d_wse: float):
        """Compute total top width at a given water-surface elevation (WSE)."""
        return (
            _calculate_side_top_width(d_wse, self.da_xs_profile1[:self.xs1_n], self.d_ordinate_dist) +
            _calculate_side_top_width(d_wse, self.da_xs_profile2[:self.xs2_n], self.d_ordinate_dist)
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
            
        return _find_bank_inflection_point_helper(da_xs_smooth, i_cross_section_number, self.d_ordinate_dist)

    def _is_valid_bathymetry_target(self, value: float | None) -> bool:
        """Return ``True`` when an optional bathymetry target can be used.

        The drainage-area bathymetry workflow passes precomputed target depth
        and width values into the cross-section logic. This helper centralizes
        the validity check so the main bathymetry routines can focus on their
        search order and hydraulic decisions.
        """
        return value is not None and np.isfinite(value) and value > 0.0

    def _is_single_cell_bathymetry_feasible(self, d_bathy_target_width: float | None) -> bool:
        """Return ``True`` when ARC should accept a one-cell channel.

        The one-cell triangular fallback is only accepted when 
        a drainage-area width prior indicates that the expected
        bankfull width is no greater than two times the cross-section sample
        spacing. This keeps one-cell bathymetry limited to intentionally
        unresolved small channels instead of letting broader channels collapse
        into the triangle fallback by accident.
        """
        if not self._is_valid_bathymetry_target(d_bathy_target_width):
            return False
        if self.xs1_n < 2 or self.xs2_n < 2:
            return False

        if float(self.d_ordinate_dist) < 15.0:
            return float(d_bathy_target_width) <= (2.0 * float(self.d_ordinate_dist))
        elif float(self.d_ordinate_dist) > 15.0:
            return float(d_bathy_target_width) <= (1.0 * float(self.d_ordinate_dist))

    def _find_bank_by_target_width(self, target_width: float) -> tuple[int, int, int]:
        """Infer bank indices from an externally estimated bankfull width.

        ARC still prefers direct evidence from the DEM or land cover. This
        helper is only used after those bank searches fail and a drainage-area
        power law supplied an estimated bankfull width. The target width is
        distributed across both sides of the sampled profile while respecting
        the amount of profile actually available on each side.
        """
        if not self._is_valid_bathymetry_target(target_width):
            return 0, 0, 1
        if self.xs1_n < 2 or self.xs2_n < 2:
            return 0, 0, 1

        half_width = 0.5 * target_width
        left_available = max(self.xs1_n - 1, 0) * self.d_ordinate_dist
        right_available = max(self.xs2_n - 1, 0) * self.d_ordinate_dist
        left_target = min(half_width, left_available)
        right_target = min(half_width, right_available)

        remaining_width = max(target_width - (left_target + right_target), 0.0)
        left_capacity = max(left_available - left_target, 0.0)
        right_capacity = max(right_available - right_target, 0.0)
        total_capacity = left_capacity + right_capacity
        if remaining_width > 0.0 and total_capacity > 0.0:
            left_target += remaining_width * (left_capacity / total_capacity)
            right_target += remaining_width * (right_capacity / total_capacity)

        i_bank_1_index = min(max(int(round(left_target / self.d_ordinate_dist)), 1), self.xs1_n - 1)
        i_bank_2_index = min(max(int(round(right_target / self.d_ordinate_dist)), 1), self.xs2_n - 1)
        i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
        if i_total_bank_cells <= 1:
            return 0, 0, 1
        return i_bank_1_index, i_bank_2_index, i_total_bank_cells

    def _clamp_bank_index(self, index: int, xs_n: int) -> int:
        """Clamp a bank index to the valid sampled range for one profile side."""
        if xs_n <= 0:
            return 0
        return min(max(int(index), 0), xs_n - 1)

    def _build_bank_search_result(
        self,
        function_used: str | None,
        i_bank_1_index: int,
        i_bank_2_index: int,
        bank_elev_1: float | None = None,
        bank_elev_2: float | None = None,
        allow_single_cell: bool = False,
    ) -> dict:
        """Normalize bank-search output for staged ARC preprocessing.

        ARC now samples and caches every stream-cell cross section first, then
        performs bank detection across the cached sections, and only after that
        applies bathymetry. This helper packages the bank-search output so the
        bathymetry stage can replay those bank choices without rerunning the
        full search hierarchy.
        """
        i_bank_1_index = self._clamp_bank_index(i_bank_1_index, self.xs1_n)
        i_bank_2_index = self._clamp_bank_index(i_bank_2_index, self.xs2_n)
        raw_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
        is_valid = function_used is not None and (
            raw_total_bank_cells > 1 or (allow_single_cell and raw_total_bank_cells == 1)
        )
        if bank_elev_1 is None:
            bank_elev_1 = float(self.da_xs_profile1[i_bank_1_index]) if self.xs1_n > 0 else float("nan")
        if bank_elev_2 is None:
            bank_elev_2 = float(self.da_xs_profile2[i_bank_2_index]) if self.xs2_n > 0 else float("nan")

        return {
            "function_used": function_used,
            "i_bank_1_index": int(i_bank_1_index),
            "i_bank_2_index": int(i_bank_2_index),
            "i_total_bank_cells": int(raw_total_bank_cells if raw_total_bank_cells > 1 else 1),
            "bank_elev_1": float(bank_elev_1),
            "bank_elev_2": float(bank_elev_2),
            "is_valid": bool(is_valid),
        }

    def get_top_width_from_bank_search_result(self, bank_search_result: dict | None) -> float:
        """Return the bank-to-bank top width represented by a bank result.

        ARC stores local bank placement as side-specific indices. For the
        staged reach filters, the corresponding top width is therefore the
        resolved number of bankfull cells multiplied by the cross-section
        sample spacing.
        """
        if not isinstance(bank_search_result, dict):
            return float("nan")
        if not bool(bank_search_result.get("is_valid", False)):
            return float("nan")

        i_total_bank_cells = int(bank_search_result.get("i_total_bank_cells", 1))
        if i_total_bank_cells < 1:
            return float("nan")
        return float(i_total_bank_cells) * float(self.d_ordinate_dist)

    def build_bank_search_result_from_target_width(
        self,
        existing_bank_search_result: dict | None,
        target_width: float,
        function_used: str | None,
    ) -> dict:
        """Rebuild bank indices so the sampled section matches a target width.

        This helper is used by reach-scale bank-width filters that decide a
        local top width is an outlier for its reach. The replacement geometry
        keeps the sampled cross section itself, but reassigns bank indices so
        the resulting top width matches the supplied reach-median width as
        closely as the sampled resolution allows.
        """
        local_result = dict(existing_bank_search_result) if isinstance(existing_bank_search_result, dict) else {}
        original_top_width = self.get_top_width_from_bank_search_result(local_result)

        allow_single_cell = False
        if (
            self._is_valid_bathymetry_target(target_width)
            and target_width <= float(self.d_ordinate_dist)
            and self.xs1_n > 1
            and self.xs2_n > 1
        ):
            i_bank_1_index = 1
            i_bank_2_index = 1
            allow_single_cell = True
        else:
            i_bank_1_index, i_bank_2_index, i_total_bank_cells = self._find_bank_by_target_width(target_width)
            if i_total_bank_cells <= 1:
                return local_result

        result = self._build_bank_search_result(
            function_used if function_used is not None else "filter_bank_width_to_reach_median",
            i_bank_1_index,
            i_bank_2_index,
            allow_single_cell=allow_single_cell,
        )
        result["reach_top_width_filter_applied"] = True
        result["reach_top_width_filter_original_function_used"] = local_result.get("function_used")
        result["reach_top_width_filter_original_i_bank_1_index"] = int(local_result.get("i_bank_1_index", 0))
        result["reach_top_width_filter_original_i_bank_2_index"] = int(local_result.get("i_bank_2_index", 0))
        result["reach_top_width_filter_original_top_width"] = float(original_top_width)
        result["reach_top_width_filter_target_top_width"] = float(target_width)
        return result

    def get_wse_or_lc_bank_search_result(
        self,
        d_bathy_target_width: float | None = None,
    ) -> dict:
        """Find bathymetry banks for the WSE/land-cover workflow.

        This method exposes the same bank-search hierarchy used by
        :meth:`Calculate_Bathymetry_Based_on_WSE_or_LC`, but without applying
        bathymetry. ARC uses it during the prepass that identifies banks for
        every sampled stream cell before the channel geometry is modified. A
        one-cell feasibility gate based on adjacent-cell relief and the
        drainage-area width prior runs first; only sections that fail that
        gate continue into the wider bank-search hierarchy.
        """
        if not self.is_valid():
            return self._build_bank_search_result(None, 0, 0)

        if self._is_single_cell_bathymetry_feasible(d_bathy_target_width):
            return self._build_bank_search_result(
                "find_single_cell_bathymetry_by_target_width",
                1,
                1,
                allow_single_cell=True,
            )

        function_used = None
        i_bank_1_index = 0
        i_bank_2_index = 0
        i_total_bank_cells = 0

        if self.b_FindBanksBasedOnLandCover:
            (_, i_bank_1_index, i_bank_2_index) = self._find_wse_and_banks_by_lc()
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells >= 1:
                function_used = "find_wse_and_banks_by_lc"

        if i_total_bank_cells < 1:
            (i_bank_1_index, i_bank_2_index) = _find_bank_using_width_to_depth_ratio(
                self.get_thalweg(),
                self.da_xs_profile1,
                self.da_xs_profile2,
                self.xs1_n,
                self.xs2_n,
                self.d_ordinate_dist,
            )
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells > 1:
                function_used = "find_bank_using_width_to_depth_ratio"

        if i_total_bank_cells < 1:
            i_bank_1_index = self._find_bank(self.da_xs_profile1, self.xs1_n, wse=True)
            i_bank_2_index = self._find_bank(self.da_xs_profile2, self.xs2_n, wse=True)
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells > 1:
                function_used = "find_wse_and_banks_by_flat_water"

        return self._build_bank_search_result(function_used, i_bank_1_index, i_bank_2_index)

    def get_bank_elevation_search_result(
        self,
        d_bathy_target_width: float | None = None,
    ) -> dict:
        """Find bathymetry banks for the bank-elevation workflow.

        This mirrors the bank-search hierarchy used by
        :meth:`Calculate_Bathymetry_Based_on_RiverBank_Elevations`, including
        the bank-elevation values needed to compute bankfull elevation once the
        bathymetry stage begins. As in the WSE workflow, ARC first tests
        whether the geometry and drainage-area width prior support a one-cell
        triangular channel before it attempts any wider bank-detection method.
        """
        if not self.is_valid():
            return self._build_bank_search_result(None, 0, 0)

        if self._is_single_cell_bathymetry_feasible(d_bathy_target_width):
            return self._build_bank_search_result(
                "find_single_cell_bathymetry_by_target_width",
                1,
                1,
                bank_elev_1=float(self.da_xs_profile1[1]),
                bank_elev_2=float(self.da_xs_profile2[1]),
                allow_single_cell=True,
            )

        function_used = None
        i_landcover_for_bathy = self.ia_lc_xs1[0]
        i_bank_1_index = 0
        i_bank_2_index = 0
        i_total_bank_cells = 0
        bank_elev_1 = 0.0
        bank_elev_2 = 0.0

        if self.b_FindBanksBasedOnLandCover:
            if self.xs1_n >= 1 and i_landcover_for_bathy == self.i_lc_water_value:
                bank_elev_1 = 0.0
                for i in range(1, self.xs1_n):
                    if self.ia_lc_xs1[i] != self.i_lc_water_value:
                        bank_elev_1 = self.da_xs_profile1[i]
                        i_bank_1_index = i - 1
                        break
            if self.xs2_n >= 1 and i_landcover_for_bathy == self.i_lc_water_value:
                bank_elev_2 = 0.0
                for i in range(1, self.xs2_n):
                    if self.ia_lc_xs2[i] != self.i_lc_water_value:
                        bank_elev_2 = self.da_xs_profile2[i]
                        i_bank_2_index = i - 1
                        break
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells > 1:
                function_used = "find_wse_and_banks_by_lc"
            else:
                i_total_bank_cells = 1

        if i_total_bank_cells <= 1:
            (i_bank_1_index, i_bank_2_index) = _find_bank_using_width_to_depth_ratio(
                self.get_thalweg(),
                self.da_xs_profile1,
                self.da_xs_profile2,
                self.xs1_n,
                self.xs2_n,
                self.d_ordinate_dist,
            )
            bank_elev_1 = self.da_xs_profile1[i_bank_1_index]
            bank_elev_2 = self.da_xs_profile2[i_bank_2_index]
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells > 1:
                function_used = "find_bank_using_width_to_depth_ratio"
            else:
                i_total_bank_cells = 1
        
        if i_total_bank_cells <= 1:
            i_bank_1_index = self._find_bank(self.da_xs_profile1, self.xs1_n)
            i_bank_2_index = self._find_bank(self.da_xs_profile2, self.xs2_n)
            bank_elev_1 = self.da_xs_profile1[i_bank_1_index]
            bank_elev_2 = self.da_xs_profile2[i_bank_2_index]
            i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
            if i_total_bank_cells > 1:
                function_used = "find_wse_and_banks_by_flat_water"

        return self._build_bank_search_result(
            function_used,
            i_bank_1_index,
            i_bank_2_index,
            bank_elev_1=bank_elev_1,
            bank_elev_2=bank_elev_2,
        )

    def get_representative_bank_indices(self) -> tuple[int, int]:
        """Estimate bank indices using ARC's DEM/land-cover search hierarchy.

        ARC's representative cross-section export is now built from median
        hydraulic databases rather than from sampled DEM profiles. This helper
        remains available for profile diagnostics and for any legacy workflows
        that still need bank locations from the sampled section itself.
        """
        if not self.is_valid():
            return 0, 0

        if self.b_FindBanksBasedOnLandCover:
            (_, i_bank_1_index, i_bank_2_index) = self._find_wse_and_banks_by_lc()
            if i_bank_1_index + i_bank_2_index - 1 > 1:
                return (
                    self._clamp_bank_index(i_bank_1_index, self.xs1_n),
                    self._clamp_bank_index(i_bank_2_index, self.xs2_n),
                )

        if self.b_bathy_use_banks:
            i_bank_1_index = self._find_bank(self.da_xs_profile1, self.xs1_n)
            i_bank_2_index = self._find_bank(self.da_xs_profile2, self.xs2_n)
        else:
            i_bank_1_index = self._find_bank(self.da_xs_profile1, self.xs1_n, wse=True)
            i_bank_2_index = self._find_bank(self.da_xs_profile2, self.xs2_n, wse=True)

        if i_bank_1_index + i_bank_2_index - 1 > 1:
            return (
                self._clamp_bank_index(i_bank_1_index, self.xs1_n),
                self._clamp_bank_index(i_bank_2_index, self.xs2_n),
            )

        i_bank_1_index, i_bank_2_index = _find_bank_using_width_to_depth_ratio(
            self.get_thalweg(),
            self.da_xs_profile1,
            self.da_xs_profile2,
            self.xs1_n,
            self.xs2_n,
            self.d_ordinate_dist,
        )
        if i_bank_1_index + i_bank_2_index - 1 > 1:
            return (
                self._clamp_bank_index(i_bank_1_index, self.xs1_n),
                self._clamp_bank_index(i_bank_2_index, self.xs2_n),
            )

        return (
            self._clamp_bank_index(i_bank_1_index, self.xs1_n),
            self._clamp_bank_index(i_bank_2_index, self.xs2_n),
        )

    def get_representative_inflect_curve(self) -> np.ndarray:
        """Return the INFLECT-style second-derivative width curve for this section.

        The representative cross-section export uses the reach-average of these
        curves to define the flood-terrace depth for each reach. ARC then
        rebuilds representative hydraulics every 0.10 m above the sampled
        thalwegs up to that terrace depth. The curve is also useful for
        diagnostic analysis when comparing the representative geometry against
        the sampled DEM sections.
        """
        if not self.is_valid():
            return np.empty(0, dtype=np.float64)

        (_, curve) = self.get_representative_inflect_curve_with_depths()
        return curve

    def get_representative_inflect_curve_with_depths(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the INFLECT curve together with its sampled depth axis.

        The temporary reach-INFLECT plotting workflow needs the actual depth
        samples used to build each `d2W/dy2` curve because the staged
        bank-height depth iteration no longer follows the older fixed 0.10 m
        spacing. ARC therefore exposes both the curve and the aligned depth
        values here so diagnostic plots can render the adjusted geometry
        accurately.
        """
        if not self.is_valid():
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        return _calculate_inflect_curve_with_depths(
            self.get_thalweg(),
            self.da_xs_profile1,
            self.da_xs_profile2,
            self.xs1_n,
            self.xs2_n,
            self.d_ordinate_dist,
        )

    def _find_bank_using_reach_scale_inflection(self) -> tuple[int, int]:
        """Convert a reach-average INFLECT bank depth into local bank indices.

        ARC precomputes one representative INFLECT bank depth for each reach by
        averaging the ``d2W_dy2`` signal and its aligned depth axis across the
        reach. This helper maps that shared depth back onto the current sampled
        cross section by converting the reach-scale depth into a
        water-surface elevation and then measuring the wetted top-width on
        each side at that elevation.
        """
        if not self.is_valid() or self.reach_scale_inflect_bank_depth < 0.0:
            return 0, 0

        bank_depth = float(self.reach_scale_inflect_bank_depth)
        wse = self.get_thalweg() + bank_depth
        t1 = _calculate_side_top_width(wse, self.da_xs_profile1[:self.xs1_n], self.d_ordinate_dist)
        t2 = _calculate_side_top_width(wse, self.da_xs_profile2[:self.xs2_n], self.d_ordinate_dist)

        i_bank_1_index = 0 if self.xs1_n <= 1 else min(max(int(round(t1 / self.d_ordinate_dist)), 0), self.xs1_n - 1)
        i_bank_2_index = 0 if self.xs2_n <= 1 else min(max(int(round(t2 / self.d_ordinate_dist)), 0), self.xs2_n - 1)
        return i_bank_1_index, i_bank_2_index

    def _find_bank_indices_from_elevation(self, bank_elevation: float) -> tuple[int, int, int]:
        """Convert a target bank elevation into local bank indices and width.

        ARC's staged bank-elevation smoothing pass creates one longitudinally
        smoothed bank elevation for every sampled cross section in a reach.
        This helper converts that elevation back into side-specific bank
        indices by recomputing top width at the target elevation.
        """
        if not self.is_valid() or not np.isfinite(bank_elevation):
            return 0, 0, 1

        bank_elevation = max(float(bank_elevation), float(self.get_thalweg()))
        t1 = _calculate_side_top_width(bank_elevation, self.da_xs_profile1[:self.xs1_n], self.d_ordinate_dist)
        t2 = _calculate_side_top_width(bank_elevation, self.da_xs_profile2[:self.xs2_n], self.d_ordinate_dist)

        i_bank_1_index = 0 if self.xs1_n <= 1 else self._clamp_bank_index(int(round(t1 / self.d_ordinate_dist)), self.xs1_n)
        i_bank_2_index = 0 if self.xs2_n <= 1 else self._clamp_bank_index(int(round(t2 / self.d_ordinate_dist)), self.xs2_n)
        i_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
        if i_total_bank_cells <= 1:
            return 0, 0, 1
        return i_bank_1_index, i_bank_2_index, i_total_bank_cells

    def build_bank_search_result_from_smoothed_elevation(
        self,
        existing_bank_search_result: dict | None,
        bank_elevation: float,
        function_used: str | None,
    ) -> dict:
        """Overlay a smoothed bathymetry elevation onto local bank geometry.

        ARC first determines local bank indices and widths from the sampled
        cross section. The reach smoother then updates only the vertical
        bathymetry control elevation while preserving those locally determined
        bank indices and top widths. The returned ``function_used`` value is
        the dedicated staged-smoothing tag so downstream bathymetry routines
        can distinguish the smoothed-elevation path from the original local
        bank-search method while still retaining the local source method and
        geometry for diagnostics.
        """
        local_result = dict(existing_bank_search_result) if isinstance(existing_bank_search_result, dict) else {}
        local_function_used = local_result.get("function_used")
        i_bank_1_index = self._clamp_bank_index(int(local_result.get("i_bank_1_index", 0)), self.xs1_n)
        i_bank_2_index = self._clamp_bank_index(int(local_result.get("i_bank_2_index", 0)), self.xs2_n)
        raw_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
        i_total_bank_cells = int(raw_total_bank_cells if raw_total_bank_cells > 1 else 1)
        local_is_valid = bool(local_result.get("is_valid", False))
        local_bank_elev_1 = float(local_result.get("bank_elev_1", self.da_xs_profile1[i_bank_1_index] if self.xs1_n > 0 else np.nan))
        local_bank_elev_2 = float(local_result.get("bank_elev_2", self.da_xs_profile2[i_bank_2_index] if self.xs2_n > 0 else np.nan))

        result = dict(local_result)
        result["function_used"] = "find_bank_using_smoothed_reach_bank_elevation"
        result["i_bank_1_index"] = int(i_bank_1_index)
        result["i_bank_2_index"] = int(i_bank_2_index)
        result["i_total_bank_cells"] = i_total_bank_cells
        result["bank_elev_1"] = float(bank_elevation)
        result["bank_elev_2"] = float(bank_elevation)
        result["is_valid"] = local_is_valid
        result["smoothed_bank_elevation"] = float(bank_elevation)
        result["smoothed_bank_source_function"] = function_used
        result["local_function_used"] = local_function_used
        result["local_i_bank_1_index"] = int(i_bank_1_index)
        result["local_i_bank_2_index"] = int(i_bank_2_index)
        result["local_i_total_bank_cells"] = int(i_total_bank_cells)
        result["local_bank_elev_1"] = float(local_bank_elev_1)
        result["local_bank_elev_2"] = float(local_bank_elev_2)
        result["local_total_bank_width"] = float(i_total_bank_cells) * float(self.d_ordinate_dist)
        return result

    def Calculate_Bathymetry_Based_on_WSE_or_LC(
        self,
        output_bathymetry: np.ndarray,
        bank_search_result: dict | None = None,
    ):
        """Burn a staged depth using the previously selected WSE/LC banks.

        This method performs no bank search and no hydraulic-depth solve.
        ``bathymetry_depth`` must already be present in ``bank_search_result``.
        """

        (
            _function_used,
            i_bank_1_index,
            i_bank_2_index,
            i_total_bank_cells,
            _bank_elev_1,
            _bank_elev_2,
            _smoothed_bank_elevation,
        ) = self._get_precomputed_bathymetry_bank_result(
            bank_search_result,
        )
        if not isinstance(bank_search_result, dict):
            return 0, 0, 1, 0.0, self.get_thalweg()
        should_apply = bool(
            bank_search_result.get(
                "bathymetry_should_apply",
                "bathymetry_depth" in bank_search_result,
            )
        )
        if not should_apply:
            return 0, 0, 1, 0.0, self.get_thalweg()

        d_total_bank_dist = i_total_bank_cells * self.d_ordinate_dist
        d_h_dist = self.d_bathymetry_trapzoid_height * d_total_bank_dist
        d_trap_base = d_total_bank_dist - 2.0 * d_h_dist
        d_y_depth = float(bank_search_result.get("bathymetry_depth", 0.0))
        d_y_bathy = self.get_thalweg()

        # Match the former combined solve/burn routine: non-finite and
        # unrealistic depths were rejected, while an exact zero was allowed.
        if not np.isfinite(d_y_depth) or d_y_depth >= 25.0:
            return 0, 0, 1, 0.0, self.get_thalweg()

        if i_total_bank_cells > 1:
            d_y_bathy = self.get_thalweg() - d_y_depth
            _adjust_one_side_for_bathymetry(
                i_bank_1_index,
                d_total_bank_dist,
                d_trap_base,
                d_h_dist,
                self.ia_xc_row1_index_main,
                self.ia_xc_column1_index_main,
                self.da_xs_profile1,
                output_bathymetry,
                0.0,
                d_y_bathy,
                d_y_depth,
                self.d_ordinate_dist,
                self.dm_elevation,
                self.b_bathy_use_banks,
            )
            _adjust_one_side_for_bathymetry(
                i_bank_2_index,
                d_total_bank_dist,
                d_trap_base,
                d_h_dist,
                self.ia_xc_row2_index_main,
                self.ia_xc_column2_index_main,
                self.da_xs_profile2,
                output_bathymetry,
                0.0,
                d_y_bathy,
                d_y_depth,
                self.d_ordinate_dist,
                self.dm_elevation,
                self.b_bathy_use_banks,
            )
        elif i_total_bank_cells == 1:
            if self.xs1_n <= 1 or self.xs2_n <= 1:
                return 0, 0, 1, 0.0, self.get_thalweg()
            i_bank_1_index = 1
            i_bank_2_index = 1
            d_y_bathy = self.get_thalweg() - d_y_depth
            self.da_xs_profile1[0] = d_y_bathy
            output_bathymetry[self.ia_xc_row1_index_main[0], self.ia_xc_column1_index_main[0]] = self.da_xs_profile1[0]
            self.da_xs_profile2[0] = d_y_bathy
            output_bathymetry[self.ia_xc_row2_index_main[0], self.ia_xc_column2_index_main[0]] = self.da_xs_profile2[0]

        return i_bank_1_index, i_bank_2_index, i_total_bank_cells, d_y_depth, d_y_bathy
    
    def set_mannings_n_values(self, dm_manning_n_raster: np.ndarray):
        """Sample Manning's n along both sides of the current cross-section.

        Parameters
        ----------
        dm_manning_n_raster : numpy.ndarray
            Raster of Manning's n values aligned to the padded DEM grid.
        """
        self.mannings_n1 = dm_manning_n_raster[self.ia_xc_row1_index_main[:self.xs1_n], self.ia_xc_column1_index_main[:self.xs1_n]]
        self.mannings_n2 = dm_manning_n_raster[self.ia_xc_row2_index_main[:self.xs2_n], self.ia_xc_column2_index_main[:self.xs2_n]]
    
    def get_flood_increment_args(self):
        """Return the tuple of arrays needed by flood-increment calculations."""
        return self.da_xs_profile1, self.xs1_n, self.mannings_n1, self.da_xs_profile2, self.xs2_n, self.mannings_n2, self.d_ordinate_dist

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
        """Return the tuple of arrays needed by :func:`calculate_discharge_from_wse`."""
        return self.da_xs_profile1, self.xs1_n, self.mannings_n1, self.da_xs_profile2, self.xs2_n, self.mannings_n2, self.d_ordinate_dist

    def _get_precomputed_bathymetry_bank_result(
        self,
        bank_search_result: dict | None,
    ) -> tuple[str | None, int, int, int, float, float, float]:
        """Normalize the staged bank result used by the bathymetry burn step.

        ARC now completes bank detection before either bathymetry routine
        runs. These routines therefore read bank geometry directly from the
        supplied ``bank_search_result`` instead of rerunning the bank-search
        hierarchy from the current cross-section profile.
        """
        if not isinstance(bank_search_result, dict):
            return None, 0, 0, 1, float(self.da_xs_profile1[0]), float(self.da_xs_profile2[0]), float("nan")

        function_used = bank_search_result.get("function_used")
        i_bank_1_index = self._clamp_bank_index(int(bank_search_result.get("i_bank_1_index", 0)), self.xs1_n)
        i_bank_2_index = self._clamp_bank_index(int(bank_search_result.get("i_bank_2_index", 0)), self.xs2_n)
        raw_total_bank_cells = i_bank_1_index + i_bank_2_index - 1
        i_total_bank_cells = int(raw_total_bank_cells if raw_total_bank_cells > 1 else 1)
        bank_elev_1 = float(
            bank_search_result.get(
                "bank_elev_1",
                self.da_xs_profile1[i_bank_1_index] if self.xs1_n > 0 else np.nan,
            )
        )
        bank_elev_2 = float(
            bank_search_result.get(
                "bank_elev_2",
                self.da_xs_profile2[i_bank_2_index] if self.xs2_n > 0 else np.nan,
            )
        )
        smoothed_bank_elevation = float(bank_search_result.get("smoothed_bank_elevation", np.nan))

        if not bool(bank_search_result.get("is_valid", False)):
            return None, 0, 0, 1, bank_elev_1, bank_elev_2, smoothed_bank_elevation

        return (
            function_used,
            i_bank_1_index,
            i_bank_2_index,
            i_total_bank_cells,
            bank_elev_1,
            bank_elev_2,
            smoothed_bank_elevation,
        )
    
    def _compute_bank_bathymetry_geometry(
        self,
        i_total_bank_cells,
        i_bank_1_index,
        i_bank_2_index,
        d_bankfull_elevation,
    ):
        """Compute the trapezoid dimensions implied by the detected banks."""
        d_side1_dist = self._calc_side_distance(self.da_xs_profile1, i_bank_1_index, d_bankfull_elevation)
        d_side2_dist = self._calc_side_distance(self.da_xs_profile2, i_bank_2_index, d_bankfull_elevation)
        d_total_bank_dist = i_total_bank_cells * self.d_ordinate_dist + d_side1_dist + d_side2_dist
        d_h_dist = self.d_bathymetry_trapzoid_height * d_total_bank_dist
        d_trap_base = d_total_bank_dist - 2.0 * d_h_dist
        return d_side1_dist, d_side2_dist, d_total_bank_dist, d_h_dist, d_trap_base

    def calculate_hydraulic_bathymetry_depth(
        self,
        d_q_baseflow: float,
        d_slope_use: float,
        bank_search_result: dict | None,
    ) -> float:
        """Solve hydraulic depth from baseflow after bank detection is complete.

        The method reads only the cached cross section and its staged bank
        result. It does not alter either cross-section profile or the output
        bathymetry raster, allowing every depth to be computed before any
        neighboring stream cell is burned.
        """
        if d_q_baseflow <= 0.0:
            return 0.0

        (
            _function_used,
            i_bank_1_index,
            i_bank_2_index,
            i_total_bank_cells,
            _bank_elev_1,
            _bank_elev_2,
            smoothed_bank_elevation,
        ) = self._get_precomputed_bathymetry_bank_result(bank_search_result)

        if i_total_bank_cells == 1:
            if self.xs1_n <= 1 or self.xs2_n <= 1:
                return 0.0
            if self.b_bathy_use_banks:
                # The bank-elevation method uses its smoothed elevation as the
                # triangular section's center and both edge elevations.
                stream_elevation = smoothed_bank_elevation
                left_bank_elevation = smoothed_bank_elevation
                right_bank_elevation = smoothed_bank_elevation
            else:
                stream_elevation = self.da_xs_profile1[0]
                left_bank_elevation = self.da_xs_profile1[1]
                right_bank_elevation = self.da_xs_profile2[1]
            d_y_depth = find_depth_of_bathymetry_triangle(
                d_q_baseflow,
                self.d_ordinate_dist,
                stream_elevation,
                left_bank_elevation,
                right_bank_elevation,
                d_slope_use,
                0.03,
            )
        else:
            if self.b_bathy_use_banks:
                (
                    _d_side1_dist,
                    _d_side2_dist,
                    d_total_bank_dist,
                    _d_h_dist,
                    d_trap_base,
                ) = self._compute_bank_bathymetry_geometry(
                    i_total_bank_cells,
                    i_bank_1_index,
                    i_bank_2_index,
                    smoothed_bank_elevation,
                )
            else:
                d_total_bank_dist = i_total_bank_cells * self.d_ordinate_dist
                d_h_dist = self.d_bathymetry_trapzoid_height * d_total_bank_dist
                d_trap_base = d_total_bank_dist - 2.0 * d_h_dist

            d_y_depth = find_depth_of_bathymetry(
                d_q_baseflow,
                d_trap_base,
                d_total_bank_dist,
                d_slope_use,
                0.03,
            )

        # Do not filter the solver result here. The old combined routine
        # performed these checks immediately before burning; leaving them to
        # the burn method preserves those exact numerical acceptance rules.
        return float(d_y_depth)

    def Calculate_Bathymetry_Based_on_RiverBank_Elevations(
        self,
        dm_output_bathymetry: np.ndarray,
        bank_search_result: dict | None = None,
    ):
        """Burn a staged depth below the smoothed bank elevation.

        Hydraulic depth and bank detection are intentionally absent here.
        This final geometry step consumes ``bathymetry_depth`` from the staged
        bank result and writes the lowered profile to the output raster.
        """

        (
            function_used,
            i_bank_1_index,
            i_bank_2_index,
            i_total_bank_cells,
            bank_elev_1,
            bank_elev_2,
            smoothed_bank_elevation,
        ) = self._get_precomputed_bathymetry_bank_result(
            bank_search_result,
        )
        if not isinstance(bank_search_result, dict):
            return 0, 0, 1, 0.0, 0.0
        should_apply = bool(
            bank_search_result.get(
                "bathymetry_should_apply",
                "bathymetry_depth" in bank_search_result,
            )
        )
        if not should_apply:
            return 0, 0, 1, 0.0, 0.0

        d_y_depth = float(bank_search_result.get("bathymetry_depth", 0.0))
        d_y_bathy = 0.0

        d_bankfull_elevation = smoothed_bank_elevation
        if (
            not np.isfinite(d_y_depth)
            or d_y_depth >= 25.0
            or d_y_depth < 0.0
        ):
            return 0, 0, 1, 0.0, 0.0

        if i_total_bank_cells == 1:
            if self.xs1_n <= 1 or self.xs2_n <= 1:
                return 0, 0, 1, 0.0, 0.0
            d_side1_dist = 0.0
            d_side2_dist = 0.0
            d_total_bank_dist = self.d_ordinate_dist
            d_h_dist = 0.0
            d_trap_base = 0.0
            i_bank_1_index = 1
            i_bank_2_index = 1
        else:
            (
                d_side1_dist,
                d_side2_dist,
                d_total_bank_dist,
                d_h_dist,
                d_trap_base,
            ) = self._compute_bank_bathymetry_geometry(
                i_total_bank_cells,
                i_bank_1_index,
                i_bank_2_index,
                d_bankfull_elevation,
            )
        d_y_bathy = d_bankfull_elevation - d_y_depth

        if i_total_bank_cells > 1:
            _adjust_one_side_for_bathymetry(
                i_bank_1_index + 1,
                d_total_bank_dist,
                d_trap_base,
                d_h_dist,
                self.ia_xc_row1_index_main,
                self.ia_xc_column1_index_main,
                self.da_xs_profile1,
                dm_output_bathymetry,
                d_side1_dist,
                d_y_bathy,
                d_y_depth,
                self.d_ordinate_dist,
                self.dm_elevation,
                self.b_bathy_use_banks,
            )
            _adjust_one_side_for_bathymetry(
                i_bank_2_index + 1,
                d_total_bank_dist,
                d_trap_base,
                d_h_dist,
                self.ia_xc_row2_index_main,
                self.ia_xc_column2_index_main,
                self.da_xs_profile2,
                dm_output_bathymetry,
                d_side2_dist,
                d_y_bathy,
                d_y_depth,
                self.d_ordinate_dist,
                self.dm_elevation,
                self.b_bathy_use_banks,
            )
        elif i_total_bank_cells == 1:
            self.da_xs_profile1[0] = d_y_bathy
            dm_output_bathymetry[self.ia_xc_row1_index_main[0], self.ia_xc_column1_index_main[0]] = self.da_xs_profile1[0]
            self.da_xs_profile2[0] = d_y_bathy
            dm_output_bathymetry[self.ia_xc_row2_index_main[0], self.ia_xc_column2_index_main[0]] = self.da_xs_profile2[0]

        return i_bank_1_index, i_bank_2_index, i_total_bank_cells, d_y_depth, d_y_bathy
    
@njit(cache=True)
def _calculate_all(da_xs_profile1: np.ndarray, xs1_n: int, mannings_n1: np.ndarray, da_xs_profile2: np.ndarray, xs2_n: int, mannings_n2: np.ndarray, d_ordinate_dist: float, wse: float, sqrt_slope: float):
    wse = np.round(wse, 3)
    A1, P1, np1, T1 = _calculate_stream_geometry_and_topwidth(da_xs_profile1[:xs1_n], wse, d_ordinate_dist, mannings_n1)
    A2, P2, np2, T2 = _calculate_stream_geometry_and_topwidth(da_xs_profile2[:xs2_n], wse, d_ordinate_dist, mannings_n2)

    T = np.round(T1 + T2, 3)
    A = np.round(A1 + A2, 3)
    P = np.round(P1 + P2, 3)

    if A <= 0.0 or P <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, T

    # Estimate mannings n
    d_composite_n = np.round(((np1 + np2) / P)**(2 / 3), 4)

    # use Manning's equation to estimate the flow
    Q = np.round((1 / d_composite_n) * A * (A / P)**(2 / 3) * sqrt_slope, 3)
    V = np.round(Q / A, 3)

    return A, P, V, Q, T

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

        for i in range(i_xs_length_indice):
            row_main = ia_xc_row_index_main[i]
            col_main = ia_xc_column_index_main[i]
            row_second = ia_xc_row_index_second[i]
            col_second = ia_xc_column_index_second[i]

            profile[i] = (
                dm_elevation[row_main, col_main] * da_xc_main_fract[i] +
                dm_elevation[row_second, col_second] * da_xc_second_fract[i]
            )
            lc_profile[i] = dm_land_use[row_main, col_main]

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
def _get_stream_depths(d_wse: float, profile: np.ndarray):
    da_y_depth = d_wse - profile

    if da_y_depth.shape[0] <= 0 or da_y_depth[0] <= 1e-16:
        return None
    
    return da_y_depth

@njit(cache=True)
def _calculate_side_top_width(d_wse: float, profile: np.ndarray, d_ordinate_dist: float):
    da_y_depth = _get_stream_depths(d_wse, profile)

    if da_y_depth is None:
        return 0

    lt_0_in_depths, i_target_index = _check_for_negative_depths(da_y_depth)

    if lt_0_in_depths:
        i_target_index += 1
        d_dist_use = _get_distance_to_use(da_y_depth, i_target_index, d_ordinate_dist)
        return np.round(_calculate_top_width_up_to_point(i_target_index, d_dist_use, d_ordinate_dist), 3)
    else:
        return np.round(_calculate_top_width_from_all(da_y_depth, d_ordinate_dist), 3)


@njit(cache=True)
def _calculate_stream_geometry(da_xs_profile: np.ndarray,
                                d_wse: float,
                                d_ordinate_dist: float,
                                da_n_profile: np.ndarray = None,) -> tuple[float, ...]:
    # Initial output
    d_area, d_perimeter, d_composite_n = 0.0, 0.0, 0.0

    # Estimate the depth of the stream
    da_y_depth = _get_stream_depths(d_wse, da_xs_profile)

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
    da_y_depth = _get_stream_depths(d_wse, da_xs_profile)

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
    """Compute discharge (Q) at a given WSE using Manning's equation.

    Parameters
    ----------
    wse : float
        Water surface elevation.
    sqrt_slope : float
        Square root of slope (i.e., ``sqrt(slope)``). Passing this avoids
        recomputing ``sqrt`` inside tight loops.
    profile1, profile2 : numpy.ndarray
        Cross-section elevation profiles for each side of the channel.
    xs1_n, xs2_n : int
        Number of valid ordinates in each profile.
    mannings_n1, mannings_n2 : numpy.ndarray
        Manning's n values sampled along each profile.
    d_ordinate_dist : float
        Distance between successive ordinates along the cross-section.

    Returns
    -------
    float
        Discharge corresponding to the given WSE.
    """
    # Calculate the geometry
    A1, P1, np1 = _calculate_stream_geometry(profile1[:xs1_n], wse, d_ordinate_dist, mannings_n1)
    A2, P2, np2 = _calculate_stream_geometry(profile2[:xs2_n], wse, d_ordinate_dist, mannings_n2)

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
    Precompute index offsets and interpolation fractions for one angle.

    Parameters
    ----------
    ia_xc_dr_index_main: ndarray
        Output array of row offsets for the main interpolation sample.
    ia_xc_dc_index_main: ndarray
        Output array of column offsets for the main interpolation sample.
    ia_xc_dr_index_second: ndarray
        Output array of row offsets for the secondary interpolation sample.
    ia_xc_dc_index_second: ndarray
        Output array of column offsets for the secondary interpolation sample.
    da_xc_main_fract: ndarray
        Output array of weights for the main interpolation sample (0-1).
    da_xc_second_fract: ndarray
        Output array of weights for the secondary interpolation sample (0-1).
    d_xs_direction: float
        Orientation of the cross section (radians).
    i_centerpoint: int
        Number of ordinates to compute (centerpoint distance in cells).
    d_dx: float
        Cell resolution in the x direction.
    d_dy: float
        Cell resolution in the y direction.

    Returns
    -------
    d_distance_z: float
        Distance between successive ordinates along the cross-section direction.

    """
    
    
    '''
    Assume there are 4 quadrants:
            Q3 | Q4      r<0 c<0  |  r<0 c>0
            Q2 | Q1      r>0 c<0  |  r>0 c>0
    
    These quadrants are inversed about the x-axis due to rows being positive in the downward direction
    '''
    
    
    # Very small floating-point remnants at exactly horizontal/vertical angles
    # can otherwise shift the "secondary" sample one full cell away and assign
    # it nearly all the interpolation weight. That creates artificial end-cell
    # asymmetry when a stream terminates at the rasterized reach boundary.
    d_axis_tolerance = 1.0e-12

    # Determine the best direction to perform calcualtions
    #  Row-Dominated
    if d_xs_direction >= (math.pi / 4) and d_xs_direction <= (3 * math.pi / 4):
        if math.fabs(math.cos(d_xs_direction)) < d_axis_tolerance:
            ia_xc_dr_index_main[0:i_centerpoint] = np.arange(i_centerpoint)
            ia_xc_dc_index_main[0:i_centerpoint] = 0
            ia_xc_dr_index_second[0:i_centerpoint] = np.arange(i_centerpoint)
            ia_xc_dc_index_second[0:i_centerpoint] = 0
            da_xc_main_fract[0:i_centerpoint] = 1.0
            da_xc_second_fract[0:i_centerpoint] = 0.0
            return d_dy

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
        if math.fabs(math.sin(d_xs_direction)) < d_axis_tolerance:
            column_pos_or_neg = 1
            if d_xs_direction >= (math.pi / 2):
                column_pos_or_neg = -1

            ia_xc_dr_index_main[0:i_centerpoint] = 0
            ia_xc_dc_index_main[0:i_centerpoint] = np.arange(i_centerpoint) * column_pos_or_neg
            ia_xc_dr_index_second[0:i_centerpoint] = 0
            ia_xc_dc_index_second[0:i_centerpoint] = np.arange(i_centerpoint) * column_pos_or_neg
            da_xc_main_fract[0:i_centerpoint] = 1.0
            da_xc_second_fract[0:i_centerpoint] = 0.0
            return d_dx

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
def find_depth_of_bathymetry_triangle(d_baseflow: float, d_distance_between_ordinates: float, d_elev_streamcell: float, d_elev_left_bank: float, d_elev_right_bank:float, d_slope: float, d_mannings_n: float):
    """
    Estimates the depth iteratively by comparing the calculated flow to the baseflow.  Uses a triangular approximation of the stream cross-section instead of a trapezoidal approximation.  This is used when the stream width is less than or equal to a sincle raster cell.

    Parameters
    ----------
    d_baseflow: float
        Baseflow input for flow convergence calculation
    d_distance_between_ordinates: float
        Distance between ordinates in the cross-section
    d_elev_streamcell: float
        elevation on the streamcell of the stream.
    d_elev_left_bank: float
        elevation on the left side of the stream.  Not necesarrily the bank, but the elevation of the first ordinate on the left side of the stream.
    d_elev_right_bank: float
        elevation on the right side of the stream.  Not necesarrily the bank, but the elevation of the first ordinate on the right side of the stream.
    d_slope: float
        Slope of the stream
    d_mannings_n: float
        Manning's roughness of the stream

    Returns
    -------
    d_working_depth: float
        Estimated depth of the stream

    """

    if d_baseflow<=0.0:
        return 0.0

    YLU = d_elev_left_bank - d_elev_streamcell
    YRU = d_elev_right_bank - d_elev_streamcell
    if YLU<0.0:
        YLU = 0.0
    if YRU<0.0:
        YRU = 0.0


    d_working_depth = 0.0
    d_max_depth = 25.0
    d_dy = 0.1
    d_flow_calculated = 0.0
    while d_flow_calculated <= d_baseflow and d_working_depth < d_max_depth:
        d_working_depth = d_working_depth + d_dy

        #Work on the left side of the triangle (bathymetry) first.
        YL = YLU + d_working_depth
        XLU = d_distance_between_ordinates * (YLU / YL)
        #AL = 0.5 * d_working_depth * (d_distance_between_ordinates-XLU)
        #WL = calculate_hypotnuse((d_distance_between_ordinates-XLU), d_working_depth)

        #Work on the right side of the triangle (bathymetry)
        YR = YRU + d_working_depth
        XRU = d_distance_between_ordinates * (YRU / YR)
        #AR = 0.5 * d_working_depth * (d_distance_between_ordinates-XRU)
        #WR = calculate_hypotnuse((d_distance_between_ordinates-XRU), d_working_depth)

        d_area = 0.5 * d_working_depth * (d_distance_between_ordinates-XLU) + 0.5 * d_working_depth * (d_distance_between_ordinates-XRU)
        d_perimeter = calculate_hypotnuse((d_distance_between_ordinates-XLU), d_working_depth)
        d_perimeter = d_perimeter + calculate_hypotnuse((d_distance_between_ordinates-XRU), d_working_depth)
        d_hydraulic_radius = d_area / d_perimeter
        d_flow_calculated = (1.0 / d_mannings_n) * d_area * d_hydraulic_radius**(2 / 3) * d_slope**0.5
    
    #print(d_working_depth)

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
    Determine bankfull elevation from candidate bank elevations.

    Parameters
    ----------
    base_elev : float
        Reference elevation (typically the thalweg or WSE at the stream cell).
    bank_elev_1, bank_elev_2 : float
        Candidate bank elevations for each side of the cross-section.

    Returns
    -------
    float
        The minimum valid bank elevation that is greater than or equal to
        ``base_elev``. If neither bank elevation is valid, returns
        ``base_elev``.
    """
    valid_banks = [elev for elev in (bank_elev_1, bank_elev_2) if is_valid_number(elev) and elev >= base_elev]
    return min(valid_banks, default=base_elev)

@njit(cache=True)
def _find_bank_inflection_point_helper(da_xs_smooth: np.ndarray, i_cross_section_number: int, d_ordinate_dist: float) -> int:
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
            total_width += d_ordinate_dist
            entry += 1  # move forward
        else:
            # Found the bank – go back one if needed
            return entry  # or return entry - 1 if you want the previous one

    # Return to the calling function
    return 0

@njit(cache=True)
def _find_bank_using_width_to_depth_ratio(d_bottom_elevation: float, da_xs_profile1: np.ndarray, da_xs_profile2: np.ndarray, xs1_n: int, xs2_n: int, d_ordinate_dist: float) -> tuple[int, int]:
    """
    Find banks at the minimum observed width-to-depth ratio.

    The search now evaluates only physically sampled candidate bank heights:
    every positive elevation above ``d_bottom_elevation`` from either side of
    the cross section is treated as a potential stage. ARC computes top width
    at each of those stages and keeps the deepest stage for which the
    width-to-depth ratio is still decreasing. When the first increase is
    detected, ARC returns to the previous sampled depth and refines upward in
    0.01 m increments until the ratio again reaches that increased value. This
    keeps the bank search tied to sampled DEM geometry while allowing a finer
    stage selection than the raw sampled bank-height candidates alone.
    """
    if xs1_n <= 1 or xs2_n <= 1 or d_ordinate_dist <= 0.0:
        return 0, 0

    bank_heights = np.empty(xs1_n + xs2_n, dtype=np.float64)
    bank_height_count = 0

    for i in range(xs1_n):
        candidate_depth = da_xs_profile1[i] - d_bottom_elevation
        if candidate_depth > 0.0:
            bank_heights[bank_height_count] = candidate_depth
            bank_height_count += 1

    for i in range(xs2_n):
        candidate_depth = da_xs_profile2[i] - d_bottom_elevation
        if candidate_depth > 0.0:
            bank_heights[bank_height_count] = candidate_depth
            bank_height_count += 1

    if bank_height_count == 0:
        return 0, 0

    bank_heights = np.sort(bank_heights[:bank_height_count])

    best_depth = 0.0
    best_t1 = 0.0
    best_t2 = 0.0
    best_ratio = np.inf
    last_depth = 0.0
    last_t1 = 0.0
    last_t2 = 0.0
    last_ratio = np.inf

    for i_bank in range(bank_height_count):
        d_depth = bank_heights[i_bank]
        if d_depth <= 0.0:
            continue
        if d_depth > 25.0:
            break

        d_wse = d_bottom_elevation + d_depth
        t1 = _calculate_side_top_width(d_wse, da_xs_profile1[:xs1_n], d_ordinate_dist)
        t2 = _calculate_side_top_width(d_wse, da_xs_profile2[:xs2_n], d_ordinate_dist)
        tw = t1 + t2
        width_to_depth_ratio = np.round(tw / d_depth, 3)

        if width_to_depth_ratio > last_ratio:
            target_increased_ratio = width_to_depth_ratio
            best_depth = last_depth
            best_t1 = last_t1
            best_t2 = last_t2
            break

        last_depth = d_depth
        last_t1 = t1
        last_t2 = t2
        last_ratio = width_to_depth_ratio

    if best_t1 <= 0.0 and best_t2 <= 0.0:
        return 0, 0

    # we want the index before the inflection, so - 1 was added here.
    i_bank_1_index = int(best_t1 / d_ordinate_dist) 
    i_bank_2_index = int(best_t2 / d_ordinate_dist) 

    if i_bank_1_index < 0:
        i_bank_1_index = 0
    elif i_bank_1_index >= xs1_n:
        i_bank_1_index = 0

    if i_bank_2_index < 0:
        i_bank_2_index = 0
    elif i_bank_2_index >= xs2_n:
        i_bank_2_index = 0

    return i_bank_1_index, i_bank_2_index

# def multipoint_slope(windowsize, timeseries, xvals):
#     dw = np.zeros(len(timeseries))
#     lr_window = int(windowsize/2) # indexing later requires this to be an integer
#     for n in range(lr_window, len(timeseries) - lr_window):
#         x = xvals[n - lr_window:n + lr_window]
#         y = timeseries[n - lr_window:n + lr_window]
#         # Begin derivative calcs once all width measurements are non-zero
#         if all(val != 0 for val in y):
#             # remove nans with a mask, if there are at least two real data points
#             nancount = sum(1 for x in y if isinstance(x, float) and math.isnan(x))
#             if nancount > 2:
#                 mask = ~np.isnan(x) & ~np.isnan(y)
#                 slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x[mask], np.array(y)[mask])
#             else: 
#                 slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x, np.array(y))
#             dw[n] = slope1
#         else:
#             dw[n] = 0 
#     return dw   

@njit(cache=True)
def slope_only(x, y):
    """Numba-compatible replacement for scipy.stats.linregress slope."""
    n = len(x)
    if n < 2:
        return 0.0

    sum_x = 0.0
    sum_y = 0.0
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
    mean_x = sum_x / n
    mean_y = sum_y / n

    ss_xy = 0.0
    ss_xx = 0.0
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        ss_xy += dx * dy
        ss_xx += dx * dx

    if ss_xx == 0.0:
        return 0.0

    return ss_xy / ss_xx


@njit(cache=True)
def multipoint_slope(windowsize, timeseries, xvals, derivative_order):
    dw = np.zeros(len(timeseries))
    lr_window = int(windowsize / 2)

    # if derivative_order == 1, we can use the native lr_window, 
    # but if derivative_order == 2, we need to adjust to not use the padded values
    # at the edges of the first derivative output
    # so we need to start the loop later and end it earlier if derivative_order == 2
    if derivative_order == 1:
        start_index = lr_window
        end_index = len(timeseries) - lr_window
    elif derivative_order == 2:
        start_index = windowsize
        end_index = len(timeseries) - windowsize + 1
        
    for n in range(start_index, end_index):
        x = xvals[n - lr_window:n + lr_window]
        y = timeseries[n - lr_window:n + lr_window]

        nancount = 0
        for i in range(len(y)):
            if math.isnan(y[i]):
                nancount += 1

        if nancount > 2:
            # build mask manually (numba supports this fine for float arrays)
            mask = np.empty(len(x), dtype=np.bool_)
            for i in range(len(x)):
                mask[i] = (not math.isnan(x[i])) and (not math.isnan(y[i]))
            x_masked = x[mask]
            y_masked = y[mask]
            dw[n] = slope_only(x_masked, y_masked)
        else:
            dw[n] = slope_only(x, y)


    return dw

@njit(cache=True)
def compute_stream_derivatives(W, D, dy):
    """
    Calculate first and second derivatives for uniformly spaced samples.

    Upstream INFLECT estimates derivatives with a moving-window linear
    regression rather than with a raw pointwise finite difference. ARC mirrors
    that behavior here so the representative-cross-section workflow smooths the
    width-depth signal before identifying inflection structure, while staying
    fully Numba compatible.

    Parameters
    ----------
    W : numpy.ndarray
        One-dimensional array of width values sampled at a constant depth
        interval.
    D : numpy.ndarray
        One-dimensional array of depth values.
    dy : float
        Constant spacing between successive depth samples.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        First derivative and second derivative arrays.
    """
    # dW_dy = _moving_window_regression_slope(W, dy, INFLECT_REGRESSION_WINDOW, derivative_order=1)
    # d2W_dy2 = _moving_window_regression_slope(dW_dy, dy, INFLECT_REGRESSION_WINDOW, derivative_order=2)

    dW_dy = multipoint_slope(INFLECT_REGRESSION_WINDOW, W, D, derivative_order=1)
    d2W_dy2 = multipoint_slope(INFLECT_REGRESSION_WINDOW, dW_dy, D, derivative_order=2)

    return dW_dy, d2W_dy2

@njit(cache=True)
def _calculate_inflect_curve(
    d_bottom_elevation: float,
    da_xs_profile1: np.ndarray,
    da_xs_profile2: np.ndarray,
    xs1_n: int,
    xs2_n: int,
    d_ordinate_dist: float,
) -> np.ndarray:
    """Return only the INFLECT ``d2W/dy2`` curve for legacy callers."""
    (_, d2W_dy2) = _calculate_inflect_curve_with_depths(
        d_bottom_elevation,
        da_xs_profile1,
        da_xs_profile2,
        xs1_n,
        xs2_n,
        d_ordinate_dist,
    )
    return d2W_dy2


@njit(cache=True)
def _calculate_inflect_curve_with_depths(
    d_bottom_elevation: float,
    da_xs_profile1: np.ndarray,
    da_xs_profile2: np.ndarray,
    xs1_n: int,
    xs2_n: int,
    d_ordinate_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the INFLECT depth axis and ``d2W/dy2`` curve together."""
    if xs1_n <= 0 or xs2_n <= 0 or d_ordinate_dist <= 0.0:
        empty = np.zeros(1, dtype=np.float64)
        return empty, empty

    # initialize a width list to store the widths at each depth increment
    width_list = []
    width_list.append(0.0)  # Start with a width of 0 at the bottom elevation

    # initialize a depth list to store the depths at each depth increment
    depth_list = []
    depth_list.append(0.0)  # Start with a depth of 0 at the bottom elevation

    bank_heights = np.empty(xs1_n + xs2_n, dtype=np.float64)
    bank_height_count = 0

    for i in range(xs1_n):
        candidate_depth = da_xs_profile1[i] - d_bottom_elevation
        if candidate_depth > 0.0:
            bank_heights[bank_height_count] = candidate_depth
            bank_height_count += 1

    for i in range(xs2_n):
        candidate_depth = da_xs_profile2[i] - d_bottom_elevation
        if candidate_depth > 0.0:
            bank_heights[bank_height_count] = candidate_depth
            bank_height_count += 1

    if bank_height_count == 0:
        empty = np.zeros(1, dtype=np.float64)
        return empty, empty

    bank_heights = np.sort(bank_heights[:bank_height_count])

    max_depth = min(1.0, bank_heights[bank_height_count - 1])
    d_refine_depth = 0.01
    last_depth = 0.0

    for i_bank in range(bank_height_count):
        candidate_depth = bank_heights[i_bank]
        if candidate_depth <= last_depth:
            continue
        if candidate_depth > max_depth:
            candidate_depth = max_depth

        d_depth = last_depth + d_refine_depth
        while d_depth < candidate_depth:
            d_wse = d_bottom_elevation + d_depth
            T1 = _calculate_side_top_width(d_wse, da_xs_profile1[:xs1_n], d_ordinate_dist)
            T2 = _calculate_side_top_width(d_wse, da_xs_profile2[:xs2_n], d_ordinate_dist)
            width_list.append(T1 + T2)
            depth_list.append(d_depth)
            d_depth += d_refine_depth

        d_wse = d_bottom_elevation + candidate_depth
        T1 = _calculate_side_top_width(d_wse, da_xs_profile1[:xs1_n], d_ordinate_dist)
        T2 = _calculate_side_top_width(d_wse, da_xs_profile2[:xs2_n], d_ordinate_dist)
        width_list.append(T1 + T2)
        depth_list.append(candidate_depth)
        last_depth = candidate_depth

        if last_depth >= max_depth:
            break

    width_array = np.asarray(width_list, dtype=np.float64)
    depth_array = np.asarray(depth_list, dtype=np.float64)

    # Use INFLECT-style moving-window regression smoothing while keeping the
    # helper fully Numba compatible for ARC's representative workflow.
    _, d2W_dy2 = compute_stream_derivatives(width_array, depth_array, d_refine_depth)

    return depth_array, d2W_dy2
