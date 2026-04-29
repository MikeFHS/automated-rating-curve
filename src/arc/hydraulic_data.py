"""Hydraulic output schema and writers.

ARC stores per-cell hydraulic results in a single 2D NumPy array and then
derives requested output products (VDT database, AP database, curve file, and
optional cross-section export) from that array.
"""

import numpy as np
import pandas as pd
from numba import njit
from numba.core.errors import TypingError
from scipy.optimize import curve_fit

from arc.cross_section import CrossSection
from arc import LOG

class HydraulicData:
    """Helper for assembling ARC outputs and writing output files.

    Parameters
    ----------
    params : dict
        ARC parameter dictionary (parsed from the MIF / overrides). This class
        reads output-path and increment settings from ``params``.
    """
    def __init__(self,  params: dict):
        """Initialize output configuration from ``params``."""
        self.ap_file: str = params['s_output_ap_database']
        self.vdt_file: str = params["s_output_vdt_database"]
        self.curve_file: str = params["s_output_curve_file"]
        self.i_number_of_increments: int = params['i_number_of_increments']
        self.b_reach_average_curve_file: bool = params['b_reach_average_curve_file']
        self.s_xs_output_file: str = params['s_xs_output_file']
        self.b_modified_dem: bool = params['b_modified_dem']

    def associate_with_cross_section(self, x_section: CrossSection):
        """Attach the current :class:`~arc.cross_section.CrossSection` instance."""
        self.x_section = x_section

    def associate_with_output_data(self, output_data: np.ndarray):
        """Attach the shared output array to populate in-place."""
        self.output_data = output_data
    
    def add_empty_x_section_for_curve_file(self,i_cell_comid: int, d_slope_use: float, i_entry_cell: int):
        """Initialize the metadata row used by reach-average curve workflows."""
        if self.output_data is None:
            return
        if not self.b_reach_average_curve_file:
            return
        
        i_row_cell, i_column_cell = self.x_section.get_row_col()
        self.output_data[i_entry_cell, 0:4] = [
            i_cell_comid, 
            i_row_cell - self.x_section.i_boundary_number, 
            i_column_cell - self.x_section.i_boundary_number, 
            self.x_section.dm_elevation[i_row_cell, i_column_cell]  # DEM elevation
        ]
        self.output_data[i_entry_cell, 5:8] = [
            d_slope_use, 
            self.x_section.d_xs_direction,
            self.x_section.dm_elevation[i_row_cell,i_column_cell] # Base elevation
        ]

    def set_q_at_index(self, n: int, q: float, i_entry_cell: int):
        """Set discharge ``q`` for increment ``n`` in the output array."""
        if self.output_data is None:
            return
        self.output_data[i_entry_cell, 8 + ((n-1) * 5)] = q

    def is_start_q_greater_than_baseflow(self, i_start_elevation_index: int, d_q_baseflow: float, i_entry_cell: int):
        """Return ``True`` if the stored starting Q is greater than baseflow."""
        if self.output_data is None:
            return False
        idx = i_start_elevation_index + 1
        return self.output_data[i_entry_cell, 8 + ((idx-1) * 5)] >= d_q_baseflow

    def set_vdt_data(self,i_cell_comid: int,  d_q_baseflow: float, d_slope_use: float, i_entry_cell: int, i_number_of_elevations: int):
        """Populate the VDT metadata columns for a stream cell."""
        if self.output_data is None:
            return
        da_total_q_half_sum = np.sum(self.output_data[i_entry_cell, range(8, (i_number_of_elevations // 2) * 5, 5)])
        i_row_cell, i_column_cell = self.x_section.get_row_col()
        if da_total_q_half_sum <= 1e-16 or self.x_section.dm_elevation[i_row_cell, i_column_cell] <= 1e-16:
            return

        self.output_data[i_entry_cell, 0:8] = [
            i_cell_comid, 
            i_row_cell - self.x_section.i_boundary_number, 
            i_column_cell - self.x_section.i_boundary_number, 
            self.x_section.dm_elevation[i_row_cell, i_column_cell] - 100 if self.b_modified_dem else self.x_section.dm_elevation[i_row_cell, i_column_cell],  # DEM elevation
            d_q_baseflow, 
            d_slope_use, 
            self.x_section.d_xs_direction,
            self.x_section.get_thalweg()-100 if self.b_modified_dem else self.x_section.get_thalweg() # Base elevation
        ]
        
    def set_non_vdt_data(self, print_curve_file: bool, i_start_elevation_index: int, i_last_elevation_index: int,
                          i_cell_comid: int, i_row_cell: int, i_column_cell: int, d_slope_use: float, d_dem_low_point_elev: float, i_entry_cell: int):
        """Populate curve-file metadata for non-VDT configurations."""
        if self.output_data is None:
            return
        if self.b_reach_average_curve_file:
            self._set_curve_data(i_cell_comid, i_row_cell, i_column_cell, d_slope_use, d_dem_low_point_elev, i_entry_cell)
        elif print_curve_file and self.curve_file and i_start_elevation_index>=0 and i_last_elevation_index>(i_start_elevation_index+1):
            self._set_curve_data(i_cell_comid, i_row_cell, i_column_cell, d_slope_use, d_dem_low_point_elev, i_entry_cell)

    def _set_curve_data(self, i_cell_comid: int, i_row_cell: int, i_column_cell: int, d_slope_use: float, d_dem_low_point_elev: float, i_entry_cell: int):
        if self.output_data is None:
            return
        self.output_data[i_entry_cell, 0:4] = [
            i_cell_comid, 
            i_row_cell - self.x_section.i_boundary_number, 
            i_column_cell - self.x_section.i_boundary_number, 
            d_dem_low_point_elev-100 if self.b_modified_dem else d_dem_low_point_elev # DEM elevation
        ]
        self.output_data[i_entry_cell, 5:8] = [
            d_slope_use, 
            self.x_section.d_xs_direction,
            self.x_section.get_thalweg()-100 if self.b_modified_dem else self.x_section.get_thalweg() # Base elevation
        ]

    def get_cross_section_data(self, i_cell_comid: int, i_row_cell: int, i_column_cell: int,):
        """Collect the current cross-section sample for optional export.

        Parameters
        ----------
        i_cell_comid : int
            Reach/cell identifier for the stream cell.
        i_row_cell, i_column_cell : int
            Stream cell row/column indices (in the padded raster arrays).

        Returns
        -------
        tuple
            Row tuple for the cross-section export file.
        """
        return (
            i_cell_comid,
            i_row_cell - self.x_section.i_boundary_number,
            i_column_cell - self.x_section.i_boundary_number,
            self.x_section.da_xs_profile1[0:self.x_section.xs1_n].copy()-100 if self.b_modified_dem else self.x_section.da_xs_profile1[0:self.x_section.xs1_n].copy(),
            self.x_section.d_ordinate_dist,
            self.x_section.mannings_n1[:self.x_section.xs1_n].copy(),
            self.x_section.da_xs_profile2[0:self.x_section.xs2_n].copy()-100 if self.b_modified_dem else self.x_section.da_xs_profile2[0:self.x_section.xs2_n].copy(),
            self.x_section.mannings_n2[:self.x_section.xs2_n].copy(),
            self.x_section.ia_xc_row1_index_main[self.x_section.xs1_n-1]-self.x_section.i_boundary_number,
            self.x_section.ia_xc_column1_index_main[self.x_section.xs1_n-1]-self.x_section.i_boundary_number,
            self.x_section.ia_xc_row2_index_main[self.x_section.xs2_n-1]-self.x_section.i_boundary_number,
            self.x_section.ia_xc_column2_index_main[self.x_section.xs2_n-1]-self.x_section.i_boundary_number
        )
    
    def add_cross_section_data(self, data):
        """Attach the list/array used to store cross-section export rows."""
        self.xs_data = data

    def has_vdt_data(self):
        """Return ``True`` if the output array contains any populated increments."""
        # Check if there are any non nan values in the last column of the output data, which would indicate that at least some VDT data was generated
        if getattr(self, "output_data", None) is None:
            return False
        return np.any(~np.isnan(self.output_data[:, -1]))

    def _linear_regression_power_function(self, da_x_input: np.ndarray, da_y_input: np.ndarray, init_guess: list = [1.0, 1.0]):
        """
        Performs a curve fit to a power function

        Parameters
        ----------
        da_x_input: np.ndarray
            X values input to the fit
        da_y_input: np.ndarray
            Y values input to the fit

        Returns
        -------
        d_coefficient: float
            Coeffient of the fit
        d_power: float
            Power of the fit
        d_R2: float
            Goodness of fit

        """
        # Default values in case of failure
        d_coefficient, d_power, d_R2 = -9999.9, -9999.9, -9999.9

        # Attempt to calculate the fit
        try:
            (d_coefficient, d_power), dm_pcov = curve_fit(
                power_func, 
                da_x_input,
                da_y_input, 
                p0=init_guess)
        except TypingError as e:
            LOG.error(e)
        except RuntimeError as e:
            pass

        # Return to the calling function
        return d_coefficient, d_power, d_R2
    
    def save_files(self, id_flow_dict):
        """Write all configured output products.

        Parameters
        ----------
        id_flow_dict : dict
            Mapping from reach ID to (baseflow, qmax) or similar flow metadata.
            Used when building curve-file outputs.
        """
        vdt_df = None
        if self.vdt_file:
            vdt_df = self.save_vdt()
        if self.ap_file:
            self.save_ap()
        if self.b_reach_average_curve_file:
            self.save_reach_average_curve_file(vdt_df, id_flow_dict)
        elif self.curve_file:
            self.save_curve_file(id_flow_dict)
        if self.s_xs_output_file:
            self.save_cross_section_file()
    
    def save_vdt(self):
        """Save the VDT database to disk (CSV or Parquet)."""
        colorder = ['COMID', 'Row', 'Col', 'Elev', 'QBaseflow', 'Slope', 'XS_Angle', 'BaseElev'] + [
            f"{prefix}_{i}" for i in range(1, self.i_number_of_increments + 1) for prefix in ['q', 'v', 't', 'wse', 'p']
        ]

        # Combine the data first (without rounding yet)
        vdt_df = pd.DataFrame(self.output_data, columns=colorder)

        # Remove perimeter columns and base elevation column
        vdt_df = vdt_df.drop(columns=[col for col in vdt_df.columns if col.startswith('p_') or col == 'BaseElev'])
        
        # Remove rows with NaN values
        vdt_df = vdt_df.dropna()

        # Drop duplicate rows
        vdt_df = vdt_df.drop_duplicates()

        # Make First 3 columns int
        for col in ['COMID', 'Row', 'Col']:
            vdt_df[col] = vdt_df[col].astype(int)

        # Round all numeric columns to 3, except 'Slope'
        for col in vdt_df.columns:
            if col not in ('Slope', ):
                vdt_df[col] = vdt_df[col].round(3)

        # Now round Slope separately to 8
        vdt_df['Slope'] = vdt_df['Slope'].round(8)

        # # Remove rows where any column has a negative value except wse or elevation
        # Select columns NOT starting with 'wse' or 'Elev'
        cols_to_check = [col for col in vdt_df.columns if (col.startswith('q') or col.startswith('t') or col.startswith('v'))]
        # Remove rows where any of the selected columns have a negative value
        vdt_df = vdt_df.loc[~(vdt_df[cols_to_check] < 0).any(axis=1)]
        if self.vdt_file.endswith('.parquet'):
            vdt_df.to_parquet(self.vdt_file, compression='brotli', index=False) # Brotli does very well with VDT data
        else:
            vdt_df.to_csv(self.vdt_file, index=False)    
        LOG.info('Finished writing ' + str(self.vdt_file))
        return vdt_df

    def save_ap(self):
        """Save the area/perimeter (AP) database to disk (CSV or Parquet)."""
        o_ap_file_df = pd.DataFrame(self.output_data, columns=['COMID', 'Row', 'Col', 'Elev', 'QBaseflow', 'Slope', 'XS_Angle', 'BaseElev'] + [
            f"{prefix}_{i}" for i in range(1, self.i_number_of_increments + 1) for prefix in ['q', 'v', 't', 'wse', 'p']
        ])

        o_ap_file_df = o_ap_file_df.drop(columns=['Elev', 'QBaseflow', 'Slope', 'XS_Angle', 'BaseElev'] + [col for col in o_ap_file_df.columns if col.startswith('t_') or col.startswith('wse_')])

        # Remove rows with NaN values, and duplicates
        o_ap_file_df = o_ap_file_df.dropna()
        o_ap_file_df = o_ap_file_df.drop_duplicates()

        # Set first 3 columns as int
        for col in ['COMID', 'Row', 'Col']:
            o_ap_file_df[col] = o_ap_file_df[col].astype(int)

        # Calculate area columns based on q and v columns
        for i in range(1, self.i_number_of_increments + 1):
            o_ap_file_df[f'a_{i}'] = o_ap_file_df[f'q_{i}'].div(o_ap_file_df[f'v_{i}'], fill_value=0)
            o_ap_file_df.loc[ o_ap_file_df[f'v_{i}'] == 0, f'a_{i}'] = 0 # Fill in area with 0 where velocity is 0 to avoid infinite area values

        # Reorder columns to have q, a, p together for each increment
        column_order = ['COMID', 'Row', 'Col'] + [col for i in range(1, self.i_number_of_increments + 1) for col in (f'q_{i}', f'a_{i}', f'p_{i}')]
        o_ap_file_df = o_ap_file_df[column_order]

        o_ap_file_df = o_ap_file_df.round(3)

        # # Remove rows where any column has a negative value except wse or elevation
        # Select columns NOT starting with 'wse' or 'Elev'
        cols_to_check = [col for col in o_ap_file_df.columns if (col.startswith('q') or col.startswith('a') or col.startswith('p'))]
        # Remove rows where any of the selected columns have a negative value
        o_ap_file_df = o_ap_file_df.loc[~(o_ap_file_df[cols_to_check] < 0).any(axis=1)]
        if self.ap_file.endswith('.parquet'):
            o_ap_file_df.to_parquet(self.ap_file, compression='brotli', index=False) # Brotli does very well with AP data
        else:
            o_ap_file_df.to_csv(self.ap_file, index=False)
        LOG.info('Finished writing ' + str(self.ap_file))

    def save_reach_average_curve_file(self, vdt_df: pd.DataFrame, id_flow_dict: dict):
        """Save a reach-averaged curve file derived from per-cell results."""
        # Creating the DataFrame
        reach_average_curvefile_df = pd.DataFrame(self.output_data[:, 0:8], columns=['COMID', 'Row', 'Col', 'Elev', 'QBaseflow', 'Slope', 'XS_Angle', 'BaseElev'])
        reach_average_curvefile_df = reach_average_curvefile_df.dropna(how='all')
        reach_average_curvefile_df = reach_average_curvefile_df[['COMID', 'Row', 'Col', 'BaseElev', 'Elev', 'QBaseflow', 'Slope', 'XS_Angle']]

        # First columns as int
        for col in ['COMID', 'Row', 'Col']:
            reach_average_curvefile_df[col] = reach_average_curvefile_df[col].astype(int)

        # rename baseflow as qmax and set values
        reach_average_curvefile_df = reach_average_curvefile_df.rename(columns={'QBaseflow': 'QMax', 'Elev': 'DEM_Elev'})
        reach_average_curvefile_df['QMax'] = reach_average_curvefile_df['COMID'].map(pd.DataFrame.from_dict(id_flow_dict, orient='index').iloc[:, 1])

        # All columns but slope rounded to 3
        for col in reach_average_curvefile_df.columns:
            if col != 'Slope':
                reach_average_curvefile_df[col] = reach_average_curvefile_df[col].round(3)

        reach_average_curvefile_df['Slope'] = reach_average_curvefile_df['Slope'].round(8)

        # Dynamically select columns, starting with prefixes
        q_prefixes = [f'q_{i}' for i in range(1, self.i_number_of_increments + 1)]
        t_prefixes = [f't_{i}' for i in range(1, self.i_number_of_increments + 1)]
        v_prefixes = [f'v_{i}' for i in range(1, self.i_number_of_increments + 1)]
        wse_prefixes = [f'wse_{i}' for i in range(1, self.i_number_of_increments + 1)]

        # Initialize lists to store regression coefficients
        comid_list = []
        d_t_a_list, d_t_b_list = [], []
        d_v_a_list, d_v_b_list = [], []
        d_d_a_list, d_d_b_list = [], []

        # Extract all unique COMID values
        for comid, group in vdt_df.groupby("COMID"):
            # Create a MultiIndex from the current group's Row and Col for precise matching
            group_index = pd.MultiIndex.from_arrays([group["Row"].values, group["Col"].values], names=["Row", "Col"])

            # Filter reach_average_curvefile_df using COMID and matching Row-Col pairs
            matching_reach = reach_average_curvefile_df[
                (reach_average_curvefile_df["COMID"] == comid) &
                (pd.MultiIndex.from_frame(reach_average_curvefile_df[["Row", "Col"]]).isin(group_index))
            ]

            matching_reach = matching_reach.drop_duplicates(subset=["Row", "Col", "COMID"])

            if matching_reach.empty:
                LOG.warning(f"No matching BaseElev values found for COMID {comid}. Skipping...")
                continue

            # Get the BaseElev values for subtraction
            base_elev_values = matching_reach.set_index(["Row", "Col"])["BaseElev"]
                        
            group_indexed = group.set_index(["Row", "Col"])

            # Align once
            aligned = group_indexed.join(base_elev_values, how="inner")

            # Depths (vectorized), Combine WSE_ values and subtract BaseElev
            depth_combined_values = np.concatenate([
                aligned[prefix].values - aligned["BaseElev"].values
                for prefix in wse_prefixes
            ])

            # Q, T, V (vectorized)
            q_combined_values = np.concatenate([group[p].values for p in q_prefixes])
            t_combined_values = np.concatenate([group[p].values for p in t_prefixes])
            v_combined_values = np.concatenate([group[p].values for p in v_prefixes])

            # Calculate regression coefficients
            try:
                (d_t_a, d_t_b, d_t_R2) = self._linear_regression_power_function(q_combined_values, t_combined_values, [12, 0.3])
                (d_v_a, d_v_b, d_v_R2) = self._linear_regression_power_function(q_combined_values, v_combined_values, [1, 0.3])
                (d_d_a, d_d_b, d_d_R2) = self._linear_regression_power_function(q_combined_values, depth_combined_values, [0.2, 0.5])
            except Exception as e:
                # Handle cases where regression fails (e.g., insufficient data)
                LOG.warning(f"Regression failed for COMID {comid}: {e}")
                d_t_a, d_t_b, d_v_a, d_v_b, d_d_a, d_d_b = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            # Append results to lists
            comid_list.append(comid)
            d_t_a_list.append(np.round(d_t_a, 3) if not np.isnan(d_t_a) else np.nan)
            d_t_b_list.append(np.round(d_t_b, 3) if not np.isnan(d_t_b) else np.nan)
            d_v_a_list.append(np.round(d_v_a, 3) if not np.isnan(d_v_a) else np.nan)
            d_v_b_list.append(np.round(d_v_b, 3) if not np.isnan(d_v_b) else np.nan)
            d_d_a_list.append(np.round(d_d_a, 3) if not np.isnan(d_d_a) else np.nan)
            d_d_b_list.append(np.round(d_d_b, 3) if not np.isnan(d_d_b) else np.nan)

        # Create a DataFrame with regression coefficients
        regression_df = pd.DataFrame({
            "COMID": comid_list,
            "depth_a": d_d_a_list,
            "depth_b": d_d_b_list,
            "tw_a": d_t_a_list,
            "tw_b": d_t_b_list,
            "vel_a": d_v_a_list,
            "vel_b": d_v_b_list,
        })

        # Merge the regression_df into reach_average_curvefile_df based on COMID
        reach_average_curvefile_df = reach_average_curvefile_df.merge(regression_df, on="COMID", how="left")

        # Drop all rows with any NaN values
        reach_average_curvefile_df = reach_average_curvefile_df.dropna()

        # Write the output file
        if self.curve_file.endswith('.parquet'):
            reach_average_curvefile_df.to_parquet(self.curve_file, compression='brotli', index=False)
        else:
            reach_average_curvefile_df.to_csv(self.curve_file, index=False)        
        LOG.info('Finished writing ' + str(self.curve_file))

    def save_curve_file(self, id_flow_dict: dict):
        """Save per-cell power-law curve coefficients for depth/width/velocity."""
        o_curve_file_df = pd.DataFrame(self.output_data[:, 0:8], columns=['COMID', 'Row', 'Col', 'DEM_Elev', 'QBaseflow', 'Slope', 'XS_Angle', 'BaseElev'])

        # Reorder
        o_curve_file_df = o_curve_file_df[['COMID', 'Row', 'Col', 'BaseElev', 'DEM_Elev', 'QBaseflow', 'Slope', 'XS_Angle']]

        # Remove rows with NaN values
        o_curve_file_df = o_curve_file_df.dropna()

        # First 3 cols as int
        for col in ['COMID', 'Row', 'Col']:
            o_curve_file_df[col] = o_curve_file_df[col].astype(int)

        # rename baseflow as qmax and set values
        o_curve_file_df = o_curve_file_df.rename(columns={'QBaseflow': 'QMax'})
        o_curve_file_df['QMax'] = o_curve_file_df['COMID'].map(pd.DataFrame.from_dict(id_flow_dict, orient='index').iloc[:, 1])
        
        # Round all numeric columns to 3, except 'Slope'
        for col in o_curve_file_df.columns:
            if col not in ('Slope', ):
                o_curve_file_df[col] = o_curve_file_df[col].round(3)

        # Now round Slope separately to 8
        o_curve_file_df['Slope'] = o_curve_file_df['Slope'].round(8)

        # Now, we need to loop through the rows of the curve file and perform the regression for each row
        depth_a = []
        depth_b = []
        tw_a = []
        tw_b = []
        vel_a = []
        vel_b = []

        for i in o_curve_file_df.index:
            idx = np.arange(8, 8 + self.i_number_of_increments * 5, 5)
            da_total_q = self.output_data[i, idx]
            da_total_v = self.output_data[i, idx + 1]
            da_total_t = self.output_data[i, idx + 2]
            da_total_wse = self.output_data[i, idx + 3]
            base_elev = o_curve_file_df.loc[i, 'BaseElev']
            da_total_depth = da_total_wse - base_elev

            mask = ~np.isnan(da_total_q)

            if not mask.all():
                # There are nans at the start or end, so we need to trim those off before performing the regression
                da_total_q = da_total_q[mask]
                da_total_t = da_total_t[mask]
                da_total_v = da_total_v[mask]
                da_total_depth = da_total_depth[mask]

            (d_t_a, d_t_b, d_t_R2) = self._linear_regression_power_function(da_total_q, da_total_t, [12, 0.3])
            (d_v_a, d_v_b, d_v_R2) = self._linear_regression_power_function(da_total_q, da_total_v, [1, 0.3])
            (d_d_a, d_d_b, d_d_R2) = self._linear_regression_power_function(da_total_q, da_total_depth, [0.2, 0.5])

            depth_a.append(d_d_a)
            depth_b.append(d_d_b)
            tw_a.append(d_t_a)
            tw_b.append(d_t_b)
            vel_a.append(d_v_a)
            vel_b.append(d_v_b)

        regression_df = pd.DataFrame({
            'depth_a': depth_a,
            'depth_b': depth_b,
            'tw_a': tw_a,
            'tw_b': tw_b,
            'vel_a': vel_a,
            'vel_b': vel_b,
        }).round(3)
        o_curve_file_df = pd.concat([o_curve_file_df.reset_index(drop=True), regression_df], axis=1)

        # Remove rows where any column has negative a coefficient value
        o_curve_file_df = o_curve_file_df.loc[(o_curve_file_df['depth_a'] > 0) & (o_curve_file_df['tw_a'] > 0) & (o_curve_file_df['vel_a'] > 0)]
        if self.curve_file.endswith('.parquet'):
            o_curve_file_df.to_parquet(self.curve_file, compression='brotli', index=False)
        else:
            o_curve_file_df.to_csv(self.curve_file, index=False)            
        LOG.info('Finished writing ' + str(self.curve_file))

    def save_cross_section_file(self):
        """Save the cross-section export file (tab-delimited)."""
        cross_section_data = [item for item in self.xs_data if item is not None]
        pd.DataFrame(cross_section_data, columns=[
            'COMID', 'Row', 'Col', 'XS1_Profile', 'Ordinate_Dist', 'Manning_N_Raster1', 'XS2_Profile', 'Manning_N_Raster2', 'r1', 'c1', 'r2', 'c2'
        ]).to_csv(self.s_xs_output_file, index=False, sep='\t', float_format='%.6f')

        LOG.info('Finished writing ' + str(self.s_xs_output_file))

@njit(cache=True)
def add_hydraulic_data(output_data: np.ndarray, n: int, wse: float, t: float, p: float, q: float, v: float, i_entry_cell: int, b_modified_dem: bool):
    """Write one increment (q, v, t, wse, p) into the output array."""
    output_data[i_entry_cell, 8 + ((n-1) * 5):8 + ((n-1) * 5) + 5] = [q, v, t, wse - 100 if b_modified_dem else wse, p]

# Power function equation
@njit(cache=True)
def power_func(d_value: np.ndarray, d_coefficient: float, d_power: float):
    """
    Define a general power function that can be used for fitting

    Parameters
    ----------
    d_value: float
        Current x value
    d_coefficient: float
        Coefficient at the lead of the power function
    d_power: float
        Power value

    Returns
    -------
    d_power_value: float
        Calculated value

    """

    # Calculate the power
    d_power_value = d_coefficient * (d_value ** d_power)

    # Return to the calling function
    return d_power_value
