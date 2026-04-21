import numpy as np
import pandas as pd
from numba import njit
from numba.core.errors import TypingError
from scipy.optimize import curve_fit

from arc.cross_section import CrossSection
from arc import LOG

class HydraulicData:
    def __init__(self,  params: dict, b_modified_dem: bool):
        self._has_vdt_data = False
        self.ap_file: str = params['s_output_ap_database']
        self.vdt_file: str = params["s_output_vdt_database"]
        self.curve_file: str = params["s_output_curve_file"]
        self.i_number_of_increments: int = params['i_number_of_increments']
        self.b_reach_average_curve_file: bool = params['b_reach_average_curve_file']
        self.s_xs_output_file: str = params['s_xs_output_file']
        self.b_modified_dem: bool = b_modified_dem

        self.da_total_t = np.zeros(self.i_number_of_increments + 1, dtype=float)
        self.da_total_a = np.zeros(self.i_number_of_increments + 1, dtype=float)
        self.da_total_p = np.zeros(self.i_number_of_increments + 1, dtype=float)
        self.da_total_v = np.zeros(self.i_number_of_increments + 1, dtype=float)
        self.da_total_q = np.zeros(self.i_number_of_increments + 1, dtype=float)
        self.da_total_wse = np.zeros(self.i_number_of_increments + 1, dtype=float)

        # instantiate the lists we will use to create the XS File
        self.XS_COMID_List = []
        self.XS_Row_List = []
        self.XS_Col_List = []
        # da_xs_profile1_str
        self.XS_da_xs_profile1 = []
        self.XS_da_xs_profile2 = []
        # dm_manning_n_raster1_str
        self.XS_dm_manning_n_raster1 = []
        self.XS_dm_manning_n_raster2 = []
        # d_ordinate_dist
        self.XS_d_ordinate_dist = []
        # r1, c1, r2, c2
        self.XS_r1 = []
        self.XS_c1 = []
        self.XS_r2 = []
        self.XS_c2 = []

        # Create the dictionary and lists that will be used to create our ATW database
        self.o_ap_file_dict: dict[str, list] = {}
        self.o_ap_file_dict['COMID'] = []
        self.o_ap_file_dict['Row'] = []
        self.o_ap_file_dict['Col'] = []
        self.comid_ap_dict_list = self.o_ap_file_dict['COMID']
        self.row_ap_dict_list = self.o_ap_file_dict['Row']
        self.col_ap_dict_list = self.o_ap_file_dict['Col']
        self.ap_column_names = []
        for i in range(1, self.i_number_of_increments+1):
            self.o_ap_file_dict[f'q_{i}'] = []
            self.o_ap_file_dict[f'a_{i}'] = []
            self.o_ap_file_dict[f'p_{i}'] = []
            self.ap_column_names.append((f'q_{i}', f'a_{i}', f'p_{i}'))

    def associate_with_cross_section(self, x_section: CrossSection):
        self.x_section = x_section
    
    def add_empty_x_section_for_curve_file(self,i_cell_comid: int, d_slope_use: float, i_entry_cell: int, vdt_array: np.ndarray):
        if not self.b_reach_average_curve_file:
            return
        
        i_row_cell, i_column_cell = self.x_section.get_row_col()
        vdt_array[i_entry_cell, 0:4] = [
            i_cell_comid, 
            i_row_cell - self.x_section.i_boundary_number, 
            i_column_cell - self.x_section.i_boundary_number, 
            self.x_section.dm_elevation[i_row_cell, i_column_cell]  # DEM elevation
        ]
        vdt_array[i_entry_cell, 5:8] = [
            d_slope_use, 
            self.x_section.d_xs_direction,
            self.x_section.dm_elevation[i_row_cell,i_column_cell] # Base elevation
        ]

    def add_hydraulic_data(self, n: int, wse: float, t: float, a: float, p: float, q: float, v: float, vdt_array: np.ndarray, i_entry_cell: int):
        vdt_array[i_entry_cell, 8 + ((n-1) * 4):8 + ((n-1) * 4) + 4] = [q, v, t, wse - 100 if self.b_modified_dem else wse]
        self.da_total_wse[n] = wse
        self.da_total_t[n] = t
        self.da_total_a[n] = a
        self.da_total_p[n] = p
        self.da_total_q[n] = q
        self.da_total_v[n] = v

    def set_q_at_index(self, n: int, q: float, vdt_array: np.ndarray, i_entry_cell: int):
        vdt_array[i_entry_cell, 8 + ((n-1) * 4)] = q
        self.da_total_q[n] = q

    def reset_hydraulic_data(self):
        self.da_total_t.fill(0)
        self.da_total_a.fill(0)
        self.da_total_p.fill(0)
        self.da_total_v.fill(0)
        self.da_total_q.fill(0)
        self.da_total_wse.fill(0)

    def is_start_q_greater_than_baseflow(self, i_start_elevation_index: int, d_q_baseflow: float):
        idx = i_start_elevation_index + 1
        return idx < len(self.da_total_q) and self.da_total_q[idx] >= d_q_baseflow

    def set_vdt_data(self,i_cell_comid: int,  d_q_baseflow: float, d_slope_use: float, i_number_of_elevations: int, vdt_array: np.ndarray, i_entry_cell: int):
        self._has_vdt_data = True
        i_row_cell, i_column_cell = self.x_section.get_row_col()
        if self.ap_file:
            self.comid_ap_dict_list.append(i_cell_comid)
            self.row_ap_dict_list.append(i_row_cell - self.x_section.i_boundary_number)
            self.col_ap_dict_list.append(i_column_cell - self.x_section.i_boundary_number)
        
            for i, (q_name, a_name, p_name) in enumerate(self.ap_column_names[:i_number_of_elevations - 1], start=1):
                self.o_ap_file_dict[q_name].append(self.da_total_q[i])
                self.o_ap_file_dict[a_name].append(self.da_total_a[i])
                self.o_ap_file_dict[p_name].append(self.da_total_p[i])

        vdt_array[i_entry_cell, 0:8] = [
            i_cell_comid, 
            i_row_cell - self.x_section.i_boundary_number, 
            i_column_cell - self.x_section.i_boundary_number, 
            self.x_section.dm_elevation[i_row_cell, i_column_cell] - 100 if self.b_modified_dem else self.x_section.dm_elevation[i_row_cell, i_column_cell],  # DEM elevatin
            d_q_baseflow, 
            d_slope_use, 
            self.x_section.d_xs_direction,
            self.x_section.get_thalweg()-100 if self.b_modified_dem else self.x_section.get_thalweg() # Base elevation
        ]
        
    def set_non_vdt_data(self, print_curve_file: bool, i_start_elevation_index: int, i_last_elevation_index: int,
                         i_cell_comid: int, i_row_cell: int, i_column_cell: int, d_slope_use: float, d_dem_low_point_elev: float, d_q_maximum: float, i_entry_cell: int, vdt_array: np.ndarray):
        if self.b_reach_average_curve_file:
            self._set_curve_data(i_cell_comid, i_row_cell, i_column_cell, d_q_maximum, d_slope_use, d_dem_low_point_elev, i_entry_cell, vdt_array)
        elif print_curve_file and self.curve_file and i_start_elevation_index>=0 and i_last_elevation_index>(i_start_elevation_index+1):
            self._set_curve_data(i_cell_comid, i_row_cell, i_column_cell, d_q_maximum, d_slope_use, d_dem_low_point_elev, i_entry_cell, vdt_array)
        if self.s_xs_output_file:
            self._set_cross_section_data(i_cell_comid, i_row_cell, i_column_cell)

    def _set_curve_data(self, i_cell_comid: int, i_row_cell: int, i_column_cell: int, d_q_maximum: float, d_slope_use: float, d_dem_low_point_elev: float, i_entry_cell: int, vdt_array: np.ndarray):
        vdt_array[i_entry_cell, 0:4] = [
            i_cell_comid, 
            i_row_cell - self.x_section.i_boundary_number, 
            i_column_cell - self.x_section.i_boundary_number, 
            d_dem_low_point_elev-100 if self.b_modified_dem else d_dem_low_point_elev # DEM elevation
        ]
        vdt_array[i_entry_cell, 5:8] = [
            d_slope_use, 
            self.x_section.d_xs_direction,
            self.x_section.get_thalweg()-100 if self.b_modified_dem else self.x_section.get_thalweg() # Base elevation
        ]

    def _set_cross_section_data(self, i_cell_comid: int, i_row_cell: int, i_column_cell: int,):
        self.XS_COMID_List.append(i_cell_comid)
        self.XS_Row_List.append(i_row_cell - self.x_section.i_boundary_number)
        self.XS_Col_List.append(i_column_cell - self.x_section.i_boundary_number)
        if self.b_modified_dem:
            # This is to remove the +100 if a negative value was in the DEM elevation
            self.XS_da_xs_profile1.append(self.x_section.da_xs_profile1[0:self.x_section.xs1_n]-100)
            self.XS_da_xs_profile2.append(self.x_section.da_xs_profile2[0:self.x_section.xs2_n]-100) 
        else:
            self.XS_da_xs_profile1.append(self.x_section.da_xs_profile1[0:self.x_section.xs1_n])
            self.XS_da_xs_profile2.append(self.x_section.da_xs_profile2[0:self.x_section.xs2_n])

        # calculate the location of the cross-section end points
        r1 = self.x_section.ia_xc_row1_index_main[self.x_section.xs1_n-1]-self.x_section.i_boundary_number
        c1 = self.x_section.ia_xc_column1_index_main[self.x_section.xs1_n-1]-self.x_section.i_boundary_number
        r2 = self.x_section.ia_xc_row2_index_main[self.x_section.xs2_n-1]-self.x_section.i_boundary_number
        c2 = self.x_section.ia_xc_column2_index_main[self.x_section.xs2_n-1]-self.x_section.i_boundary_number
      
        # dm_manning_n_raster1_str
        self.XS_dm_manning_n_raster1.append(self.x_section.mannings_n1[:self.x_section.xs1_n].copy())
        self.XS_dm_manning_n_raster2.append(self.x_section.mannings_n2[:self.x_section.xs2_n].copy())
        # d_ordinate_dist
        self.XS_d_ordinate_dist.append(self.x_section.d_ordinate_dist)
        # r1, c1, r2, c2
        self.XS_r1.append(r1)
        self.XS_c1.append(c1)
        self.XS_r2.append(r2)
        self.XS_c2.append(c2)

    def has_vdt_data(self):
        return self._has_vdt_data

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
    
    def save_files(self, vdt_data, id_flow_dict):
        if not self.has_vdt_data():
            LOG.warning('No VDT data was generated, so no output VDT database file will be created.')
            return
        
        vdt_df = None
        if self.vdt_file:
            vdt_df = self.save_vdt(vdt_data)
        if self.ap_file:
            self.save_ap()
        if self.b_reach_average_curve_file:
            self.save_reach_average_curve_file(vdt_df, vdt_data, id_flow_dict)
        elif self.curve_file:
            self.save_curve_file(vdt_data, id_flow_dict)
        if self.s_xs_output_file:
            self.save_cross_section_file()

    def save_vdt(self, vdt_array: np.ndarray):
        colorder = ['COMID', 'Row', 'Col', 'Elev', 'QBaseflow', 'Slope', 'XS_Angle'] + [
            f"{prefix}_{i}" for i in range(1, self.i_number_of_increments + 1) for prefix in ['q', 'v', 't', 'wse']
        ]

        # Combine the data first (without rounding yet)
        vdt_df = pd.DataFrame(np.delete(vdt_array, [7], axis=1), columns=colorder)
        
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
            vdt_df.to_parquet(self.vdt_file, compression='brotli', index=False, engine='fastparquet') # Brotli does very well with VDT data
        else:
            vdt_df.to_csv(self.vdt_file, index=False)    
        LOG.info('Finished writing ' + str(self.vdt_file))
        return vdt_df

    def save_ap(self):
        # Write the output VDT Database file
        dtypes = {
                    "COMID": 'int64',
                    "Row": 'int64',
                    "Col": 'int64',
        }
        for i in range(1, self.i_number_of_increments + 1):
            self.o_ap_file_dict[f'a_{i}'] = np.round(self.o_ap_file_dict[f'a_{i}'], 3)
            self.o_ap_file_dict[f'q_{i}'] = np.round(self.o_ap_file_dict[f'q_{i}'], 3)
            self.o_ap_file_dict[f'p_{i}'] = np.round(self.o_ap_file_dict[f'p_{i}'], 3)

        o_ap_file_df = pd.DataFrame(self.o_ap_file_dict).astype(dtypes)
        # Remove rows with NaN values
        o_ap_file_df = o_ap_file_df.dropna()
        # # Remove rows where any column has a negative value except wse or elevation
        # Select columns NOT starting with 'wse' or 'Elev'
        cols_to_check = [col for col in o_ap_file_df.columns if (col.startswith('q') or col.startswith('a') or col.startswith('p'))]
        # Remove rows where any of the selected columns have a negative value
        o_ap_file_df = o_ap_file_df.loc[~(o_ap_file_df[cols_to_check] < 0).any(axis=1)]
        if self.ap_file.endswith('.parquet'):
            o_ap_file_df.to_parquet(self.ap_file, compression='brotli', index=False, engine='fastparquet') # Brotli does very well with AP data
        else:
            o_ap_file_df.to_csv(self.ap_file, index=False)
        LOG.info('Finished writing ' + str(self.ap_file))

    def save_reach_average_curve_file(self, vdt_df: pd.DataFrame, vdt_data: np.ndarray, id_flow_dict: dict):
        # Creating the DataFrame
        reach_average_curvefile_df = pd.DataFrame(vdt_data[:, 0:8], columns=['COMID', 'Row', 'Col', 'Elev', 'QBaseflow', 'Slope', 'XS_Angle', 'BaseElev'])
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
        unique_comids = vdt_df["COMID"].unique()

        # Process each unique COMID
        for comid in unique_comids:
            group = vdt_df[vdt_df["COMID"] == comid]
            
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

            # Combine WSE_ values and subtract BaseElev
            depth_combined_values_list = []
            for prefix in wse_prefixes:
                # Match rows using Row and Col from the group
                wse_values = group.set_index(["Row", "Col"])[prefix]
                depth_values = wse_values - base_elev_values
                depth_combined_values_list.extend(depth_values.values)
            d_combined_values = np.array(depth_combined_values_list)

            # Combine Q_ values
            q_combined_values_list = []
            for prefix in q_prefixes:
                q_combined_values_list.extend(group[prefix].values)
            q_combined_values = np.array(q_combined_values_list)

            # Combine T_ values
            t_combined_values_list = []
            for prefix in t_prefixes:
                t_combined_values_list.extend(group[prefix].values)
            t_combined_values = np.array(t_combined_values_list)

            # Combine V_ values
            v_combined_values_list = []
            for prefix in v_prefixes:
                v_combined_values_list.extend(group[prefix].values)
            v_combined_values = np.array(v_combined_values_list)

            # Calculate regression coefficients
            try:
                (d_t_a, d_t_b, d_t_R2) = self._linear_regression_power_function(q_combined_values, t_combined_values, [12, 0.3])
                (d_v_a, d_v_b, d_v_R2) = self._linear_regression_power_function(q_combined_values, v_combined_values, [1, 0.3])
                (d_d_a, d_d_b, d_d_R2) = self._linear_regression_power_function(q_combined_values, d_combined_values, [0.2, 0.5])
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
            reach_average_curvefile_df.to_parquet(self.curve_file, compression='brotli', index=False, engine='fastparquet')
        else:
            reach_average_curvefile_df.to_csv(self.curve_file, index=False)        
        LOG.info('Finished writing ' + str(self.curve_file))

    def save_curve_file(self, vdt_array: np.ndarray, id_flow_dict: dict):
        o_curve_file_df = pd.DataFrame(vdt_array[:, 0:8], columns=['COMID', 'Row', 'Col', 'DEM_Elev', 'QBaseflow', 'Slope', 'XS_Angle', 'BaseElev'])

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
            idx = np.arange(8, 8 + self.i_number_of_increments * 4, 4)
            da_total_q = vdt_array[i, idx]
            da_total_v = vdt_array[i, idx + 1]
            da_total_t = vdt_array[i, idx + 2]
            da_total_wse = vdt_array[i, idx + 3]
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
            o_curve_file_df.to_parquet(self.curve_file, compression='brotli', index=False, engine='fastparquet')
        else:
            o_curve_file_df.to_csv(self.curve_file, index=False)            
        LOG.info('Finished writing ' + str(self.curve_file))

    def save_cross_section_file(self):
        pd.DataFrame({
            'COMID': self.XS_COMID_List,
            'Row': self.XS_Row_List,
            'Col': self.XS_Col_List,
            'XS1_Profile': self.XS_da_xs_profile1,
            'Ordinate_Dist': self.XS_d_ordinate_dist,
            'Manning_N_Raster1': self.XS_dm_manning_n_raster1,
            'XS2_Profile': self.XS_da_xs_profile2,
            'Manning_N_Raster2': self.XS_dm_manning_n_raster2,
            'r1': self.XS_r1,
            'c1': self.XS_c1,
            'r2': self.XS_r2,
            'c2': self.XS_c2,
        }).to_csv(self.s_xs_output_file, index=False, sep='\t', float_format='%.6f')

        LOG.info('Finished writing ' + str(self.s_xs_output_file))

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
