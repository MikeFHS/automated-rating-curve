"""Hydraulic output schema and writers.

ARC stores per-cell hydraulic results in a single 2D NumPy array and then
derives requested output products (VDT database, AP database, curve file, and
optional cross-section exports) from that array.
"""

import numpy as np
import pandas as pd
from numba import njit
from numba.core.errors import TypingError
from scipy.optimize import curve_fit

from arc.cross_section import CrossSection, _calculate_all
from arc import LOG

REPRESENTATIVE_DEPTH_INCREMENT = 0.10

REPRESENTATIVE_CROSS_SECTION_COLUMNS = [
    'COMID',
    'Cross_Section_Count',
    'Hydraulic_Sample_Count',
    'Depth_Stage_Index',
    'Depth_Stage_Meters',
    'Stream_Slope',
    'Reach_Inflect_Terrace_Depth',
    'Representative_Thalweg_Elevation',
    'Mean_Discharge',
    'Mean_Depth',
    'Mean_Velocity',
    'Representative_Velocity',
    'Mean_Top_Width',
    'Mean_Cross_Sectional_Area',
    'Mean_WSE',
    'Representative_Cross_Sectional_Area',
    'Representative_Depth_Increment',
    'Representative_Depth',
    'Representative_Top_Width',
    'Representative_Stage_Elevation',
    'Representative_Left_Station',
    'Representative_Right_Station',
]

XS_EXPORT_COLUMNS = [
    'COMID',
    'Row',
    'Col',
    'XS1_Profile',
    'Ordinate_Dist',
    'Manning_N_Raster1',
    'XS2_Profile',
    'Manning_N_Raster2',
    'r1',
    'c1',
    'r2',
    'c2',
]


def _calculate_increment_area(discharge: pd.Series, velocity: pd.Series) -> pd.Series:
    """Compute hydraulic area while protecting against divide-by-zero cases."""
    area = discharge.div(velocity.replace(0, np.nan))
    return area.replace([np.inf, -np.inf], np.nan)


def _monotonic_cumulative_max(values: np.ndarray) -> np.ndarray:
    """Force a 1D array to be non-decreasing while preserving NaNs.

    Reach means are computed independently at each 0.10 m depth stage, so
    tiny non-monotonic artifacts can appear after aggregation. The
    representative geometry uses a cumulative maximum to keep the staged
    cross-section envelope physically ordered from the thalweg outward.
    """
    result = values.astype(np.float64, copy=True)
    running_max = -np.inf
    for i in range(result.size):
        if np.isnan(result[i]):
            continue
        if result[i] < running_max:
            result[i] = running_max
        else:
            running_max = result[i]
    return result


def _derive_depths_from_width_and_area(
    representative_widths: np.ndarray,
    representative_areas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert staged width/area means into cross-section depth dimensions.

    The representative cross section is now defined from its staged hydraulic
    dimensions rather than from mean DEM profile points. ARC treats each
    0.10 m depth step as one stage on the representative section. Starting at
    the thalweg with zero width and zero area, it computes the incremental
    depth needed to grow from the previous stage to the current one assuming
    the width varies linearly over that increment:

    ``delta_area = ((width_prev + width_curr) / 2) * delta_depth``

    Solving that equation yields the depth increment needed to honor both the
    representative top width and representative area at each stage.
    """
    stage_count = representative_widths.size
    depth_increments = np.zeros(stage_count, dtype=np.float64)
    cumulative_depths = np.zeros(stage_count, dtype=np.float64)
    previous_width = 0.0
    previous_area = 0.0
    previous_depth = 0.0

    for i in range(stage_count):
        current_width = float(representative_widths[i])
        current_area = float(representative_areas[i])
        delta_area = max(current_area - previous_area, 0.0)
        denominator = previous_width + current_width

        if denominator > 0.0 and delta_area > 0.0:
            delta_depth = (2.0 * delta_area) / denominator
        else:
            delta_depth = 0.0

        depth_increments[i] = delta_depth
        cumulative_depths[i] = previous_depth + delta_depth
        previous_width = current_width
        previous_area = current_area
        previous_depth = cumulative_depths[i]

    return depth_increments, cumulative_depths


def _build_reach_average_inflect_curve(group: list[dict]) -> np.ndarray | None:
    """Average the representative INFLECT diagnostic curves for one reach."""
    inflect_curves = []
    for record in group:
        curve = record.get('Inflect_D2W_Dy2')
        if curve is None:
            continue
        curve_array = np.asarray(curve, dtype=np.float64)
        if curve_array.size == 0:
            continue
        inflect_curves.append(curve_array)

    if not inflect_curves:
        return None

    min_length = min(curve.shape[0] for curve in inflect_curves)
    if min_length <= 0:
        return None

    aligned_curves = np.vstack([curve[:min_length] for curve in inflect_curves])
    return np.nanmean(aligned_curves, axis=0)


def _get_inflect_minimum_index(mean_d2w_dy2: np.ndarray) -> int:
    """Locate the flood-terrace depth index from a reach-average INFLECT curve."""
    if mean_d2w_dy2.size == 0:
        return 0
    return int(np.argmin(mean_d2w_dy2))


def _filter_stage_samples_to_two_standard_deviations(
    areas: np.ndarray,
    velocities: np.ndarray,
    top_widths: np.ndarray,
) -> np.ndarray:
    """Keep stage samples whose key hydraulics are near the reach mean.

    The representative section is meant to summarize the reach, so a sampled
    cross-section must have area, velocity, and top width all within two
    standard deviations of that stage's mean before it contributes to the
    final reach mean.
    """
    keep_mask = np.isfinite(areas) & np.isfinite(velocities) & np.isfinite(top_widths)
    for values in (areas, velocities, top_widths):
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return np.zeros(values.shape, dtype=bool)

        mean_value = float(np.nanmean(finite_values))
        standard_deviation = float(np.nanstd(finite_values))
        if standard_deviation == 0.0:
            keep_mask &= values == mean_value
        else:
            keep_mask &= np.abs(values - mean_value) <= (2.0 * standard_deviation)

    return keep_mask


def _build_representative_hydraulic_rows_for_reach(
    comid: int,
    group: list[dict],
    max_depth: float,
) -> list[dict]:
    """Build staged hydraulic means for one reach up to a fixed depth cap.

    ARC evaluates each contributing cross section every 0.10 meters above its
    local thalweg up to ``max_depth`` meters, computing area, perimeter,
    velocity, discharge, and top width from Manning's equation at each stage.
    If any non-finite hydraulic value appears, the sweep stops early and the
    last successful depth becomes the effective cap for that reach.
    """
    max_depth = float(min(max_depth, 25.0))
    stage_count = max(int(round(max_depth / REPRESENTATIVE_DEPTH_INCREMENT)), 0)
    if stage_count <= 0:
        return []

    representative_thalweg = float(
        np.nanmean([float(record['Thalweg']) for record in group if record.get('Thalweg') is not None])
    )
    valid_slopes = [
        float(record['Slope'])
        for record in group
        if record.get('Slope') is not None and float(record['Slope']) > 0.0
    ]
    representative_stream_slope = float(np.nanmean(valid_slopes)) if valid_slopes else np.nan
    rows: list[dict] = []
    last_valid_stage_depth = 0.0

    for stage_index in range(1, stage_count + 1):
        stage_depth = float(stage_index * REPRESENTATIVE_DEPTH_INCREMENT)
        discharges = []
        velocities = []
        top_widths = []
        areas = []
        wses = []

        for record in group:
            thalweg = record.get('Thalweg')
            slope = record.get('Slope')
            if thalweg is None or slope is None or slope <= 0.0:
                continue

            left_profile = np.asarray(record['XS1_Profile'], dtype=np.float64)
            right_profile = np.asarray(record['XS2_Profile'], dtype=np.float64)
            left_n = np.asarray(record['Manning_N_Raster1'], dtype=np.float64)
            right_n = np.asarray(record['Manning_N_Raster2'], dtype=np.float64)
            ordinate_dist = float(record['Ordinate_Dist'])
            wse = float(thalweg + stage_depth)
            sqrt_slope = float(slope) ** 0.5

            area, perimeter, velocity, discharge, top_width, d_composite_n = _calculate_all(
                left_profile,
                left_profile.size,
                left_n,
                right_profile,
                right_profile.size,
                right_n,
                ordinate_dist,
                wse,
                sqrt_slope,
            )

            if not all(np.isfinite(value) for value in (area, perimeter, velocity, discharge, top_width, d_composite_n)):
                if rows:
                    for row in rows:
                        row['Reach_Inflect_Terrace_Depth'] = last_valid_stage_depth
                return rows

            if area <= 0.0 or top_width <= 0.0 or discharge < 0.0:
                continue

            discharges.append(float(discharge))
            velocities.append(float(velocity))
            top_widths.append(float(top_width))
            areas.append(float(area))
            wses.append(float(wse))

        if not areas:
            continue

        area_values = np.asarray(areas, dtype=np.float64)
        velocity_values = np.asarray(velocities, dtype=np.float64)
        top_width_values = np.asarray(top_widths, dtype=np.float64)
        discharge_values = np.asarray(discharges, dtype=np.float64)
        wse_values = np.asarray(wses, dtype=np.float64)

        # Remove hydraulic outliers before averaging the retained stage
        # samples. A sample must pass all three checks so odd geometry in one
        # metric cannot still influence the representative reach curve.
        sample_keep_mask = _filter_stage_samples_to_two_standard_deviations(
            area_values,
            velocity_values,
            top_width_values,
        )
        if not sample_keep_mask.any():
            continue

        rows.append(
            {
                'COMID': int(comid),
                'Cross_Section_Count': int(len(group)),
                'Hydraulic_Sample_Count': int(np.count_nonzero(sample_keep_mask)),
                'Depth_Stage_Index': int(stage_index),
                'Depth_Stage_Meters': stage_depth,
                'Stream_Slope': representative_stream_slope,
                'Reach_Inflect_Terrace_Depth': max_depth,
                'Representative_Thalweg_Elevation': representative_thalweg,
                'Mean_Discharge': float(np.nanmean(discharge_values[sample_keep_mask])),
                'Mean_Depth': stage_depth,
                'Mean_Velocity': float(np.nanmean(velocity_values[sample_keep_mask])),
                'Mean_Top_Width': float(np.nanmean(top_width_values[sample_keep_mask])),
                'Mean_Cross_Sectional_Area': float(np.nanmean(area_values[sample_keep_mask])),
                'Mean_WSE': float(np.nanmean(wse_values[sample_keep_mask])),
            }
        )
        last_valid_stage_depth = stage_depth

    if rows:
        for row in rows:
            row['Reach_Inflect_Terrace_Depth'] = last_valid_stage_depth
    return rows


def build_representative_cross_section_dataframe(
    cross_section_data: list[dict],
) -> pd.DataFrame:
    """Build representative cross sections from filtered staged hydraulic means.

    When representative cross sections are enabled, ARC now ignores the qmax-
    based VDT staging for this export. Instead it:

    1. Collects the sampled cross-section geometry, Manning's n arrays, slope,
       and thalweg elevation for every stream cell.
    2. Recomputes hydraulic properties every 0.10 meters above each
       cross-section thalweg up to a 25 meter maximum depth.
       The sweep truncates earlier if any non-finite hydraulic outputs appear.
    3. Removes cross sections whose area, velocity, or top width are outside
       two standard deviations of the reach mean at the current stage.
    4. Takes the mean discharge, velocity, top width, area, and WSE across the
       retained reach samples at each stage.
    5. Derives representative cross-section dimensions from the staged mean
       top width and staged mean area.

    Parameters
    ----------
    cross_section_data : list of dict
        Per-cell sampled cross-section records collected during the ARC main
        loop. These records include the profiles, Manning arrays, local slope,
        and thalweg elevation needed to rebuild the representative hydraulic
        stage database.

    Returns
    -------
    pandas.DataFrame
        Long-format representative cross-section table with one row per reach
        and 0.10 m depth stage up to the INFLECT-defined flood terrace.
    """
    if not cross_section_data:
        return pd.DataFrame(columns=REPRESENTATIVE_CROSS_SECTION_COLUMNS)

    grouped_records: dict[int, list[dict]] = {}
    for i, record in enumerate(cross_section_data):
        if record is None:
            continue
        comid = int(record['COMID'])
        grouped_records.setdefault(comid, []).append(record)

    rows: list[dict] = []
    for comid, group in grouped_records.items():
        rows.extend(_build_representative_hydraulic_rows_for_reach(int(comid), group, 25.0))

    representative_df = pd.DataFrame(rows)
    if representative_df.empty:
        return pd.DataFrame(columns=REPRESENTATIVE_CROSS_SECTION_COLUMNS)

    representative_groups = []
    for _, group in representative_df.groupby('COMID', sort=True):
        group = group.sort_values('Depth_Stage_Index').copy()

        # Independent depth-stage means can wobble slightly, so enforce a
        # monotonic staged area/width envelope before solving for the
        # representative cross-section dimensions.
        representative_area = _monotonic_cumulative_max(
            group['Mean_Cross_Sectional_Area'].to_numpy(dtype=np.float64)
        )
        width_seed = group['Mean_Top_Width'].to_numpy(dtype=np.float64)
        fallback_width = representative_area / np.maximum(group['Mean_Depth'].to_numpy(dtype=np.float64), 1e-9)
        width_seed = np.where(width_seed > 0.0, width_seed, fallback_width)
        representative_width = _monotonic_cumulative_max(width_seed)
        representative_depth_increment, representative_depth = _derive_depths_from_width_and_area(
            representative_width,
            representative_area,
        )

        group['Representative_Cross_Sectional_Area'] = representative_area
        group['Representative_Depth_Increment'] = representative_depth_increment
        group['Representative_Top_Width'] = representative_width
        group['Representative_Depth'] = representative_depth
        group['Representative_Velocity'] = group['Mean_Discharge'] / group['Representative_Cross_Sectional_Area'] 
        group['Representative_Stage_Elevation'] = (
            group['Representative_Thalweg_Elevation'].to_numpy(dtype=np.float64) + representative_depth
        )
        group['Representative_Left_Station'] = -0.5 * representative_width
        group['Representative_Right_Station'] = 0.5 * representative_width
        representative_groups.append(group)

    representative_df = pd.concat(representative_groups, ignore_index=True)
    representative_df = representative_df[REPRESENTATIVE_CROSS_SECTION_COLUMNS]

    int_columns = ['COMID', 'Cross_Section_Count', 'Hydraulic_Sample_Count', 'Depth_Stage_Index']
    representative_df[int_columns] = representative_df[int_columns].astype(int)
    return representative_df


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
        self.build_representative_cross_section: bool = params['b_build_representative_cross_section']
        self.representative_cross_section_file: str = params['s_representative_cross_section_file']
        self.b_modified_dem: bool = params['b_modified_dem']

    def associate_with_cross_section(self, x_section: CrossSection):
        """Attach the current :class:`~arc.cross_section.CrossSection` instance."""
        self.x_section = x_section

    def associate_with_output_data(self, output_data: np.ndarray):
        """Attach the shared output array to populate in-place."""
        self.output_data = output_data

    def associate_with_reach_inflect_terrace_index(self, reach_inflect_terrace_index: np.ndarray | None):
        """Attach the per-cell reach terrace indices from the INFLECT prepass."""
        self.reach_inflect_terrace_index = reach_inflect_terrace_index

    def wants_cross_section_records(self) -> bool:
        """Return ``True`` when ARC must retain sampled profile records.

        ``XS_Out_File`` needs the sampled profile arrays directly. The
        representative cross-section export also needs those records because it
        rebuilds a separate 0.10 m stage database from the cross sections up to
        the reach-level INFLECT terrace depth.
        """
        return bool(
            self.s_xs_output_file
            or (
                self.build_representative_cross_section
                and self.representative_cross_section_file
            )
        )
    
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

    def set_vdt_data(self,i_cell_comid: int,  d_q_baseflow: float, d_slope_use: float, i_entry_cell: int):
        """Populate the VDT metadata columns for a stream cell."""
        if self.output_data is None:
            return
        
        i_row_cell, i_column_cell = self.x_section.get_row_col()

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
        
    def set_non_vdt_data(self, i_start_elevation_index: int, i_last_elevation_index: int,
                          i_cell_comid: int, i_row_cell: int, i_column_cell: int, d_slope_use: float, d_dem_low_point_elev: float, i_entry_cell: int):
        """Populate curve-file metadata for non-VDT configurations."""
        if self.output_data is None:
            return
        if self.b_reach_average_curve_file:
            self._set_curve_data(i_cell_comid, i_row_cell, i_column_cell, d_slope_use, d_dem_low_point_elev, i_entry_cell)
        elif self.curve_file and i_start_elevation_index>=0 and i_last_elevation_index>(i_start_elevation_index+1):
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
        """Attach the per-cell cross-section records collected during the run."""
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
    
    def save_files(self, id_flow_dict, qmax_key: str):
        """Write all configured output products.

        Parameters
        ----------
        id_flow_dict : dict
            Mapping from reach ID to (baseflow, qmax) or similar flow metadata.
            Used when building curve-file outputs.
        qmax_key : str
            Name of the qmax field within ``id_flow_dict``.
        """
        vdt_df = None
        if self.vdt_file:
            vdt_df = self.save_vdt()
        if self.ap_file:
            self.save_ap()
        if self.b_reach_average_curve_file:
            self.save_reach_average_curve_file(vdt_df, id_flow_dict, qmax_key)
        elif self.curve_file:
            self.save_curve_file(id_flow_dict, qmax_key)
        if self.s_xs_output_file:
            self.save_cross_section_file()
        if self.build_representative_cross_section and self.representative_cross_section_file:
            self.save_representative_cross_section_file()

    def save_cross_section_outputs_only(self):
        """Write XS/representative exports even when no hydraulic array exists."""
        if self.s_xs_output_file:
            self.save_cross_section_file()
        if self.build_representative_cross_section and self.representative_cross_section_file:
            self.save_representative_cross_section_file()
    
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

        # Round most numeric columns to 3 decimals, but preserve more precision
        # for the diagnostic angle and slope metadata.
        for col in vdt_df.columns:
            if col not in ('Slope', 'XS_Angle'):
                vdt_df[col] = vdt_df[col].round(3)

        # Angle metadata benefits from finer precision than the hydraulic
        # output columns because small orientation changes matter in debugging.
        vdt_df['XS_Angle'] = vdt_df['XS_Angle'].round(7)

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

    def save_reach_average_curve_file(self, vdt_df: pd.DataFrame, id_flow_dict: dict, qmax_key: str):
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
        flow_df = pd.DataFrame.from_dict(id_flow_dict, orient='index')
        reach_average_curvefile_df['QMax'] = reach_average_curvefile_df['COMID'].map(flow_df[qmax_key])

        # Most columns are rounded to 3 decimals, but keep more precision for
        # the cross-section angle metadata and the local slope metadata.
        for col in reach_average_curvefile_df.columns:
            if col not in ('Slope', 'XS_Angle'):
                reach_average_curvefile_df[col] = reach_average_curvefile_df[col].round(3)

        reach_average_curvefile_df['XS_Angle'] = reach_average_curvefile_df['XS_Angle'].round(7)
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

    def save_curve_file(self, id_flow_dict: dict, qmax_key: str):
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
        flow_df = pd.DataFrame.from_dict(id_flow_dict, orient='index')
        o_curve_file_df['QMax'] = o_curve_file_df['COMID'].map(flow_df[qmax_key])
        
        # Round most numeric columns to 3 decimals, but preserve more precision
        # for the diagnostic angle and slope metadata.
        for col in o_curve_file_df.columns:
            if col not in ('Slope', 'XS_Angle'):
                o_curve_file_df[col] = o_curve_file_df[col].round(3)

        o_curve_file_df['XS_Angle'] = o_curve_file_df['XS_Angle'].round(7)

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
        df = pd.DataFrame(cross_section_data)
        df = df[XS_EXPORT_COLUMNS] if not df.empty else pd.DataFrame(columns=XS_EXPORT_COLUMNS)

        # Prepare numpy columns for printing
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: np.array2string(x, precision=6, max_line_width=np.inf, threshold=np.inf, floatmode='fixed')
                if isinstance(x, np.ndarray) else x
            )

        df.to_csv(self.s_xs_output_file, index=False, sep='\t')

        LOG.info('Finished writing ' + str(self.s_xs_output_file))

    def save_representative_cross_section_file(self):
        """Save the reach-level representative cross-section export as CSV.

        The representative cross section is now a long-format hydraulic summary
        with one row per reach and 0.10 m depth stage. Each row stores the
        mean hydraulic properties rebuilt from the sampled cross sections up
        to a 25 meter maximum depth, plus the representative dimensions
        derived from staged top width and staged area.
        """
        cross_section_data = list(self.xs_data) if getattr(self, "xs_data", None) is not None else []
        df = build_representative_cross_section_dataframe(cross_section_data)
        if not df.empty:
            numeric_columns = [
                col
                for col in df.columns
                if col not in ('COMID', 'Cross_Section_Count', 'Hydraulic_Sample_Count', 'Depth_Stage_Index')
            ]
            df[numeric_columns] = df[numeric_columns].round(6)
        df.to_csv(self.representative_cross_section_file, index=False)
        LOG.info('Finished writing ' + str(self.representative_cross_section_file))

@njit(cache=True)
def add_hydraulic_data(output_data: np.ndarray, n: int, wse: float, t: float, p: float, q: float, v: float, i_entry_cell: int, b_modified_dem: bool):
    """Write one increment (q, v, t, wse, p) into the output array."""
    index = 8 + ((n-1) * 5)
    output_data[i_entry_cell, index:index + 5] = [q, v, t, wse - 100 if b_modified_dem else wse, p]
    
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
