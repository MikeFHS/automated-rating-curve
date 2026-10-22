"""ARC's inputs: the model input file (MIF), as YAML, as text, or as a dictionary."""
import os
import difflib
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints
from dataclasses import dataclass, field, fields

import yaml
import numpy as np

from arc._log import LOG

StreamSlopeMethod = Literal["local_average", "local_average_corrected", "reach_average", "end_points", "local_fit"]

# curve2flood's keys (curve2flood.config.Config) that aren't ARC's: one input file often drives both models, so they
# are allowed and ignored
_CURVE2FLOOD_KEYS = {
    'comid_flow_file', 'outfld', 'outdep', 'outwse', 'outvel', 'make_output_gpkg', 'mapper', 'tw_multfact',
    'topwidthplausiblelimit', 'localfloodoption', 'flood_waterlc_and_strm_cells', 'flood_bathy_mask', 'land_watervalue',
    'set_depth', 'flow_direction_file', 'filled_dem_file', 'stream_info_file', 'fldpln_library', 'fsoutbathy',
    'fldpln_min_flood_depth',
    # Its multi-point interpolation mapper's options
    'mpi_topwidth_threshold_m', 'mpi_xs_point_spacing_m', 'mpi_remove_hwm_outliers', 'mpi_hwm_outlier_method',
    'mpi_hwm_outlier_threshold', 'mpi_hwm_outlier_group_by', 'mpi_hwm_outlier_min_samples', 'mpi_corridor_buffer_m',
    'mpi_anchor_buffer_m', 'mpi_use_topwidth_buffers', 'mpi_corridor_topwidth_factor', 'mpi_anchor_topwidth_factor',
    'mpi_connectivity', 'mpi_k', 'mpi_power', 'mpi_max_distance_m', 'mpi_use_topwidth_max_distance',
    'mpi_maxdist_topwidth_factor', 'mpi_min_topwidth_m', 'mpi_max_topwidth_m', 'mpi_fallback_topwidth_m',
    'mpi_smooth_sigma_pixels', 'mpi_apply_wse_sanity_filter', 'mpi_wse_sanity_tolerance_m', 'mpi_fast_mode',
    'mpi_one_based_vdt_rc',
    # Its other names for set_depth and localfloodoption, from FloodSpreader
    'floodspreader_specifydepth', 'floodlocalonly',
    # Its retired keys, which it ignores with a warning
    'max_wse_rise', 'fldpln_max_wse_rise', 'median_filter_size', 'fldpln_median_filter_size', 'max_drop_below_source',
    'fldpln_max_drop_below_source',
    # Keys nencarta writes into the input files it shares with curve2flood, which neither model uses
    'outshp', 'fs_adjust_flow_by_fraction'}
# Keys that ARC input files often carry (Create_ARC_Model_Input_File writes some of them) but ARC doesn't use
_IGNORED_KEYS = {'spatial_units', 'str_limit_val', 'print_vdt'}
# Keys ARC used to read and no longer does, with why: they're ignored with a warning
_RETIRED_KEYS = {'x_section_dist': "each cross section reaches as far as its water needs, out to the raster's edge "
                                   "(see arc.pipeline)"}


@dataclass(kw_only=True, frozen=True)
class Configs:
    """ARC's inputs, checked and typed.

    Make one with from_file (YAML, or text with a key and its value on each line) or from_mapping (a dictionary).
    Keys are case-insensitive, so DEM_File and dem_file are the same. Keys that curve2flood reads from the same file
    are allowed and ignored; any other unknown key raises ValueError, naming the closest known key. The fields after
    compression are worked out from the others, by validate_options.
    """
    # Inputs files
    dem_file: os.PathLike | None = None
    stream_file: os.PathLike | None = None
    strmshp_file: os.PathLike | None = None
    flow_file: os.PathLike | None = None
    lu_raster_sameres: os.PathLike | None = None
    lu_manning_n: os.PathLike | None = None
    manual_cross_sections_file: os.PathLike | None = None
    # Where the water is (its non-zero cells), on the DEM's grid: the banks are where its water ends. curve2flood
    # reads it too, only to flood its cells (Flood_Bathy_Mask): it burns the bathymetry in wherever ARC carved.
    bathywatermask: os.PathLike | None = None

    # Input file field names
    reach_id: str = "Reach_ID"
    downstream_reach_id: str = "Downstream_Reach_ID"
    flow_file_id: str | None = None
    flow_file_bf: str | None = None
    flow_file_qmax: str | None = None
    lc_water_value: int = 80
    drainage_area_field: str | None = None
    coefficient_depth: float | None = None
    exponent_depth: float | None = None
    coefficient_width: float | None = None
    exponent_width: float | None = None

    # Cross section orientation
    degree_manip: float = 1.1
    degree_interval: float = 1.0
    low_spot_range: int = 0
    gen_dir_dist: int = 10
    gen_slope_dist: int = 10
    stream_slope_method: StreamSlopeMethod = "local_average_corrected"
    slope_low_percentile: int = 25
    slope_high_percentile: int = 75

    # Discharge adjustments
    slope_adjustment_factor: float = 1.0
    # Scale Manning's n with the depth of the water by k_decay, shallow_factor and deep_factor
    # (arc.hydraulics.DepthRoughness). False keeps each land cover's n at every depth.
    depth_varying_n: bool = True
    k_decay: float = 6.0
    shallow_factor: float = 2.0
    deep_factor: float = 1.0

    # Bathymetry configuration
    bathy_trap_h: float = 0.2
    bathy_use_banks: bool = False
    # Hold the smoothed channel bed to rising or falling no more than MAX_SLOPE (0.5 m) per metre along the stream
    # (arc.bathymetry.smooth_channel_depths)
    bathy_bed_cap: bool = True
    # Smooth each reach's channel widths and bank elevations (arc.bathymetry.smooth_bank_elevations), and its bed
    # (arc.bathymetry.smooth_channel_depths). False leaves each cross section's own. Unless given, the bed is smoothed
    # only without the standard step and without bank elevations (Bathy_Use_Banks): a smoothed bed is a neighbour's,
    # too shallow for a narrow channel to carry the baseflow in bank and too deep for a wide one. The standard step's
    # water rides above the dips that leaves in the normal depths (on the Karnali's main stem, 0.9 m higher at the
    # baseflow). Without it, each rating curve is its own cross section's, so with bank elevations, where the baseflow
    # should fill each channel to its banks, the water lands metres above or below them: on the Karnali, where
    # water-mask banks 54 m apart sit next to the reach median's 210 m, 14 m above them, and 465 cells' water below
    # the DEM.
    bathy_smooth_banks: bool = True
    bathy_smooth_bed: bool | None = None
    # With Bathy_Use_Banks, a cross section whose banks weren't found (none, or a single cell from the width prior)
    # and whose baseflow is below this (m³/s) carries its baseflow at its stream cell's level rather than at its
    # reach's bank elevation (arc.pipeline.apply_bathymetry). 0 turns it off.
    bathy_bankless_max_baseflow: float = 20.0
    findbanksbasedonlandcover: bool = False

    # Synthetic rating curve configuration
    vdt_database_numiterations: int = 15
    # Link the water surfaces along the stream network by the standard-step method (arc.standard_step), instead of
    # each cross section's normal depth
    standard_step: bool = False
    # How far downstream (m) the standard step reckons each step's eddy loss over (arc.standard_step.EDDY_LENGTH):
    # 0 charges each step the change from the cross section below alone
    standard_step_eddy_length: float = 300.0
    # Take the overbanks' flow on lines out from the banks square to the valley, at the valley's slope, and the
    # channel's on the cell's own cross section (arc.xsection.valley): over the floodplain of a meandering river the
    # water runs down the valley, not along the channel, and Manning's equation takes the flow square to the section
    valley_sections: bool = False
    # How far each way along the stream the valley's direction and sinuosity are taken over (a Gaussian of half that,
    # arc.xsection.valley.valley_axes), about a meander's length: as many of the reach's channel widths (its cross
    # sections' median bank-to-bank width), or with 0, or for a reach without banks, Valley_Dir_Dist cells
    valley_dir_widths: float = 16.0
    valley_dir_dist: int = 40

    # Output files
    print_vdt_database: os.PathLike | None = None
    print_ap_database: os.PathLike | None = None
    print_curve_file: os.PathLike | None = None
    xs_out_file: os.PathLike | None = None

    reach_average_curve_file: bool = False
    build_representative_cross_section: bool = False
    representative_cross_section_file: os.PathLike | None = None
    bathy_out_file: os.PathLike | None = None
    aroutflood: os.PathLike | None = None

    compression: str = "ZSTD"

    # Derived by validate_options(), so they cannot be set from the inputs
    makes_rating_curves: bool = field(init=False, default=True)
    use_representative_baseflow_bathymetry: bool = field(init=False, default=False)
    use_bathymetry_powerlaw: bool = field(init=False, default=False)
    use_bathymetry_powerlaw_width: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._coerce_types()
        self.validate_options()


    @classmethod
    def from_mapping(cls, params: dict):
        """The inputs from a dictionary of the input file's keys and values."""
        return cls(**cls._normalize_keys(params))

    @classmethod
    def from_file(cls, file_path: os.PathLike):
        """The inputs from an input file: YAML for a .yaml or .yml file, and otherwise text."""
        if Path(file_path).suffix.lower() in ('.yaml', '.yml'):
            return cls.from_yaml(file_path)
        else:
            return cls.from_text_file(file_path)


    @classmethod
    def from_yaml(cls, yaml_name: os.PathLike):
        with open(yaml_name, 'r') as o_input_file:
            params = yaml.load(o_input_file, Loader=yaml.CSafeLoader)
        return cls.from_mapping(params or {})

    @classmethod
    def from_text_file(cls, mif_name: os.PathLike):
        """The inputs from a text file with a key, a space or tab, and its value on each line. Blank lines and lines
        starting with # are skipped."""
        with open(mif_name, 'r') as o_input_file:
            params = {}
            for line in o_input_file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    params[key.strip()] = value.strip()

        return cls.from_mapping(params)

    @classmethod
    def _normalize_keys(cls, params: dict) -> dict:
        # Keys are case-insensitive, so DEM_File is dem_file
        params = {str(k).lower(): v for k, v in params.items()}

        # AROutBATHY is another name for BATHY_Out_File, and wins when both are given
        aroutbathy = params.pop("aroutbathy", None)
        if aroutbathy:
            params["bathy_out_file"] = aroutbathy

        for obsolete_name in ('f_min', 'alpha_boost', 'alpha_low'):
            if obsolete_name in params:
                raise ValueError(f'{obsolete_name} has been replaced by shallow_factor and deep_factor; update your ARC inputs.')

        for retired, reason in _RETIRED_KEYS.items():
            if retired in params:
                LOG.warning(f"{retired} is no longer used, and is ignored: {reason}.")

        known = {f.name for f in fields(cls) if f.init}
        unknown = sorted(set(params) - known - _CURVE2FLOOD_KEYS - _IGNORED_KEYS - set(_RETIRED_KEYS))
        if unknown:
            described = []
            for key in unknown:
                close = difflib.get_close_matches(key, sorted(known), n=1)
                described.append(f'{key!r}' + (f' (did you mean {close[0]!r}?)' if close else ''))
            raise ValueError(f'Unrecognized config key(s): {", ".join(described)}')
        # The curve2flood, ignored and retired keys are allowed in the inputs, but aren't fields
        return {key: value for key, value in params.items() if key in known}

    def _coerce_types(self) -> None:
        # Text input files give every value as a string, so convert each one to its field's type
        hints = get_type_hints(type(self))
        for f in fields(self):
            if f.init:
                object.__setattr__(self, f.name, _coerce(f.name, getattr(self, f.name), hints[f.name]))

    def validate_options(self):
        """Check the inputs, and work out the bathymetry options and whether the run makes rating curves."""
        if self.stream_slope_method not in get_args(StreamSlopeMethod):
            raise ValueError(
                f"stream_slope_method must be one of {', '.join(get_args(StreamSlopeMethod))}, "
                f"not {self.stream_slope_method!r}."
            )
        if self.stream_slope_method == "end_points" and not self.strmshp_file:
            raise ValueError("The 'end_points' stream slope method requires a strmshp_file to be specified.")
        if self.standard_step and not self.strmshp_file:
            raise ValueError("The standard step needs the stream network: StrmShp_File, with reach_id and "
                             "downstream_reach_id.")
        if self.stream_slope_method != "end_points" and self.gen_slope_dist < 1:
            raise ValueError(f"gen_slope_dist must be at least 1 for the {self.stream_slope_method!r} stream slope "
                             "method, which measures slopes between stream cells that many cells apart or nearer.")

        if self.build_representative_cross_section and not self.representative_cross_section_file:
            raise ValueError("Building a representative cross section requires a representative_cross_section_file to be specified.")

        # A run makes rating curves unless it only builds representative cross sections. Asked for an output of the
        # rating curves as well, it makes both from the same cross sections, and carves its channels as it would
        # without the representative cross sections, so that they don't change the rating curves.
        makes_rating_curves = bool(not self.build_representative_cross_section or self.print_vdt_database
                                   or self.print_ap_database or self.print_curve_file)
        representative_only = self.build_representative_cross_section and not makes_rating_curves

        # Check if the coefficients are provided and drainage area. Otherwise, raise an error.
        provided_flags = {
            'drainage_area_field': bool(self.drainage_area_field),
            'coefficient_depth': self.coefficient_depth is not None,
            'exponent_depth': self.exponent_depth is not None,
            'coefficient_width': self.coefficient_width is not None,
            'exponent_width': self.exponent_width is not None,
        }
        depth_provided = provided_flags['coefficient_depth'] or provided_flags['exponent_depth']
        width_provided = provided_flags['coefficient_width'] or provided_flags['exponent_width']
        any_powerlaw_provided = depth_provided or width_provided

        if any_powerlaw_provided and not provided_flags['drainage_area_field']:
            raise ValueError(
                'drainage_area_field is required when bathymetry power-law '
                'coefficients or exponents are provided.'
            )

        if depth_provided and provided_flags['coefficient_depth'] != provided_flags['exponent_depth']:
            raise ValueError(
                'The depth bathymetry power law requires coefficient_depth and '
                'exponent_depth together.'
            )

        if width_provided and provided_flags['coefficient_width'] != provided_flags['exponent_width']:
            raise ValueError(
                'The width bathymetry power law requires coefficient_width and '
                'exponent_width together.'
            )

        if any_powerlaw_provided and not self.strmshp_file:
            raise ValueError(
                "StrmShp_File is required when configuring drainage-area bathymetry "
                "parameters because ARC reads the drainage area attribute from that dataset."
            )

        # A run that only builds representative cross sections carves its channels for the baseflow when it has the
        # flow file, its ID and its baseflow column, ahead of the depth power law
        representative_baseflow_inputs = {
            'Flow_File': self.flow_file,
            'Flow_File_ID': self.flow_file_id,
            'Flow_File_BF': self.flow_file_bf,
        }
        b_use_representative_baseflow_bathymetry = bool(
            representative_only
            and all(representative_baseflow_inputs.values())
        )
        if (
            representative_only
            and any(representative_baseflow_inputs.values())
            and not b_use_representative_baseflow_bathymetry
        ):
            missing = [
                name for name, value in representative_baseflow_inputs.items()
                if not value
            ]
            LOG.warning(
                'Representative baseflow bathymetry requires Flow_File, '
                'Flow_File_ID, and Flow_File_BF together. Missing: '
                + ', '.join(missing)
                + '. Falling back to drainage-area power-law bathymetry.'
            )

        b_use_bathymetry_powerlaw = bool(
            any_powerlaw_provided
            and not b_use_representative_baseflow_bathymetry
        )
        b_use_bathymetry_powerlaw_width = bool(
            provided_flags['coefficient_width']
            and provided_flags['exponent_width']
        )

        if representative_only:
            has_bathymetry_source = bool(
                b_use_representative_baseflow_bathymetry
                or b_use_bathymetry_powerlaw
                or b_use_bathymetry_powerlaw_width
            )
        else:
            has_bathymetry_source = bool(
                self.flow_file_bf or b_use_bathymetry_powerlaw
            )
        if self.bathy_out_file and not has_bathymetry_source:
            LOG.warning(
                'Neither complete baseflow inputs nor drainage-area bathymetry '
                'parameters were provided; disabling bathymetry estimation.'
            )
            object.__setattr__(self, 'bathy_out_file', None)
        if self.bathy_out_file:
            if not self.reach_id or not self.downstream_reach_id:
                raise ValueError(
                    'Bathymetry output requires both reach_id and downstream_reach_id '
                    'to be provided in the input file.'
                )

        # The dataclass is frozen, so derived values are set with object.__setattr__
        if self.bathy_smooth_bed is None:
            object.__setattr__(self, 'bathy_smooth_bed', not (self.standard_step or self.bathy_use_banks))
        object.__setattr__(self, 'makes_rating_curves', makes_rating_curves)
        object.__setattr__(self, 'use_representative_baseflow_bathymetry', b_use_representative_baseflow_bathymetry)
        object.__setattr__(self, 'use_bathymetry_powerlaw', b_use_bathymetry_powerlaw)
        object.__setattr__(self, 'use_bathymetry_powerlaw_width', b_use_bathymetry_powerlaw_width)

        # A reach-average curve file is only written when there is a curve file to write
        if self.reach_average_curve_file and not self.print_curve_file:
            object.__setattr__(self, 'reach_average_curve_file', False)

        if not np.isfinite(self.slope_adjustment_factor) or self.slope_adjustment_factor <= 0.0:
            raise ValueError('slope_adjustment_factor must be finite and positive.')
        if not np.isfinite(self.k_decay) or self.k_decay <= 0.0:
            raise ValueError('k_decay must be finite and positive.')
        if not self.bathy_bankless_max_baseflow >= 0.0:
            raise ValueError('Bathy_Bankless_Max_Baseflow must be 0 or more.')
        if not (np.isfinite(self.standard_step_eddy_length) and self.standard_step_eddy_length >= 0.0):
            raise ValueError('Standard_Step_Eddy_Length must be finite and 0 or more.')
        if self.valley_sections and not self.valley_dir_dist >= 2:
            raise ValueError('Valley_Dir_Dist must be at least 2 cells.')
        if self.valley_sections and not (np.isfinite(self.valley_dir_widths) and self.valley_dir_widths >= 0.0):
            raise ValueError('Valley_Dir_Widths must be finite and 0 or more.')
        if not np.isfinite(self.shallow_factor) or self.shallow_factor < 1.0:
            raise ValueError('shallow_factor must be finite and >= 1.')
        if not np.isfinite(self.deep_factor) or not 0.0 < self.deep_factor <= 1.0:
            raise ValueError('deep_factor must be finite and in (0, 1].')


def _to_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _coerce(name: str, value, annotation):
    """Convert a raw input value to the type given by its field annotation."""
    members = get_args(annotation) if get_origin(annotation) in (Union, UnionType) else (annotation,)
    if type(None) in members and (value is None or value == ''):
        return None
    if bool in members:
        return _to_bool(value)
    for kind, description in ((int, 'an integer'), (float, 'a number')):
        if kind in members:
            try:
                if kind is int and isinstance(value, float) and not value.is_integer():
                    raise ValueError
                return kind(value)
            except (TypeError, ValueError):
                raise ValueError(f'{name} must be {description}, not {value!r}.') from None
    if str in members and value is not None:
        return str(value)
    return value
