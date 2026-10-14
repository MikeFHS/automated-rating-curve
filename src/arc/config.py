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

StreamSlopeMethod = Literal["local_average", "local_average_corrected", "reach_average", "end_points"]

_CURVE2FLOOD_KEYS = {'comid_flow_file', 'make_output_gpkg', 'strmorder_field', 'outfld', 'outdep', 'outvel', 'outwse', 'mapper', 'topwidthplausiblelimit', 'tw_multfact', 'set_depth', 'localfloodoption', 'fsoutbathy', 'flood_waterlc_and_strm_cells', 'flow_direction_file', 'filled_dem_file', 'stream_info_file', 'fldpln_library', 'max_wse_rise', 'fldpln_median_filter_size', 'fldpln_max_drop_below_source', 'fs_adjust_flow_by_fraction',
                     # curve2flood's aliases, and the keys nencarta writes into the ARC input files it shares with curve2flood
                     'floodspreader_specifydepth', 'floodlocalonly', 'land_watervalue', 'fldpln_max_wse_rise', 'median_filter_size', 'max_drop_below_source', 'outshp'}
# Keys that ARC input files often carry (Create_ARC_Model_Input_File writes some of them) but ARC doesn't use
_IGNORED_KEYS = {'spatial_units', 'str_limit_val', 'print_vdt'}


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
    # reads it too, and keeps the bathymetry only inside it.
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
    x_section_dist: float = 5000.0
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
    findbanksbasedonlandcover: bool = False

    # Synthetic rating curve configuration
    vdt_database_numiterations: int = 15
    # Link the water surfaces along the stream network by the standard-step method (arc.standard_step), instead of
    # each cross section's normal depth
    standard_step: bool = False

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

        known = {f.name for f in fields(cls) if f.init}
        unknown = sorted(set(params) - known - _CURVE2FLOOD_KEYS - _IGNORED_KEYS)
        if unknown:
            described = []
            for key in unknown:
                close = difflib.get_close_matches(key, sorted(known), n=1)
                described.append(f'{key!r}' + (f' (did you mean {close[0]!r}?)' if close else ''))
            raise ValueError(f'Unrecognized config key(s): {", ".join(described)}')
        # The curve2flood and ignored keys are allowed in the inputs, but aren't fields
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
