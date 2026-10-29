"""Bathymetry: finding each cross section's banks, how deep its channel is, and carving the channel in.

The steps:

1. find_banks: each cross section's banks (arc.bathymetry.banks).
2. smooth_bank_elevations: along each reach, widths far from the reach's median (its bank tops', where it has enough)
   replaced by it, and the bank elevation smoothed, then ramped to meet the next reach's at each junction on a main
   stem (arc.bathymetry.smoothing).
3. bathymetry_depth: how deep each channel is, to carry its baseflow or from a drainage-area power law
   (arc.bathymetry.depth).
4. smooth_channel_depths: along each reach, missing depths filled in and the channel bed smoothed
   (arc.bathymetry.bed_smoothing).
5. carve_channel: the channel carved into the cross section (arc.bathymetry.channel); burn_into_raster writes it into
   a sparse raster and fill_sparse_gaps fills its gaps (arc.bathymetry.raster).

Banks are distances from the stream cell, so a bank can fall between ordinates and a channel can
be any width, including narrower than a cell. Every channel is a trapezoid, and the hydraulics see its exact shape.
"""
from arc.bathymetry.banks import (
                                  Banks,
                                  bank_control_elevation,
                                  banks_by_flat_water,
                                  banks_by_land_cover,
                                  banks_by_width_to_depth_ratio,
                                  banks_for_width,
                                  find_banks,
                                  in_bank,
                                  set_bank_distances,
                                  set_in_bank_roughness,
                                  single_cell_banks,
)
from arc.bathymetry.bed_smoothing import (
                                  ChannelDepths,
                                  fill_reach_depths,
                                  smooth_channel_depths,
                                  smooth_reach_bed,
)
from arc.bathymetry.channel import carve_channel
from arc.bathymetry.depth import (
                                  BathymetryDepth,
                                  bathymetry_depth,
                                  channel_depth,
                                  power_law_geometry,
                                  trapezoid_depth,
)
from arc.bathymetry.raster import (
                                  Burns,
                                  SparseRaster,
                                  burn_into_raster,
                                  fill_sparse_gaps,
                                  ordinate_cells,
                                  sample_land_cover,
)
from arc.bathymetry.smoothing import (
                                  ReachSections,
                                  ReachWidths,
                                  SmoothedReach,
                                  reach_network,
                                  smooth_bank_elevations,
)

__all__ = [
    "Banks", "BathymetryDepth", "Burns", "ChannelDepths", "ReachSections", "ReachWidths", "SmoothedReach",
    "SparseRaster", "bank_control_elevation", "banks_by_flat_water", "banks_by_land_cover",
    "banks_by_width_to_depth_ratio", "banks_for_width", "bathymetry_depth", "burn_into_raster", "carve_channel",
    "channel_depth", "fill_reach_depths", "fill_sparse_gaps", "find_banks", "in_bank",
    "ordinate_cells", "power_law_geometry", "reach_network", "sample_land_cover", "set_bank_distances",
    "set_in_bank_roughness", "single_cell_banks", "smooth_bank_elevations", "smooth_channel_depths", "smooth_reach_bed",
    "trapezoid_depth",
]
