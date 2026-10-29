"""The cross section and its exact ground profile, as the hydraulics see them."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from arc.xsection.valley import Valley


class Profile(NamedTuple):
    """A cross section's ground as the hydraulics see it: a polyline through vertices at these distances from the
    stream cell (negative to the left), with these elevations and Manning's n. The distances never fall, and two
    vertices at the same distance make a vertical face. The stream cell's vertex, at distance 0, is at index center."""
    stations: np.ndarray
    elevations: np.ndarray
    mannings_n: np.ndarray
    center: int


@dataclass(eq=False, repr=False)
class XSection:
    """A cross section: its ground elevations and Manning's n at ordinates ordinate_distance metres apart, from its
    left end to its right (see arc.xsection.sampling for the convention), with the stream cell in the middle. The bank
    distances are -1 until they're set from the banks the hydraulics use (arc.bathymetry.set_bank_distances), which
    arc.pipeline does after the bathymetry."""
    elevations: np.ndarray
    mannings_n: np.ndarray
    ordinate_distance: float

    # The following fields are derived later
    left_bank_distance: float = field(init=False, default=-1.0)
    right_bank_distance: float = field(init=False, default=-1.0)
    # A carved channel's exact shape, whose corners can fall between the ordinates (arc.bathymetry.carve_channel).
    # When it's set, the hydraulics use it, and the elevations are only its values at the ordinates.
    profile: Profile | None = field(init=False, default=None)
    # With Valley_Sections, the overbanks from lines out from the banks square to the valley (arc.xsection.valley):
    # when it's set, the hydraulics take the channel from this cross section and the overbanks from those
    valley: "Valley | None" = field(init=False, default=None)
    # With Subcell_Channels (arc.xsection.subcell): the ordinates as they were before the bathymetry first carved them,
    # and the banks it carved between, and whether the profile has the channel reconstructed below the cell scale
    ground: np.ndarray | None = field(init=False, default=None)
    carved_banks: tuple[float, float] | None = field(init=False, default=None)
    subcell: bool = field(init=False, default=False)
    # The last spliced profile and what it was spliced from (arc.hydraulics.hydraulic_profile), which it knows by
    # identity: arrays changed in place after a splice aren't seen, so replace them instead. Only the most recent
    # splices are kept (arc.hydraulics.KEPT_SPLICES)
    _spliced: tuple | None = field(init=False, default=None)
