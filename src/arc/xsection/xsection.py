"""The cross section and its exact ground profile, as the hydraulics see them."""
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


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
    distances are -1 until banks are found (arc.bathymetry.set_bank_distances)."""
    elevations: np.ndarray
    mannings_n: np.ndarray
    ordinate_distance: float

    # The following fields are derived later
    left_bank_distance: float = field(init=False, default=-1.0)
    right_bank_distance: float = field(init=False, default=-1.0)
    # A carved channel's exact shape, whose corners can fall between the ordinates (arc.bathymetry.carve_channel).
    # When it's set, the hydraulics use it, and the elevations are only its values at the ordinates.
    profile: Profile | None = field(init=False, default=None)
