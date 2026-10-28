# Limitations

What ARC can't do, or does with a known bias, and what to watch for. The [decision log and QA flags](outputs.md#decision-log)
mark the cells where several of these apply.

!!! danger "Match `Flow_File_BF` to `Bathy_Use_Banks`"
    The most common way to get the bathymetry badly wrong is a baseflow that doesn't go with the level the channel is
    taken to be full to. `Flow_File_BF` is the flow that fills the channel exactly to that level: the DEM's water
    surface with `Bathy_Use_Banks: False` (a low flow), the bank elevation with `Bathy_Use_Banks: True` (bankfull). A
    mismatched pair carves the bed metres too deep or fills it in metres too high, with no error or warning, since
    ARC can't tell which flow was meant. A water mask doesn't change which flow: with `Bathy_Use_Banks` the banks are
    the bank tops, mask or not. See [Water surface or banks](bathymetry.md#water-surface-or-banks).

## Narrow channels and the DEM's cells

A DEM's cell is the mean of the ground over it, and ARC joins its cross sections' cells with straight lines, which
turns a narrow channel's corners into ramps. The channel holds too little water, so the water that carries a flow
stands too high, and most of all at a constriction, where the narrowest channel holds the backwater up. On a synthetic
channel narrowing from a 60 m bed to 30 m, ARC's backwater peak was 0.07 m too high on 5 m cells, 0.30 m on 10 m,
1.05 m on 20 m and 1.91 m on 30 m, against 3.4 m of real backwater. Normal depth is as far out.

As a rule, where the water level matters the narrowest channel's bed should be about six cells across or more. On a
30 m DEM that is most rivers under about 180 m wide. `Subcell_Channels: True` reconstructs the ground of channels
under 10 cells across below the cell scale (see [Model Input File](model_input_file.md#cross-sections-and-slopes)): it brings the
synthetic errors down, but it costs about half a millisecond a narrow cross section. Cells whose
channel is narrow and not reconstructed carry the `NARROW_CHANNEL` flag.

## Steep reaches

ARC's hydraulics are subcritical. With `Standard_Step`, a reach too steep for subcritical flow gets critical depth,
which is above its real, supercritical water: in a wide channel critical depth is Fr<sup>2/3</sup> times the
supercritical depth, 1.6 times at a Froude number of 2. On Chow's steep reach ARC gives 2.15 ft where the water is
2.0 ft. There's no hydraulic jump either: below a fall the water stays at critical depth until the tailwater's energy
reaches it. Cells where this happens carry the `STEP_CRITICAL` flag. Without `Standard_Step`, each cross section is at
Manning's normal depth whatever the flow's regime, so a steep reach's water is its normal depth, but nothing links it
to the water above or below.

## Slopes

- **DEM noise makes slopes steep.** The local and reach-average slope methods measure the falls between stream cells,
  and count each rise between them, from noise or a dip, as a fall the other way: 2 cm of noise along a test channel
  made their slopes two to three times the bed's. These are the strict expected failures in
  `testing/test_dem_resilience.py`. `local_fit` is the least affected (see
  [Which method to use](stream_slope_methods.md#which-method-to-use)).
- **Oblique streams' slopes are gentle.** A stream running across the grid at an angle is longer than its cells'
  steps make it, and its slopes came out 7% too gentle at 30 degrees.
- **Hydro-conditioned DEMs.** Streams burned into a DEM can leave dead-flat runs, where the slope is the floor, and
  trenches deeper than the real channel. ARC takes the DEM as it is.
- **A reach without a slope of its own** takes its neighbours' in the stream network (`SLOPE_FROM_NEIGHBOURS`); one
  that has neither is skipped (`NO_SLOPE`), and has no rating curves.

## Banks and bathymetry

- **Banks the DEM can't show.** A channel too narrow for the water mask, the land cover or the DEM to show is as wide
  as the width power law says, or one cell without it (`NO_BANKS_FOUND`). On 30 m DEMs that is most small streams, so
  give the width power law.
- **Canyons and levees.** With `Bathy_Use_Banks`, no bank is taken above the water that carries the maximum flow, so a
  canyon's bench above any flood isn't, but one the maximum flow reaches is, as is the rim of a trench burned into a
  conditioned DEM. A levee's crest is missed: the banks stop at the channel's edge, about 0.3 m below it. A compound
  channel's banks are often its inner terrace, which can be the true bankfull level.
- **With a maximum flow below bankfull,** the banks are no higher than its water, below the true bank tops: with
  `Bathy_Use_Banks` at the highest level below it, and without bathymetry at its edges. The hydraulics divide any water
  the standard step backs up above it there, behind a lake, a confluence or an outlet's given water: on Chow's channel
  behind a lake, the water came out up to 0.06 ft (2 cm) low.
- **Banks found far apart.** The carved channel spans the banks found, so banks found on the far side of a valley, or
  on the walls of a canyon, carve ground that isn't channel. The bank dip rule and the width filter catch some of
  these, not all.
- **Banks without a channel.** A cross section without banks of its own, on a small flow, is carved to carry its
  baseflow at its stream's level (`CARVED_TO_STREAM`, `Bathy_Bankless_Max_Baseflow`) rather than at its reach's bank
  height, which could stand above the ground beside it.

## Rating curves and flood maps

- **Cells without rating curves.** A cross section whose water spills over a bank before it carries the maximum flow,
  to a discharge more than half the maximum flow away, gets no rating curve (`NO_RATING_CURVE`); one whose top
  discharge isn't the maximum flow but within half of it is flagged `CURVE_SPILLS`. Curve2Flood fills between the
  rated cells, and its flood maps can spread thin films of water from unrated cells at a channel's lip.
- **Cross sections only.** There are no bridges, culverts, weirs, levees with gaps, or junction losses: the ground
  the DEM shows is all there is.
- **Steady flow, one flow a reach.** Each reach carries its own flows all along it, with no inflow between its ends
  and no storage or attenuation.
- **Representative cross sections.** They average the cross sections whose area, velocity and top width are within
  two standard deviations of their reach's means. On cross sections that are all alike, rounding decides which those
  are, so their counts can change from run to run.
