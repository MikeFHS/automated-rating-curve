# Representative cross sections

With `Build_Representative_Cross_Section: true`, ARC exports one representative
hydraulic-stage table per reach and a companion celerity-K coefficient table.

The calculations are implemented in `arc.outputs.representative`
(`src/arc/outputs/representative.py`). The pipeline calls
`build_representative_cross_section_dataframe` to compute staged hydraulics and
`write_representative_tables` to write the stage and coefficient CSVs.
`REPRESENTATIVE_STAGE_COLUMNS` defines the 22-column stage schema.

ARC samples the cross sections, detects banks, estimates bathymetry and smooths
the channel before calculating representative hydraulics. It can also generate
rating curves and link their water surfaces using the standard-step method.
Representative stage discharge is calculated independently of those curves.

```yaml
Build_Representative_Cross_Section: true
Representative_Cross_Section_File: representative.csv
StrmShp_File: streams.gpkg
Stream_Length_m: length_m
```

`Stream_Length_m` names an existing attribute containing finite positive full
reach lengths in **meters**. The input alias is `s_stream_length_field`.
No lengths are estimated from the processed geometry or converted between units.
The typed input API also accepts `s_stream_length_field` and
`b_build_representative_cross_section` as aliases for their external keys.
Missing/duplicate IDs and invalid lengths raise errors.

The export consumes the local XS file's two uniform outward-facing profiles,
roughness arrays, thalweg, slope and local bank indices. Native continuous bank
distances are mapped to the nearest sampled ordinate for the discrete bank
indices used by the representative calculation. The local cross sections themselves are never modified by this adapter.

Representative hydraulics are computed every 0.10 m to 25 m, stopping
if a nonfinite hydraulic result occurs. At each stage they filter area, velocity,
top width, perimeter and discharge within two standard deviations before taking
means. `k_decay`, `shallow_factor`, `deep_factor` and `slope_adjustment_factor`
control depth-varying roughness and discharge scaling for the representative calculation.
Depth scaling is intrinsic to that calculation even when the local rating-curve
engine's `Depth_Varying_N` is false. Local bathymetry and rating curves retain
their original roughness settings.

Representative geometry uses the filtered mean area, width and depth at each
stage. These values are calculated directly from the cross-section profiles.
`Representative_Depth_Increment` contains the absolute stage depth in meters.
Representative velocity is the channel velocity when valid banks exist, and
whole-section velocity otherwise.

`representative.csv` has 22 stage columns, with six-decimal numeric
rounding. `representative_exponents.csv` has `COMID`, `Cross_Section_Count`,
`Celerity_K_Power_Law_A` and `Celerity_K_Power_Law_p`. The fit is
`K = A Q^p`, using `K = length / ((5/3) * velocity)`. A is positive; p keeps its
fitted sign. Failed fits leave NaN coefficients while preserving stage rows.
Both files are CSV, even if the requested filename has a `.parquet` suffix.

`Results.representative` contains the unrounded combined calculation table,
including repeated coefficient columns for internal use. The writer separates
those columns into the companion file. With `pipeline.run(..., write=False)`
the table is returned without writing either file.

The local API's existing output choices remain: asking for VDT, AP or curve
files alongside this export still generates them using the native workflow.
Without those outputs, the existing representative-only flow/bathymetry rules
apply and no QMax is needed. NenCarta's `representative_cross_section_creation`
mode omits those rating-curve outputs and disables all mapper calls.
