# Python API

`Arc` and the tools that make ARC's inputs are importable from `arc`. `Configs` is in `arc.config`, and `run` and
`Results` are in `arc.pipeline`.

## Running a model

`Arc` runs ARC from Python, as the `arc` command does.

```python
from arc import Arc

results = Arc("ARC_Input_File.yaml").run()
```

::: arc.arc.Arc

## Settings

`Configs`' fields are the input-file keys in lower case, and [Model Input File](model_input_file.md) describes each
one. A `Configs` can't be changed, but `dataclasses.replace` makes a copy with some fields changed (and checks it
again):

```python
from dataclasses import replace

from arc.config import Configs
from arc.pipeline import run

configs = Configs.from_file("ARC_Input_File.yaml")
results = run(replace(configs, vdt_database_numiterations=30))
```

::: arc.config.Configs
    options:
      merge_init_into_class: false
      group_by_category: false
      separate_signature: false

## Running from settings

`run` runs ARC on a `Configs`. It and `Arc.run` return what the run made, whether or not they write it; with
`write=False`, nothing is written:

```python
results = run(configs, write=False)
table = results.curves.output_array()  # a row per stream cell
columns = results.curves.columns()
```

::: arc.pipeline.run

::: arc.pipeline.Results
    options:
      merge_init_into_class: false

## Quality flags

A run's `Results` hold each cell's quality flags (`qa_flags`, see [Decision log](outputs.md#decision-log)), and
`decision_log_dataframe` makes the decision log from them without writing it:

```python
from arc.outputs.decisions import QA, decision_log_dataframe

log = decision_log_dataframe(results)
narrow = log[(log.QA_Flag & int(QA.NARROW_CHANNEL)) > 0]
```

::: arc.outputs.decisions.QA

::: arc.outputs.decisions.decision_log_dataframe

## Preparing inputs

`Create_ARC_Streamflow_Input` makes the flow file from GEOGLOWS's flows (see
[Produce Streamflow Inputs](produce_streamflow_inputs.md)), and `Process_ARC_Geospatial_Data` the rasters, the
Manning's n table and a starter input file from a DEM, land cover and stream lines (see
[Produce Geospatial Inputs](produce_geospatial_inputs.md)).

::: arc.streamflow_processing.Create_ARC_Streamflow_Input

::: arc.process_geospatial_data.Process_ARC_Geospatial_Data

## Flood-mapping inputs

`Run_Main_Curve_to_GEOJSON_Program_Stream_Vector` turns the curve file into the water surface at each stream cell, and
the SEED points, for FIST (see [Making Inputs for FIST](making_inputs_for_fist.md)).

::: arc.create_geojson.Run_Main_Curve_to_GEOJSON_Program_Stream_Vector
