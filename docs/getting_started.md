# Getting Started

ARC reads its settings from a model input file (MIF): a YAML file, or a text file with a key and its value on each
line. See [Model Input File](model_input_file.md) for the keys.

## Run from the command line

Installing ARC adds the `arc` command:

```bash
arc path/to/ARC_Input_File.yaml
```

Add `-q` to hide the progress bars and every message but errors, or `-l info` (or `debug`) to see more of what ARC
is doing. `arc -h` lists the options.

## Run from Python

```python
from arc import Arc

Arc("path/to/ARC_Input_File.yaml").run()
```

You can also give the inputs as a dictionary, with the same keys as the input file:

```python
from arc import Arc

Arc(args={
    "DEM_File": "path/to/dem.tif",
    "Stream_File": "path/to/streams.tif",
    # ...
}).run()
```

`run()` writes the outputs and also returns them (the cross sections, rating curves, bathymetry and so on), so a
script can use them directly. To change settings in a script, or to run without writing anything, see
[Python API](api.md).

## Next

- [Model Input File](model_input_file.md): every input and what it does.
- [Outputs](outputs.md): what each output file contains.
- [Python API](api.md): running ARC from a script.
