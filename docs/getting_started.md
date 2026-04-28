# Getting Started

After installing ARC, you can run it from the command line or from Python. Instructions for both are provided below.

## Run from the command line

ARC installs a console script named `arc`.

```bash
arc "path/to/ARC_Input_File.txt"
```

To run with parallel processing (recommended for large domains at high resolution):

```bash
arc "path/to/ARC_Input_File.txt" --processes 4
```

Where `4` is the number of processes to use. If you have a large domain and want to use all available cores, you can set `--processes` to `-1`:
```bash
arc "path/to/ARC_Input_File.txt" --processes -1
```

You can also set processes to 'auto' to determine the number of processes automatically:
```bash
arc "path/to/ARC_Input_File.txt" --processes auto
```

## Run from Python

```python
from arc import Arc

Arc(mifn="path/to/ARC_Input_File.txt").run()
```

Alternatively, you can pass the input parameters as a dictionary instead of an input file:

```python
from arc import Arc

Arc(args={
    "param1": "value1",
    "param2": "value2"
}).run()
```

## Next

- See [Model Input File](model_input_file.md) for the full list of input arguments.
- See [Outputs](outputs.md) for what each output file contains.
