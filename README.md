# Malawi ES COVID analyses

Analyses of environmental surveillance and cost-effectiveness using Covasim in Malawi.

## Structure

User scripts are in `scripts`. The Python library code is in `malawi_es`. The webapp code is in `webapp`.

## Installation

1. Ensure you have Python installed (if you haven't installed Python already, the easiest is to use [Anaconda](https://www.anaconda.com/products/individual)).

2. In a terminal (or Windows command prompt), type `pip install -e .` to install (note the "`.`" at the end of the command, this is critical!).

3. If you're using R, you need to have R installed. You also need to install `reticulate` (which allows R to communicate with Python): `install.packages("reticulate")`. **Note:** Even if you're using R, you still need to follow the first two steps to install the Python package.

## Usage

The scripts are in the `scripts` folder. To run a sweep with default values, run `python simple.py`. To use custom values, use `python sweep.py`. To use R instead, use `Rscript sweep.R` or run via Rstudio.

## Troubleshooting

### Variant not implemented

```
NotImplementedError: The selected variant "omicron" is not implemented; choices are:
None
```

This means that you are using an incompatible version of Covasim. **Note:** The Covasim version that includes Omicron has not been publicly released yet. To install, clone the (private) [Covasim repo](https://github.com/amath-idm/covasim) and check out branch `gr-feb2022`. This will be released as the official public Covasim version in the next several weeks.

### Module not found

If you get an error like in either Python or R:

```
Traceback (most recent call last):
  File "malawi/scripts/simple.py", line 5, in <module>
    import malawi_es as mes
ModuleNotFoundError: No module named 'malawi_es'
```

then it means the Python library hasn't been installed properly. Ensure the first two steps of the installation have been complete without errors, and that the Python version that you installed with is the same as the one you're using.