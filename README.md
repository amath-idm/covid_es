# Cost-effectiveness of wastewater-based environmental surveillance for SARS-CoV-2 in Blantyre, Malawi and Kathmandu, Nepal: a model-based study

This code is released in support of the above manuscript, which is currently under review. The full citation will be added upon acceptance.


## Structure

All code used to generate model-based results is in this repository. The structure is as follows:
- The Python library code is in `covasim_es`. 
- Script files used to run and process the simulations for Malawi are in `malawi_scripts`. These scripts are computationally intensive, requiring roughly 1,000 core-hours on a high-performance compute cluster. Do not try to run on your laptop without first setting `debug = True`. The main file used to generate the results is `sweep_lines.py`, but other scripts are included for completeness.
- Equivalent scripts for Nepal are in `nepal_scripts`.
- Scripts used to produce the figures in the manuscript are in `figures`.
- The pre-generated data files loaded by the scripts are stored in `results`. These are in compressed binary format, but can be loaded as pandas dataframes (and then exported to Excel if desired) using [Sciris](https://sciris.org) (specifically, `sc.load()`).


## Installation

1. Ensure you have Python installed (if you haven't installed Python already, the easiest is to use [Anaconda](https://www.anaconda.com); an out-of-the-box system Python installation is unlikely to work).

2. In a terminal (or Windows command prompt), type `pip install -e .` to install (note the "`.`" at the end of the command, this is critical!).

3. To test your installation, `import covasim_es` should work from a Python prompt.

If you're using R, you need to have R installed. You also need to install `reticulate` (which allows R to communicate with Python): `install.packages("reticulate")`. **Note:** Even if you're using R, you still need to follow the first two steps to install the Python package.


## Usage

The code is provided on an as-is basis and there may be practical challenges in getting it to run on a particular system (see note about compute requirements above). However, it should at least be possible to regenerate the figures by running the scripts in the `figures` folder. All scripts should take only seconds to run, except for `fig1_timeseries.py` (which takes 5-60 minutes to run, depending on your laptop).


## Disclaimer

The code in this repository was developed by [IDM](https://idmod.org), [PATH](https://path.org), [MLW](https://www.mlw.mw/), and [ENPHO](https://enpho.org/). We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.
