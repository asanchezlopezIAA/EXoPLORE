# Guided EXoPLORE notebooks

[Your first ANDES transit simulation](01_first_ANDES_simulation.ipynb) is the
executable companion to [Tutorial 1](../docs/tutorial.md). It runs the 76-order
ANDES YJHK experiment and follows spectral preparation, cross-correlation and
the Kp–Vrest map through commented code and diagnostic figures.

Save the notebook in the repository's `notebooks/` directory and select a
Jupyter kernel from the EXoPLORE environment. The [installation guide](../docs/installation.md)
and [input guide](../docs/input_files.md) describe the required opacity tables,
PHOENIX spectra and instrument files. `tutorial.prepare_tutorial()` checks their
local paths and creates the run directory. The helper functions are in
[_tutorial_support.py](_tutorial_support.py).

The reference configuration sets Gaussian-noise and throughput base seeds to
12345. With both the Gaussian-noise and throughput seeds fixed, the stochastic
realisation is reproducible. End-to-end floating-point results may still differ
at machine-precision level, so bitwise-identical output is not guaranteed.

`SAVE_MATRICES` selects optional per-order arrays for saving. The simulator
calculates the experiment and exports a compact combined CCF for the animation.
Each run saves its configuration, overrides, log, products and software/input
record under `outputs/notebooks/01_first_ANDES_simulation/`.

Figures and the CCF-to-map animation are embedded in the notebook. Matplotlib
and Pillow render the animation. PyMuPDF provides inline PDF previews.
Tutorial 1 links directly to the notebook download. Sphinx copies the file
from `notebooks/` into its download directory when building the documentation.
