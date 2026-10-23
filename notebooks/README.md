# Guided EXoPLORE notebooks

Start with [Your first ANDES transit simulation](01_first_ANDES_simulation.ipynb), the executable companion to [RTD Tutorial 1](../docs/tutorial.md). It runs all 76 ANDES YJHK orders, then follows the real preparation, CCF trail and Kp–Vrest map through short code blocks and figures.

Use a Jupyter kernel in the existing EXoPLORE scientific environment. Opacity tables, PHOENIX stellar spectra and instrument inputs must already be available; see the [installation](../docs/installation.md) and [input](../docs/input_files.md) guides. The commented setup cell calls `tutorial.prepare_tutorial()` to check local inputs and create a fresh run directory. It installs or downloads nothing.

The reference configuration sets Gaussian-noise and throughput base seeds to 12345. With both the Gaussian-noise and throughput seeds fixed, the stochastic realisation is reproducible. End-to-end floating-point results may still differ at machine-precision level, so bitwise-identical output is not guaranteed.

`SAVE_MATRICES=False` skips optional large per-order arrays. The full calculation still runs, and a compact combined numerical CCF bundle is exported immediately afterward for the animation. Each run keeps its resolved configuration, explicit overrides, log, products and provenance under `outputs/notebooks/01_first_ANDES_simulation/`. No cached or artificial spectra replace the calculation.

The four scientific figures and the CCF-to-map GIF are embedded in the notebook outputs, so viewing the downloaded file needs no sidecar images. The setup cell imports readable utilities from [_tutorial_support.py](_tutorial_support.py), included in the EXoPLORE checkout. Running the notebook requires placing it within an EXoPLORE checkout and using that checkout's scientific environment and inputs. No RTD tooling is involved.

The GIF and static companion in `assets/` remain convenient regeneration products. Rendering uses Matplotlib/Pillow, and inline PDF previews use PyMuPDF. EXoPLORE performs the scientific calculations; the utilities handle presentation and the isolated combined-CCF export adapter.

RTD Tutorial 1 provides a direct notebook download using Sphinx's standard download role. There is one canonical notebook under `notebooks/`; Sphinx copies it into its generated download directory at build time. Notebook 02 has not been created.
