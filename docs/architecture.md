# Architecture and code flow

Use this guide to understand what EXoPLORE does when you run a simulation
or analyse observations. It follows your configuration through the calculation
to the resulting spectra, figures and detection maps, showing which files
perform each step.

The [code reading guide](code_reading_guide.md) walks through the function calls.
The [file index](module_guide.md) explains what each module does.

## Run an experiment

```python
from exoplore.config import SimulationConfig
from exoplore.core import ExoploreSimulator

# Read the planet, instrument, atmosphere, observation and analysis settings.
config = SimulationConfig.from_json("configs/hd189733b_andes_transit_clean.json")

# Select and execute the workflow specified by those settings.
simulator = ExoploreSimulator(config)
simulator.run()
```

`ExoploreSimulator.run()` coordinates the calculation and returns `None`.
The selected output products are written under `config.paths.output_root`.
The command-line entry point in `core/cli.py` uses these same classes: it
prints a preview by default and executes the experiment when given `--run`.

```text
SimulationConfig
      |
      v
ExoploreSimulator.run()
      |
      +--- Molecular spectroscopy: core/simulator.py
      |       |
      |       +--- Read system, instrument and observing inputs       1–2
      |       |
      |       +--- For each selected spectral order:
      |       |       atmospheric spectrum                            3
      |       |           v
      |       |       stellar/planetary time series                   4
      |       |           v
      |       |       tellurics, throughput, noise and preparation     5
      |       |           v
      |       |       cross-correlation                               6
      |       |
      |       +--- Combine CCFs and calculate velocity maps            7
      |       +--- Save selected numerical products                    8
      |       +--- Run retrieval when enabled                          9
      |
      +--- Helium simulation: core/helium_simulation.py
      |                       core/carmenes_helium_simulation.py
      |
      +--- Direct sodium: core/atomic_simulation.py
      |       spatial stellar/planetary transfer
      |           v
      |       tellurics, native sampling and ETC uncertainties
      |           v
      |       Molecfit and pipelines/seidel22.py
      |           v
      |       transmission spectra and line light curves
      |
      +--- Observed helium: pipelines/palle2020.py
      |                     pipelines/czesla2024.py
      |
      +--- Helium grid fitting: retrieval/helium_grid.py
```

`core/spectroscopy_mode.py` selects the workflow. The block numbers in the
molecular path label sections of `core/simulator.py`. Blocks 3–6 process one
spectral order before advancing to the next. The helium runners each manage
their own observing, preparation and analysis sequence. The direct sodium
runner is selected by `atomic_transmission.enabled`. It reads the existing
atmospheric, stellar, instrument and noise configuration and writes the full
resolved settings beside its outputs. Seidel22 uncertainty resampling repeats
continuum fitting and shared-reference preparation on native-pixel noise draws.

## Where to find a calculation

Paths in this table are relative to `src/exoplore/`.

| Package | Contents | Starting points |
|---|---|---|
| `config/` | Simulation settings and JSON loading | `models.py`, instrument and helium configuration modules |
| `core/` | Experiment execution and workflow selection | `simulator.py`, `spectroscopy_mode.py`, `cli.py` |
| `planets/` | Planetary-system parameters and derived quantities | `models.py`, `catalog.py` |
| `instruments/` | Instrument information, wavelength grids and observing inputs | `base.py`, `wavegrid.py`, instrument readers |
| `atmosphere/` | Radiative transfer, chemistry, temperature, ray transmissions and broadening | `prt.py`, `transmission.py`, `chemistry.py`, `temperature.py`, `winds.py`, helium solvers |
| `stellar/` | Local stellar intensities, rotating disk integration and joint atmospheric transfer | `surface.py`, `observation.py`, `coupled.py` |
| `observation/` | Exposure times, velocities, airmass and noise | `timing.py`, `velocity.py`, `airmass.py`, `noise.py`, `randomness.py` |
| `pipelines/` | Spectral masks, preparation and model filtering | `prepare.py`, `masking.py`, preparation-method modules |
| `ccf/` | Cross-correlation and velocity-map calculations | `kernels.py`, `compute.py`, `statistics.py` |
| `retrieval/` | Likelihoods, priors, sampling interfaces and helium grid fitting | `likelihood.py`, `priors.py`, `samplers.py`, `helium_grid.py` |
| `analysis/` | Statistical summaries and comparisons of calculated products | `stats.py`, `multi_night.py`, `diagnostics.py` |
| `plotting/` | Spectral, CCF, detection-map and retrieval figures | `matrices.py`, `kpvsys.py`, instrument and helium plotting modules |
| `io/` | File readers, output paths and saved products | `paths.py`, `stellar.py`, `stellar_surface.py`, `utils.py`, `results.py` |

The optional `rm_clv` section selects stellar line distortions in the standard
transit workflow. `io/stellar_surface.py` validates local intensity inputs.
`stellar/surface.py` integrates the rotating and occulted disk.
`stellar/observation.py` averages exposures, applies the instrumental response
and constructs the observer-frame model. Block 4 inserts the distortion when
requested. Block 5 applies a requested correction to flux and uncertainty
before preparation. The [RM/CLV reference](config_reference.md#19-stellar-rm-and-clv)
describes both coupling choices and their supported routes.
With `coupling: spatial`, `atmosphere/transmission.py` exports native ray
transmissions from pRT and checks their integrated area. `stellar/coupled.py`
weights those rays by the local stellar spectrum and integrates the joint
flux. Block 4 uses that completed flux, which already includes the continuum
transit. The existing CCF template and analysis windows remain available.
`pipelines/runner.py` supplies the preparation result structure consumed by
the observing loop. For fixed-telluric preparation it carries forward the
calculated flux/error pair together with the unchanged column selection.

The observing loop and several retrieval-model and likelihood functions are
inside `core/simulator.py`. The runner imports `spec_to_mat_fraction()` from
`core/observation.py` to build spectral matrices.

At the repository root, `configs/` holds example experiments, `scripts/` holds
command-line tools, `notebooks/` holds executable tutorials and `docs/` holds
the documentation. The [input guide](input_files.md) describes the external
scientific datasets. Runs write their products to the configured output root.

## How inputs reach a calculation

`SimulationConfig` holds the choices read from JSON. `PlanetParameters` holds
the physical system and its derived quantities. A calculation receives the
section or numerical quantities it uses.

| Input object | File | Used for |
|---|---|---|
| `InstrumentInputs` | `instruments/base.py` | Paths and observing settings used to read an instrument. |
| `RadiativeTransferInputs` | `atmosphere/models.py` | System quantities and units shared by atmospheric spectra. |
| `PreparationInputs` | `pipelines/models.py` | Method settings and order/night information for preparation. |
| `HeliumObservation` | `pipelines/models.py` | Exposure velocities, dates and correction flags for observed helium. |
| `NightStudyInputs` | `ccf/models.py` | Night counts, reference motion and choices for statistical comparisons. |
| `TimeResolvedCCFInputs` | `ccf/models.py` | Settings for CCF comparisons between transit intervals. |
| `DetectionPlotSettings` | `plotting/models.py` | Figure labels, velocity ranges and output location. |
| `LimbRetrievalState` | `retrieval/models.py` | Regional abundances retained during likelihood evaluation. |

Configuration objects describe the experiment. Arrays carry calculated spectra,
uncertainties, masks and CCFs. Dictionaries remain useful for species-indexed
abundances, registries and collections of saved products.

The retrieval likelihood choices are `BL19`, `Blain24`, `Blain24_beta` and
`Gibson22`. `retrieval/likelihood.py` exposes the array-based functions.
Block 9 of `core/simulator.py` connects the selected likelihood to atmospheric
models, prepared observations and sampler coordinates. `Blain24_beta` samples
one global uncertainty scale alongside the four ordinary 1D parameters.
Its configuration is described in the
[retrieval reference](config_reference.md#blain24-with-a-fitted-uncertainty-scale).

## Read completed results

After a successful molecular run:

```python
# Read the combined time–velocity CCF and the saved S/N map.
ccf = simulator.get_combined_ccf()
detection = simulator.get_detection_map(metric="snr")

# CCF axes: velocity, exposure. Map axes: Vrest, Kp.
print(ccf.ccf.shape, detection.values.shape)
print(detection.peak_value, detection.peak_kp_kms, detection.peak_v_rest_kms)
```

`CCFResult` and `DetectionMapResult`, defined in `results.py`, hold the arrays,
axes, units and run metadata. The peak properties locate the strongest map
pixel. Statistical uncertainties require a separate inference calculation.

`get_combined_ccf()` returns copies by default. With `copy=False`, it returns
read-only views that follow changes to the underlying arrays. The result
objects have fixed field assignments. Their NumPy arrays and metadata retain
their own mutability rules.

For different observing sequences, specify a zero-based `night_index`. The
CCF reader selects that night's valid exposures. Nights sharing one geometry
are coadded in the combined CCF. `get_detection_map(night_index=0)` reads the
corresponding per-night map when that product was saved. `metric="snr"` and
`metric="welch"` select their respective archives.

These methods support the ordinary two-dimensional molecular CCFs and maps.
Requests for products with extra injection or iteration axes raise an error.
Results become available after a successful run. An unsuccessful new attempt
clears that availability. Missing map files also raise an error.

The private `_state` dictionary holds intermediate arrays for the running
calculation. The public result methods capture the completed experiment's
metadata, including its order selection and velocity conventions.

To save or read the compact CCF:

```python
from exoplore.io.results import load_ccf_result, load_detection_map, save_ccf_result

# Save the CCF, its axes, selected exposures and observing metadata.
save_ccf_result(ccf, "combined_ccf.npz")
ccf_from_disk = load_ccf_result("combined_ccf.npz")

# Read the map archive written in the run's matrices directory.
map_from_disk = load_detection_map("path/to/ccf_tot_sn_map_RUN_NAME.npz")
```

The save function requires `overwrite=True` to replace an existing file.
The CCF reader also accepts compact notebook exports. Map grids are stored in
the map archive and in the notebook animation bundle.

## Import the calculation you need

```python
from exoplore.core.simulator import ExoploreSimulator
from exoplore.observation.timing import orbital_phase
from exoplore.observation.velocity import planet_radial_velocity
from exoplore.retrieval.priors import PriorSet
from exoplore.retrieval.samplers import run_multinest
from exoplore.plotting.kpvsys import plot_Kp_Vrest
```

## Settings, arrays and random draws

Configuration dataclasses hold named groups of settings. Dictionaries hold
collections with variable keys, such as chemical species and instrument names.
`CCFResult` and `DetectionMapResult` give the completed arrays named fields and
explicit axes. Calling `dataclasses.asdict()` recursively copies their values,
including arrays.

`observation/randomness.py` selects seeds and calls the throughput function in
`observation/noise.py`. The throughput draw uses a fractional white-jitter
scale of 0.02. Gaussian noise draws and their scaling are applied in the
observing loop. Ordinary multi-night Gaussian draws use fresh random
initialization. Cheverall26 assigns separate seeded draws. Throughput streams
use physical order IDs, while Cheverall26 Gaussian streams use positions in
the selected-order list.

For the single-night Tutorial 1 configuration, fixing both the Gaussian-noise
and throughput seeds makes the stochastic realisation reproducible. End-to-end
floating-point results may still differ at machine-precision level, so
bitwise-identical output is not guaranteed. The multi-night seed policies
described above determine repeatability for those runs.

Velocity functions document their argument units individually.
`planet_radial_velocity_eccentric()` takes its periastron argument in radians.
`get_V_eccentric()` takes degrees and uses its documented phase convention.


## Read and build the documentation

`docs/index.rst` defines the RTD navigation. The linked Markdown pages contain
installation, tutorials and reference material. `docs/conf.py` selects Sphinx,
MyST and the theme, and reads the package version from `src/exoplore/__init__.py`.
ReadTheDocs installs its build dependencies from `docs/requirements.txt` using
`.readthedocs.yaml`.

To inspect a local documentation build with those dependencies installed:

```bash
python -m sphinx -E -a -b html docs docs/_build/html
```

Tutorial 1 offers the canonical notebook as a direct download. Sphinx checks
its saved outputs and copies the file into `_downloads/`. It does not execute
notebook cells. The notebook runs locally from a source checkout with the
project environment and `notebooks/_tutorial_support.py` available.
