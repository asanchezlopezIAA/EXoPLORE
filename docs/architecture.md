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
      +--- Observed helium: pipelines/palle2020.py
      |                     pipelines/czesla2024.py
      |
      +--- Helium grid fitting: retrieval/helium_grid.py
```

`core/spectroscopy_mode.py` selects the workflow. The block numbers in the
molecular path label sections of `core/simulator.py`. Blocks 3–6 process one
spectral order before advancing to the next. The helium runners each manage
their own observing, preparation and analysis sequence.

## Where to find a calculation

Paths in this table are relative to `src/exoplore/`.

| Package | Contents | Starting points |
|---|---|---|
| `config/` | Simulation settings and JSON loading | `models.py`, instrument and helium configuration modules |
| `core/` | Experiment execution and workflow selection | `simulator.py`, `spectroscopy_mode.py`, `cli.py` |
| `planets/` | Planetary-system parameters and derived quantities | `models.py`, `catalog.py` |
| `instruments/` | Instrument information, wavelength grids and observing inputs | `base.py`, `wavegrid.py`, instrument readers |
| `atmosphere/` | Radiative transfer, chemistry, temperature and broadening | `prt.py`, `chemistry.py`, `temperature.py`, `winds.py`, helium solvers |
| `observation/` | Exposure times, velocities, airmass and noise | `timing.py`, `velocity.py`, `airmass.py`, `noise.py`, `randomness.py` |
| `pipelines/` | Spectral masks, preparation and model filtering | `prepare.py`, `masking.py`, preparation-method modules |
| `ccf/` | Cross-correlation and velocity-map calculations | `kernels.py`, `compute.py`, `statistics.py` |
| `retrieval/` | Likelihoods, priors, sampling interfaces and helium grid fitting | `likelihood.py`, `priors.py`, `samplers.py`, `helium_grid.py` |
| `analysis/` | Statistical summaries and comparisons of calculated products | `stats.py`, `multi_night.py`, `diagnostics.py` |
| `plotting/` | Spectral, CCF, detection-map and retrieval figures | `matrices.py`, `kpvsys.py`, instrument and helium plotting modules |
| `io/` | File readers, output paths and saved products | `paths.py`, `stellar.py`, `utils.py`, `results.py` |

The observing loop and several retrieval-model and likelihood functions are
inside `core/simulator.py`. The runner imports `spec_to_mat_fraction()` from
`core/observation.py` to build spectral matrices.

At the repository root, `configs/` holds example experiments, `scripts/` holds
command-line tools, `notebooks/` holds executable tutorials and `docs/` holds
the documentation. The [input guide](input_files.md) describes the external
scientific datasets. Runs write their products to the configured output root.

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

With both the Gaussian-noise and throughput seeds fixed, the stochastic
realisation is reproducible. End-to-end floating-point results may still differ
at machine-precision level, so bitwise-identical output is not guaranteed.

Velocity functions document their argument units individually.
`planet_radial_velocity_eccentric()` takes its periastron argument in radians.
`get_V_eccentric()` takes degrees and uses its documented phase convention.
