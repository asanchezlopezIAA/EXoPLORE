# Architecture and code flow

This guide is for researchers reading or contributing to EXoPLORE. It explains
where calculations live, how a run connects them, and which interfaces must stay
compatible when the implementation is reorganized.

The [tutorials](tutorial.md) explain how to run experiments. The
[configuration reference](config_reference.md) explains their settings. Start
here when you want to find the code behind a step in an experiment.

## Follow a run

The public entry point stays small:

```python
from exoplore.config import SimulationConfig
from exoplore.core import ExoploreSimulator

# Parse the experiment into named configuration sections.
config = SimulationConfig.from_json("configs/hd189733b_andes_transit_clean.json")

# Validate the selected route, then execute its configured workflow.
simulator = ExoploreSimulator(config)
simulator.run()
```

The simulator coordinates the calculation. Its `run()` method returns `None`;
products are written to the configured output directory. The private `_state`
dictionary is working storage and is not a stable public result interface.

### Inspect completed products through public accessors

After a successful standard molecular run:

```python
# These methods expose existing products; they do not run the analysis again.
ccf = simulator.get_combined_ccf()
detection = simulator.get_detection_map(metric="snr")

# Axis order is explicit: CCF (velocity, exposure), map (Vrest, Kp).
print(ccf.ccf.shape, detection.values.shape)
print(detection.peak_value, detection.peak_kp_kms, detection.peak_v_rest_kms)
```

`CCFResult` and `DetectionMapResult` live in `exoplore.results`, independent of
the simulator, IO and plotting modules. They name already computed arrays and
their conventions; neither class performs preparation, coaddition or map
normalization. The map's peak is a maximum pixel, not an uncertainty estimate.

The CCF accessor returns detached arrays by default. Use `copy=False` to obtain
read-only views when avoiding a copy matters; those views still reflect later
changes to the underlying arrays. The container's frozen fields do not deeply
freeze NumPy arrays or metadata. Result arrays read from disk are detached.

The map accessor reads the explicitly named S/N or Welch archive. In an
all-metrics run the old `ccf_tot_sig` working variable can contain the last Welch
result, so using it as an S/N map would be ambiguous. Missing files produce an
error; the accessor does not substitute another metric or recalculate a map.

For nights with different observing sequences, request the CCF with
`night_index=0` (zero-based); the accessor trims documented exposure padding.
Nights sharing one geometry have already been coadded in that CCF. Saved
per-night maps can be requested with `get_detection_map(night_index=0)` when
the run produced them. Injection/iteration cubes are rejected rather than
silently selecting an extra axis. Helium workflows do not yet expose these
accessors. Access before success, or after a failed new attempt, raises an error.

Read or export compact products without a live simulator:

```python
from exoplore.io.results import load_ccf_result, load_detection_map, save_ccf_result

# Export the CCF, phase, selected exposures, BERV and velocity/provenance metadata.
# Existing files are protected unless overwrite=True is explicitly requested.
save_ccf_result(ccf, "combined_ccf.npz")
ccf_from_disk = load_ccf_result("combined_ccf.npz")

# Use the exact S/N archive path from the run's matrices directory.
map_from_disk = load_detection_map("path/to/ccf_tot_sn_map_RUN_NAME.npz")
```

The CCF reader also accepts existing compact notebook exports. The new compact
export stores the CCF itself; map grids remain in the separate map archive.
It does not replace the notebook animation bundle, which includes additional
map grids.

Provenance is captured when a standard run succeeds, so changing the config
for a subsequent experiment does not relabel the completed CCF. The existing
`run()` return value, files, calculation order and scientific defaults are
preserved.

```text
JSON experiment
      |
      v
SimulationConfig -> ExoploreSimulator -> select spectroscopy route
                                         |
             +---------------------------+--------------------------+
             |                           |                          |
             v                           v                          v
    Standard molecular run       Helium simulation          Observed helium
    core/simulator.py            core/helium_simulation.py  pipeline recipe
             |
             |                     Helium grid is a separate route:
             |                     retrieval/helium_grid.py
             v
    Inputs + instrument + planet + observing sequence       Blocks 1–2
             |
             v
    +------ for each selected spectral order --------------------+
    | Atmospheric forward model                              3  |
    |         v                                                 |
    | Project star/planet through the observing sequence      4  |
    |         v                                                 |
    | Tellurics, throughput, noise and spectral preparation   5  |
    |         v                                                 |
    | Cross-correlate prepared spectra and template           6  |
    +-----------------------------------------------------------+
             |
             v
    Combine CCFs; search trial orbital motions                  7
             |
             v
    Save configured matrices and detection products            8
             |
             v
    Atmospheric retrieval, when enabled                        9
```

The numbered blocks are the labels currently used in `core/simulator.py`.
Blocks 3–6 share an order loop: the code follows an order through those stages
before advancing to the next. Enabled features and real-data inputs determine
which operations actually run. Observed helium recipes currently live in
`pipelines/palle2020.py` and `pipelines/czesla2024.py`; they are separate runners,
not additional steps appended to the molecular route.

Preparation changes the data searched by the CCF. Retrieval may also reprocess
its model to account for that preparation. Moving these calls requires checking
their dependencies and preserving their ordering; visual similarity between
functions is insufficient evidence that they can be exchanged.

## Where to find a calculation

All package paths below are relative to `src/exoplore/`.

| Package | Responsibility | Useful starting points |
|---|---|---|
| `config/` | User settings and JSON conversion | `models.py`, route-specific configuration modules |
| `core/` | Public simulator, route selection and workflow coordination | `simulator.py`, `spectroscopy_mode.py`, `cli.py` |
| `planets/` | System parameters and their loading | `models.py` |
| `instruments/` | Instrument registry, wavelength grids, ETC inputs and instrument-specific adapters | `base.py`, `wavegrid.py`, instrument modules |
| `atmosphere/` | Atmospheric spectra, chemistry interfaces and broadening | `prt.py`, `winds.py`, helium providers |
| `observation/` | Timing, velocities, airmass and noise | `timing.py`, `velocity.py`, `airmass.py`, `noise.py`, `randomness.py` |
| `pipelines/` | Masking, preparation and model filtering | `prepare.py`, `masking.py`, `sysrem.py` |
| `ccf/` | Correlation kernels, CCF assembly and velocity-space analysis | `kernels.py`, `compute.py`, `statistics.py` |
| `retrieval/` | Likelihoods, priors and sampler adapters | `likelihood.py`, `priors.py`, `samplers.py` |
| `analysis/` | Existing analysis and diagnostic routines | `stats.py`, `multi_night.py`, `diagnostics.py` |
| `plotting/` | Scientific figure construction | `matrices.py`, `kpvsys.py`, route-specific figures |
| `io/` | Input readers, serialization and output paths | `paths.py`, `stellar.py`, `utils.py` |

`results.py` contains shared named products; `io/results.py` handles compact
product serialization and the temporary adapter from the current runner's
dictionary state. Keep new calculations out of both files.

These are the current homes of the code, not a claim that every boundary has
already been separated. Large workflow blocks and retrieval likelihoods still
reside inside `core/simulator.py`. Some analysis routines also calculate, save
and plot in one function. Extract those responsibilities incrementally while
keeping the public entry point stable.

Outside the package:

- `configs/` contains reference experiments; check Git tracking when distinguishing
  maintained examples from local research configurations.
- `scripts/` contains command-line tools. New reusable calculations should live
  in the package, with scripts responsible for arguments and execution.
- `notebooks/` contains executable teaching material and presentation helpers.
  Notebook 01 is a published artifact; preserve its approved saved outputs.
- `tests/` contains automated checks. Scientific inputs that live outside the
  checkout are described in the [input guide](input_files.md).
- `outputs/` contains generated products, not package source. Do not move or
  delete existing experiments as part of source reorganization.

## Canonical modules and compatibility imports

Older import paths remain available:

| Existing import | Canonical implementation |
|---|---|
| `exoplore.simulator` | `exoplore.core.simulator` |
| `exoplore.velocity` | `exoplore.observation.velocity` |
| `exoplore.priors` | `exoplore.retrieval.priors` |
| `exoplore.samplers` | `exoplore.retrieval.samplers` |

The compatibility modules import the same functions/classes directly. They do
not wrap or copy their calculations. Make changes in the canonical module and
keep the exported names working. This also allows older serialized priors to
resolve their original module paths. Newly serialized classes identify their
canonical module, so byte-identical pickle files are not an API guarantee.

The simulator's existing `_apply_throughput` helper also remains as a thin
wrapper. `observation/randomness.py` now owns the choice of throughput and
Gaussian seeds; `observation/noise.py` still implements the throughput draw and
its unchanged 2% scale. Gaussian draws and uncertainty scaling remain at their
original positions in the simulation.

Different nights need different noise. The extraction preserves both existing
ways of obtaining it: fresh unseeded Gaussian draws for ordinary multi-night
runs, and separately seeded draws for Cheverall26. Throughput seeds retain
physical order IDs; Cheverall26 Gaussian seeds retain selected-order loop
positions. These choices are documented so moving code does not accidentally
change a night's noise or which order receives it.

Some similarly named functions are **not** interchangeable. For example,
`planet_radial_velocity_eccentric` accepts its periastron argument in radians,
whereas `get_V_eccentric` accepts degrees and has a different phase-anchor
implementation. Both remain as implemented. Any scientific reconciliation
requires its own review, separate from an architecture change.

## Settings, results and dictionaries

Configuration dataclasses describe fixed, named settings. They are appropriate
for planet, atmosphere, observation and analysis options. Keep species-to-
abundance mappings and instrument registries as dictionaries: their keys
represent a variable collection of species or instruments.

For new fixed-schema stage results, prefer a small named record over a large
dictionary whose required keys are known only to its callers. Document:

- units, including whether wavelength is in microns and velocity in km/s;
- array axes and whether nights/orders can have different lengths;
- whether a mask stores indices or booleans, and what `True` means;
- whether arrays are shared references, views or copies.

The CCF and detection-map records above implement this boundary for completed
diagnostics. Prepared-order and other stage records remain future incremental
work; current scientific array layouts are preserved. Avoid copying large
arrays simply to put them in a dataclass. In particular, `dataclasses.asdict()`
recursively copies values and should not be used casually on large results.

Existing simulator variables such as `_mini_*` dictionaries translate typed
settings into older function arguments. `_b5`/`_b7` suffixes refer to workflow
blocks, and `h` is used as the selected-order loop position. Keep those names
until a stage can be extracted and reviewed in isolation. Renaming everything
at once would obscure the evidence that calculations remain unchanged.

## Comments that help a scientific reviewer

Explain why a call is needed, where its inputs came from and what conventions
its outputs follow. Put a stage overview at its entry point, and document
non-obvious transformations beside the code that performs them.

For example, explain that BERV is subtracted under the existing velocity
convention, or that a preparation operator is applied to a retrieval model to
match processing of the data. Do not infer units or mask semantics from a
variable name: verify the producer and consumer before documenting them.

Use descriptive names for new interfaces. Preserve existing public names,
keywords, serialized field names and configuration keys through compatibility
adapters when internal names change. Comments must describe the calculation
that exists, including limitations, rather than a planned implementation.

## Preservation checks for refactoring

Before moving a stage, capture its existing behavior. After the move, compare
spectra, uncertainties, masks, grids, CCFs and maps as applicable. Keep equations,
operation order, normalization, output conventions, defaults and random streams
unchanged. Configuration validation changes need a separate review because
they can alter which existing configurations are accepted.

With both the Gaussian-noise and throughput seeds fixed, the stochastic
realisation is reproducible. End-to-end floating-point results may still differ
at machine-precision level, so bitwise-identical output is not guaranteed.

Run the source checkout's automated tests with:

```bash
PYTHONPATH=src python -m pytest
```

Check both component tests and the documented tutorials when changing a stage.
