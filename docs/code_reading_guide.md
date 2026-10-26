# Reading the code: what calls what, and why

Start with the JSON file you use to run EXoPLORE. We will follow one spectral
order through the code, connecting your settings to the functions that use
them and the results they produce.

The [architecture guide](architecture.md) shows the full calculation in a diagram.
The [file index](module_guide.md) explains what each module does. Paths below
are relative to `src/exoplore/`.

## Read the experiment settings

Open `configs/hd189733b_andes_transit_clean.json`, then read
`SimulationConfig.from_json()` in `config/models.py`.

The JSON specifies the planet, instrument, atmosphere, observing sequence and
analysis. The loader constructs configuration objects with named fields:
`cfg.observation.exposure_time_seconds`, for example, supplies the exposure
time. Species abundances and the instrument registry use dictionaries because
their keys depend on the selected species or instrument.

The loader ignores unknown keys. `from_json()` includes checks in addition to
those in `from_dict()`. Read the method used by your calling code when tracing
validation. The [configuration reference](config_reference.md) describes the
individual settings.

## Start the simulator

Read `core/cli.py:main()` and `core/simulator.py:ExoploreSimulator`.

```text
Command line                         Notebook / Python script
    |                                      |
    +---------- SimulationConfig ----------+
                       |
               ExoploreSimulator
                       |
                  summary()       print the experiment settings
                       |
                    run()         execute the experiment
```

The command line runs the calculation with `--run`. A notebook or Python
script calls `simulator.run()` directly. Both use the configuration loader
and simulator shown above.

## Select the workflow

`core/spectroscopy_mode.py:spectroscopy_route()` selects the runner for
molecular spectroscopy, helium simulation, observed helium or helium grid
fitting. The selected runner executes that experiment. `use_real_data`
specifies the source of spectra within the applicable workflow. The selector
accepts the explicit helium mode and the supported helium configuration fields.

## Load the system and instrument

Blocks 1–2 of `ExoploreSimulator.run()` load the inputs and construct the
observing sequence.

| Call or module | Result used by the simulation |
|---|---|
| `io.paths.build_output_tree()` | Creates the configured output directories. |
| `planets.load_planet()` | Loads system parameters for geometry and atmospheric calculations. |
| `instruments.load_instrument()` | Supplies the instrument resolution and input locations. |
| `instruments.wavegrid.get_WaveGrid()` | Supplies wavelength arrays and associated observing inputs. |
| `io.stellar.LoadPhoenix()` | Reads, broadens and interpolates the stellar model onto the instrument grid. |
| `observation.timing`, `observation.airmass` | Supply exposure times, event membership and observing conditions. |

Instrument readers first identify files and metadata. The wavelength-grid
reader loads the corresponding arrays. PHOENIX wavelengths are converted from
Angstrom to microns in the stellar reader. Planetary calculations also use
cgs quantities, as documented in their parameter definitions.

## Follow one spectral order

Find `for h in range(n_orders)` in `core/simulator.py`.

The loop selects a wavelength segment and processes it through the atmospheric
model, observing sequence, preparation and cross-correlation. These are
Blocks 3–6.

### Calculate the atmospheric spectrum

Follow `call_pRT()` or `call_pRT_limbs()` into `atmosphere/prt.py`.

A petitRADTRANS object holds the opacity tables. The calling function supplies
temperature and chemistry settings and requests the spectrum. Calls produce
the injected planet, the cross-correlation template or a trial retrieval model,
according to the supplied configuration. The injected atmosphere and search
template have their own species lists and physical parameters.

`RadiativeTransferInputs`, in `atmosphere/models.py`, supplies the masses,
radii, internal temperature and instrumental resolving power used by these
calculations. Masses and radii are in CGS units. The atmospheric configuration
supplies the regional temperatures, species and chemistry choices.

### Construct the spectral time series

Follow `spec_to_mat_fraction()` into `core/observation.py`.
`core/simulator.py` and `pipelines/prepare.py` import it directly.

The function places spectra onto the exposure/wavelength grid using the
configured velocities and event geometry. The runner combines the stellar,
planetary and telluric contributions and applies throughput and noise. For
observed-data analysis, the selected reader supplies the measured spectra.

`observation/randomness.py` chooses the configured seeds, and
`observation/noise.py:add_throughput()` generates throughput variations. The
runner draws Gaussian noise and applies the configured uncertainty scaling.
Multi-night runs draw separate noise realizations for their nights.

### Prepare the spectra

Follow `preparing_pipeline()` into `pipelines/prepare.py`.

Each call processes an order/night matrix whose rows are exposures and columns
are wavelength channels. The selected method returns prepared spectra,
uncertainties, column selections and the detrending information needed by later
steps. Read the corresponding assignment in the caller to identify each
returned array.

`PreparationInputs`, in `pipelines/models.py`, carries the method and telluric
settings together with the effective SYSREM count for this call. Some methods
also use the selected order's theoretical telluric transmission. For observed
helium spectra, `HeliumObservation` supplies exposure velocities and records
the corrections already applied to the data.

For Tutorial 1, the call sequence is:

```text
pipeline.name = "Blain24"
            |
cfg.pipeline.name
            |
preparation_inputs.pipeline.name
            |
preparing_pipeline(...) -> Blain24 branch in pipelines/prepare.py
            |
remove_throughput_fit(data, ..., uncertainties=noise)
            |
remove_telluric_lines_fit(data_prepared, ..., uncertainties=propag_noise)
            |
prepared spectra + propagated uncertainties + column selections
```

The telluric fit receives the throughput-corrected spectra and their propagated
uncertainties. In the data branch, masking and uncertainty correction are
enabled. Model-processing flags specify the corrections applied to templates.
Excluded columns receive a placeholder flux value. The returned masks and
retained-column arrays identify which channels enter the analysis.

### Calculate the CCFs

Follow the selected `call_ccf_*()` function in `ccf/compute.py` into its kernel
in `ccf/kernels.py`. The instrument and configuration select the weighting
and normalization used by that kernel.

The inputs are prepared residual spectra and their uncertainty estimates.
The ordinary result has axes `(velocity, exposure)`. Some optimization modes
include extra injection or detrending-iteration axes.

The orbital-alignment functions in `ccf/compute.py` receive the velocity grids,
exposure selections and orbital quantities explicitly. Their keyword arguments
specify the eccentricity convention and the noise-normalisation region.
`NightStudyInputs`, in `ccf/models.py`, groups the configuration and reference
motion needed to compare nights. `TimeResolvedCCFInputs` supplies the settings
for comparisons between transit intervals.

`DetectionPlotSettings`, in `plotting/models.py`, holds plot labels, velocity
ranges and the output directory. The plotting functions read the completed
arrays supplied by the runner.

## Read the loop variables and selections

| Name in the runner | Meaning |
|---|---|
| `h` | Position in the selected-order list. |
| `order_selection[h]` | Instrument order ID at that position. |
| `b` | Night index. |
| `_nn` | Number of nights. |
| `_dn` | Whether nights have different observing sequences. |
| `_nl` | Whether the run is noiseless. |
| `_urd` | Whether observed spectra are used. |
| `pm` / `cc` | Planet-model / CCF-template settings. |
| `rt_inputs` | Masses, radii, temperature and resolving power shared by atmospheric calculations. |
| `preparation_inputs` | Method settings, masks and effective SYSREM count for one order/night. |
| `retrieval_orders` | Order IDs retained for the atmospheric retrieval. |
| `detection_plot` | Output labels and velocity ranges shared by detection figures. |
| `_b5`, `_b7`, `_b9` suffixes | The workflow block associated with a variable. |
| `per_order_results` | Dictionaries holding the products for each order. |

`per_order_results` stores references to the order dictionaries. Updates made
during preparation and correlation remain visible when the runner combines
those products.

`with_signal` and `without_signal` select exposures. Masks and retained-pixel
lists select wavelength columns. Function documentation identifies whether a
particular selection uses integer indices or a boolean array.

## Align and combine the CCFs

In Block 7, follow `get_shifted_ccf_matrix()` into `ccf/compute.py`.

For each trial orbital velocity amplitude, this function samples the selected
exposure CCFs along the predicted planetary velocity track. The caller sums
the aligned exposures to make a detection map. The shifted array has axes
`(Vrest, selected exposure, Kp)`. The sum has axes `(Vrest, Kp)`.

`get_max_CCF_peak()` normalizes the map using its configured noise region and
locates the maximum. Interpreting its S/N as a false-alarm probability requires
calibration against an appropriate null distribution.

The same module also defines `compute_kp_vsys_map()`, which takes exposures,
phases and velocity grids for a smaller map calculation. Its arguments and
normalization are documented separately. The standard runner uses
`get_shifted_ccf_matrix()` and its BERV convention.

## Save and plot the products

Read Block 8, `io/paths.py`, `plotting/matrices.py` and `plotting/kpvsys.py`.

Output flags select which numerical arrays are saved. Diagnostic figures use
in-memory arrays during the calculation. To inspect a figure numerically after
the run, enable saving of its underlying arrays.

The [result methods](architecture.md#read-completed-results) return completed
CCFs and detection maps. `results.py` defines their fields and axes.
`io/results.py` reads and writes these products.

## Follow the retrieval

Read Block 9 when `retrieval.enabled` is true.

The sampler proposes parameters, the atmospheric model calculates a spectrum,
and the likelihood compares it with the prepared data. Model reprocessing
accounts for the preparation applied to the observations. Its flags and the
selected likelihood determine which operations are applied to each trial.

`retrieval/` contains likelihood functions, priors and sampler interfaces.
Several model and likelihood functions are defined inside `core/simulator.py`,
where Block 9 also sets up MultiNest or emcee. The functions in
`retrieval/samplers.py` provide callable interfaces for those samplers.
MultiNest receives a likelihood and a separate prior transform. Emcee receives
a log-posterior that includes the prior and likelihood.
