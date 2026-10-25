# Reading the code: what calls what, and why

Use this guide alongside your editor. Read one stage, follow its named call,
then return to the simulator. You do not need to read the entire simulator in
one sitting. The [architecture overview](architecture.md) gives the folder map
and complete workflow diagram.

For a file-by-file lookup, use [which file should I open?](module_guide.md).
It covers the maintained package and scripts, including the instrument and
helium branches outside the main Tutorial 1 path.

The paths below are relative to `src/exoplore/`. This describes the current
implementation, including older interfaces that remain in use.

## Start with the experiment

Open the Tutorial 1 JSON at `configs/hd189733b_andes_transit_clean.json`, then
read `SimulationConfig.from_json()` in `config/models.py`.

The file contains choices: the planet, instrument, atmosphere, observation and
analysis. The loader turns its nested JSON dictionaries into named objects.
That is why the runner can write `cfg.observation.exposure_time_seconds`
instead of repeatedly looking up strings in nested dictionaries.

Some dictionaries should remain dictionaries. Species names form a variable
collection of abundances, and instrument names select registered loader
functions. Those are different from a fixed configuration section with known
fields. See the [configuration reference](config_reference.md) for the meaning
of individual settings.

The current loader ignores unknown keys and has some checks in `from_json()`
that are absent from `from_dict()`. Comments now identify those boundaries;
this documentation pass does not change which configurations are accepted.

## Enter through the same interface as a notebook

Read `core/cli.py:main()` and then `core/simulator.py:ExoploreSimulator`.

```text
Command line                         Notebook / Python script
    |                                      |
    +---------- SimulationConfig ----------+
                       |
               ExoploreSimulator
                       |
                  summary()       inspect the selected experiment
                       |
                    run()         perform the calculation
```

The command line requires `--run` before executing. A notebook calls `run()`
explicitly. Keeping both on the same interface avoids implementing scientific
setup twice.

## Choose the workflow before calculating spectra

Read `core/spectroscopy_mode.py:spectroscopy_route()`.

It decides which runner should handle the experiment: the standard molecular
workflow, a helium simulation, an observed-helium recipe or a helium grid.
These are different routes, not successive stages of one run. `use_real_data`
selects an input source within the applicable route.

The explicit helium switch and older helium settings are both recognized for
compatibility. A selected route returns control to its own runner rather than
continuing into the molecular simulation afterward.

## Load the system and instrument

Return to Blocks 1–2 of `ExoploreSimulator.run()`.

| Call or module | Why it is needed |
|---|---|
| `io.paths.build_output_tree()` | Keeps products for an experiment under its output root. Directory creation is explicit. |
| `planets.load_planet()` | Supplies the planetary/system parameters used by geometry and atmosphere calculations. |
| `instruments.Load_Instrumental_Info()` | Resolves instrument information and input locations. |
| `instruments.wavegrid.get_WaveGrid()` | Reads or constructs the wavelength grid and associated observing inputs. |
| `io.stellar.LoadPhoenix()` | Reads, broadens and interpolates the stellar model onto the instrument grid. |
| `observation.timing` and `observation.airmass` | Supply event/exposure geometry and observing conditions needed downstream. |

Instrument metadata and actual arrays have different jobs. Knowing an
instrument's resolution or a file location does not yet construct an observed
time series. Follow the reader after the metadata loader to find the arrays.

PHOENIX wavelengths are converted from Angstrom to microns in the stellar
reader. More generally, check units at the function boundary; planetary
calculations also use cgs quantities internally. Do not infer a unit from a
short legacy variable name.

## Follow one order through the calculation

Find `for h in range(n_orders)` in `core/simulator.py`.

An order is one wavelength segment. The runner takes one selected segment
through atmosphere generation, observation, preparation and cross-correlation
before moving to the next. These are Blocks 3–6.

### Atmospheric model

Follow `call_pRT()` or `call_pRT_limbs()` into `atmosphere/prt.py`.

The petitRADTRANS object contains the loaded opacity information. The wrapper
supplies the temperature/chemistry and requests the spectrum. It is called for
different purposes: injected atmospheric truth, a search template, or proposed
retrieval models. Identify which settings were passed before interpreting the
returned spectrum.

The injected atmosphere and H₂O search template can have different species and
parameters. The template is a hypothesis being compared with the data; it is
not automatically a copy of everything injected.

### Observed spectral time series

Follow `spec_to_mat_fraction()` into `core/observation.py`. The similarly named
function imported from `io/stellar.py` is a forwarding shim to that implementation.

This step places the model on exposure/wavelength coordinates using the
configured motion and event geometry. Later steps combine the stellar,
planetary and telluric contributions and apply the configured throughput/noise.
The exact branches depend on whether data are synthetic or observed.

`observation/randomness.py` chooses the existing seeds;
`observation/noise.py:add_throughput()` makes the throughput draw. Gaussian
draws remain in the runner. Different nights retain their existing different
noise realizations. The helpers' location has changed without changing those
draw choices or their amplitudes.

### Preparation

Follow `preparing_pipeline()` into `pipelines/prepare.py`.

One call handles an order/night matrix. Rows are exposures; columns are
wavelength channels. The dispatcher selects the configured preparation method
and passes back prepared data, uncertainties and column selections, together
with detrending information used by later stages.

Read the chosen branch, not every pipeline branch at once. The current return
signatures can vary, so also read how the caller unpacks that branch's result.

For a concrete example, trace the Blain24 setting in Tutorial 1:

```text
JSON: pipeline.name = "Blain24"
            |
cfg.pipeline.name
            |
_mini_pp["preparing_pipeline"]
            |
preparing_pipeline(...) -> Blain24 branch in pipelines/prepare.py
            |
remove_throughput_fit(data, ..., uncertainties=noise)
            |
remove_telluric_lines_fit(data_prepared, ..., uncertainties=propag_noise)
            |
prepared matrix + propagated uncertainties + updated column selections
            |
runner stores the products for CCFs and later analysis
```

The order of those two corrections matters: the second receives the output
and propagated uncertainties of the first. In the configured data branch,
masking and uncertainty correction are enabled. Model-reprocessing branches
can pass different flags, so follow the arguments as well as the function name.

The Blain24 branch writes a placeholder value into excluded columns. A flux
value alone therefore does not establish whether a channel is valid: the
mask/retained-column selections carry that information.

### Cross-correlation

Follow the selected `call_ccf_*()` wrapper in `ccf/compute.py` to its numerical
kernel in `ccf/kernels.py`. Configuration and instrument branches determine
which wrapper is called; similarly named kernels can implement different
weighting/normalization choices.

The data passed onward are prepared residuals, not the original raw spectra.
Uncertainty estimates enter the configured correlation calculation. The
result retains one CCF per exposure, with axes `(velocity, exposure)` on the
ordinary path. Extra injection/iteration axes exist for some optimization modes.

## Understand the bookkeeping without confusing it with physics

| Name in the runner | Meaning |
|---|---|
| `h` | Position in the selected-order list. |
| `order_selection[h]` | Actual instrument order ID at that position. |
| `b` | Night index in the per-night loop. |
| `_nn` | Number of nights used by the standard runner. |
| `_dn` | Whether nights have different observing sequences. |
| `_nl` | The run's noiseless flag. |
| `_urd` | Whether existing observed data are used. |
| `pm` / `cc` | Planet-model / CCF-template configuration objects. |
| `_mini_*` | Dictionaries translating typed settings into older function argument names. |
| `_b5`, `_b7`, `_b9` suffixes | Variables associated with the numbered workflow blocks. |
| `per_order_results` | Working products accumulated for later combination. |

The per-order list contains dictionaries that are updated as an order moves
through the stages. Appending the dictionary does not make a copy; later
updates intentionally remain visible to the combination stage.

Two selections also need distinguishing:

- `with_signal` / `without_signal` select **exposures**.
- Masks and useful-pixel lists select **wavelength columns**. Many interfaces
  use integer indices; temporary boolean selections also exist. Check the
  producer/consumer rather than assuming one universal mask representation.

## Combine information after preserving the motion

Return to Block 7 and follow `get_shifted_ccf_matrix()` in `ccf/compute.py`.

The CCFs still contain the exposure-dependent velocity pattern. For each trial
orbital amplitude, the function samples each selected exposure along its
predicted velocity track. The caller then sums the aligned exposures. Summing
before alignment would discard the changing planetary velocity.

The ordinary shifted array has axes `(Vrest, selected exposure, Kp)`. Summing
over its exposure axis produces a `(Vrest, Kp)` map.

`get_max_CCF_peak()` applies the configured map-S/N normalization and locates
a maximum. Read its noise-region options when inspecting that statistic. It
does not establish a calibrated false-alarm probability merely by returning
a numerical S/N.

The smaller helper `compute_kp_vsys_map()` near the top of the same file is a
different interface. It processes the supplied exposures and lacks the full
runner's BERV argument and map-S/N normalization. Its similar name is not a
reason to substitute it into Tutorial 1.

## Saving and plotting

Read Block 8, `io/paths.py`, `plotting/matrices.py` and `plotting/kpvsys.py`.

Saving flags decide which numerical products remain on disk. The calculation
can still run when large intermediate arrays are not saved. Many diagnostics
are plotted from in-memory products during the run; a saved figure alone does
not imply that its complete numerical input was also persisted.

For ordinary completed CCF/map products, use the public accessors described in
the [architecture guide](architecture.md#inspect-completed-products-through-public-accessors).
`results.py` defines their named containers; `io/results.py` reads/writes them.
They expose calculated values without repeating the scientific pipeline.

## Why retrieval can prepare a model again

Read Block 9 only when `retrieval.enabled` is relevant to your experiment.

A retrieval proposes parameters, constructs the corresponding model and
evaluates it against the prepared data. Depending on the selected route, model
reprocessing accounts for the data preparation. A repeated preparation-related
call can therefore concern the model rather than accidentally process the
observations twice. Check the passed flags and the selected likelihood path.

Likelihood definitions, priors and sampler adapters live in `retrieval/`.
Several model/likelihood closures still live inside the simulator; their
location is a remaining architecture task, not an invitation to change their
mathematics while moving them.

Block 9 also still calls MultiNest/emcee through inline setup. The public
adapters in `retrieval/samplers.py` do not yet replace all of those calls.
MultiNest takes a likelihood and separate prior transform; emcee takes a
log-posterior containing both prior and likelihood. Check which interface you
are reading before deciding where the prior is evaluated.

## A practical way to read a function

For each unfamiliar call, establish five things:

1. Which configuration choice selected this branch?
2. Which earlier step produced each input?
3. What are the units, array axes and selection conventions?
4. Does the function calculate, transform, display or save something?
5. Which next step consumes its return values?

If the code does not establish an answer, flag the uncertainty rather than add
an assumed explanation. The comments and diagrams must describe the current
calculation, including compatibility paths, so readers can follow its
scientific intent.
