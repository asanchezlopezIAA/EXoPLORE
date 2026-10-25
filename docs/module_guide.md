# Which file should I open?

Use this as a lookup while browsing the repository. For the sequence of a run,
start with the [code reading guide](code_reading_guide.md); for package boundaries,
see the [architecture overview](architecture.md).

The package paths below are relative to `src/exoplore/`. “Import examples” are
source files that explicitly import that module, found by inspecting Python
imports (including conditional imports). They help locate related code; they
are **not** an execution order, a complete dynamic call graph, or evidence that
every listed route runs in your experiment. Start with the module's “Reading
this file” comments, then follow the function actually called by your route.

The maintained library lives under `src/exoplore/`; `scripts/` contains commands
that use it. A similarly named local script elsewhere is not automatically the
implementation imported by a tutorial. Follow the import path when in doubt.

This index covers maintained package files and tracked scripts. Local research
copies, outputs and ignored files are not a second supported public interface.
A file-purpose note does not replace reading scientific assumptions and array
conventions inside the relevant function.

## Package entry points

| File | Why it is here | Import examples |
|---|---|---|
| `__init__.py` | This top-level module identifies the package and its version. Start a run through config.SimulationConfig and core.ExoploreSimulator; importing the package name alone is not the same operation as executing an experiment. | No explicit importer found; inspect public exports or external callers. |
| `priors.py` | Compatibility reexport of retrieval/priors.py. Old parameter/prior imports and older serialized prior names remain resolvable. | No explicit importer found; inspect public exports or external callers. |
| `results.py` | Named containers for completed CCF/map arrays, with explicit axes, units and copy conventions. They do not repeat the scientific calculation. | `core/simulator.py`, `io/results.py` |
| `samplers.py` | Compatibility reexport of retrieval/samplers.py. Optional sampler packages are needed when their run functions are called. | No explicit importer found; inspect public exports or external callers. |
| `simulator.py` | Compatibility reexport of the simulator in core/simulator.py. Existing imports keep working; make implementation changes in core. | No explicit importer found; inspect public exports or external callers. |
| `timing.py` | Older top-level timing implementations. The current runner uses observation/timing.py; this file is not yet a compatibility reexport. | No explicit importer found; inspect public exports or external callers. |
| `velocity.py` | Compatibility reexport of observation/velocity.py. Keep one canonical implementation of the descriptive velocity functions. | No explicit importer found; inspect public exports or external callers. |

## analysis

| File | Why it is here | Import examples |
|---|---|---|
| `analysis/__init__.py` | These imports provide convenient analysis names, not a required sequence of analysis steps. This package currently imports plotting-oriented modules as well as numerical routines; follow the selected function into its module when checking inputs, plotting configuration or saved-file handling. | `core/simulator.py` |
| `analysis/ccf_time.py` | These routines examine how CCF profiles change across parts of an event. They mix selections, combinations and figure construction; they are not simply image viewers. Start with the function's exposure/frame inputs and follow the selected time intervals before interpreting a change between panels. | `analysis/__init__.py` |
| `analysis/diagnostics.py` | These functions compare already available models, matrices or retrieval products. Some also calculate summaries or generate illustrative examples. Check the function's input/source before treating a figure as a diagnostic of your actual simulation rather than a separate demonstration. | `analysis/__init__.py` |
| `analysis/multi_night.py` | This file combines/comparisons retrieval posteriors across nights or instruments and makes plots. That is different from adding exposure CCFs in the simulator. Check which posterior parameters/priors and input runs are being combined before treating the resulting distribution as a joint inference. | `analysis/__init__.py` |
| `analysis/retrieval_plots.py` | These routines read retrieval products and compare samples, evidences or derived abundance ratios. A ratio derived from sampled abundances is postprocessing, not another forward atmospheric fit. Keep the source runs and parameter conventions attached when comparing their figures. | `analysis/__init__.py` |
| `analysis/stats.py` | Several statistics live here: night-stack summaries, correlations, iteration selection and evidence conversions. Follow the specific statistic's inputs instead of treating every output named significance as the same quantity. Plotting and numerical postprocessing are still mixed in some functions. | `core/simulator.py` |
| `analysis/utils.py` | The cleanup function deletes directory contents. Read its caller and target path before using it: existing files are not recoverable through this function. | `analysis/__init__.py` |

## atmosphere

| File | Why it is here | Import examples |
|---|---|---|
| `atmosphere/__init__.py` | This collects public atmospheric helpers. The actual implementation is in the named pressure, temperature, chemistry, broadening and pRT modules. The runner still uses some older call interfaces alongside descriptive ones; read the actual caller before deciding which helper represents its calculation. | No explicit importer found; inspect public exports or external callers. |
| `atmosphere/chemistry.py` | Temperature and pressure alone do not specify a spectrum: the opacity calculation also needs abundances at every atmospheric layer. These helpers produce those abundance arrays. Keep volume and mass fractions distinct, and check the selected chemistry path before interpreting an abundance parameter. | `atmosphere/__init__.py` |
| `atmosphere/helium_provider.py` | The helium runner asks this small selector for its configured physical backend. Imports happen inside that choice so a molecular run need not load these optional solvers. HeliumTransit here is a factory function: it returns the selected backend object, rather than defining another atmosphere model. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `atmosphere/helium_pwinds.py` | This backend solves a physical outflow and evaluates how it absorbs stellar light during transit. The transit object caches expensive optical-depth information for repeated phases. Detector sampling, observational noise and transmission preparation belong to the observing runner, not this solver. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `atmosphere/helium_sunbather.py` | This is the alternative helium provider selected by helium_provider.py. It handles the external atmosphere solution and returns planet-frame spectra; the observing runner subsequently shifts and samples them. Retained solver files let a researcher inspect convergence rather than just the final figure. | `atmosphere/helium_provider.py`, `retrieval/helium_grid.py` |
| `atmosphere/helium_sunbather_metals.py` | These helpers belong to the metal-bearing Sunbather atmosphere path. They keep the configured composition and the solver's mean-mass iteration together. A failed convergence is an atmosphere failure, not a missing plot; changing the iteration or abundances would be a scientific change. | `atmosphere/helium_sunbather.py` |
| `atmosphere/pressure.py` | This is a small building block: choose the atmospheric pressure levels before evaluating temperatures, chemistry and opacities on those same levels. The returned values are pressures in bar, not altitude coordinates. | `atmosphere/__init__.py` |
| `atmosphere/prt.py` | Radiative-transfer wrappers, chemistry interfaces and convolution helpers. Calls can represent injected atmospheres, search templates or retrieval proposals. | `core/observation.py`, `core/simulator.py` |
| `atmosphere/temperature.py` | The atmosphere solver needs one temperature for each pressure level. This file supplies the selected temperature profile; chemistry and radiative transfer consume it afterward. Several older entry points remain, so follow the actual caller rather than substituting a similarly named profile helper. | `atmosphere/prt.py` |
| `atmosphere/winds.py` | These functions describe broadening and the changing contributions of planetary limbs. Broadening spreads a line over velocities; shifting a whole exposure along an orbital track is a separate observation operation. The runner combines these ingredients, so preserve their order and units. | `core/simulator.py`, `pipelines/sysrem.py` |

## ccf

| File | Why it is here | Import examples |
|---|---|---|
| `ccf/__init__.py` | These imports expose kernels, wrappers and statistics in one place. They have different jobs: wrappers prepare arguments, kernels evaluate correlations, and statistics combine/interpret the resulting CCF values. Importing all names does not mean a run uses every available statistic. | No explicit importer found; inspect public exports or external callers. |
| `ccf/compute.py` | Bridge correlation kernels to analysis and orbital-track sampling. Descriptive helpers and current runner-facing functions coexist; their conventions are not interchangeable. | `core/simulator.py`, `analysis/ccf_time.py` |
| `ccf/kernels.py` | These are numerical inner loops: wrappers in compute.py choose inputs and allocate results before entering them. At every boundary check exposure, pixel and velocity axes plus the weighting choice. Numba acceleration changes execution mechanics; it is not a reason to change normalization or arithmetic. | `ccf/compute.py` |
| `ccf/statistics.py` | This module contains Welch-map calculations and night-combination statistics. A CCF value, a map S/N and a Welch-derived value have different definitions. Read the selected branch and the supplied in/out-of-trail samples before comparing numbers or interpreting repeated night combinations. | `core/simulator.py`, `analysis/stats.py` |

## config

| File | Why it is here | Import examples |
|---|---|---|
| `config/__init__.py` | This is the current public configuration entry point. Its loader below delegates to SimulationConfig.from_json. An older file named config/load.py also exists; do not infer that it is the implementation behind this public function merely because it has a similar filename. | `core/cli.py`, `scripts/plot_crires_czesla2024.py` |
| `config/carmenes_helium.py` | These settings describe the CARMENES direct-helium preparation branch. They reuse shared geometry/selection fields while keeping that branch's normalization and averaging choices separate from the CRIRES recipe. The configuration states the choices; pipelines/carmenes_helium.py applies them. | `config/models.py`, `pipelines/palle2020.py` |
| `config/czesla2024.py` | This configuration records the direct-helium recipe's scientific choices. The pipeline derives exposure membership from observation times and geometry. Distinguish those derived indices from options a user supplies in JSON when following reference-spectrum and in-transit selections. | `config/carmenes_helium.py`, `config/models.py` |
| `config/helium.py` | These are explicit inputs to a helium atmosphere/observation route. Selecting a planet name alone does not supply its irradiation or wind model. Validation here catches incompatible options before the optional atmosphere solver or external observing tools are started. | `atmosphere/helium_pwinds.py`, `config/models.py` |
| `config/helium_grid.py` | A grid fit starts with a measured, prepared helium spectrum and explores explicit model axes. These settings specify that inference stage, not a new noisy observation. axis_values expands the user's grid description before retrieval/helium_grid.py evaluates its points. | `core/simulator.py`, `config/models.py` |
| `config/helium_oh.py` | OH emission is sky contamination near the helium feature. These checks resolve the requested treatment and whether sky-fibre data/calibration are required. They prevent an instrument-incompatible correction from being selected before loading spectra; they do not perform the correction. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `config/load.py` | This is an older loader implementation with field names predating the current model schema. The current public exoplore.config.load_simulation_config is defined in config/__init__.py and delegates to SimulationConfig.from_json. Use that public route for current tutorials; this older file is not an alternative current-schema recipe. Its behavior is left unchanged here. | No explicit importer found; inspect public exports or external callers. |
| `config/models.py` | Named simulation settings, defaults and JSON conversion. Unknown-key compatibility and constructor validation boundaries are documented in the code. | `core/simulator.py`, `scripts/plot_carmenes_palle2020.py` |
| `config/sunbather.py` | This records installation and physical-solver choices for the optional Sunbather backend. The selected backend consumes it through helium_provider; ordinary molecular runs do not need a Sunbather calculation. | `atmosphere/helium_sunbather.py`, `atmosphere/helium_sunbather_metals.py` |

## core

| File | Why it is here | Import examples |
|---|---|---|
| `core/__init__.py` | This exposes the user-facing simulator and forwards older observation helper names on demand through __getattr__. Actual science belongs to the selected runner/helper; the list of exported names is not an execution order. | No explicit importer found; inspect public exports or external callers. |
| `core/carmenes_helium_simulation.py` | This runner connects the helium atmosphere to CARMENES detector inputs and direct-transmission preparation. Full-channel ETC information is consumed before selecting an order, so order selection must not relabel detector/noise arrays. The ordinary molecular simulator remains a separate route. | `core/helium_multi_night.py`, `core/helium_simulation.py` |
| `core/cli.py` | Shared command-line entry point. Read JSON, validate/preview the selected experiment, and execute only when --run is requested. | `scripts/run_exoplore.py` |
| `core/helium_etc.py` | The detector simulation needs noise amplitudes at actual pixels and a selection covering the helium feature. These helpers check/convert those inputs; drawing noise and generating an atmosphere happen elsewhere. A stated S/N must be interpreted with its pixel/exposure convention. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `core/helium_multi_night.py` | This runner coordinates separate synthetic helium nights and then combines their prepared products. Each night retains its own observation noise, telluric fit and reference spectrum. Read the covariance propagation alongside the mean: reusing a reference creates correlations that averaging alone misses. | `core/carmenes_helium_simulation.py` |
| `core/helium_simulation.py` | Read this as the helium counterpart of the molecular run coordinator: validate inputs, obtain an atmosphere/control, simulate detector exposures, prepare the transmission spectrum, then save diagnostics. The physical spectrum provider and the observing adapter have different responsibilities. | `core/carmenes_helium_simulation.py`, `core/helium_multi_night.py` |
| `core/helium_synthetic_observation.py` | This adapter packages synthetic extracted spectra for the existing observed-spectrum molecfit path. It connects simulated exposure times/flux to external telluric fitting. These are extracted spectral inputs, not fabricated raw detector frames or a replacement for upstream raw-data reduction. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `core/observation.py` | These are observation-building functions retained under core by older interfaces. In particular, spec_to_mat_fraction is still reached through the io.stellar forwarding function. Follow the supplied velocity, event fraction and model flags to distinguish injected spectra from templates/retrieval models. | `io/stellar.py`, `pipelines/prepare.py` |
| `core/simulator.py` | Main public coordinator and current standard molecular workflow. Follow its numbered stages into the domain functions, then return to the caller. | `core/cli.py`, `scripts/prepare_carmenes_palle2020.py` |
| `core/spectroscopy_mode.py` | Select one spectroscopy runner before calculation. Spectroscopy choice and synthetic/observed input choice have separate roles. | `core/helium_simulation.py`, `core/simulator.py` |

## instruments

| File | Why it is here | Import examples |
|---|---|---|
| `instruments/__init__.py` | Resolve instrument names through the registry and expose metadata/grid interfaces. Registration and complete end-to-end support are separate checks. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `instruments/base.py` | InstrumentInfo is the small, fixed record passed from a loader to the runner. It describes resolution and input locations rather than containing the observation calculation. Check wavegrid.py and runner branches as well as registration when adding an instrument with a new data layout. | `instruments/carmenes_nir.py`, `instruments/carmenes_vis.py` |
| `instruments/carmenes_helium_molecfit.py` | This is the external-tool bridge for telluric fitting of synthetic CARMENES sections. Fitting sections separately respects detector boundaries. Configured fit gates determine whether a correction is accepted; a failed fit must not be interpreted as a successful unity-transmission atmosphere. | `core/carmenes_helium_simulation.py` |
| `instruments/carmenes_nir.py` | This loader selects NIR input locations for the requested observing mode. get_WaveGrid reads the arrays afterward. A user grid can take precedence over the bundled reference; a fixed S/N can bypass an S/N file. Those choices must be distinguished from generating or preparing the spectral time series. | `instruments/__init__.py` |
| `instruments/carmenes_oh.py` | Fibre B measures the sky contribution; fibre A contains the science spectrum. These routines fit/transfer that sky contribution with explicit calibration. A sky amplitude and an inter-fibre flux conversion are different quantities; inspect their units before applying a subtraction. | `config/helium_oh.py`, `instruments/carmenes_oh_oot.py` |
| `instruments/carmenes_oh_oot.py` | This optional OH fit uses out-of-transit science information together with the sky amplitudes. A common stellar spectrum and a varying sky component are fitted in their respective frames. Acceptance thresholds are needed to check whether that decomposition is constrained by the supplied data. | `config/helium_oh.py`, `pipelines/palle2020.py` |
| `instruments/carmenes_palle2020.py` | This is the CARACAL ingestion/telluric-fitting adapter for observed CARMENES helium work. Keep native pixel orientation and matched fibre/exposure membership before correction. The later transmission recipe is a separate stage from reading extracted spectra and fitting tellurics. | `pipelines/palle2020.py` |
| `instruments/carmenes_vis.py` | This loader does the VIS-channel input selection, using its own bundled wavelength grid when a user grid is absent. Shared CARMENES reading logic in wavegrid.py consumes the resulting locations. NIR and VIS do not share one interchangeable order/pixel layout. | `instruments/__init__.py` |
| `instruments/crires_czesla2024.py` | Upstream cr2res extraction has already happened before this adapter. Here the selected segment's telluric anchors are protected against predicted helium absorption, then external fits are checked. The resulting corrections feed the direct-helium recipe rather than the raw-frame reduction. | `core/helium_synthetic_observation.py`, `scripts/prepare_crires_czesla2024.py` |
| `instruments/crires_molecfit.py` | This module connects extracted CRIRES+ spectra to external molecfit recipes, wavelength checks and telluric models. It is downstream of raw-frame reduction. Read the fit result/acceptance path as well as the launcher: an external command returning does not by itself explain which correction is used. | `scripts/prepare_crires_night.py`, `instruments/crires_czesla2024.py` |
| `instruments/crires_plus.py` | A CRIRES+ observation depends on its selected wavelength setting, not one universal full-band grid. This loader resolves the corresponding inputs and resolution/order information. Reduction and molecfit preparation live in separate modules/scripts and are not performed merely by loading metadata. | `instruments/__init__.py` |
| `instruments/igrins.py` | This loader describes IGRINS reference-night or observed-data inputs. The preparation script converts upstream PLP products into that input format. The instrument's order layout then has to stay consistent across wavelength, flux, uncertainty and timing arrays. | `instruments/__init__.py` |
| `instruments/wavegrid.py` | Metadata loaders identify files; get_WaveGrid turns those inputs into arrays, and the ANDES helpers construct order grids from ETC information. Pay attention to each instrument branch's axis order and unit conversions. The runner has compatibility transposes, so changing an orientation here can silently relabel pixels/orders even when array sizes look plausible. | `core/simulator.py`, `analysis/ccf_time.py` |

## io

| File | Why it is here | Import examples |
|---|---|---|
| `io/__init__.py` | These public names make file/path helpers convenient to import. Look in the implementing module to distinguish reading, writing, name construction and numerical helpers retained under IO. A path returned by a helper does not by itself establish that the file already exists. | No explicit importer found; inspect public exports or external callers. |
| `io/naming.py` | This naming helper is available to postprocessing without importing the simulator. The standard runner still has its own name builder in core/simulator. Trace which one a caller uses; matching function names do not establish that all naming behavior has already been consolidated. | No explicit importer found; inspect public exports or external callers. |
| `io/paths.py` | Construct related output locations and optionally create their directories. A returned Path is not a calculated scientific product. | `core/simulator.py` |
| `io/results.py` | Read/write compact results and bridge the current runner dictionaries to typed public products. No reweighting, correlation or normalization is performed here. | `core/simulator.py` |
| `io/stellar.py` | Read, convolve and interpolate stellar models, plus retained stellar/projection interfaces. The planet projection shim forwards to core/observation.py. | `core/simulator.py` |
| `io/utils.py` | These are shared file/array helpers, including saving and numerical summaries. Check whether a particular helper writes files or samples/bootstrap indices before calling it during inspection. Formatting and serialization should not be confused with the scientific operation that produced an array. | `core/simulator.py`, `analysis/stats.py` |

## observation

| File | Why it is here | Import examples |
|---|---|---|
| `observation/__init__.py` | This groups the observing ingredients used by runners: times, motion, airmass and noise. They must refer to the same exposure sequence when combined. The package exports the helpers; the runner determines when and with which arrays each is called. | `core/simulator.py` |
| `observation/airmass.py` | Airmass and water-vapour inputs describe how observing conditions vary through the exposure sequence. Some functions construct synthetic curves; others write SkyCalc inputs or PWV products. Check which one is called before expecting a returned array versus files or an external-tool launcher. | `core/simulator.py`, `scripts/generate_skycalc_inputs.py` |
| `observation/ephemeris.py` | Different observing dates have different sky geometry and Earth motion. These helpers calculate dates, airmass and BERV from target/site information for synthetic different-night observations. Returned velocities must retain the same BERV sign convention used by observation.velocity. | `core/simulator.py` |
| `observation/noise.py` | This file contains numerical noise/amplitude helpers and the actual multiplicative throughput draw. randomness.py separately chooses the existing seeds. The standard runner also has its own Gaussian draw/scaling branches; these convenience helpers do not silently replace every one of those calls. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `observation/randomness.py` | Choose the existing throughput/Gaussian seeds while preserving different-night noise and existing stream identities. The actual draws are separate. | `core/simulator.py` |
| `observation/timing.py` | Choose the exposure times and identify which exposures contain the planetary event before constructing spectra or combining CCFs. The selected indices refer to exposures, not wavelengths. This file also retains light-curve helpers used by the current runner; a timing selection and an absorption weight are related inputs, not interchangeable arrays. | `core/simulator.py`, `scripts/generate_skycalc_inputs.py` |
| `observation/velocity.py` | Orbital velocity helpers with documented units/signs. Keep the distinct eccentric interfaces and their angle/phase conventions separate. | `core/simulator.py`, `analysis/stats.py` |

## pipelines

| File | Why it is here | Import examples |
|---|---|---|
| `pipelines/__init__.py` | These imports expose preparation building blocks. Selecting a pipeline happens in prepare.py or a direct-helium runner, rather than by applying every exported function. Keep masks, propagated uncertainties and any model-filtering information connected to the selected preparation. | No explicit importer found; inspect public exports or external callers. |
| `pipelines/bl19.py` | These are preparation operations selected by pipelines/prepare.py. Normalization and the subsequent telluric fits act in a particular sequence; the output of one becomes the input of the next. Preserve that sequence when following data, uncertainties and excluded wavelength columns. | `pipelines/cheverall26.py`, `pipelines/prepare.py` |
| `pipelines/blain24.py` | The preparation dispatcher calls these polynomial corrections in order: throughput over wavelength, then telluric behaviour against airmass. The corrected data and propagated uncertainties travel together. Model-processing calls can use different flags, so inspect arguments before assuming data and model are being treated identically in every branch. | `pipelines/prepare.py` |
| `pipelines/carmenes_helium.py` | This direct-transmission recipe normalizes spectra, forms the stellar reference, then aligns/combines the selected planetary spectra. It is separate from the molecular CCF pipeline. The reference membership and arithmetic averaging choices are part of the calculation and must travel with its inputs. | `core/carmenes_helium_simulation.py`, `scripts/plot_carmenes_palle2020.py` |
| `pipelines/cheverall26.py` | This file provides the individual preparation operations and timing checks used by the Cheverall26 branch. The dispatcher/runner determines their sequence and which data enter them. Normalization, outlier removal and a signal-duration diagnostic are different tasks despite sharing this module. | `core/simulator.py`, `pipelines/prepare.py` |
| `pipelines/czesla2024.py` | This file contains the direct-helium recipe and its observed-data runner. Exposure selections, reference division and frame changes precede the final combined spectrum. Fit/plot helpers here are downstream diagnostics; inspect their inputs rather than assuming every call repeats preparation. | `core/carmenes_helium_simulation.py`, `core/helium_multi_night.py` |
| `pipelines/helium_math.py` | These small numerical helpers keep direct-helium wavelength-frame and resampling conventions together. Wavelengths are vacuum nm and velocities km/s here; molecular routines may use microns. Preserve valid-pixel/gap masks when interpolating rather than inventing data across detector gaps. | `core/carmenes_helium_simulation.py`, `core/helium_multi_night.py` |
| `pipelines/helium_noise_propagation.py` | Preparation reuses a reference spectrum and interpolates pixels, so errors become correlated. These routines propagate the supplied variances through those operations without drawing new noise. They hold fitted continuum and telluric parameters fixed; the covariance does not include all possible calibration-parameter uncertainties. | `core/carmenes_helium_simulation.py`, `core/helium_multi_night.py` |
| `pipelines/helium_significance.py` | This measures a configured helium band relative to continuum-noise behaviour. The bandwidth, continuum selections and binning are part of that statistic. It is a separate calculation from the molecular CCF-map S/N and from a statistical bootstrap experiment. | `core/helium_multi_night.py`, `core/helium_simulation.py` |
| `pipelines/helium_sysrem.py` | This is an optional alternative preparation for synthetic helium data. It removes components in the Earth frame before later transmission analysis. Its differential residuals are not a theoretical telluric transmittance, so read which product a caller consumes before using it as a correction. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `pipelines/masking.py` | Identify excluded/retained spectral columns and repair/check selected data conditions. Read each function to distinguish integer indices, boolean selections and corrected matrices. | `core/simulator.py`, `scripts/prepare_crires_night.py` |
| `pipelines/nortmann26.py` | These are the Nortmann-specific preparation/model-filtering operations. The driving branch is in prepare.py; wavelength/telluric inputs can come from the CRIRES preparation adapters. Inspect whether a call acts on observations, a model, or already calculated CCFs before interpreting its output. | `pipelines/prepare.py` |
| `pipelines/palle2020.py` | This is the observed-CARMENES workflow around the direct-transmission recipe. It resolves exposure membership and OH treatment, reads correction products, and then uses the shared CARMENES preparation. Sky-fibre data locate contamination; they are not substituted for the science flux. | `core/simulator.py`, `scripts/prepare_carmenes_palle2020.py` |
| `pipelines/pca.py` | PCA removes selected singular components from the exposure/pixel matrix. Its call signature resembles SYSREM so the dispatcher can choose either, but PCA does not use the uncertainty array as inverse-variance weights. Interface compatibility does not make their scientific operators identical. | `pipelines/prepare.py` |
| `pipelines/prepare.py` | Choose the preparation branch and return residuals, uncertainties, selections and filtering information. It is shared with configured model-reprocessing paths. | `core/simulator.py` |
| `pipelines/sysrem.py` | SYSREM fits/removes weighted time-wavelength components; model-filtering helpers account for that operation when comparing a proposed signal to prepared data. Selection of how many passes to use is a separate decision. Follow the returned components/projector as well as the residual matrix. | `core/simulator.py`, `pipelines/helium_sysrem.py` |
| `pipelines/tellurics.py` | Telluric transmittance is an input to synthetic observations, while pipeline_fixedTellurics is a preparation path for the configured fixed case. Loading a theoretical atmosphere and removing structure from observed spectra are different operations, even though both concern terrestrial absorption. | `core/simulator.py`, `pipelines/prepare.py` |

## planets

| File | Why it is here | Import examples |
|---|---|---|
| `planets/__init__.py` | Use load_planet to obtain the system-parameter object consumed by the runner. Configuration selects the observation/analysis separately; the planet object records the physical system and derived quantities used across stages. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `planets/catalog.py` | This reader converts a planet JSON into PlanetParameters, including conversion from the supported older file convention. It loads system facts; the simulation JSON separately selects how to observe and analyse that system. Read the relevant format branch before interpreting a short legacy field name. | `planets/__init__.py` |
| `planets/models.py` | A planet object keeps system parameters and derived orbital quantities together so the atmosphere and timing code use the same system. Public fields with unit suffixes are the clearest starting point. Compatibility properties convert values for older routines; they are not new independent measurements. | `planets/catalog.py` |

## plotting

| File | Why it is here | Import examples |
|---|---|---|
| `plotting/__init__.py` | These are convenient names for several figure families. Find the individual function to check expected axes, units and whether it loads or reconstructs data before drawing. The public import list is not a recipe for deciding which scientific analysis to perform. | `scripts/run_detectability_maps.py` |
| `plotting/calibration.py` | These figures inspect the relationship between extracted spectra, telluric corrections and measured resolution. They help check an upstream calibration stage; drawing the comparison is not a new external molecfit run. | `plotting/__init__.py` |
| `plotting/carmenes_oh.py` | These plots explain the saved sky correction and its A/B calibration. They consume correction products rather than refitting the atmosphere or re-estimating a sky transfer factor as part of displaying a figure. | `scripts/plot_carmenes_palle2020.py`, `pipelines/palle2020.py` |
| `plotting/detectability.py` | The simulation sweep writes one statistic per atmospheric grid point; this reader assembles those points into a figure. The plotted colour is the chosen detection statistic, not a posterior abundance probability or a fresh simulation performed by the plotting function. | `plotting/__init__.py` |
| `plotting/detection.py` | This file preserves older plotting imports by forwarding names to kpvsys.py/matrices.py. Find the real function there when reading or changing a plot; adding another implementation here would recreate duplicated code. | No explicit importer found; inspect public exports or external callers. |
| `plotting/helium.py` | These functions show prepared helium observations and their model comparisons with consistent axes. Identify the wavelength frame and the prepared model used for residuals. The figure renderer is separate from physical atmosphere generation and direct-transmission preparation. | `core/helium_multi_night.py`, `core/helium_simulation.py` |
| `plotting/helium_atmosphere.py` | These diagnostics let a researcher inspect the actual solved atmospheric profiles rather than only the observed line. They accompany the optional Sunbather result; their quantities are atmosphere profiles, not detector counts or a measured transmission spectrum. | `atmosphere/helium_sunbather.py` |
| `plotting/helium_grid.py` | The grid runner has already evaluated model points. This file displays those fit results and temperature information; it does not infer a detection from the colour scale or silently run an extra physical grid. | `retrieval/helium_grid.py` |
| `plotting/helium_molecfit.py` | These routines read saved per-exposure fit reports to show accepted correction/kernel behaviour. Distinguish a fitted kernel width from a fixed configured width, and keep detector-section/nod membership when comparing exposures. Reading reports does not repeat external fits. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `plotting/kpvsys.py` | This file contains both descriptive plot helpers and older runner-facing plot functions. Check each function's expected map orientation and its supplied velocity convention: a systemic velocity and a residual Vrest offset are not interchangeable labels. The map statistic is calculated upstream. | `core/simulator.py`, `plotting/detection.py` |
| `plotting/matrices.py` | These functions display wavelength/exposure or velocity/exposure arrays. Axis orientation, masks and colour scaling determine what a panel means; check the call's actual arrays and display options. A displayed residual is not automatically a clean planetary spectrum. | `core/simulator.py` |
| `plotting/steps.py` | These diagnostics connect the simulated spectra to prepared residuals. Some helpers reconstruct intermediate SYSREM stages from stored components: that is deliberate numerical reconstruction, unlike loading a saved PDF. Display-only options must be distinguished from the arrays used for analysis. | `core/simulator.py`, `scripts/plot_carmenes_palle2020.py` |

## retrieval

| File | Why it is here | Import examples |
|---|---|---|
| `retrieval/__init__.py` | This exposes likelihoods, priors and sampler adapters. A likelihood evaluates a supplied model against data; a prior supplies parameter support or a coordinate transform; a sampler explores proposals. Importing these objects is separate from enabling and executing a retrieval. | No explicit importer found; inspect public exports or external callers. |
| `retrieval/emulator.py` | This is an explicitly experimental spectrum surrogate, not the default physical solver. Training/prediction accuracy and retrieval accuracy are different checks. The warning above remains applicable; moving or documenting this file does not validate it for scientific retrievals. | `core/simulator.py`, `scripts/emulator/compare_emulator.py` |
| `retrieval/helium_grid.py` | This inference runner consumes a prepared helium spectrum and evaluates physical atmosphere points while retaining the supplied sampling/covariance context. It does not generate a new noisy observation. Follow native-input replay in _night_prediction when checking how model and data are compared. | `core/simulator.py` |
| `retrieval/likelihood.py` | Evaluate supplied data/model arrays using the selected likelihood. No atmosphere generation or sampler exploration happens in these functions. | `retrieval/__init__.py` |
| `retrieval/likelihood_emulator.py` | This experimental surrogate predicts likelihood values rather than running every atmospheric calculation. That can alter an inferred posterior if approximation errors matter. Read its existing validation warning before considering it as a substitute for the actual likelihood. | `core/simulator.py` |
| `retrieval/priors.py` | Prior transforms and prior-density evaluation serve different samplers. The ordered parameter names connect sampler coordinates to physical model arguments; reordering them without changing the consumer changes the experiment. This file defines distributions, not atmospheric spectra or data likelihoods. | `priors.py` |
| `retrieval/samplers.py` | Public sampler adapters. MultiNest receives a likelihood and prior transform; emcee receives a log-posterior. Some runner setup remains inline. | `samplers.py` |

## Maintained command-line scripts

Paths here are relative to `scripts/`. Scripts can generate inputs, launch real
calculations, inspect saved outputs, or build teaching illustrations. These are
different jobs; choose the entry point matching the step you intend to perform.
No script was executed while constructing this index.

| Script | Role |
|---|---|
| `carmenes_etc_to_fits.py` | This converts an ETC table into the S/N FITS input expected by EXoPLORE. It prepares an input noise description; it does not simulate the transit or measure a planet from spectra. Channel/order layout must match the later loader. |
| `emulator/compare_emulator.py` | This compares predicted spectra against actual atmospheric-model calculations. Spectral agreement is one diagnostic; it does not establish that the experimental emulator yields an accurate retrieval posterior. |
| `emulator/compare_retrievals.py` | This compares saved emulator and physical-model retrieval results. The emulator remains experimental; inspecting timings and corner plots here is part of assessing its limitations, not switching the default solver. |
| `emulator/generate_training_data.py` | This generates physical spectra used to train the experimental emulator. Its sampling, batches and stored coefficients define the training set. It is a model-generation job, not a light plotting-only command. |
| `emulator/train_emulator.py` | This fits the experimental surrogate to its supplied training products. Training changes the approximation, not the physical opacity model itself. Use separate validation/retrieval checks before drawing scientific conclusions. |
| `generate_skycalc_inputs.py` | This prepares telluric input spectra through the SkyCalc service. It is an input-generation step before the simulation reads transmittances, not a fit to the observed planetary signal. Read its observing/PWV choices first. |
| `illustrate_doppler_trick.py` | This makes a schematic teaching figure about moving spectral lines. The arrays are illustrative, unlike the numerical products of a real simulator run. Use it to understand the concept, not to validate a measured detection. |
| `illustrate_likelihood_weighting.py` | This controlled teaching example compares likelihood behaviour on a toy spectrum. It helps explain a statistical choice; it is not a replacement for running a real-data preparation and atmospheric retrieval. |
| `illustrate_model_spectrum.py` | This runs atmospheric-model calculations to make documentation figures. It is more than a viewer of existing PNG files: the configured inputs determine the physical spectra generated when the script is executed. |
| `illustrate_significance_sampling.py` | This teaching calculation explores how correlated velocity samples affect a significance statistic. Its demonstration concerns statistical sampling; interpreting a real dataset still requires its own validation. |
| `make_planet_params.py` | This writes a planetary-system input JSON after calculating derived quantities from the user-input section. Check those physical inputs and units before executing; the later simulator reads the resulting system description. |
| `make_wave_carmenes_nir.py` | This converts an upstream time-dependent CARACAL wavelength array to the reference order/pixel layout. Averaging those wavelength solutions is an explicit preparation choice, separate from processing a planetary spectrum. |
| `plot_carmenes_palle2020.py` | This reads saved correction/transmission products to make diagnostic figures. It is downstream of preparation; using it does not run another atmosphere solver or refit each exposure with molecfit. |
| `plot_corner_overlay.py` | This compares saved posterior samples from selected retrievals. Check which parameters and source runs are overlaid; plotting a distribution does not launch a sampler or make different priors equivalent. |
| `plot_crires_czesla2024.py` | This displays saved telluric-correction and helium-preparation products. The fitting/acceptance decisions happened upstream. Display scaling in a panel must be distinguished from the correction applied to scientific flux. |
| `plot_multinight_stacking.py` | This reads per-night and combined maps already produced by a run. Slices are taken at each map's chosen Kp, so the figure is a comparison of saved profiles rather than an additional simulation or a new posterior fit. |
| `plot_significance_study.py` | This plots statistics saved over repeated realizations. Identify which metric and realization table are supplied before interpreting the cloud of points; displaying the table does not generate another noise ensemble. |
| `prepare_carmenes_night.py` | This converts extracted CARACAL observations into the reference-night arrays consumed by the instrument loader. Keep exposure order and matching wavelength/flux/error arrays together. Raw detector reduction is upstream. |
| `prepare_carmenes_palle2020.py` | This is a command-line entry for the observed CARMENES helium input/fit workflow. Preview and execution choices differ; follow the requested action before expecting either an inspection summary or external fitting products. |
| `prepare_crires_czesla2024.py` | This connects command-line choices to the selected-segment helium molecfit adapter. Extracted spectra and a manifest are inputs; raw cr2res reduction is a separate earlier stage. The pilot concerns a selected exposure. |
| `prepare_crires_night.py` | This ingests externally extracted CRIRES+ exposures into EXoPLORE inputs, with the configured wavelength/telluric preparation. It belongs after raw reduction and before the reference-night simulation or observed-data analysis. |
| `prepare_igrins_night.py` | This reads upstream PLP H/K products and writes the reference-night layout used by the IGRINS loader. The conversion must keep time, order and pixel axes aligned across all saved inputs; it does not reduce raw frames. |
| `reduce_crires_night.py` | This launches upstream ESO cr2res reduction of raw exposures and their calibrations. Its extracted spectra are inputs to prepare_crires_night.py. It is a different stage from later atmospheric preparation/cross-correlation. |
| `run_detectability_maps.py` | This drives repeated simulations over atmospheric grid points. It can therefore be substantially more expensive than one run. The plotted grid is a sweep of recovered statistics, not an atmosphere posterior from one sampler. |
| `run_exoplore.py` | This source-checkout launcher delegates to the shared command-line interface. Without --run it previews the configuration; with --run it starts the same simulator used by the notebook. Keep scientific logic in the package. |
| `verify_clean_run.py` | This inspects existing run outputs against its programmed checks. Read its path settings and assumptions before using it for another experiment; an archived verification script is not a universal guarantee for every route. |

## What the comments do and do not certify

The orientation notes explain ownership and connections. They do not certify
that an experimental model is validated, that an older loader implements the
current schema, or that all functions with similar names are equivalent.
Experimental emulators and older interfaces remain labelled as such. Numerical
behavior, scientific defaults and published notebook contents are preserved.
