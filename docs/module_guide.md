# Which file should I open?

This index lists the package modules and command-line scripts. Use the
[code reading guide](code_reading_guide.md) to follow a run and the
[architecture guide](architecture.md) for the folder map.

Package paths are relative to `src/exoplore/`. The last column lists source
files with explicit imports of that module, including imports inside functions.
Follow the selected calling function to see how its arguments are constructed.

## Package entry points

| File | Function | Used by |
|---|---|---|
| `__init__.py` | Define the package version and author. |  |
| `results.py` | Store completed CCFs and maps with named arrays, axes, units and observing metadata. | `core/simulator.py`, `io/results.py` |

## analysis

| File | Function | Used by |
|---|---|---|
| `analysis/__init__.py` | Export analysis and diagnostic plotting functions from the listed modules. | `core/simulator.py` |
| `analysis/ccf_time.py` | Select time intervals from CCF products, combine their information and plot the resulting profiles. Exposure selections and velocity frames determine which parts of the event are compared. | `analysis/__init__.py` |
| `analysis/diagnostics.py` | Compare model spectra, spectral matrices and retrieval products, and generate the selected diagnostic or illustrative figure. | `analysis/__init__.py` |
| `analysis/multi_night.py` | Combine and compare retrieval posteriors from selected nights or instruments. Parameter definitions, priors and source runs specify the distributions being combined. | `analysis/__init__.py` |
| `analysis/retrieval_plots.py` | Read retrieval products and plot posterior distributions, evidences and abundance ratios. Derived quantities use the samples from the selected source runs. | `analysis/__init__.py` |
| `analysis/stats.py` | Plot statistical summaries and correlation diagnostics, and convert evidence ratios. Night-study calculations are in ccf/statistics.py. | `analysis/__init__.py`, `core/simulator.py` |
| `analysis/utils.py` | Delete the contents of a specified directory. This removes its files and subdirectories. | `analysis/__init__.py` |

## atmosphere

| File | Function | Used by |
|---|---|---|
| `atmosphere/__init__.py` | Export atmospheric functions from the pressure, temperature, chemistry, wind and pRT modules. |  |
| `atmosphere/chemistry.py` | Calculate abundance arrays at the atmospheric pressure and temperature levels. The radiative-transfer calculation uses these abundances to weight each species' opacity. Mass fractions and volume mixing ratios have separate conversion functions. | `atmosphere/__init__.py` |
| `atmosphere/helium_provider.py` | Select the configured helium atmosphere solver. HeliumTransit returns an instance of that solver. Imports occur when the solver is selected. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `atmosphere/helium_pwinds.py` | Solve the planetary outflow and calculate its absorption during transit. The transit object caches optical depths for repeated phase calculations. The observing code samples these spectra onto detector pixels. | `atmosphere/helium_provider.py`, `atmosphere/helium_sunbather.py`, `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `atmosphere/helium_sunbather.py` | Run the Sunbather atmosphere solver and return planet-frame spectra to the observing code. Saved solver files contain the profiles and convergence information for the atmosphere. | `atmosphere/helium_provider.py`, `retrieval/helium_grid.py` |
| `atmosphere/helium_sunbather_metals.py` | Calculate the composition and mean-mass iteration for metal-bearing Sunbather atmospheres. Convergence checks determine whether the solved atmosphere can be used. | `atmosphere/helium_sunbather.py` |
| `atmosphere/models.py` | Define the system quantities and units shared by atmospheric spectrum calculations. | `atmosphere/prt.py`, `core/simulator.py`, `scripts/emulator/compare_emulator.py`, `scripts/emulator/generate_training_data.py`, `scripts/illustrate_model_spectrum.py` |
| `atmosphere/pressure.py` | Choose the pressure levels, in bar, on which temperature, chemistry and radiative transfer are evaluated. | `atmosphere/__init__.py` |
| `atmosphere/prt.py` | Radiative-transfer wrappers, chemistry interfaces and convolution helpers. Calls can represent injected atmospheres, search templates or retrieval proposals. | `analysis/ccf_time.py`, `atmosphere/__init__.py`, `core/observation.py`, `core/simulator.py`, `io/stellar.py`, `pipelines/tellurics.py`, `retrieval/helium_grid.py`, `scripts/emulator/compare_emulator.py`, `scripts/emulator/generate_training_data.py`, `scripts/illustrate_model_spectrum.py` |
| `atmosphere/temperature.py` | Calculate the configured temperature at each atmospheric pressure level. Chemistry and radiative transfer use the resulting profile. | `atmosphere/__init__.py`, `atmosphere/prt.py` |
| `atmosphere/winds.py` | Calculate velocity broadening and the changing contributions of planetary limbs. The runner combines these profiles with the exposure-dependent orbital velocity. | `atmosphere/__init__.py`, `core/simulator.py`, `pipelines/sysrem.py` |

## ccf

| File | Function | Used by |
|---|---|---|
| `ccf/__init__.py` | Export CCF kernels, calculation functions and statistics. |  |
| `ccf/compute.py` | Calculate CCFs, combine orders and sample orbital velocity tracks using the kernels and configured velocity grids. | `analysis/ccf_time.py`, `analysis/stats.py`, `ccf/__init__.py`, `ccf/statistics.py`, `core/simulator.py` |
| `ccf/kernels.py` | Calculate exposure CCFs over the trial velocity grid. compute.py supplies the arrays and allocates the results. These Numba loops evaluate the selected weighting and correlation formula. | `ccf/__init__.py`, `ccf/compute.py` |
| `ccf/models.py` | Define the configuration and reference motion used by night studies and time-resolved CCF diagnostics. | `analysis/ccf_time.py`, `ccf/statistics.py`, `core/simulator.py` |
| `ccf/statistics.py` | Calculate Welch detection maps and night-combination statistics from CCFs. The selected in-trail and out-of-trail samples define each statistic. | `analysis/stats.py`, `ccf/__init__.py`, `core/simulator.py` |

## config

| File | Function | Used by |
|---|---|---|
| `config/__init__.py` | Export configuration classes and load_simulation_config(), which calls SimulationConfig.from_json(). | `core/cli.py`, `scripts/plot_crires_czesla2024.py` |
| `config/carmenes_helium.py` | Define the normalization, reference selection and averaging settings for CARMENES direct-helium preparation. pipelines/carmenes_helium.py applies these settings to the spectra. | `config/models.py`, `pipelines/palle2020.py`, `retrieval/helium_grid.py` |
| `config/czesla2024.py` | Define the scientific settings for CRIRES+ direct-helium preparation. The pipeline uses observation times and orbital geometry to select the reference and in-transit exposures. | `config/carmenes_helium.py`, `config/models.py`, `instruments/crires_czesla2024.py`, `pipelines/czesla2024.py`, `retrieval/helium_grid.py` |
| `config/helium.py` | Define the helium atmosphere and observing inputs. Validation checks the irradiation, wind and observing settings before starting the selected solver or external tools. | `atmosphere/helium_pwinds.py`, `config/models.py` |
| `config/helium_grid.py` | Define the atmospheric axes and input spectrum for a helium grid fit. axis_values expands the configured axes into the values evaluated by retrieval/helium_grid.py. | `config/models.py`, `core/simulator.py`, `plotting/helium_grid.py`, `retrieval/helium_grid.py` |
| `config/helium_oh.py` | Validate the OH-emission treatment and its required sky-fibre data. The selected correction is applied later by the instrument preparation code. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py`, `core/spectroscopy_mode.py`, `pipelines/czesla2024.py`, `pipelines/palle2020.py` |
| `config/models.py` | Define simulation settings, defaults, JSON loading and field validation. | `ccf/models.py`, `config/__init__.py`, `core/simulator.py`, `instruments/__init__.py`, `instruments/base.py`, `io/paths.py`, `pipelines/models.py`, `scripts/generate_skycalc_inputs.py`, `scripts/plot_carmenes_palle2020.py`, `scripts/prepare_carmenes_palle2020.py`, `scripts/run_detectability_maps.py` |
| `config/sunbather.py` | Define the installation paths and physical settings consumed by the optional Sunbather atmosphere solver. | `atmosphere/helium_sunbather.py`, `atmosphere/helium_sunbather_metals.py`, `config/helium.py`, `config/helium_grid.py` |

## core

| File | Function | Used by |
|---|---|---|
| `core/__init__.py` | Export the simulator and load the listed observation helpers when requested. |  |
| `core/carmenes_helium_simulation.py` | Generate CARMENES helium observations from the selected atmosphere, detector inputs and direct-transmission preparation. Read full-channel ETC information before selecting the order so its noise arrays keep their detector indices. | `core/helium_multi_night.py`, `core/helium_simulation.py` |
| `core/cli.py` | Shared command-line entry point. Read JSON, validate/preview the selected experiment, and execute the experiment with --run. | `scripts/run_exoplore.py` |
| `core/helium_etc.py` | Read and convert detector noise inputs and select pixels covering the helium feature. S/N values carry their pixel and exposure conventions into the observation calculation. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `core/helium_multi_night.py` | Simulate and prepare separate helium nights, then combine their products. Each night supplies its noise, telluric correction and reference spectrum. Covariance propagation accounts for the shared-reference errors. | `core/carmenes_helium_simulation.py` |
| `core/helium_simulation.py` | Run a helium experiment: validate inputs, calculate the atmosphere or control spectrum, generate detector exposures, prepare the transmission spectrum and save diagnostics. | `core/carmenes_helium_simulation.py`, `core/helium_multi_night.py`, `core/simulator.py` |
| `core/helium_synthetic_observation.py` | Write synthetic extracted spectra for the observed-spectrum molecfit interface. Exposure times, wavelengths and flux arrays provide the inputs to the external telluric fit. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `core/observation.py` | Construct spectral time-series matrices from velocity tracks, event fractions and model spectra. The simulator and preparation pipeline call spec_to_mat_fraction here. | `core/simulator.py`, `pipelines/prepare.py` |
| `core/simulator.py` | Run the molecular spectroscopy workflow. Numbered blocks load inputs, calculate spectra, prepare them, calculate CCFs and maps, save results and run enabled retrievals. | `core/__init__.py`, `core/cli.py`, `scripts/prepare_carmenes_palle2020.py` |
| `core/spectroscopy_mode.py` | Select one spectroscopy runner before calculation. Spectroscopy choice and synthetic/observed input choice have separate roles. | `core/helium_simulation.py`, `core/simulator.py` |

## instruments

| File | Function | Used by |
|---|---|---|
| `instruments/__init__.py` | Resolve instrument names through the registry and expose metadata/grid interfaces. Registration and complete end-to-end support are separate checks. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `instruments/base.py` | Define InstrumentInfo, which passes an instrument's resolution and input locations to the runner. wavegrid.py reads the corresponding arrays. | `instruments/carmenes_nir.py`, `instruments/carmenes_vis.py` |
| `instruments/carmenes_helium_molecfit.py` | Fit telluric transmission separately in the synthetic CARMENES detector sections. The configured acceptance checks determine which fitted corrections can be used. | `core/carmenes_helium_simulation.py` |
| `instruments/carmenes_nir.py` | Resolve CARMENES NIR wavelength and S/N inputs for the selected observing mode. get_WaveGrid reads the selected files. A supplied grid or fixed S/N takes precedence over the corresponding reference file. | `instruments/__init__.py` |
| `instruments/carmenes_oh.py` | Fit the sky spectrum in fibre B and calibrate its contribution to the science spectrum in fibre A. The fitted sky amplitude and inter-fibre flux factor determine the subtraction. | `config/helium_oh.py`, `instruments/carmenes_oh_oot.py`, `pipelines/palle2020.py` |
| `instruments/carmenes_oh_oot.py` | Fit a common stellar spectrum and time-varying OH emission using out-of-transit science spectra and sky amplitudes in their respective wavelength frames. Acceptance checks test whether the data constrain the fitted components. | `config/helium_oh.py`, `pipelines/palle2020.py` |
| `instruments/carmenes_palle2020.py` | Read extracted CARACAL exposures for CARMENES helium analysis and fit their telluric corrections. Preserve native pixel orientation and match fibres by exposure before constructing transmission spectra. | `pipelines/palle2020.py` |
| `instruments/carmenes_vis.py` | Resolve CARMENES VIS input locations and its reference wavelength grid. wavegrid.py reads these inputs using the VIS order and pixel layout. | `instruments/__init__.py` |
| `instruments/crires_czesla2024.py` | Fit telluric corrections in the selected extracted CRIRES+ segment. Protect the predicted helium wavelength intervals during fitting and check the corrections before direct-transmission preparation. | `core/helium_synthetic_observation.py`, `instruments/carmenes_helium_molecfit.py`, `instruments/carmenes_palle2020.py`, `scripts/prepare_crires_czesla2024.py` |
| `instruments/crires_molecfit.py` | Prepare extracted CRIRES+ spectra for the external molecfit recipes, run the fits and check their wavelengths and transmission models. | `instruments/crires_czesla2024.py`, `pipelines/czesla2024.py` |
| `instruments/crires_plus.py` | Resolve CRIRES+ input files, spectral orders and resolution for the selected wavelength setting. The reduction and preparation scripts produce the files read here. | `instruments/__init__.py` |
| `instruments/igrins.py` | Resolve the IGRINS reference-night or observed-data files. The reader uses the order and pixel layout written by the PLP preparation script. | `instruments/__init__.py` |
| `instruments/wavegrid.py` | Read wavelength grids and their associated noise, timing and airmass arrays. ANDES branches construct order grids from ETC data. Instrument-specific branches define their axis order and unit conversions. | `analysis/ccf_time.py`, `core/simulator.py` |

## io

| File | Function | Used by |
|---|---|---|
| `io/__init__.py` | Export file readers, path constructors, saving functions and numerical utilities. |  |
| `io/paths.py` | Construct experiment output paths and create their directories when requested. | `core/simulator.py`, `io/__init__.py` |
| `io/results.py` | Read and write compact results. Construct CCF and detection-map objects from the arrays and metadata stored by the simulator. | `core/simulator.py` |
| `io/stellar.py` | Read PHOENIX spectra and construct Doppler-shifted stellar matrices. | `core/simulator.py`, `io/__init__.py` |
| `io/utils.py` | Read, save and format shared file and array products. Numerical helpers in this module also calculate summaries and sampling indices for analysis functions. | `analysis/stats.py`, `ccf/statistics.py`, `core/simulator.py`, `io/__init__.py` |

## observation

| File | Function | Used by |
|---|---|---|
| `observation/__init__.py` | Export the observing functions for times, motion, airmass and noise. They must refer to the same exposure sequence when combined. The package exports the helpers. The runner determines when and with which arrays each is called. | `core/simulator.py` |
| `observation/airmass.py` | Construct airmass and water-vapour sequences and prepare the corresponding SkyCalc inputs. The selected function returns arrays or writes the input files used by the observing calculation. | `core/simulator.py`, `observation/__init__.py`, `scripts/generate_skycalc_inputs.py` |
| `observation/ephemeris.py` | Calculate observable transit dates, airmass and BERV from the target and observatory coordinates. The observing code uses these velocities with the sign convention defined in observation.velocity. | `core/simulator.py` |
| `observation/noise.py` | Generate Gaussian noise and multiplicative throughput variations. randomness.py selects their configured seeds. The simulator applies its observation-specific noise scaling. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py`, `core/simulator.py`, `observation/__init__.py`, `observation/randomness.py` |
| `observation/randomness.py` | Choose throughput and Gaussian-noise seeds for each order and night. observation/noise.py generates the random values. | `core/simulator.py`, `observation/__init__.py` |
| `observation/timing.py` | Calculate exposure times, event membership and light-curve weights. Exposure-index arrays select the spectra used to construct and combine the planetary signal. | `analysis/__init__.py`, `core/simulator.py`, `observation/__init__.py`, `pipelines/czesla2024.py`, `pipelines/palle2020.py`, `scripts/generate_skycalc_inputs.py` |
| `observation/velocity.py` | Orbital velocity helpers with documented units/signs. Keep the distinct eccentric interfaces and their angle/phase conventions separate. | `ccf/compute.py`, `ccf/statistics.py`, `core/simulator.py`, `observation/__init__.py`, `pipelines/prepare.py`, `pipelines/sysrem.py` |

## pipelines

| File | Function | Used by |
|---|---|---|
| `pipelines/__init__.py` | Export spectral masks, preparation operations and model-filtering functions. |  |
| `pipelines/bl19.py` | Apply the BL19 normalization and telluric fits in the order selected by pipelines/prepare.py. Each step passes its corrected data, propagated uncertainties and wavelength selections to the next. | `pipelines/__init__.py`, `pipelines/cheverall26.py`, `pipelines/prepare.py`, `plotting/steps.py` |
| `pipelines/blain24.py` | Apply wavelength-dependent throughput corrections followed by the configured airmass fit. Data and uncertainty arrays are propagated together. Model-processing flags select the operations used for templates. | `pipelines/__init__.py`, `pipelines/prepare.py` |
| `pipelines/carmenes_helium.py` | Normalize direct-helium spectra, construct the stellar reference and align and combine the selected planetary spectra. Configuration fields specify reference membership and arithmetic averaging. | `core/carmenes_helium_simulation.py`, `instruments/carmenes_palle2020.py`, `pipelines/palle2020.py`, `plotting/helium.py`, `retrieval/helium_grid.py`, `scripts/plot_carmenes_palle2020.py` |
| `pipelines/cheverall26.py` | Apply the preparation operations and timing checks used by Cheverall26. The dispatcher selects their order, including normalization, outlier removal and the signal-duration diagnostic. | `core/simulator.py`, `pipelines/prepare.py` |
| `pipelines/czesla2024.py` | Run the CRIRES+ direct-helium analysis: select exposures, divide by the reference spectrum, transform wavelength frames and combine the planetary spectra. Fit and plotting helpers inspect the resulting products. | `core/carmenes_helium_simulation.py`, `core/helium_multi_night.py`, `core/helium_simulation.py`, `core/helium_synthetic_observation.py`, `core/simulator.py`, `instruments/crires_czesla2024.py`, `pipelines/carmenes_helium.py`, `pipelines/prepare.py`, `plotting/helium.py`, `retrieval/helium_grid.py`, `scripts/plot_crires_czesla2024.py` |
| `pipelines/helium_math.py` | Convert wavelength frames and resample direct-helium spectra. These functions use vacuum wavelengths in nm and velocities in km/s, and carry valid-pixel and detector-gap masks through interpolation. | `core/carmenes_helium_simulation.py`, `core/helium_multi_night.py`, `core/helium_simulation.py`, `instruments/carmenes_helium_molecfit.py`, `instruments/carmenes_oh_oot.py`, `instruments/carmenes_palle2020.py`, `instruments/crires_czesla2024.py`, `pipelines/carmenes_helium.py`, `pipelines/czesla2024.py`, `pipelines/helium_noise_propagation.py`, `pipelines/helium_sysrem.py`, `pipelines/palle2020.py`, `plotting/helium.py`, `retrieval/helium_grid.py`, `scripts/plot_carmenes_palle2020.py` |
| `pipelines/helium_noise_propagation.py` | Propagate spectral variances through normalization, reference division and interpolation to calculate their covariance. Fitted continuum and telluric parameters are held fixed in this calculation. | `core/carmenes_helium_simulation.py`, `core/helium_multi_night.py`, `pipelines/czesla2024.py`, `pipelines/palle2020.py`, `retrieval/helium_grid.py` |
| `pipelines/helium_significance.py` | Measure the helium absorption statistic for the configured band, continuum intervals and binning. The selected continuum samples supply the noise estimate. | `config/helium.py`, `core/helium_multi_night.py`, `core/helium_simulation.py` |
| `pipelines/helium_sysrem.py` | Remove Earth-frame components from synthetic helium spectra with SYSREM. The resulting differential residuals enter the subsequent transmission analysis. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `pipelines/masking.py` | Identify excluded/retained spectral columns and repair/check selected data conditions. Read each function to distinguish integer indices, boolean selections and corrected matrices. | `core/simulator.py`, `pipelines/__init__.py`, `pipelines/prepare.py`, `scripts/prepare_crires_night.py` |
| `pipelines/models.py` | Define order/night preparation inputs and exposure information for observed helium spectra. | `core/simulator.py`, `pipelines/czesla2024.py`, `pipelines/palle2020.py`, `pipelines/prepare.py` |
| `pipelines/nortmann26.py` | Apply the Nortmann preparation and model-filtering operations selected by prepare.py. CRIRES+ preparation supplies the wavelength and telluric inputs for these calculations. | `pipelines/prepare.py` |
| `pipelines/palle2020.py` | Prepare observed CARMENES helium spectra: select exposures, apply the configured OH and telluric corrections and construct the transmission spectrum with the shared CARMENES recipe. | `core/simulator.py`, `pipelines/prepare.py`, `scripts/prepare_carmenes_palle2020.py` |
| `pipelines/pca.py` | Remove selected singular components from the exposure/pixel matrix with PCA. The call signature matches the preparation dispatcher. The decomposition uses the spectral matrix directly. | `pipelines/prepare.py` |
| `pipelines/prepare.py` | Choose the preparation branch and return residuals, uncertainties, selections and filtering information. It is shared with configured model-reprocessing paths. | `core/simulator.py`, `pipelines/__init__.py` |
| `pipelines/sysrem.py` | Fit and remove uncertainty-weighted time-wavelength components with SYSREM. Model-filtering functions use the fitted components or projector when evaluating a trial signal against the prepared data. | `analysis/stats.py`, `core/simulator.py`, `pipelines/__init__.py`, `pipelines/helium_sysrem.py`, `pipelines/nortmann26.py`, `pipelines/prepare.py`, `plotting/steps.py` |
| `pipelines/tellurics.py` | Read telluric transmission for synthetic observations and apply the configured fixed-telluric preparation through pipeline_fixedTellurics. | `core/simulator.py`, `pipelines/__init__.py`, `pipelines/prepare.py` |

## planets

| File | Function | Used by |
|---|---|---|
| `planets/__init__.py` | Use load_planet to obtain the system-parameter object consumed by the runner. Configuration selects the observation/analysis separately. The planet object records the physical system and derived quantities used across stages. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py` |
| `planets/catalog.py` | Read planetary-system JSON files into PlanetParameters. Format-specific branches convert input names and units to the fields used by the atmosphere and observing code. | `planets/__init__.py` |
| `planets/models.py` | Store planetary-system parameters and calculate derived orbital quantities. Unit-labelled fields supply the atmosphere and timing code. Compatibility properties convert them to the conventions used by their callers. | `planets/__init__.py`, `planets/catalog.py` |

## plotting

| File | Function | Used by |
|---|---|---|
| `plotting/__init__.py` | Export map, spectral-matrix and calibration plotting functions. | `scripts/run_detectability_maps.py` |
| `plotting/calibration.py` | Plot extracted spectra, telluric corrections and resolution measurements to inspect the instrument calibration. | `plotting/__init__.py` |
| `plotting/carmenes_oh.py` | Plot saved OH-subtraction products and the calibration between fibres A and B. | `pipelines/palle2020.py`, `scripts/plot_carmenes_palle2020.py` |
| `plotting/detectability.py` | Read detection statistics from atmospheric grid points and assemble the detectability figure. The colour scale shows the statistic saved by the simulation sweep. | `plotting/__init__.py` |
| `plotting/helium.py` | Plot prepared helium spectra and their model comparisons. Function arguments specify the wavelength frame and the model used to calculate residuals. | `core/helium_multi_night.py`, `core/helium_simulation.py`, `pipelines/czesla2024.py`, `pipelines/palle2020.py`, `scripts/plot_carmenes_palle2020.py` |
| `plotting/helium_atmosphere.py` | Plot the atmospheric profiles calculated by Sunbather, including the quantities saved with the solved atmosphere. | `atmosphere/helium_sunbather.py` |
| `plotting/helium_grid.py` | Plot the fit and temperature results saved by the helium grid calculation. | `retrieval/helium_grid.py` |
| `plotting/helium_molecfit.py` | Read saved per-exposure molecfit reports and plot the correction and instrumental-kernel diagnostics. Reports distinguish fitted and fixed widths and identify the detector section or nod. | `core/carmenes_helium_simulation.py`, `core/helium_simulation.py`, `pipelines/czesla2024.py`, `pipelines/palle2020.py`, `scripts/plot_carmenes_palle2020.py`, `scripts/plot_crires_czesla2024.py` |
| `plotting/kpvsys.py` | Plot detection maps and velocity profiles using the map orientation and velocity convention supplied to each function. Labels distinguish systemic velocity from the residual Vrest offset. | `core/simulator.py`, `plotting/__init__.py` |
| `plotting/matrices.py` | Display wavelength/exposure and velocity/exposure matrices using the supplied masks, colour scales and axis conventions. | `core/simulator.py`, `plotting/__init__.py` |
| `plotting/models.py` | Define output labels, reference motion and velocity ranges for detection figures. | `core/simulator.py`, `plotting/kpvsys.py`, `plotting/matrices.py` |
| `plotting/steps.py` | Plot the simulated spectra and preparation stages. SYSREM diagnostics reconstruct intermediate matrices from saved components. Display options control how the arrays appear in each panel. | `core/simulator.py`, `pipelines/palle2020.py`, `plotting/helium.py`, `scripts/plot_carmenes_palle2020.py` |

## retrieval

| File | Function | Used by |
|---|---|---|
| `retrieval/__init__.py` | Export likelihoods, priors and sampler interfaces. A likelihood evaluates a supplied model against data. A prior supplies parameter support or a coordinate transform. A sampler explores proposals. Importing these objects is separate from enabling and executing a retrieval. |  |
| `retrieval/emulator.py` | Train and evaluate an experimental emulator of atmospheric spectra. Compare its predicted spectra and retrieval posteriors with the physical model before scientific use. | `core/simulator.py`, `scripts/emulator/compare_emulator.py` |
| `retrieval/helium_grid.py` | Fit a prepared helium spectrum by evaluating atmospheric grid points with its sampling and covariance information. _night_prediction samples each model using the stored observing inputs. | `core/simulator.py` |
| `retrieval/likelihood.py` | Calculate the selected likelihood from the supplied data, model and uncertainty arrays. | `retrieval/__init__.py` |
| `retrieval/likelihood_emulator.py` | Train and evaluate an experimental approximation to the likelihood. Validate the inferred posterior against the physical-model likelihood before scientific use. | `core/simulator.py` |
| `retrieval/models.py` | Retain regional abundances evaluated by the retrieval likelihood and read by its diagnostic plots. | `core/simulator.py` |
| `retrieval/priors.py` | Define parameter distributions, prior transforms and prior-density calculations for the samplers. The ordered parameter names map sampling coordinates to the model arguments. | `retrieval/__init__.py` |
| `retrieval/samplers.py` | Public sampler adapters. MultiNest receives a likelihood and prior transform. Emcee receives a log-posterior. Some runner setup remains inline. | `retrieval/__init__.py` |

## Maintained command-line scripts

Paths here are relative to `scripts/`. Scripts can generate inputs, launch real
calculations, inspect saved outputs, or build teaching illustrations. These are
different jobs. Choose the entry point matching the step you intend to perform.

| Script | Role |
|---|---|
| `carmenes_etc_to_fits.py` | Convert an instrument ETC table to the S/N FITS arrays read by EXoPLORE. Preserve the table's spectral-order and pixel layout when preparing these inputs. |
| `emulator/compare_emulator.py` | Compare emulator spectra with spectra calculated by the atmospheric model. The residuals measure the emulator's spectral approximation errors. |
| `emulator/compare_retrievals.py` | Read emulator and physical-model retrieval outputs and compare their runtimes and posterior distributions. |
| `emulator/generate_training_data.py` | Calculate atmospheric spectra for the emulator training set. The sampling, batches and saved coefficients specify which models enter that set. |
| `emulator/train_emulator.py` | Fit the experimental emulator to a set of calculated spectra. Evaluate its predictions and retrieval results against the atmospheric model before scientific use. |
| `generate_skycalc_inputs.py` | Download telluric transmission spectra from SkyCalc for the simulation's wavelength range, observatory and water-vapour settings. The simulator reads these files when building the observations. |
| `illustrate_doppler_trick.py` | Draw a schematic of stellar, telluric and moving planetary lines. The script constructs illustrative spectra to explain Doppler separation. |
| `illustrate_likelihood_weighting.py` | Compare likelihood weighting choices on an illustrative spectrum with known uncertainties. |
| `illustrate_model_spectrum.py` | Calculate planetary transmission spectra and use them to draw the model-spectrum and chemistry figures in the documentation. |
| `illustrate_significance_sampling.py` | Calculate a statistical example showing how correlated velocity samples affect detection-significance estimates. |
| `make_planet_params.py` | Calculate derived planetary-system quantities and write the parameter JSON read by the simulator. Set the physical inputs and their units in the user-input section. |
| `make_wave_carmenes_nir.py` | Average time-dependent CARACAL wavelength solutions and save the order/pixel grid used as the CARMENES NIR reference. |
| `plot_carmenes_palle2020.py` | Read saved CARMENES helium correction and transmission products and draw their diagnostic figures. |
| `plot_corner_overlay.py` | Read posterior samples from the selected runs and overlay their marginal and joint distributions. The labels identify each run's parameters and priors. |
| `plot_crires_czesla2024.py` | Read saved CRIRES+ telluric fits and helium preparation products and display their spectra, residuals and diagnostics. Display options control the panel scales. |
| `plot_multinight_stacking.py` | Read per-night and combined detection maps and plot velocity profiles at each map's selected Kp. |
| `plot_significance_study.py` | Read detection statistics from repeated noise realizations and plot their distributions. The selected table and metric determine which statistic is shown. |
| `prepare_carmenes_night.py` | Convert extracted CARACAL observations to reference-night arrays for the instrument reader. Match the wavelength, flux and uncertainty arrays by exposure and spectral order. |
| `prepare_carmenes_palle2020.py` | Prepare observed CARMENES helium spectra. Preview the inputs and fit settings, or run the configured external telluric fits with --run. |
| `prepare_crires_czesla2024.py` | Prepare telluric corrections for the selected CRIRES+ helium segment from extracted spectra and an exposure manifest. The pilot option fits one selected exposure. |
| `prepare_crires_night.py` | Read extracted CRIRES+ exposures and prepare the wavelength, telluric and reference-night inputs used by EXoPLORE. |
| `prepare_igrins_night.py` | Convert reduced IGRINS PLP H/K spectra to reference-night inputs. Keep exposure times and the wavelength, flux and uncertainty arrays aligned. |
| `reduce_crires_night.py` | Run the ESO cr2res recipes on raw CRIRES+ exposures and calibration frames. Pass the extracted spectra to prepare_crires_night.py for EXoPLORE input preparation. |
| `run_detectability_maps.py` | Run a simulation at each atmospheric grid point, collect its detection statistic and plot the detectability map. |
| `run_exoplore.py` | Load a simulation JSON through the shared command-line interface. The default action prints a preview. Use --run to execute the configured experiment. |
| `verify_clean_run.py` | Read saved run products and check them against the settings and assumptions defined in this script. |
