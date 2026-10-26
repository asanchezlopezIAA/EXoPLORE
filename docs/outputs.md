# EXoPLORE Outputs Reference

This page describes the molecular simulation products written by
`core/simulator.py`. The [CRIRES+](crires_reduction.md),
[IGRINS](igrins_pipeline.md) and [helium tutorials](tutorial.md#tutorial-12-he-i-transmission-spectroscopy-with-crires-wasp-121-b)
describe their additional reduction and direct-transmission products.

## Output directory structure

`paths.output_root` selects the output location. A molecular run creates:

```text
<output_root>/<planet>/<instrument>/<event>/<run_name>/
  matrices/          spectral arrays, masks, velocity grids and detection maps
  plots/             diagnostic figures and retrieval corner plots
  correlations/      directory available for correlation products
  warnings/          fully masked order reports
  inputs/            directory available for input metadata
```

The runner writes its molecular CCF and map archives into `matrices/`.
The presence of `inputs/` alone does not establish that configuration snapshots
were saved. Keep the configuration used for your run. The notebook writes its
own resolved configuration and provenance record.

The run name includes the pipeline, signal and noise choices, number of nights,
statistic and stacking mode. For example:

```text
BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1
```

Several products use this name as their suffix. Per-night and combined maps can
have additional suffixes. Inspect the files in your run directory before
selecting an archive for comparison. Reusing a run directory can replace
products, and existing timing files can be retained. Use a fresh output root
for a changed experiment.

## Timing arrays

These files are in `matrices/`:

| File | Contents |
|---|---|
| `syn_jd.fits` | Exposure BJD timestamps written for generated observing sequences. |
| `phase.fits` | Orbital phases for a common observing sequence. |
| `phase.npz` | Separate arrays named `phase_0`, `phase_1`, etc., for different observing sequences. |

For the common sequence, phase is `(BJD - T0) / Period`, with zero at
mid-transit. `with_signal` and `without_signal` are exposure selections used
by the runner. Their construction depends on the event and observing geometry.
The public CCF result includes the selected signal exposures.

Airmass is calculated or read during observation setup. Reference-night
`airmass_0.fits` files and per-night SkyCalc `airmass.fits` files are
[input products](input_files.md). The molecular output writer does not
unconditionally save an `airmass.fits` file in each run.

## Spectral matrix files

The per-order files below are compressed NumPy archives in `matrices/`.
Each stores its array under key `a`:

```python
import numpy as np

# Replace this path with a per-order file from your completed run.
with np.load("path/to/mat_res_order_0_RUN_NAME.npz") as archive:
    residuals = archive["a"].copy()
print(residuals.shape)
```

`K` in the filenames is the instrument order ID, selected by
`instrument.order_indices`. Residuals and propagated uncertainties ordinarily have axes
`(night, exposure, wavelength pixel)`. The shifted `mat_back` template has
axes `(exposure, wavelength pixel)` for the common observing sequence.
Extra injection or iteration axes can appear in optimization modes. Different-night runs can include padding for
unequal exposure counts.

| File | Contents | Save flag |
|---|---|---|
| `mat_res_order_{K}_{run}.npz` | Prepared residual spectra. | `save_mat_res` |
| `propag_noise_order_{K}_{run}.npz` | Propagated uncertainties used by the analysis. | `save_propag_noise` |
| `mat_back_order_{K}_{run}.npz` | Velocity-shifted CCF template retained for model processing. | `save_mat_back` |
| `mat_noise_order_{K}_{run}.npz` | Noise-control matrix. | `save_mat_noise` |
| `std_noise_order_{K}_{run}.npz` | Noise estimates. | `save_std_noise` |
| `mat_cc_order_{K}_{run}.npz` | Prepared template matrix, when supplied by the method. | `save_mat_cc` |
| `ccf_store_order_{K}_{run}.npz` | Per-order CCF, ordinarily `(night, velocity, exposure)`. | `save_ccf_store` |

The `output` block controls these optional files. It leaves the in-memory
calculation and diagnostic plotting active:

```json
"output": {
  "save_mat_res": true,
  "save_mat_back": true,
  "save_ccf_store": true,
  "save_propag_noise": true,
  "save_U_sysrem": true,
  "save_mat_cc": false,
  "save_mat_noise": false,
  "save_std_noise": false
}
```

These are the defaults. The previously measured full 76-order example used
approximately 334 MB for residuals, 298 MB for background spectra, 141 MB for
per-order CCFs and 416 MB for propagated uncertainties. These are estimates
for that experiment. Exposure counts, masks and compression change the sizes.
Enabling every optional matrix used approximately 2.2 GB in that measurement.

Setting all flags to `false` reduces saved products. Timing arrays, masks,
velocity grids, selected detection products and figures are written separately.
Retrieval reads saved residuals, background spectra and propagated uncertainties,
and its model filtering can require SYSREM information. Keep those products
when enabling retrieval.

### Global mask arrays

The following archives use key `a`, ordinarily with shape
`(night, selected order, wavelength pixel)`:

| File | Meaning of `True` |
|---|---|
| `mask_{run}.npz` | Excluded channel in the main preparation mask. |
| `useful_spectral_points_{run}.npz` | Retained channel. |
| `mask_snr_{run}.npz` | Channel excluded by the S/N selection. |
| `useful_spectral_points_snr_{run}.npz` | Channel retained by the S/N selection. |
| `mask_inter_{run}.npz` | Channel excluded by the intersection mask used for model processing. |
| `useful_spectral_points_inter_{run}.npz` | Channel retained by that intersection mask. |

The order axis follows the selected-order list. Use the saved mask to identify
excluded channels numerically. Display colours depend on the plotting function.

### Supporting arrays and grids

| File | Contents |
|---|---|
| `U_sysrem_{run}.npz` | Fitted temporal components, saved when allocated and `save_U_sysrem` is enabled. |
| `mat_star_forfile_{run}.npz` | Stellar matrix retained by the runner, when available. |
| `v_ccf_{run}.npz` | Earth-frame CCF velocity grid, in km/s. |
| `kp_range_{run}.npz` | Trial orbital velocity amplitudes, in km/s. Read the array for its bounds and spacing. |

## Diagnostic plots (`plots/`)

| File pattern | Contents |
|---|---|
| `pipeline_steps_{run}.pdf` | Simulation and preparation diagnostic for the selected illustrative order. |
| `sn_map_*.pdf` | Detection map with the configured reference motion. |
| `1D_CCF_*.pdf` | Velocity profile through the selected Kp row. |
| `CC_ERF_*.pdf` | Time–velocity CCF in the Earth rest frame. |
| `retrieval*_corner.pdf` | Retrieval posterior figure when produced by the selected sampler route. |

### Pipeline-steps figure

`plotting.pipeline_steps_order` selects the displayed instrument order. An unset
value selects the first processed order. `plotting.pipeline_steps_xlim_um`
selects a wavelength window in the top spectrum panel.

The standard molecular figure has four panels: representative noiseless and
noisy spectra, a noiseless time series, a noisy time series and prepared
residuals. SYSREM methods can show a sequence of intermediate iterations,
selected by `plotting.pipeline_steps_sysrem_iterations`. The notebook's helper
also provides its own preparation preview.

Rows in the image panels are exposures and columns are wavelength channels.
Excluded channels are displayed as white gaps. Red timing markers identify
the selected event exposures. Image panels use separate display scales.
The actual noisy spectrum keeps its simulated amplitude. If
`plotting.pipeline_steps_noise_illustration_factor` is set, an additional
labelled curve magnifies the deviation for illustration only.

## CCF detection products

For the ordinary two-dimensional S/N analysis, use
`ccf_tot_sn_map_*.npz`. This archive contains named arrays:

| Key | Shape or meaning |
|---|---|
| `ccf_tot_sn` | Normalised map, axes `(Vrest, Kp)`. |
| `v_rest` | Residual velocity offsets in km/s. |
| `kp_range` | Trial Kp values in km/s. |
| `order_indices` | Selected instrument order IDs. |
| `v_rest_convention` | Convention recorded by the writer. |
| `sysrem_opt` | Whether extra optimization axes are present. |
| `n_components_per_order` | Detrending component counts when available. |

Welch analysis uses `ccf_tot_welch_map_*.npz` with map key
`ccf_tot_welch`. The reader requires selecting the corresponding metric.
Map archives for optimization modes can have additional axes.

Repeated-night statistical studies also write `stats_*.npz`,
`stats_planet_pos_*.npz`, `stats_planet_area_*.npz` and
`ccf_tot_sn_stat_*.npz`. These hold study distributions and per-night statistics.
Their arrays depend on the study mode. Use the named map archive above for
plotting an ordinary two-dimensional detection map.

### Read a map and its strongest pixel

```python
from pathlib import Path
from exoplore.io.results import load_detection_map

# Point to one specific map from your run's matrices directory.
map_path = Path("path/to/ccf_tot_sn_map_RUN_NAME.npz")
detection = load_detection_map(map_path)
print(f"Peak map S/N = {detection.peak_value:.2f}")
print(f"Kp = {detection.peak_kp_kms:.1f} km/s")
print(f"Vrest = {detection.peak_v_rest_kms:.1f} km/s")
```

The map already contains its axes. Its Vrest grid differs from the Earth-frame
velocity grid used to calculate exposure CCFs.

### Plot the map and a velocity slice

```python
import matplotlib.pyplot as plt

# Transpose (Vrest, Kp) to put velocity on the horizontal plot axis.
fig, ax = plt.subplots()
image = ax.pcolormesh(detection.v_rest_kms, detection.kp_kms,
                      detection.values.T, shading="auto", cmap="RdBu_r")
fig.colorbar(image, ax=ax, label="Map S/N")
ax.set_xlabel("Vrest (km/s)")
ax.set_ylabel("Kp (km/s)")
plt.show()

# Select the Kp column containing the strongest map pixel.
velocity_index, kp_index = detection.peak_indices
fig, ax = plt.subplots()
ax.plot(detection.v_rest_kms, detection.values[:, kp_index])
ax.set_xlabel("Vrest (km/s)")
ax.set_ylabel("Map S/N")
plt.show()
```

### Read and plot the combined CCF

After a successful molecular run, `simulator.get_combined_ccf()` exposes the
combined CCF and its velocity, phase and event selections. Save it if you want
to inspect it later:

```python
from exoplore.io.results import save_ccf_result

# Use the simulator that has just completed your experiment.
ccf = simulator.get_combined_ccf()
save_ccf_result(ccf, "combined_ccf.npz")
```

The save function protects an existing file unless `overwrite=True` is passed.
For different observing sequences, select `night_index` as explained in the
[architecture guide](architecture.md#read-completed-results).

```python
from exoplore.io.results import load_ccf_result

# Read the compact export, including its velocity conventions.
ccf = load_ccf_result("combined_ccf.npz")
fig, ax = plt.subplots()
ax.pcolormesh(ccf.velocity_kms, ccf.phase, ccf.ccf.T,
              shading="auto", cmap="RdBu_r")
ax.set_xlabel("Earth-frame velocity (km/s)")
ax.set_ylabel("Orbital phase")
plt.show()
```

## Understanding the Kp–Vrest map

The alignment calculation samples exposure CCFs along each trial orbital track
and sums the selected exposures. A compatible moving line pattern adds
coherently near its orbital solution. Noise, other spectral structure and
processing effects also contribute to the map.

`Vrest` is an offset relative to the expected planetary track, after accounting
for the systemic and barycentric velocities with the analysis convention.
Regional atmospheric velocities and their changing contributions can shift or
broaden a peak. In observations, model and ephemeris uncertainties also matter.

For the default peak-centred S/N normalisation, each Kp column in the stored
array is centred by subtracting the off-peak mean and divided by the off-peak
standard deviation. The excluded velocity interval is set by
`cross_correlation.snr_exclude_kms`. `snr_noise_source` can select signal-free
Kp rows for the noise estimate. Cheverall26 uses point-centred exclusions.
See the [configuration reference](config_reference.md#9-cross_correlation)
for these choices.

Report the maximum as a **peak map S/N under the adopted normalisation**.
A false-alarm significance requires calibration against a suitable null
distribution. The [validation tutorial](tutorial.md#tutorial-8-detection-significance-and-statistical-validation)
explains the significance tests. Atmospheric abundances require inference with
an atmospheric model and likelihood.

## Warnings

`warnings/fully_masked_orders_{run}.txt` lists fully excluded orders when found.
Read the order ID, night and wavelength information alongside the saved masks
and run log. Entirely excluded spectra contribute no usable CCF information.

## Retrieval outputs

Molecular retrieval products are written directly in the run's `matrices/`
directory. A single-night MultiNest run normally uses the prefix
`{retrieval_name}_night_0_`, where `retrieval_name` defaults to `retrieval`.
Other dimensions and time intervals can add suffixes.

| File pattern | Contents |
|---|---|
| `*_post_equal_weights.dat` | Equal-weight parameter samples followed by log-likelihood. |
| `*_.txt` | Weighted posterior, with sample weight, minus twice the log-likelihood, then parameter columns. |
| `*_stats.dat` / `*_stats.json` | Sampler summaries when written by that route. |
| `*_summary.txt` | MultiNest summary. |

Parameter columns follow the selected retrieval dimensionality. Read their
order in the [retrieval configuration reference](config_reference.md#10-retrieval)
and the chosen run. For example, the first parameter of a standard molecular
VMR retrieval is log abundance. A C/O retrieval uses different parameters.

```python
import numpy as np

# Select the equal-weight samples for the retrieval you ran.
samples = np.loadtxt("path/to/retrieval_night_0_post_equal_weights.dat", ndmin=2)
parameters = samples[:, :-1]
log_likelihood = samples[:, -1]
print(parameters.shape, log_likelihood.shape)
```

The sampler adapters and the runner's retrieval block determine which files are
written. Emcee products have their own chains and summary outputs.

## Log file

Capture the command-line output while retaining progress in the terminal:

```bash
python -u scripts/run_exoplore.py configs/hd189733b_andes_transit_clean.json --run 2>&1 | tee run.log
```

The `-u` option makes progress available without Python's normal stdout buffering.
