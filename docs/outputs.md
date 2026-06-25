# EXoPLORE Outputs Reference

In the following we describe every file EXoPLORE writes, where to find it, and how to read and interpret it.

---

## Output directory structure

All outputs are written under `paths.output_root` in the simulation config. Everything for a single simulation run lives inside one folder named after that run:

```
<output_root>/
  <PlanetName>/
    <InstrumentName>/
      <event_type>/
        <run_name>/
          matrices/          ← spectral matrices, masks, CCF grids, detection maps
          plots/             ← PDF diagnostic figures
          correlations/      ← CCF combined products, Kp-Vsys maps
          warnings/          ← masked-order reports and flagged spectra
          inputs/            ← parameter snapshots for reproducibility
```

The `run_name` (also called `Simulation_name`) is derived from the simulation settings. For the default HD 189733 b config it is:

```
BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1
```

The pattern is: `{pipeline}_{signal_flag}_{n_nights}nights_{signif_flag}_{stack_flag}_{real_flag}_{noise_flag}_stdnoisex{noise_scaling}`.

For the full example config the complete path would be:

```
<output_root>/HD189733b/ANDES_YJHK/transit/BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/
```

---

## Timing arrays

Located in `<run_name>/matrices/`.

### `syn_jd.fits`

Shape: `(n_spectra,)`, synthetic BJD timestamps for each exposure, computed from the reference T₀ and the observing cadence.

### `phase.fits` (or `phase.npz`)

Shape: `(n_spectra,)`, orbital phase for each exposure.

Phase = (BJD - T₀) / Period. Values near 0 are at transit centre. Ingress and egress occur at ±T₁₄ / (2 × Period).

```python
from astropy.io import fits
import numpy as np

run    = 'BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1'
base   = f'/your/output_root/HD189733b/ANDES_YJHK/transit/{run}/matrices/'

phase     = fits.open(base + 'phase.fits')[0].data
in_transit = np.where(np.abs(phase) < 0.01)[0]
```

### `airmass.fits`

Shape: `(n_spectra,)`, airmass at each exposure.

---

## Spectral matrix files

Located in `<run_name>/matrices/`. All files are NumPy compressed `.npz` archives. Every file stores its single array under the key `'a'`:

```python
import numpy as np
data = np.load('path/to/file.npz')['a']
```

### Per-order matrices (one file per spectral order)

For each order with absolute index `K` (from the instrument's `order_selection` array):

| File | Shape | Description |
|---|---|---|
| `mat_res_order_{K}_{run_name}.npz` | `(n_spectra, n_pixels)` | Residual spectral matrix after pipeline processing. This is the primary input to the CCF. |
| `propag_noise_order_{K}_{run_name}.npz` | `(n_spectra, n_pixels)` | Propagated noise (uncertainty) per pixel per exposure. Used as weights in the inverse-variance weighted CCF. |
| `std_noise_order_{K}_{run_name}.npz` | `(n_spectra, n_pixels)` | Standard noise estimate (σ). |
| `mat_back_order_{K}_{run_name}.npz` | `(n_spectra, n_pixels)` | Background stellar matrix (the modelled stellar spectrum after pipeline processing, without planet signal). |
| `mat_noise_order_{K}_{run_name}.npz` | `(n_spectra, n_pixels)` | Noise-only matrix (random noise component). |
| `mat_cc_order_{K}_{run_name}.npz` | `(n_spectra, n_pixels)` | CCF-ready matrix (BL19/BLASP24 pipelines only). The matrix as passed to the CCF kernel. |
| `ccf_store_order_{K}_{run_name}.npz` | `(n_nights, n_vel, n_spectra)` | Per-order CCF: CCF value at each velocity lag and each exposure, for all nights. |

### Global mask arrays

| File | Shape | Description |
|---|---|---|
| `mask_{run_name}.npz` | `(n_nights, n_orders, n_pixels)` | Combined telluric + SNR mask. `True` = masked (bad pixel), `False` = good. |
| `useful_spectral_points_{run_name}.npz` | `(n_nights, n_orders, n_pixels)` | Boolean array of usable pixels after the main mask. Complement of `mask`. |
| `mask_snr_{run_name}.npz` | `(n_nights, n_orders, n_pixels)` | SNR-only mask (pixels below `pipeline.snr_mask_threshold`). |
| `useful_spectral_points_snr_{run_name}.npz` | `(n_nights, n_orders, n_pixels)` | Usable pixels after SNR masking only. |
| `mask_inter_{run_name}.npz` | `(n_nights, n_orders, n_pixels)` | Combined mask used for retrieval. |
| `useful_spectral_points_inter_{run_name}.npz` | `(n_nights, n_orders, n_pixels)` | Usable pixels for retrieval mask. |

```python
run  = 'BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1'
base = f'/your/output_root/HD189733b/ANDES_YJHK/transit/{run}/matrices/'

mask = np.load(base + f'mask_{run}.npz')['a']
# shape: (n_nights, n_orders, n_pixels)
# True = masked (bad), False = good
masked_fraction = mask.mean()
print(f"Masked fraction: {masked_fraction*100:.1f}%")
```

### Supporting matrices

| File | Shape | Description |
|---|---|---|
| `U_sysrem_{run_name}.npz` | `(n_nights, n_spectra, sysrem_its)` | SYSREM basis vectors. Only written when pipeline is `ASL19` or `Gibson22`. |

---

## CCF grids

| File | Shape | Description |
|---|---|---|
| `v_ccf_{run_name}.npz` | `(n_vel,)` | CCF velocity grid in km/s, from -`velocity_max_kms` to +`velocity_max_kms` in steps of `velocity_step_kms`. |
| `kp_range_{run_name}.npz` | `(n_kp,)` | Kp grid for the Kp-Vsys map, from 0 to `kp_max_kms` km/s. |

```python
run  = 'BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1'
base = f'/your/output_root/HD189733b/ANDES_YJHK/transit/{run}/matrices/'

v_ccf    = np.load(base + f'v_ccf_{run}.npz')['a']
kp_range = np.load(base + f'kp_range_{run}.npz')['a']
# e.g. v_ccf shape (651,) for v_max=325 km/s, step=1 km/s
```

---

## Diagnostic plots (`plots/`)

Every run saves PDF figures to `<run_name>/plots/`. The key files are:

| File | Description |
|---|---|
| `pipeline_steps_<run_name>.pdf` | **Four-panel pipeline-steps figure** (see below). Generated for the first processed order on every run. |
| `sn_map_<run_name>_SNR.pdf` | Kp-Vsys S/N map in colour with contours. |
| `1D_CCF_<run_name>.pdf` | 1-D CCF at the best-fit Kp vs rest-frame velocity. |
| `CC_ERF_<run_name>.pdf` | CCF trail as a function of orbital phase (the "butterfly" plot). |
| `retrieval_night_0_corner.pdf` | Posterior corner plot from the Bayesian retrieval (if `retrieval.enabled: true`). |

### Pipeline-steps figure

`pipeline_steps_<run_name>.pdf` shows the successive data-simulation and preparation steps for the first processed order, mirroring Fig. 5 of Sánchez-López et al. (2022). In particular, the four panels illustrate:

- **Panel A**, 1-D spectrum at mid-transit: noiseless model (black) and noisy realisation (red). The wavelength window is controlled by `plotting.pipeline_steps_xlim_um` in the config (default 1.4862 to 1.4890 µm, a water-band window).
- **Panel B**, 2-D noiseless spectral matrix (phase × wavelength).
- **Panel C**, 2-D noisy matrix including throughput variations.
- **Panel D**, 2-D residual matrix after `preparing_pipeline`. Masked pixels are white; ingress and egress are marked with white dashed lines.

We note that this figure allows us to verify that the pipeline correctly suppresses stellar and telluric systematics while preserving the planet signal trail in Panel D.

---

## CCF detection products

These are produced by Block 7 (statistical analysis) and saved into `<run_name>/matrices/`.

| File | Shape | Description |
|---|---|---|
| `stats_{run_name}.npz` | `(n_kp, n_vel)` | The Kp-Vsys cross-correlation map (raw CCF values after co-adding the CCF trail at each Kp). This is the primary detection map. |
| `ccf_tot_sn_stat_{run_name}.npz` | `(n_kp, n_vel)` | S/N version of the Kp-Vsys map: `stats / out-of-trail RMS`. The peak value is the detection significance. |
| `v_rest_{run_name}.npz` | `(n_vel,)` | 1D collapsed CCF at the best Kp (CCF_SNR mode). |
| `stats_planet_pos_{run_name}.npz` | scalar | Planet position statistic (single-number detection metric). |
| `stats_planet_area_{run_name}.npz` | scalar | Planet area statistic (integral of the CCF peak). |
| `kp_range_{run_name}.npz` | `(n_kp,)` | Kp grid (also written by Block 8, both copies are identical). |

For the Welch t-test mode (`cross_correlation.welch_ttest: true`):

| File | Description |
|---|---|
| `v_rest_ttest_{run_name}.npz` | 1D collapsed t-test CCF |
| `stats_tvalue_{run_name}.npz` | Welch t-values |
| `stats_pvalue_{run_name}.npz` | Welch p-values |

---

## How to make a Kp-Vsys plot

```python
import numpy as np
import matplotlib.pyplot as plt

# Adjust these to your output_root and run_name
run  = 'BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1'
base = f'/your/output_root/HD189733b/ANDES_YJHK/transit/{run}/matrices/'

v_ccf    = np.load(base + f'v_ccf_{run}.npz')['a']
kp_range = np.load(base + f'kp_range_{run}.npz')['a']
snr_map  = np.load(base + f'ccf_tot_sn_stat_{run}.npz')['a']

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(v_ccf, kp_range, snr_map, cmap='RdBu_r',
                    vmin=-5, vmax=snr_map.max())
plt.colorbar(img, ax=ax, label='S/N')
ax.set_xlabel('$v_{rest}$ (km/s)')
ax.set_ylabel('$K_p$ (km/s)')
ax.set_title('HD 189733 b, ANDES 1 transit')

# Mark the expected position (no wind injected → peak at v_rest = 0; literature Kp = 152.5 km/s)
ax.axvline(0, color='white', ls='--', lw=1, label='Expected $v_{rest}$ (no wind)')
ax.axhline(152.5, color='white', ls=':',  lw=1, label='Literature $K_p$')
ax.legend()
plt.tight_layout()
plt.savefig('kpvsys.png', dpi=150)
```

---

## How to read the peak S/N

```python
run  = 'BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1'
base = f'/your/output_root/HD189733b/ANDES_YJHK/transit/{run}/matrices/'

snr_map  = np.load(base + f'ccf_tot_sn_stat_{run}.npz')['a']
kp_range = np.load(base + f'kp_range_{run}.npz')['a']
v_ccf    = np.load(base + f'v_ccf_{run}.npz')['a']

peak_snr = snr_map.max()
peak_kp_idx, peak_vsys_idx = np.unravel_index(snr_map.argmax(), snr_map.shape)
peak_kp   = kp_range[peak_kp_idx]
peak_vsys = v_ccf[peak_vsys_idx]
print(f"Peak S/N = {peak_snr:.1f} at Kp = {peak_kp:.0f} km/s, Vsys = {peak_vsys:.1f} km/s")
```

---

## How to read a 1D CCF at the best Kp

```python
kp_best = 152.5   # km/s, literature Kp for HD 189733 b (Triaud et al. 2009)
kp_idx  = np.argmin(np.abs(kp_range - kp_best))

ccf_1d = snr_map[kp_idx, :]

fig, ax = plt.subplots()
ax.plot(v_ccf, ccf_1d)
ax.axvline(0, color='r', ls='--', label='Expected $v_{rest}$ (no wind)')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('S/N')
ax.set_title(f'1D CCF at Kp = {kp_best:.0f} km/s')
ax.legend()
plt.tight_layout()
```

---

## How to plot the CCF trail (time-series)

The per-order CCF trail shows the CCF as a function of time (exposure) and velocity. The planet signal appears as a diagonal streak, Doppler-shifted by `Kp × sin(2π × phase)` at each exposure.

```python
from astropy.io import fits

run  = 'BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1'
base = f'/your/output_root/HD189733b/ANDES_YJHK/transit/{run}/matrices/'

# Load per-order CCF for order K
K         = 10    # absolute order index
ccf_trail = np.load(base + f'ccf_store_order_{K}_{run}.npz')['a']
# shape: (n_nights, n_vel, n_spectra)
# transpose to (n_spectra, n_vel) for a single night:
trail_2d = ccf_trail[0].T   # shape: (n_spectra, n_vel)

phase = fits.open(base + 'phase.fits')[0].data

fig, ax = plt.subplots(figsize=(8, 5))
ax.pcolormesh(v_ccf, phase, trail_2d, cmap='RdBu_r')
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Orbital phase')
ax.set_title(f'CCF trail, order {K}')
plt.tight_layout()
```

---

## Understanding the Kp-Vsys map

The Kp-Vsys map is built by shifting the time-series CCF trail at each orbital phase by the expected planet velocity `Kp × sin(2π × phase)`, then co-adding. At the true `(Kp, Vsys)`, the planet signal adds coherently; everywhere else it averages to zero. This allows us to simultaneously constrain the orbital velocity semi-amplitude and the systemic rest-frame velocity of the planet.

**What to look for:**

- A clear peak at the expected Kp and Vsys, this constitutes a detection.
- The expected Kp for a circular orbit is `Kp = (2π/P) × a × sin(i)` (listed in the planet parameter file as `kp_kms`).
- The expected position on the v_rest axis is 0 km/s for a planet with no atmospheric wind. The barycentric and systemic velocity corrections are already removed from the data before the CCF is computed, so v_rest encodes only residual atmospheric motion (winds). Do not confuse this axis with the stellar systemic velocity (`systemic_velocity_kms`), which is a separate parameter already accounted for.
- If `atmosphere.planet_model.wind_velocity_kms ≠ 0`, the detected peak will be offset from v_rest = 0 by the injected wind velocity.

**S/N definition:**

```
S/N = CCF_peak / std(CCF_noise_region)
```

The noise region is the part of the Kp-Vsys map more than `cross_correlation.noise_velocity_max_kms` km/s from Vsys = 0 and more than `cross_correlation.snr_exclude_kms` km/s from the identified peak.

**Interpretation guide:**

| S/N | Interpretation |
|---|---|
| < 3 | No detection |
| 3 to 5 | Marginal, requires confirmation |
| 5 to 8 | Tentative detection |
| > 8 | Firm detection |
| > 15 | High-confidence detection (typical for ANDES 1-night simulations of bright targets) |

---

## Warnings

### `<run_name>/warnings/fully_masked_orders_<run_name>.txt`

Written when one or more spectral orders are entirely masked (all pixels fail SNR + negative-value masking across all nights). This occurs in particular in deep telluric bands (e.g., around 1.4 µm and 1.9 µm for ANDES YJHK). Content:

```
EXoPLORE, fully masked orders report
Simulation: BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1
N_fully_masked: 3
order_h   order_K   night_b   wave_centre_um
   5       7         0          1.3943
  ...
```

These orders are automatically excluded from CCF co-adding and retrieval. Their NaN-filled rows in `mat_res_order_{K}` confirm they were skipped.

---

## Retrieval outputs

When retrieval is enabled, posterior files are written into `<run_name>/matrices/matrices_<run_name>/` (a sub-directory created by the retrieval block at runtime, named after the simulation to avoid overwriting forward-model matrices).

The file naming follows: `<retrieval_name>_night_<N>_<suffix>`, where `retrieval_name` defaults to `"retrieval"`.

**MultiNest raw files** are written to the same directory with the prefix `<retrieval_name>_night_<N>_`:

| File pattern | Contents |
|---|---|
| `*_.txt` | Full posterior chain (MultiNest format: log-evidence, parameters, weights) |
| `*_summary.txt` | Log-evidence Z, parameter means and standard deviations |
| `*_stats.dat` | Nested sampling statistics |

**Reading posteriors:**

```python
import numpy as np

run  = 'BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1'
base = f'/your/output_root/HD189733b/ANDES_YJHK/transit/{run}/matrices/matrices_{run}/'

# Load MultiNest posterior chain (columns: log-evidence, parameters..., weight)
chain = np.loadtxt(base + 'retrieval_night_0_.txt')
weights   = chain[:, 0]         # sample weights (from nested sampling)
co_samples = chain[:, 2]        # C/O posterior (column depends on dimensionality)
met_samples = chain[:, 3]       # metallicity posterior

# Weighted mean and std
co_mean  = np.average(co_samples,  weights=weights)
co_std   = np.sqrt(np.average((co_samples - co_mean)**2, weights=weights))
print(f"C/O = {co_mean:.2f} ± {co_std:.2f}")
```

**Corner plot** (auto-generated when `corner` is installed):
```
<run_name>/plots/retrieval_night_0_corner.pdf
```

---

## Quick reference: all output files

All paths below are relative to `<output_root>/<planet>/<instrument>/<event>/`.

| Location | File | Shape | Key | Contents |
|---|---|---|---|---|
| `{run}/matrices/` | `syn_jd.fits` | `(n_spec,)` | FITS | BJD timestamps |
| `{run}/matrices/` | `phase.fits` | `(n_spec,)` | FITS | Orbital phase |
| `{run}/matrices/` | `airmass.fits` | `(n_spec,)` | FITS | Airmass per exposure |
| `{run}/matrices/` | `mat_res_order_{K}_{run}.npz` | `(n_spec, n_pix)` | `'a'` | Residual spectrum, order K |
| `{run}/matrices/` | `propag_noise_order_{K}_{run}.npz` | `(n_spec, n_pix)` | `'a'` | Propagated noise |
| `{run}/matrices/` | `std_noise_order_{K}_{run}.npz` | `(n_spec, n_pix)` | `'a'` | Standard noise |
| `{run}/matrices/` | `mat_back_order_{K}_{run}.npz` | `(n_spec, n_pix)` | `'a'` | Background stellar matrix |
| `{run}/matrices/` | `mat_noise_order_{K}_{run}.npz` | `(n_spec, n_pix)` | `'a'` | Noise matrix |
| `{run}/matrices/` | `mat_cc_order_{K}_{run}.npz` | `(n_spec, n_pix)` | `'a'` | CCF-ready matrix |
| `{run}/matrices/` | `ccf_store_order_{K}_{run}.npz` | `(n_nights, n_vel, n_spec)` | `'a'` | Per-order CCF trail |
| `{run}/matrices/` | `mask_{run}.npz` | `(n_nights, n_orders, n_pix)` | `'a'` | Combined mask |
| `{run}/matrices/` | `mask_snr_{run}.npz` | `(n_nights, n_orders, n_pix)` | `'a'` | SNR mask |
| `{run}/matrices/` | `mask_inter_{run}.npz` | `(n_nights, n_orders, n_pix)` | `'a'` | Retrieval mask |
| `{run}/matrices/` | `v_ccf_{run}.npz` | `(n_vel,)` | `'a'` | CCF velocity axis (km/s) |
| `{run}/matrices/` | `kp_range_{run}.npz` | `(n_kp,)` | `'a'` | Kp axis (km/s) |
| `{run}/matrices/` | `stats_{run}.npz` | `(n_kp, n_vel)` | `'a'` | Kp-Vsys CCF map |
| `{run}/matrices/` | `ccf_tot_sn_stat_{run}.npz` | `(n_kp, n_vel)` | `'a'` | Kp-Vsys S/N map |
| `{run}/matrices/` | `v_rest_{run}.npz` | `(n_vel,)` | `'a'` | 1D collapsed CCF |
| `{run}/warnings/` | `fully_masked_orders_{run}.txt` | text |, | Masked-order report |
| `{run}/plots/` | `*.pdf` | PDF |, | Diagnostic figures |

---

## Log file

The simulator prints progress to stdout. In order to capture a persistent log file without losing real-time terminal output:

```bash
python -u scripts/run_exoplore.py configs/my_sim.json --run 2>&1 | tee run.log
```

The `-u` flag unbuffers Python's stdout so output appears immediately in the terminal.
