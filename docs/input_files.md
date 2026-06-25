# EXoPLORE Input Files Reference

This document lists every external file the simulator needs, where to obtain it, and where to place it.

---

## Overview

Input files fall into four categories:

1. **Instrument files**, wavelength grids and SNR tables specific to the spectrograph
2. **Observation reference files**, real data from a reference night (JD, airmass, signal, SNR arrays)
3. **Stellar atmosphere models**, PHOENIX spectra for the target star
4. **Telluric reference spectra**, SkyCalc-generated atmospheric transmission

petitRADTRANS opacity tables are a fifth category handled separately (see the [Installation guide](installation.md)).

---

## 1. Instrument files

### ANDES (ELT)

For simulated observations (`observation.specific_event: false`, the default) ANDES does **not** need reference-night data. The simulator generates its own synthetic time grid and derives the noise directly from the ETC SNR tables. Only the ETC file is required.

| File | Config key | Description |
|---|---|---|
| `ANDES_ETC_WAVE_SNR_YJHK_<target>.fits` | `paths.inputs_dir` | Combined YJH + K multi-extension FITS. Extensions: `YJH_WAVE_STARTS`, `YJH_WAVE_MIDS`, `YJH_WAVE_ENDS`, `YJH_SNR_MID`, `K_WAVE_STARTS`, `K_WAVE_MIDS`, `K_WAVE_ENDS`, `K_SNR_MID`. SNR depends on the target magnitude. **Included in repository** for HD189733b. |

> **The ETC SNR is per resolution element, not per pixel.** The `*_SNR_MID` extensions give the signal-to-noise ratio per resolution element. EXoPLORE converts this to a per-pixel SNR internally via SNR<sub>pixel</sub> = SNR<sub>resel</sub> / √m, where m = `instrument.pixels_per_resolution_element` (2.5 for ANDES). For order 35, for example, the tabulated value of ≈1256 per resolution element corresponds to ≈800 per pixel.

Individual arm files (`ANDES_ETC_WAVE_SNR_YJH_HD189733b.fits`, `…_K_…`, `…_RIZ_…`, `…_UBV_…`) are also provided in `inputs/ANDES/HD189733b/` for reference.

**Specific-event mode** (`observation.specific_event: true`): in this mode the simulator reproduces a real observed night using the actual timestamps and airmass. You must additionally provide:
- `reference_night/julian_date_0.fits`, BJD timestamps, shape `(n_spectra,)`
- `reference_night/airmass_0.fits`, airmass per exposure, shape `(n_spectra,)`

**Obtaining new ETC files:** Generate with the [ANDES ETC](https://andes-etc.brera.inaf.it/).

---

### CARMENES (CAHA)

| File | Location | Description |
|---|---|---|
| `wave_CARMENES_NIR.fits` | `src/exoplore/instruments/data/` | Wavelength solution for the CARMENES NIR channel. Shape: `(28, 4080)` = `(n_orders, n_pixels)`. Units: µm. Generated from a CARACAL wavelength calibration file. **Included in the repository.** |
| `wave_CARMENES_VIS.fits` | `src/exoplore/instruments/data/` | Wavelength solution for the CARMENES VIS channel. Shape: `(44, 4096)` = `(n_orders, n_pixels)`. Units: µm. Averaged from 95 CARMENES VIS calibration frames. **Included in the repository.** |

Both wavelength files are found automatically by the simulator, you do not need to copy or configure them. The simulator also needs per-night observation reference files (see Section 2).

The wavelength grid was extracted from real CARMENES FITS files produced by the CARACAL pipeline. Each per-exposure FITS file (e.g. `car-YYYYMMDDTHHMMSS-sci-norl-nir_A.fits`) contains four data extensions: spectra (1), continuum (2), uncertainties (3), and wavelength (4). All four have shape `(n_pixels, n_orders)` per exposure; when read across a full night they form arrays of shape `(n_spectra, n_orders, n_pixels)` after transposition. The wavelength solution is stable across exposures, so the bundled `wave_CARMENES_NIR.fits` discards the `n_spectra` dimension and has shape `(n_orders, n_pixels)` in µm (extension 4 values divided by 1e4 to convert from Å). The full `(n_spectra, n_orders, n_pixels)` arrays for spectra, uncertainties, and wavelengths are only needed when `observation.specific_event: true` (reference-night noise model) or `observation.use_real_data: true` (real-data analysis), and are provided as the `sig`, `snr`, and `observations_night_K` files described in Section 2.

---

## 2. Observation reference files

These provide the time sampling, airmass evolution, and noise properties for a real or template observing night.

For each instrument/target combination you need a `reference_night/` subdirectory inside `paths.inputs_dir` containing:

Files always use a **night index suffix** (`_0`, `_1`, …), even for single-night simulations.  A single night is night 0.

| File | Shape | Description |
|---|---|---|
| `sig_0.fits` | `(n_spectra, n_orders, n_pixels)` | Signal array from the reference night. Used when `observation.use_real_data: true`, or to provide noise scaling. |
| `snr_0.fits` | `(n_spectra, n_orders, n_pixels)` | SNR array from the reference night. |
| `julian_date_0.fits` | `(n_spectra,)` | BJD timestamps for each exposure. Used directly as `syn_jd` by the simulator. |
| `airmass_0.fits` | `(n_spectra,)` | Airmass for each exposure. The simulator reads this array verbatim, it is **not** recomputed from a model. |

### Multiple-night mode (`observation.different_nights: true`)

`different_nights: true` is supported for both **ANDES** and **CARMENES_NIR**.

Supply one file per night, with a zero-based night index suffix. When `use_real_data: true`, each night may have a **different number of spectra** `n_spectra_b` (read from its `julian_date_{b}.fits`); the simulator pads all arrays to the maximum night length with NaN and operates only on the live rows. With `use_real_data: false` all nights share the same number of spectra.

| File | Shape | Description |
|---|---|---|
| `sig_0.fits`, `sig_1.fits`, … | `(n_spectra_b, n_orders, n_pixels)` | Signal array for night `b` |
| `snr_0.fits`, `snr_1.fits`, … | `(n_spectra_b, n_orders, n_pixels)` | SNR array for night `b` |
| `julian_date_0.fits`, `julian_date_1.fits`, … | `(n_spectra_b,)` | BJD timestamps for night `b` |
| `airmass_0.fits`, `airmass_1.fits`, … | `(n_spectra_b,)` | Airmass time series for night `b` |

**Additional files required only when `observation.use_real_data: true`** (analysing real observed spectra):

| File | Shape | Description |
|---|---|---|
| `observations_berv_0.fits`, `observations_berv_1.fits`, … | `(n_spectra_b,)` | Barycentric Earth RV for each exposure of night `b` |
| `observations_night_0_order_K.fits` | `(n_spectra_0, n_pixels)` | Real observed spectra for night 0, spectral order `K` |
| `observations_night_1_order_K.fits` | `(n_spectra_1, n_pixels)` | Real observed spectra for night 1, spectral order `K` |
| … | … | One file per night per order |

Here `K` is the **absolute spectral order number** from the instrument's `order_selection` array, not a sequential counter.

> **Note on `n_spectra`:** regardless of `use_real_data`, the number of exposures per night is always determined by `julian_date_{b}.fits`, specifically, `n_spectra_b = len(julian_date_b)`. The spectral and uncertainty arrays must have a matching first dimension; the simulator does not infer `n_spectra` from them. For `different_nights: false` with `use_real_data: true`, the night suffix is omitted: `observations_order_{K}.fits` and `observations_berv.fits`.

**For the HD189733b CARMENES reference night** (Alonso-Floriano et al. 2019, A&A 621, A74, night of 2017-09-07):
The sig/snr/JD/airmass files from that published observation are in `inputs/CARMENES_NIR/HD189733b/reference_night/`. These are the files used in the CARMENES validation run. They should be cited as Alonso-Floriano et al. (2019) if used in published work.

---

## 3. Stellar atmosphere models, PHOENIX

The simulator uses PHOENIX-ACES-AGSS-COND-2011 high-resolution spectra to model the stellar continuum.

You need two files:

| File | `paths` key | Description |
|---|---|---|
| `WAVE_PHOENIX-ACES-AGSS-COND-2011.fits` | `phoenix_wave_file` | Wavelength grid, shared across all PHOENIX models. ~1.57 million wavelength points, 500 to 55,000 Å (0.05 to 5.5 µm). |
| `lte{Teff:05d}-{logg:.2f}-{met:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits` | `phoenix_flux_file` | Flux spectrum for the specific stellar parameters. |

**How to select the right flux file:** round `stellar_teff_K` to the nearest 100 K, `stellar_logg` to the nearest 0.5, and `stellar_metallicity` to the nearest 0.5. For HD189733 (Teff=5400 K, logg=4.587, [Fe/H]=-0.03) the correct file is:

```
lte05400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
```

**Downloading:** From the PHOENIX library at the Göttingen observatory:

```
https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/
```

The wavelength file is at:
```
https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
```

These files are large (the wavelength file is ~10 MB; flux files are ~500 MB each) and are **not included in the repository**. Set the absolute paths in your config under `paths.phoenix_wave_file` and `paths.phoenix_flux_file`.

---

## 4. Telluric reference spectra

EXoPLORE supports three telluric modes (see [config_reference.md §7](config_reference.md#7-tellurics) for how to select them). The files required depend on which mode you use.

> **We encourage you to try all three modes.** The choice significantly affects how realistic the atmospheric systematics are and which spectral regions get masked. Mode 1 (airmass scaling) captures the dominant effect with minimal setup. Mode 2 (per-exposure) adds independent PWV variation, bringing the simulation closest to real observing conditions, particularly important for long transits or high-airmass targets where the atmosphere can evolve substantially during the night. Mode 3 (no tellurics) is useful for clean signal tests.

---

### Mode 1, Airmass-scaled reference (default, `use_full_skycalc: false`)

Requires a single reference telluric spectrum computed at `reference_airmass` (typically 1.0). The simulator scales it to each exposure using the Beer-Lambert-Bouguer law (Bouguer 1729; Lambert 1760; Beer 1852):

```
T(X) = T_ref ^ (X / X_ref)
```

where `T_ref` is the transmission at the reference airmass `X_ref` and `X` is the airmass at each exposure. This follows from the fact that optical depth scales linearly with airmass, so `τ(X) = τ_ref × (X / X_ref)` and therefore `T(X) = exp(-τ(X)) = T_ref^(X/X_ref)`.

| File | Config key | Description |
|---|---|---|
| `tellurics/tell_ref_airmass_1.0.fits` | `tellurics.reference_telluric_file` | SkyCalc FITS file with columns `lam` (nm) and `trans` (fractional transmission 0 to 1), computed at airmass 1.0 and your target PWV. |

**Standard convention:** place the reference file in a `tellurics/` subdirectory inside your `inputs_dir`. Both bundled instruments follow this layout:
- `inputs/ANDES/HD189733b/tellurics/tell_ref_airmass_1.0.fits`
- `inputs/CARMENES_NIR/HD189733b/tellurics/tell_ref_airmass_1.0.fits`

If you leave `reference_telluric_file` unset in the config, the simulator auto-constructs the path as:
```
{inputs_dir}tellurics/tell_ref_airmass_{reference_airmass:.1f}.fits
```
This matches the standard `tellurics/` convention, so you can simply place your reference file there and omit the explicit config key.

**Generating the reference file** using `scripts/generate_skycalc_inputs.py`
(see [Mode 2, Per-exposure telluric evolution](#mode-2-per-exposure-telluric-evolution-use_full_skycalc-true)
for full script documentation):

```bash
pip install skycalc_ipy
python scripts/generate_skycalc_inputs.py configs/my_config.json
```

The script generates `tell_ref_airmass_{X.X}.fits` automatically alongside
the per-exposure files.  Alternatively use the
[ESO SkyCalc web tool](https://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC)
and save the result as a FITS table with columns `lam` (nm) and `trans` (0 to 1).

---

### Mode 2, Per-exposure telluric evolution (`use_full_skycalc: true`)

Each exposure gets its own telluric transmission spectrum with independent airmass and PWV. This is the most realistic treatment of atmospheric variability.

#### Required files

One FITS file per synthetic exposure, named sequentially and placed in the correct directory:

```
{inputs_dir}Skycalc_{flag_event}/
  Fixed_PWV/              ← when constant_pwv: true
    pwv_values.fits       ← per-exposure PWV array, shape (n_spectra,)
    tell_spec_0.fits      ← telluric spectrum for exposure 0
    tell_spec_1.fits      ← telluric spectrum for exposure 1
    ...
    tell_spec_{N-1}.fits  ← one file per synthetic exposure
  Variable_PWV/           ← when constant_pwv: false
    pwv_values.fits
    tell_spec_0.fits
    ...
```

Each `tell_spec_{n}.fits` must be a FITS file with at minimum two columns: `lam` (wavelength in nm) and `trans` (fractional transmission 0 to 1). The simulator reads whatever FITS files it finds at these paths, **you can supply spectra from any source**, not just SkyCalc. If you have telluric spectra from MOLECFIT, Tapas, or your own atmospheric model, place them here under the correct naming convention.

#### Generating files with `scripts/generate_skycalc_inputs.py`

EXoPLORE provides a ready-to-use script that queries the ESO SkyCalc REST
API directly via the `skycalc_ipy` Python package (no CLI installation
required).  All SkyCalc parameters are read from your config, no manual
editing of parameter files.

**Install the dependency once:**
```bash
pip install "exoplore[skycalc]"   # or: pip install skycalc_ipy astroplan
```

**Run the script:**
```bash
python scripts/generate_skycalc_inputs.py configs/my_config.json
```

The script writes `tell_spec_{n}.fits` and `tell_ref_airmass_{X.X}.fits` into
the directory the simulator expects, and generates `pwv_values.fits` when
`constant_pwv: false`.  Existing files are skipped; use `--overwrite` to
regenerate them.

---

#### Airmass modes, a critical choice

The airmass value assigned to each exposure determines which SkyCalc spectrum
is downloaded.  Two modes are available, selected with `--mode`:

---

**`--mode astro` (default)**

Computes airmass from true sky geometry using
[astroplan](https://astroplan.readthedocs.io) (Morris et al. 2018), querying
the planet's RA/Dec against the observatory site and the actual observation
timestamps.  This approach mirrors the methodology of the Ratri simulator
(Dash et al. (arXiv:2602.22830, 2026)), which uses astroplan
to plan and date synthetic observations from the ELT site and derives the
per-exposure airmass from real sky coordinates rather than a parametric model.

The behaviour differs depending on `observation.specific_event`:

**`specific_event: true`, you have a real observed night:**

> You should use the **real airmass from `reference_night/airmass_0.fits`**
> rather than this mode.  The simulator reads those files directly
> (`syn_jd = JD_og` in `get_event`; airmass loaded from `airmass_0.fits`).
>
> If `airmass_0.fits` is unavailable, `--mode astro` is a fallback: it loads
> the real timestamps from `reference_night/julian_date_0.fits` (the same
> JDs the simulator will use) and queries astroplan at each one.  Agreement
> with real CARMENES airmass: < 2 % (RMSE 0.014), limited by proper motion
> not being applied to the target coordinates.  At SkyCalc rounding precision
> (0.1 airmass) this error is negligible.

```bash
python scripts/generate_skycalc_inputs.py configs/hd189733b_carmenes_transit.json \
    --mode astro
```

**`specific_event: false`, fully synthetic simulation:**

> The script uses astroplan's `EclipsingSystem` to search the transit
> ephemeris for the **next observable transit from the site** (target above
> 30°, Sun below astronomical twilight) and builds the exposure JD grid with
> `observation_julian_dates()`, the same function the simulator uses in
> Block 2 with your config's `exposure_time_seconds`, `readout_time_seconds`,
> and `pre_event_hours`.  This guarantees that the SkyCalc airmass sequence
> matches the simulator's synthetic JD grid exactly.
>
> No date input is required.  Use `--search-from YYYY-MM-DD` to start the
> search from a specific date (e.g. for ELT-era planning).
>
> This is analogous to how Dash et al. (arXiv:2602.22830, 2026) plan their ELT/ANDES
> simulations: they select real future dates at which the target is observable
> from Cerro Armazones and derive the airmass sequence from the actual sky
> geometry, not a parametric model.

```bash
# Next observable transit from today:
python scripts/generate_skycalc_inputs.py configs/hd189733b_andes_transit_clean.json \
    --mode astro

# Plan for ELT era:
python scripts/generate_skycalc_inputs.py configs/hd189733b_andes_transit_clean.json \
    --mode astro --search-from 2030-01-01
```

**Required fields in planet parameter file for `--mode astro`:**
- `ra_deg`, `dec_deg`, sky coordinates (J2000)
- `transit_epoch_bjd`, `orbital_period_days`, ephemeris
- `transit_duration_hours`, optional; computed from orbital elements if absent

---

**`--mode synthetic`**

Computes airmass from the same parabolic model the simulator uses in Block 2
(`synthetic_airmass`, driven by `tellurics.airmass_limits` and
`tellurics.airmass_evolution`).  The airmass array exactly matches what the
simulator will use during the run, no inconsistency between the telluric
files and the simulation geometry.

Use when sky coordinates are unavailable or for reproducible test runs
independent of actual sky geometry.

```bash
python scripts/generate_skycalc_inputs.py configs/my_config.json --mode synthetic
```

---

#### Things to consider when choosing a mode

| Situation | Recommended approach |
|---|---|
| Fully synthetic simulation (ANDES or CARMENES, no real night) | `--mode astro` (default) |
| Reproducing a known real night (`specific_event: true`) | Use `airmass_0.fits` directly; `--mode astro` as fallback |
| Planning future observations from site geometry | `--mode astro [--search-from DATE]` |
| Coordinates unavailable or quick tests | `--mode synthetic` |

**Note on SkyCalc airmass rounding:** SkyCalc rounds airmass to one decimal
place per query.  The difference between `--mode synthetic` (airmass 1.432)
and `--mode astro` (airmass 1.445) results in the same SkyCalc call (both
round to 1.4).  For most targets the practical difference between modes is
zero at the SkyCalc precision level.

#### Using your own telluric spectra

You can bypass SkyCalc entirely and supply spectra from any tool (MOLECFIT, Tapas, a custom atmospheric model, etc.). Compute one telluric transmission spectrum per synthetic exposure and save each as a FITS table with columns `lam` (nm) and `trans` (0 to 1) named `tell_spec_{n}.fits` for `n = 0, 1, ..., N-1`. Place them under `Fixed_PWV/` or `Variable_PWV/` as appropriate and set `use_full_skycalc: true`. The simulator will load them without any modification.

#### Variable PWV (`constant_pwv: false`)

When `constant_pwv: false`, `PWV_handling` draws a per-exposure PWV sequence that varies randomly around `pwv_mm` within the SkyCalc discrete grid:

```
[0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0]  mm
```

Each exposure is assigned the reference value or one grid step above/below it, mimicking realistic intra-night PWV fluctuation. Files go to `Variable_PWV/` instead of `Fixed_PWV/`.

---

### Mode 3, No tellurics (`include_tellurics: false`)

The telluric multiplication step is skipped entirely, producing a clean absence of atmospheric contamination. No input files required for this mode. Use it to test the pipeline on a pure planetary signal, verify CCF template alignment under ideal conditions, or diagnose whether a feature in your detection map comes from the atmosphere or the planet model.

---

## 5. Summary table, what you need for each instrument/scenario

### ANDES + HD189733b transit (default simulated observation)

| File | Provided? |
|---|---|
| `ANDES_ETC_WAVE_SNR_YJHK_HD189733b.fits` | ✓ in `inputs/ANDES/HD189733b/` |
| `tellurics/tell_ref_airmass_1.0.fits` | ✓ in `inputs/ANDES/HD189733b/tellurics/` |
| `WAVE_PHOENIX-ACES-AGSS-COND-2011.fits` | ✗ download from PHOENIX library |
| `lte05400-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits` | ✗ download from PHOENIX library |
| petitRADTRANS `input_data/` | ✗ installed with petitRADTRANS |

*No reference-night files are needed for simulated ANDES observations. `different_nights: true` is supported, see the section below.*

### CARMENES NIR + HD189733b transit (Alonso-Floriano et al. 2019 night)

| File | Provided? |
|---|---|
| `wave_CARMENES_NIR.fits` | ✓ in `src/exoplore/instruments/data/` |
| `reference_night/julian_date_0.fits` | ✓ in `inputs/CARMENES_NIR/HD189733b/reference_night/` (tracked via Git LFS) |
| `reference_night/airmass_0.fits` | ✓ in `inputs/CARMENES_NIR/HD189733b/reference_night/` (tracked via Git LFS) |
| `reference_night/sig_0.fits` | ✓ in `inputs/CARMENES_NIR/HD189733b/reference_night/` (tracked via Git LFS) |
| `reference_night/snr_0.fits` | ✓ in `inputs/CARMENES_NIR/HD189733b/reference_night/` (tracked via Git LFS) |
| `tellurics/tell_ref_airmass_1.0.fits` | ✓ in `inputs/CARMENES_NIR/HD189733b/tellurics/` |
| PHOENIX wave + flux files | ✗ download from PHOENIX library |
| petitRADTRANS `input_data/` | ✗ installed with petitRADTRANS |

> **Git LFS note:** The CARMENES reference-night spectral arrays (`sig_0.fits`, `snr_0.fits`) are ~40 MB each and are tracked with [Git LFS](https://git-lfs.github.com/). Run `git lfs pull` after cloning to download them. The small arrays (`julian_date_0.fits`, `airmass_0.fits`) are tracked in regular git.

> **Order selection:** the bundled config excludes orders 9, 10, 18, 19, 20 following Alonso-Floriano et al. (2019), leaving 23 orders across the Y, J, and H windows. Orders 18 to 20 (~1.9 µm) are reliably opaque. Orders 9 to 10 (~1.4 µm) are PWV-dependent and can be recovered under dry conditions, users should assess their own data before discarding them. Set `"order_indices": []` to include all 28 and decide based on the observed telluric transmission.

---

## 6. Inputs directory layout

The repository ships with this layout under `inputs/`:

```
inputs/
  ANDES/
    HD189733b/
      ANDES_ETC_WAVE_SNR_YJHK_HD189733b.fits   ← primary ETC file (YJHK combined)
      ANDES_ETC_WAVE_SNR_YJH_HD189733b.fits     ← YJH arm only (reference)
      ANDES_ETC_WAVE_SNR_K_HD189733b.fits       ← K arm only (reference)
      ANDES_ETC_WAVE_SNR_RIZ_HD189733b.fits     ← RIZ arm only (reference)
      ANDES_ETC_WAVE_SNR_UBV_HD189733b.fits     ← UBV arm only (reference)
      tellurics/
        tell_ref_airmass_1.0.fits               ← Mode 1 reference file (standard location)
      Skycalc_full_event/                       ← Mode 2 files (use_full_skycalc: true)
        Fixed_PWV/                              ← single night, constant_pwv: true
          pwv_values.fits                       ← per-exposure PWV array (auto-generated)
          tell_spec_0.fits                      ← per-exposure SkyCalc output
          tell_spec_1.fits
          ...
          (generated by: python scripts/generate_skycalc_inputs.py config.json)
        Variable_PWV/                           ← constant_pwv: false
          pwv_values.fits
          tell_spec_0.fits
          ...
        night_0/                                ← different_nights: true, night 0
          Fixed_PWV/
            tell_spec_0.fits
            ...
        night_1/                                ← different_nights: true, night 1
          Fixed_PWV/
            tell_spec_0.fits
            ...
        (generated by: python scripts/generate_skycalc_inputs.py config.json --night 0)
        (                python scripts/generate_skycalc_inputs.py config.json --night 1)
      reference_night/                          ← only needed for specific_event=true
        julian_date_0.fits                      ← real BJDs, used verbatim as syn_jd
        airmass_0.fits                          ← real airmass, read directly by simulator
  CARMENES_NIR/
    HD189733b/
      reference_night/
        sig_0.fits          ← ~40 MB (large, not in git, host on Zenodo or download separately)
        snr_0.fits          ← ~40 MB
        julian_date_0.fits  ← 5 KB, tracked in git
        airmass_0.fits      ← 5 KB, tracked in git
      tellurics/
        tell_ref_airmass_1.0.fits
```

Each simulation config's `paths.inputs_dir` should point to the instrument+target subdirectory, e.g.:
```json
"inputs_dir": "inputs/ANDES/HD189733b/"
```
