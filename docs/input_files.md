# EXoPLORE Input Files Reference

This document lists every external file the simulator needs, where to obtain it, and where to place it.

## CARMENES ETC maximum-anchor inputs

For synthetic CARMENES observations, save the HTML response from the ETC for
your target and exposure duration. Obtain responses at airmasses bracketing
the observing sequence, keeping the other ETC inputs fixed. Package them with:

```bash
python scripts/carmenes_etc_to_fits.py \
    --planet HD189733b --channel VIS --exposure_seconds 198 \
    --etc_html 1.0=etc_X1.html --etc_html 1.35=etc_X1p35.html \
    --output_dir inputs/CARMENES_VIS/HD189733b
```

Set `noise.carmenes_etc_file` in your simulation JSON to the resulting
`CARMENES_VIS_ETC_HD189733b.npz`. Use `--channel NIR` for that channel.
The bundle stores all returned curves, their native ETC wavelength samples,
airmasses, the declared exposure duration and source-file hashes.
The converter preserves an existing bundle by refusing to overwrite it.

The simulation uses each order's maximum S/N and its wavelength as anchors.
The native EXoPLORE wavelength grid supplies the pixels-per-resolution-element
conversion. Each exposure's telluric spectrum supplies relative attenuation,
and the ANDES analytic blaze supplies the order envelope. The complete native
order is needed to position that envelope, even when a later diagnostic
displays only a small wavelength interval.

Enter the actual ETC exposure duration in `--exposure_seconds`. The simulator
requires that duration to match `observation.exposure_time_seconds` and requires
all exposure airmasses to fall within the supplied ETC conditions.
The ETC atmosphere remains part of the absolute anchor, while the simulated
atmosphere defines relative wavelength structure. See the
[noise configuration](config_reference.md#carmenes-etc-maximum-anchors) for the equations.

---

## Overview

Input files fall into four categories:

1. **Instrument files**, wavelength grids and SNR tables specific to the spectrograph
2. **Observation reference files**, real data from a reference night (JD, airmass, signal, SNR arrays)
3. **Stellar atmosphere models**, PHOENIX flux spectra and optional
   [local intensity spectra for RM and CLV](stellar_intensities.md)
4. **Telluric reference spectra**, [SkyCalc](https://skycalc-ipy.readthedocs.io)-generated atmospheric transmission

[petitRADTRANS](https://petitradtrans.readthedocs.io) opacity tables are a fifth category handled separately (see the [Installation guide](installation.md)).

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

The simulator locates both wavelength files automatically. The simulator also needs per-night observation reference files (see Section 2).

The wavelength grid was extracted from real CARMENES FITS files produced by the CARACAL pipeline. Each per-exposure FITS file (e.g. `car-YYYYMMDDTHHMMSS-sci-norl-nir_A.fits`) contains four data extensions: spectra (1), continuum (2), uncertainties (3), and wavelength (4). All four have shape `(n_pixels, n_orders)` per exposure. When read across a full night they form arrays of shape `(n_spectra, n_orders, n_pixels)` after transposition. The wavelength solution is stable across exposures, so the bundled `wave_CARMENES_NIR.fits` discards the `n_spectra` dimension and has shape `(n_orders, n_pixels)` in µm (extension 4 values divided by 1e4 to convert from Å). The full `(n_spectra, n_orders, n_pixels)` arrays for spectra, uncertainties, and wavelengths are only needed when `observation.specific_event: true` (reference-night noise model) or `observation.use_real_data: true` (real-data analysis), and are provided as the `sig`, `snr`, and `observations_night_K` files described in Section 2.

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

Supply one file per night, with a zero-based night index suffix. When `use_real_data: true`, each night may have a **different number of spectra** `n_spectra_b` (read from its `julian_date_{b}.fits`). The simulator pads all arrays to the maximum night length with NaN and operates only on the live rows. With `use_real_data: false` all nights share the same number of spectra.

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

Here `K` is the **instrument order ID** stored in `order_selection`.

> **Note on `n_spectra`:** regardless of `use_real_data`, the number of exposures per night is always determined by `julian_date_{b}.fits`, specifically, `n_spectra_b = len(julian_date_b)`. The spectral and uncertainty arrays must have a matching first dimension. Their exposure count must match the timing file. For `different_nights: false` with `use_real_data: true`, the night suffix is omitted: `observations_order_{K}.fits` and `observations_berv.fits`.

**For the HD189733b CARMENES reference night** ([Alonso-Floriano et al. 2019](https://doi.org/10.1051/0004-6361/201834339), A&A 621, A74, night of 2017-09-07):
The sig/snr/JD/airmass files from that published observation are in `inputs/CARMENES_NIR/HD189733b/reference_night/`. These are the files used in the CARMENES validation run. They should be cited as Alonso-Floriano et al. (2019) if used in published work.

---

## 3. Stellar atmosphere models, PHOENIX

The simulator uses PHOENIX-ACES-AGSS-COND-2011 high-resolution spectra to model the stellar continuum.

You need two files:

| File | `paths` key | Description |
|---|---|---|
| `WAVE_PHOENIX-ACES-AGSS-COND-2011.fits` | `phoenix_wave_file` | Wavelength grid, shared across all PHOENIX models. ~1.57 million wavelength points, 500 to 55,000 Å (0.05 to 5.5 µm). |
| `lte{Teff:05d}-{logg:.2f}-{met:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits` | `phoenix_flux_file` | Flux spectrum for the specific stellar parameters. |

**How to select the right flux file:** round `stellar_teff_K` to the nearest 100 K, `stellar_logg` to the nearest 0.5, and `stellar_metallicity` to the nearest 0.5. For HD189733 (Teff=5052 K, logg=4.587, [Fe/H]=-0.03) the nearest grid file is:

```
lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
```

**Downloading:** From the PHOENIX library at the Göttingen observatory:

```
https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/
```

The wavelength file is at:
```
https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
```

These files are large (the wavelength file is ~10 MB, flux files are ~500 MB each) and are **not included in the repository**. Set the absolute paths in your config under `paths.phoenix_wave_file` and `paths.phoenix_flux_file`.

---

## 4. Telluric reference spectra

EXoPLORE supports three telluric modes (see [config_reference.md §7](config_reference.md#7-tellurics) for how to select them). The files required depend on which mode you use.

> **We encourage you to try all three modes.** The choice significantly affects how realistic the atmospheric systematics are and which spectral regions get masked. Mode 1 (airmass scaling) captures the dominant effect with minimal setup. Mode 2 (per-exposure) adds independent PWV variation, bringing the simulation closest to real observing conditions, particularly important for long transits or high-airmass targets where the atmosphere can evolve substantially during the night. Mode 3 (no tellurics) is useful for clean signal tests.

---

### Mode 1, Airmass-scaled reference (default, `use_full_skycalc: false`)

Requires a single reference telluric spectrum computed at `reference_airmass` (typically 1.0). The simulator scales it to each exposure using the Beer-Lambert-Bouguer law (Bouguer 1729, Lambert 1760, Beer 1852):

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

(skycalc-reference-download)=
#### Generating the reference file

Use the simulation JSON to configure the download. Set
`tellurics.skycalc_wavelength_limits_um` to cover all orders you will analyse, with
some margin. For example, inside the existing `tellurics` block:

```json
"skycalc_wavelength_limits_um": [0.9, 2.5]
```

That interval is suitable for the full reference ANDES YJHK example. The
existing ANDES UBV example instead uses `[0.34, 0.65]`. Values here are in
**microns**. The generator converts them to SkyCalc's **nanometres**. This
setting controls the downloaded spectrum. Instrument input files define the observing grid.

Generate the reference with `scripts/generate_skycalc_inputs.py`
(see [Mode 2, Per-exposure telluric evolution](#mode-2-per-exposure-telluric-evolution-use_full_skycalc-true)
for full script documentation):

```bash
python -m pip install -e ".[skycalc]"
python scripts/generate_skycalc_inputs.py configs/my_config.json --ref-only --mode synthetic
```

The reference is written to
`{paths.inputs_dir}/tellurics/tell_ref_airmass_{X.X}.fits`, where `X.X` is
`tellurics.reference_airmass`. Without `--ref-only`, the script also generates
per-exposure files. Existing files are skipped unless `--overwrite` is
explicitly supplied. Back up an existing reference before replacing it with
a download using a different wavelength interval.

For a single reference, point `tellurics.reference_telluric_file` at that file.
For per-exposure tellurics, use the directory convention below and set
`use_full_skycalc: true`. EXoPLORE checks coverage of the selected orders and
raises an error if the file is too narrow. Synthetic observations load these
transmission files. Observed-data preparation uses the measured spectra. A telluric search
template explicitly requests a transmission input.

Alternatively use the
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

Each `tell_spec_{n}.fits` must be a FITS file with at minimum two columns: `lam` (wavelength in nm) and `trans` (fractional transmission 0 to 1). The simulator reads whatever FITS files it finds at these paths, **you can supply spectra from any source** using this format. If you have telluric spectra from MOLECFIT, Tapas, or your own atmospheric model, place them here under the correct naming convention.

#### Generating files with `scripts/generate_skycalc_inputs.py`

EXoPLORE provides a ready-to-use script that queries the ESO SkyCalc REST
API directly via the `skycalc_ipy` Python package (no CLI installation
required). It reads the wavelength interval, observatory, PWV and observing
choices from your simulation config. Other sky-model components and spectral
sampling retain the defaults in `_query_skycalc()`. They are not all exposed as
JSON options. No separate SkyCalc JSON is required.

**Install the dependency once:**
```bash
python -m pip install -e ".[skycalc]"
```

**Run the script:**
```bash
python scripts/generate_skycalc_inputs.py configs/my_config.json
```

The script writes `tell_spec_{n}.fits` and `tell_ref_airmass_{X.X}.fits` into
the directory the simulator expects, and generates `pwv_values.fits` when
`constant_pwv: false`.  Existing files are skipped. Use `--overwrite` to
regenerate them.

---

#### Airmass modes, a critical choice

The airmass value assigned to each exposure determines which SkyCalc spectrum
is downloaded.  Two modes are available, selected with `--mode`:

---

**`--mode config` (default)**

Uses the molecular simulator's observing samples. A single synthetic night
uses the existing parabolic airmass model. Separate synthetic nights follow
`use_accurate_airmass`, including observable transit epochs when coordinates
are available. A measured night uses its reference JD and airmass files.

```bash
python scripts/generate_skycalc_inputs.py configs/my_config.json --plan-only
```

**`--mode synthetic`**

Uses the parabolic airmass model with `airmass_limits` and
`airmass_evolution`. Use `--mode config` for separate nights with astronomical
airmasses so the generated inputs follow the simulator's choices.

**`--mode astro`**

Plans a future observable transit using astroplan, the planet's coordinates
and the observatory. `--search-from YYYY-MM-DD` sets the search date. This
explicit planning mode can produce dates different from the configured
simulation. The loader checks `exposure_jd.fits` and stops if the dates differ.
For inputs to an existing simulation, use the default `--mode config`.

The exposure table contains both the calculated airmass and the value sent to
SkyCalc. Downloads retain the existing rounding to 0.1 and clipping to 1–3.
Inspect those columns when planning high-airmass observations.

(skycalc-exposure-plan)=

#### Plan the exposures first

We can export the observing samples before downloading spectra:

```bash
python scripts/generate_skycalc_inputs.py configs/my_config.json --plan-only
```

The default `--mode config` follows the molecular simulator's observing dates
and airmass choices. `exposure_plan.csv` lists the night, exposure index, JD,
elapsed hours, exposure duration, airmass and expected `tell_spec_N.fits` name.
It also records the airmass rounded to 0.1 and clipped to 1–3 for SkyCalc.
`skycalc_request.json` gives the wavelength interval in nm, vacuum wavelength
convention and input resolving power. Planning sends no SkyCalc requests.

These times follow EXoPLORE's existing exposure sampling convention. Separate
nights require `--night N`. When a per-night cadence is configured, the table
uses that night's exposure time and the runner's corresponding time grid.

#### Using your own telluric spectra

Set the following fields in the master JSON's `tellurics` block:

```json
{
  "include_tellurics": true,
  "use_full_skycalc": true,
  "skycalc_source": "external",
  "skycalc_directory": "inputs/my_skycalc",
  "constant_pwv": false
}
```

Run `--plan-only` and generate the spectra for the listed exposures using your
chosen PWV history. Save a FITS table with columns `lam` (nm, vacuum) and
`trans` (fractional transmission) for each exposure. The files must share a
wavelength grid and cover the selected instrument orders. Separate nights
use `inputs/my_skycalc/night_N/`. Supply `airmass.fits` for each separate night
when the preparation should use the airmasses of your generated models.

The simulator checks the expected file list and wavelength coverage, then
interpolates and convolves each transmission onto the observing grid. Each
exposure's transmission also enters the ANDES per-pixel S/N calculation.
The PWV is set by the supplied spectra.

#### Downloading a variable-PWV night

A few time points describe the chosen evolution. Add these fields to the
`tellurics` block, keeping the other observing settings:

```json
{
  "include_tellurics": true,
  "use_full_skycalc": true,
  "skycalc_source": "download",
  "constant_pwv": false,
  "pwv_time_series": {
    "hours_from_observation_start": [0.0, 2.0, 5.0, 8.0],
    "pwv_mm": [2.5, 3.5, 5.0, 3.5]
  }
}
```

First inspect the exposure plan, then run the same command without
`--plan-only`. Extend the time points if the final exposure falls after the
last supplied time. EXoPLORE interpolates PWV linearly in time and requires
coverage of every exposure.

[ESO SkyCalc](https://eso.org/observing/etc/doc/skycalc/helpskycalccli.html)
provides PWV nodes at 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2.5, 3.5, 5, 7.5, 10,
20 and 30 mm. At an intermediate PWV, EXoPLORE downloads the two surrounding
nodes at the exposure's SkyCalc airmass and interpolates their `trans` columns
linearly. At a grid node it uses the downloaded model directly. This is an
approximation between the model transmissions. Saturated lines can respond
nonlinearly to PWV. A held-out test at 3.5 mm, using the wider 2.5–5 mm bracket,
gave a maximum absolute transmission error of 0.032 and RMS 0.019 over
1379–1381 nm at airmass 1.2. This test measures interpolation error in that
interval. It does not establish an accuracy bound for other wavelengths.

Downloaded models remain in `inputs_dir/skycalc_cache/`. Each cache key includes
airmass, PWV, observatory, wavelength interval, resolving power and query recipe.
Repeated exposures and later runs reuse matching downloads. Final exposure
files have request records alongside them. A changed request requires
`--overwrite`, which rebuilds those products from the cache where possible.

The plan records both PWV neighbours and their interpolation weight. FITS
headers retain the same information. Only `lam` and `trans` are consumed by
EXoPLORE. Other SkyCalc table columns in an interpolated product retain the
lower node's values and should not be used as interpolated sky radiance.
`pwv_values.fits` stores the requested per-exposure PWV, and `exposure_jd.fits`
lets the loader check that downloaded models match the simulation dates.


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
| `lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits` | ✗ download from PHOENIX library |
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
