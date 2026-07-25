# Reducing and preparing real CRIRES+ data

This page describes how to turn raw CRIRES+ nodding frames into the
`reference_night/` files EXoPLORE consumes, following the data handling of
Nortmann et al. 2026. The chain has three stages:

```
raw frames ──▶ reduce_crires_night.py ──▶ crires_molecfit ──▶ prepare_crires_night.py ──▶ reference_night/
   (cr2res)        (extraction)            (wavelength +          (ingest)
                                            telluric model)
```

EXoPLORE only **wraps** the ESO tools for these steps; it does not replace them
or their documentation. You must install the ESO pipelines yourself.

## Prerequisites: install the ESO tools

Two ESO command line pipelines must be installed and on your `PATH` (the
wrappers call `esorex`):

- **cr2res** (CRIRES+ pipeline), for the raw reduction and extraction.
  Installation and the authoritative recipe documentation:
  <https://www.eso.org/sci/software/pipelines/cr2res/>
- **molecfit**, for the telluric model and the wavelength refinement.
  Installation and documentation:
  <https://www.eso.org/sci/software/pipelines/molecfit/>

The easiest way to obtain both is the ESO pipeline installer (`esoreflex` /
`install_esoreflex`), which places `esorex` and all recipe data on your system.
Refer to the ESO pages above for the current instructions; they are the source
of truth for the reduction itself.

## Getting the raw data from the ESO archive

Request the raw science frames **with their associated raw calibrations**
(CalSelector: darks, flats, uranium-neon and Fabry-Perot wavelength frames) and
the **static calibration tables** for the setting, from the
[ESO archive](http://archive.eso.org/eso/eso_archive_main.html). Three practical
points that otherwise cause silent failures:

- **Frames arrive `.fits.Z`-compressed.** Decompress them (e.g. `uncompress` or
  `gunzip`) before reducing.
- **One delivery can bundle several nights.** Sort the frames so that **each
  night sits in its own directory** with its own calibrations; the driver reduces
  one directory at a time.
- **The static `M.CRIRES.*.fits` tables must be in the frame directory.**
  `cr2res_cal_wave` needs the per-setting emission-line catalog (`EMISSION_LINES`)
  and initial trace-wave (`UTIL_WAVE_TW`); these ship with the download as
  `M.CRIRES.*.fits` files. If they are not alongside the frames the wavelength
  step fails with *"the emission lines catalog is needed"*, and the extraction
  silently falls back to the (less accurate) flat trace-wave guess. Copy them in.

## Stage 1: raw reduction and extraction (`reduce_crires_night.py`)

`scripts/reduce_crires_night.py` drives the cr2res cascade on a directory of
raw frames and their calibrations:

```bash
python scripts/reduce_crires_night.py /path/to/raw all
```

It runs, in order, `cr2res_cal_dark`, `cr2res_cal_flat`, `cr2res_cal_wave`
(wavelength solution from the uranium neon lamp and the Fabry Perot etalon) and
`cr2res_obs_nodding` **per A/B pair**, so each nodding exposure is kept as a
separate spectrum in the time series. The calibration exposure times are read
from the frames, so no observing programme specific numbers are hard coded. You
can run a single step by replacing `all` with `dark`, `flat`, `wave` or `nod`.

Output (written next to the raw directory):

- `reduced/pair_*/cr2res_obs_nodding_extracted[AB].fits`, one extracted 1D
  spectrum per nodding exposure, in the CRIRES+ layout of 3 detectors times the
  echelle orders of the setting.
- `reduced/timeseries_manifest.txt`, one `mjd  path` line per extracted
  spectrum.

## Stage 2: wavelength refinement and telluric model (`crires_molecfit`)

`exoplore.instruments.crires_molecfit` runs molecfit on a reference spectrum per
nodding position to (a) refine the wavelength solution per detector segment and
(b) produce a theoretical telluric transmittance for the normalisation masks
(Nortmann Appendix A.1 and Sec 3.1). **You do not usually call this stage
directly**: `prepare_crires_night.py` (Stage 3) runs it automatically and caches
the result. To run it on its own:

```bash
python -m exoplore.instruments.crires_molecfit /path/to/reduced A
```

What it does automatically:

- **Reads the setting and slit width from the header** and selects the telluric
  molecules relevant to the band. Only water is fitted (it anchors the
  wavelength solution across the near infrared); the other band molecules are
  included at climatological abundance. The molecule lists per band are in
  `BAND_MOLECULES` and can be overridden.
- **Picks the highest signal to noise exposure** as the reference (all exposures
  share the same Fabry Perot solution, which does not drift over the night).
- **Runs molecfit in the configuration validated for CRIRES+**: a vacuum
  wavelength frame (cr2res delivers vacuum wavelengths; running in air offsets
  the solution by of order 80 km/s at 2 micron) and the reflex line spread
  function (a single fitted Gaussian). The optimiser tolerance is molecfit's
  documented default (`FTOL = XTOL = 1e-2` order; the wrapper uses `1e-4`), not a
  far tighter value: on telluric-poor segments (e.g. between the water bands in
  Y) a tolerance well below the noise floor never converges. A per-segment
  wall-clock timeout bounds any segment that still fails to converge; it then
  falls back to the per-order median shift (below).
- **Gates the refinement.** molecfit only constrains the wavelength where a
  segment carries usable telluric structure. Segments that are telluric
  saturated (deep band heads, no continuum to anchor) or telluric poor (almost
  no lines) are under constrained; they inherit the median shift of the
  constrained detectors of the same echelle order (consistent to well under
  1 km/s), or keep the Fabry Perot solution if no detector of the order is
  constrained.
- **Reports a wavelength consistency check**: if the constrained segments
  cluster near zero shift, the Fabry Perot solution is already correct, which is
  the same conclusion Nortmann reaches from a telluric template cross
  correlation.

This generalises to any CRIRES+ Y, J, H, K, L or M setting; only the molecule
list and the slit width are setting dependent, and both are handled from the
header.

### Specifying the telluric species

The wrapper ships with a default telluric molecule list per band
(`BAND_MOLECULES`). **Only the K band list is a published recipe** (Nortmann
et al. 2025: water, methane, carbon dioxide, nitrous oxide). The Y, J, H, L and M
lists are the standard telluric absorbers of each range and are meant as
sensible starting points, **not** as validated per setting recipes. If you use
molecfit on a band other than K, you should state the fit species explicitly and
check them against your data.

Pass them with `--species` (to `prepare_crires_night.py` or to the wrapper). The
value is a comma separated list; append `:1`/`:0` to control which are fitted,
otherwise water is fitted as the wavelength anchor:

| Band | Approx. range | Example `--species` |
|---|---|---|
| Y | 0.96 to 1.12 µm | `H2O,O2` |
| J | 1.12 to 1.35 µm | `H2O,O2,CO2` |
| H | 1.43 to 1.78 µm | `H2O,CO2,CH4` |
| K | 1.93 to 2.50 µm | `H2O,CH4,CO2,N2O` (default) |
| L | 3.4 to 4.2 µm | `H2O,CH4,N2O,CO2,O3` |
| M | 4.5 to 5.3 µm | `CO2:1,CO:0,H2O:0,N2O:0,O3:0` (fit CO2) |

```bash
python scripts/prepare_crires_night.py /path/to/reduction_dir /path/to/output_dir \
    --species "H2O,CO2,CH4"
```

Only species with structured lines across the segment should be fitted; the rest
are included at climatological abundance so their column does not run away where
their lines are absent.

Output: `molecfit/molecfit_nod[AB].npz` with `names`, `wave_refined` (nm) and
`transmittance`, one row per segment.

## Stage 3: ingest into a reference night (`prepare_crires_night.py`)

`scripts/prepare_crires_night.py` assembles the extracted time series and the
molecfit products into the `reference_night/` files, computing the barycentric
correction and barycentric Julian date per exposure:

```bash
python scripts/prepare_crires_night.py /path/to/reduction_dir /path/to/output_dir
```

`reduction_dir` is the directory containing the `reduced/` and `molecfit/`
subdirectories from Stages 1 and 2. If the molecfit products are missing, this
script runs Stage 2 automatically (so a first run includes the slow molecfit
step; later runs reuse the cached npz). Flux is kept in native pixel space and
never resampled; the per exposure wavelength solution is stored in
`wave_perframe_0.fits` so the analysis pipeline aligns the nodding positions and
shifts to the stellar rest frame itself.

Output (see [Input files](input_files.md) for the full `reference_night/`
layout): `sig_0.fits`, `snr_0.fits`, `julian_date_0.fits`, `airmass_0.fits`,
`observations_berv_0.fits`, `wave_0.fits`, `wave_perframe_0.fits`,
`telluric_model_0.fits` and the per order `observations_night_0_order_*.fits`.
