# The Nortmann26 preparing pipeline (CRIRES+ division SYSREM)

`pipeline.name: "Nortmann26"` selects the CRIRES+ analysis recipe of
Nortmann et al. (2026, A&A), developed for the GJ 1214 b K-band transits and
applicable to any CRIRES+ nodding transit time series prepared with
`scripts/prepare_crires_night.py` (see [CRIRES+ reduction](crires_reduction.md)
for the upstream chain).

## What the pipeline does

Per wavelength segment (detector × echelle order), on the real-data time
series:

1. **Common-blaze normalisation** (Nortmann 2026, Sec. 3.1). Two steps: each
   spectrum is divided by a second-order polynomial fit to its ratio with the
   master (the summed out-of-transit spectrum); then the master itself is
   fitted with a second-order polynomial under iterative stellar-line masking
   and divided out, bringing all spectra to a common blaze. NaN and >4σ
   outlier pixels are flagged.
2. **Column masking**. Columns with more than three flagged pixels are
   masked entirely; lighter flags are interpolated over time
   (`nortmann_column_mask`).
3. **Deepest-telluric mask**. Regions where the flux falls below 20% of the
   continuum are masked before detrending (Nortmann 2026, Sec. 3.1).
4. **Division SYSREM** (Gibson et al. 2020 convention; Nortmann 2026,
   Sec. 3.2). SYSREM components are accumulated into a systematics model
   M_N and the data are corrected by division, R = D / M_N, preserving the
   planet signal shape. The per-iteration time basis U is stored for model
   reprocessing. The paper uses 9 iterations for GJ 1214 b
   (`pipeline.sysrem_iterations`).
5. **Model filtering for retrievals** (`nortmann_model_filter`): the
   Gibson et al. (2022) filter in the division form of Nortmann 2026
   Appendix A.3 (Eqs. A.1–A.3), so forward models suffer the same SYSREM
   distortions as the data.

The Kp–Vsys maps, in/out-of-trail statistics and significance metrics are the
engine's standard CCF machinery; the map convention is
`v_rest = offset from the expected planet trail`.

## Extra preparation steps bundled for CRIRES+

`scripts/prepare_crires_night.py` performs, automatically, on every run:

- **Frame quality cut**: exposures whose spatial PSF FWHM (pipeline
  `ESO QC SLITFWHM` headers) or S/N are strong outliers for the night
  (adaptive-optics losses) are excluded, data-driven.
- **Cosmic-ray rejection** (Lesjak et al. 2025): per wavelength channel, a
  robust (Tukey biweight) polynomial fit over time; >5σ outliers are replaced
  by the fit value (`Robust_Outlier_Removal`).
- **Effective spectral resolution** (Nortmann et al. 2024, App. A.2; Lesjak
  et al. 2025): the stellar PSF FWHM is read from the cr2res headers; a PSF
  narrower than the slit means the star underfills it and the true resolution
  exceeds the nominal R = 100,000 ("super-resolution"),
  R = 100000 × 3.5 px / FWHM px. When the PSF **overfills** the slit (poor
  seeing) the resolution is slit-limited and clamped to the nominal value, rather
  than extrapolated below it. The measured per-segment R is stored in
  `resolution_0.fits` and the engine convolves all templates to the measured
  median R instead of the nominal value.
- **Telluric alignment check** (Nortmann et al. 2024, App. A.2): every
  exposure is cross-correlated, in continuum-normalised absorption, against a
  reference exposure over the telluric lines; the A/B offset and night drift
  are corrected only if significant, otherwise the grids are kept (the
  no-drift case of Lesjak 2025 and Nortmann 2026).

## Current implementation notes

Read before publishing results based on this pipeline:

- The theoretical telluric model is currently used in the preparation stage
  (masks, wavelength anchoring) but is **not yet forwarded into the engine's
  normalisation step**, so the 96%-continuum telluric mask of the
  normalisation fits (Nortmann 2026, Sec. 3.1) is not applied there; the
  deep (<20% of continuum) mask uses the observed flux, which matches the
  paper's wording for that mask.
- Detrending currently runs in the observer frame; the stellar rest frame
  variant preferred by Nortmann 2026 (`nortmann_shift_to_stellar_frame`) is
  available in `exoplore.pipelines.nortmann26` but not yet wired into the
  engine branch. For a single short night the difference is small; for
  multi-night stacks it matters.
- The linear super-resolution anchor extrapolates; at PSF FWHM well below
  the slit width it may overestimate R by several percent (Nortmann 2024
  quote R ≈ 140,000 at ∼2 px). The effect on a matched filter is a few
  percent of S/N at most.
- Injection-recovery is supported (`observation.simulate_planet: true` with
  `observation.scale_injection`/`pipeline.inject_scale_factor` and
  `pipeline.kp_vrest_injection`): a scaled template is injected into the real
  residuals and recovered through the full pipeline, so a non-detection can be
  turned into an upper limit. The injected signal uses the real per-exposure
  BERV, so it recovers at the planet position rather than displaced.

## Worked example: a public CRIRES+ K-band transit

The transit of L 98-59 c observed with CRIRES+ in the K2148 setting on the
night of 2022-03-23 (ESO programme 108.22PH) is public in the ESO archive and
makes a complete end-to-end example.

1. **Download** from the [ESO archive](http://archive.eso.org/eso/eso_archive_main.html):
   query instrument CRIRES, night 2022-03-23 to 2022-03-24, target L 98-59
   (programme 108.22PH). Request the raw science frames **with associated raw
   calibrations** (CalSelector: darks, flats, uranium-neon and Fabry-Perot
   wavelength frames, plus the static trace-wave, emission-line and photometric
   flux tables). Place everything in one directory, e.g. `mynight/raw/`.
2. **Install the ESO tools**: `esorex` with the `cr2res` and `molecfit`
   recipes, from the [ESO pipelines page](https://www.eso.org/sci/software/pipelines/).
   See [CRIRES+ reduction](crires_reduction.md); EXoPLORE only wraps these
   tools.
3. **Reduce**: `python scripts/reduce_crires_night.py mynight/raw all`
   → `mynight/reduced/` with one extracted spectrum per nodding exposure and
   `timeseries_manifest.txt`.
4. **Prepare**:
   `python scripts/prepare_crires_night.py mynight inputs/CRIRES_PLUS/L98-59c/reference_night`
   — runs molecfit per nodding position if not cached (slow, once), then the
   quality cut, cosmic-ray rejection, wavelength refinement and gating,
   resolution measurement and drift check, and writes the `reference_night/`
   files.
5. **Configure a run**: planet parameters in `planet_params/L9859c.json`
   (Demangeon et al. 2021); a minimal config uses

   ```json
   {
     "planet":      {"name": "L9859c", "parameter_file": "planet_params/L9859c.json"},
     "instrument":  {"name": "CRIRES+", "observatory": "paranal",
                     "pixels_per_resolution_element": 3.0,
                     "order_indices": [], "convolve_to_resolution": true},
     "observation": {"event_type": "transit", "specific_event": true,
                     "specific_T0_bjd": 2459662.7016, "use_real_data": true,
                     "exposure_time_seconds": 240.0, "n_nights": 1,
                     "simulate_planet": false, "noiseless": true},
     "pipeline":    {"name": "Nortmann26", "sysrem_iterations": 9},
     "paths":       {"inputs_dir": "inputs/CRIRES_PLUS/L98-59c/"}
   }
   ```

   with the remaining blocks (atmosphere template species, cross_correlation)
   as in [the config reference](config_reference.md).
6. **Run**: `python scripts/run_exoplore.py myconfig.json --run` — the engine
   ingests the reference night, detrends with division SYSREM, cross-correlates
   the petitRADTRANS template convolved to the measured resolution, and writes
   Kp–Vsys maps and matrices under `paths.output_root`.

The reduced GJ 1214 b spectra of Nortmann et al. (2026) are also public
(Zenodo record 19387252) and can be used to validate the analysis stage
against a published detection.

## References

- Nortmann, L., et al. 2026, A&A (arXiv:2604.15292): the reproduced pipeline
  (normalisation, masks, division SYSREM, model filter).
- Nortmann, L., et al. 2025, A&A 693, A213, WASP-127 b (arXiv:2404.12363):
  super-resolution and wavelength methodology (App. A.2, A.3).
- Lesjak, F., et al. 2025, A&A 693, A72: super-resolution usage, 5σ
  time-series outlier rejection, deep-telluric masking.
- Gibson, N. P., et al. 2020, MNRAS 493, 2215: division SYSREM convention.
- Gibson, N. P., et al. 2022, MNRAS 512, 4618: model filtering.
- Tamuz, O., Mazeh, T., & Zucker, S. 2005, MNRAS 356, 1466: SYSREM.
- Dorn, R. J., et al. 2023, A&A 671, A24: CRIRES+ instrument, nominal
  resolution and slit sampling.
- Smette, A., et al. 2015, A&A 576, A77; Kausch, W., et al. 2015, A&A 576,
  A78: molecfit.
- Demangeon, O. D. S., et al. 2021, A&A 653, A41: L 98-59 system parameters.
