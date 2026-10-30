# EXoPLORE Configuration Reference

A simulation config selects the atmosphere, observation and analysis. Its
planet parameter file supplies the orbit, stellar properties and planet size.
The sections below explain the molecular, helium and sodium controls, including
RM+CLV. Field definitions and defaults live in `src/exoplore/config/`.

Start with the Tutorial 1 reference files:

```
configs/hd189733b_andes_transit_clean.json
planet_params/HD189733b.json
```

The simulation JSON contains a `_notes` section explaining where to edit each
part of the experiment. The loader reads the scientific fields and leaves
these reading notes out of the resolved configuration saved with a run.
RM+CLV and direct sodium are disabled in this reference file. The `seidel22`
entry is `null` because Tutorial 1 uses Blain24.

For a complete sodium example, open
`configs/hd189733b_andes_sodium_rm_clv.json` and follow
{ref}`Tutorial 13 <tutorial-13-sodium>`.
It sets the optical order, atmosphere, observing sequence, stellar intensities
and telluric fit together. Check these inputs when adapting it to another target.

> **Running a simulation:** Always pass `--run` to actually execute. Without it the simulator prints a summary and exits:
> ```bash
> python scripts/run_exoplore.py configs/my_sim.json          # preview only
> python scripts/run_exoplore.py configs/my_sim.json --run    # actually runs
> ```

---

## Table of Contents

1. [Planet parameter file](#1-planet-parameter-file)
2. [Simulation config, `planet`](#2-planet)
3. [Simulation config, `instrument`](#3-instrument)
4. [Simulation config, `observation`](#4-observation)
5. [Simulation config, `noise`](#5-noise)
6. [Simulation config, `atmosphere`](#6-atmosphere)
7. [Simulation config, `tellurics`](#7-tellurics)
8. [Simulation config, `pipeline`](#8-pipeline)
9. [Simulation config, `cross_correlation`](#9-cross_correlation)
10. [Simulation config, `retrieval`](#10-retrieval)
11. [Simulation config, `statistics`](#11-statistics)
12. [Simulation config, `paths`](#12-paths)
13. [Atmosphere model block reference](#13-atmosphere-model-block-reference)
14. [Simulation config, `plotting`](#14-plotting)
15. [Simulation config, `output`](#15-output)
16. [Supported instruments](#16-supported-instruments)
17. [Adding a new planet](#17-adding-a-new-planet)
18. [Detectability sweeps and helium configurations](#18-detectability-sweeps-and-helium-configurations)
19. [Stellar RM and CLV](#19-stellar-rm-and-clv)
20. [Direct synthetic sodium observations](#direct-synthetic-sodium-observations)

---

## 1. Planet parameter file

Stored in `planet_params/<PlanetName>.json`. All values use natural astronomical units (solar, Jupiter, AU, degrees), the simulator converts internally to CGS and km/s as required by [petitRADTRANS](https://petitradtrans.readthedocs.io).

Supply the orbital, planetary, stellar, thermal and limb-darkening parameters
explicitly. Missing, `null` or non-finite required values raise a `ValueError`
that identifies the field and its source file. An explicit zero is retained.
Kp, stellar RV amplitude and transit duration can be calculated from the supplied
system parameters when omitted. Sky coordinates are optional at loading time
and are required by the observing calculations that use them.

| Field | Unit | Description |
|---|---|---|
| `name` |, | Planet name string, used for output directory naming |
| `orbital_period_days` | days | Orbital period |
| `transit_epoch_bjd` | BJD | Reference transit mid-time (T₀) |
| `semi_major_axis_au` | AU | Semi-major axis |
| `inclination_deg` | degrees | Orbital inclination (90° = edge-on) |
| `eccentricity` |, | Orbital eccentricity (0 for circular) |
| `argument_of_periastron_deg` | degrees | Argument of periastron ω (set to 90° for circular orbits) |
| `planet_radius_rjup` | R_Jup | Planet radius |
| `planet_mass_mjup` | M_Jup | Planet mass |
| `stellar_radius_rsun` | R_Sun | Stellar radius |
| `stellar_mass_msun` | M_Sun | Stellar mass |
| `stellar_teff_K` | K | Stellar effective temperature (used to select PHOENIX model) |
| `stellar_logg` | dex | Stellar surface gravity (used to select PHOENIX model) |
| `stellar_metallicity` | dex | Stellar metallicity [Fe/H] (used to select PHOENIX model) |
| `v_rotsini_kms` | km/s | Stellar projected rotational velocity v sin i |
| `systemic_velocity_kms` | km/s | Barycentric systemic radial velocity of the system |
| `kp_kms` | km/s | Planet orbital velocity semi-amplitude (used for CCF map axes and retrieval) |
| `stellar_rv_semiamplitude_kms` | km/s | Stellar RV semi-amplitude K_star |
| `equilibrium_temperature_K` | K | Planet equilibrium temperature (default for atmosphere model) |
| `t_int_K` | K | Planet internal temperature (for Guillot profile, typically 100 to 300 K) |
| `kappa_ir` | cm²/g | Infrared opacity for Guillot temperature profile |
| `gamma_guillot` |, | Ratio of optical to infrared opacity for Guillot profile |
| `limb_darkening_coeffs` |, | Quadratic limb darkening coefficients [u1, u2] for [BATMAN](https://lkreidberg.github.io/batman/docs/html/index.html) transit model |
| `ra_deg` | degrees | Right ascension (used for BERV calculation) |
| `dec_deg` | degrees | Declination (used for BERV calculation) |

---

## 2. `planet`

```json
"planet": {
  "name": "HD189733b",
  "parameter_file": "planet_params/HD189733b.json"
}
```

| Field | Description |
|---|---|
| `name` | Planet name. Must match the `name` field inside the parameter file. Used for output directory structure. |
| `parameter_file` | Path to the planet JSON file, relative to the repo root. |

---

## 3. `instrument`

```json
"instrument": {
  "name": "ANDES_YJHK",
  "observatory": "paranal",
  "pixels_per_resolution_element": 2.5,
  "order_indices": [],
  "split_detectors": false,
  "convolve_to_resolution": true
}
```

| Field | Valid values | Description |
|---|---|---|
| `name` | See table below | Instrument name. Controls the wavelength grid, resolving power, and number of spectral orders. See [Supported instruments](#16-supported-instruments). |
| `observatory` | `"paranal"`, `"lasilla"` | Observatory site for BERV computation and sky models. Set automatically by the instrument module. Override only if you need to force a different site. |
| `pixels_per_resolution_element` | typically 2 to 4 | Number of detector pixels per resolution element, denoted m. Used when convolving model spectra to instrument resolution, and to convert the ETC SNR (given per resolution element) to a per-pixel SNR via SNR<sub>pixel</sub> = SNR<sub>resel</sub> / √m. ANDES: 2.5. CARMENES NIR: 3.3. CARMENES VIS: 3.3. |
| `order_indices` | `[]` or list of ints | Spectral order indices to simulate. Empty list `[]` means all orders for the selected instrument band. Use `[0]` for single-order instruments. Example: `[0, 1, 5, 10]` simulates only those four orders (useful for fast testing). |
| `split_detectors` | `true`, `false` | If `true`, treat each detector half as an independent order. Used for raw CARMENES data where each echelle order spans two detectors. |
| `convolve_to_resolution` | `true`, `false` | If `true`, convolve the pRT model spectrum to the instrument resolution before injection. Should generally be `true`. |

---

## 4. `observation`

```json
"observation": {
  "event_type": "transit",
  "flag_event": "full_event",
  "pre_event_hours": 0.0,
  "post_event_hours": 0.0,
  "n_nights": 1,
  "different_nights": false,
  "exposure_time_seconds": 30.0,
  "readout_time_seconds": 6.0,
  "overhead_time_seconds": 0.0,
  "specific_event": false,
  "specific_T0_bjd": null,
  "use_real_data": false,
  "noise_scaling_factor": 1.0,
  "simulate_planet": true,
  "external_planet_model": false,
  "external_planet_model_file": "",
  "signal_uses_light_curve": true,
  "scale_injection": 1.0,
  "significant_eccentricity": false,
  "berv_kms": 0.0,
  "noiseless": false,
  "first_night_noiseless": false,
  "add_throughput_variations": true,
  "throughput_seed": null,
  "exposure_time_seconds_per_night": null,
  "mask_v_rotsini": false
}
```

| Field | Valid values | Description |
|---|---|---|
| `event_type` | `"transit"`, `"dayside"` | Type of observation. `"transit"`: primary transit (transmission spectroscopy). `"dayside"`: secondary eclipse / dayside emission. |
| `flag_event` | `"full_event"`, `"pre"`, `"post"` | Which part of the event to simulate. `"full_event"`: full event including baseline on both sides (controlled by `pre_event_hours` / `post_event_hours`). `"pre"`: baseline before the event only. `"post"`: baseline after the event only. |
| `pre_event_hours` | float ≥ 0 | Hours of baseline observation before the event starts (before ingress for transit). **When set to `0.0` (the default), the simulator automatically substitutes half the transit duration** as the pre-event window. Set an explicit positive value to override this. |
| `post_event_hours` | float ≥ 0 | Hours of baseline observation after the event ends (after egress for transit). Same auto-substitution behaviour as `pre_event_hours`: `0.0` → half transit duration. |
| `n_nights` | int ≥ 1 | Number of transits (or eclipses) to co-add. Multiple nights use the same config but different epochs, spaced by the orbital period. |
| `different_nights` | `true`, `false` | If `true`, each night has its own number of spectra, airmass profile, and timing, enabling genuinely distinct simulated nights. Supported for both **ANDES** and **CARMENES_NIR**. Three sub-modes exist. **Sub-mode A, per-night reference files:** provide `julian_date_N.fits`, `snr_N.fits`, and `airmass_N.fits` for each night. The length of `julian_date_N.fits` defines `n_spectra[n]` for that night. This works for both real data (`use_real_data: true`) and fully synthetic simulations, the files may be synthetically generated or taken from real observations. When different nights have different numbers of spectra, arrays are padded to the maximum length with NaN and each night is processed on its live rows only. **Sub-mode B, `exposure_time_seconds_per_night`:** provide a list of per-night exposure times. The simulator regenerates each night's JD grid synthetically with that cadence, without requiring JD files. BERV is broadcast as the scalar `berv_kms`. **Sub-mode C, fully synthetic (ANDES default):** when `specific_event: false` and no JD files are present, the simulator synthesises per-night JD grids automatically, placing each night at successive transit epochs (T₀ + n × P) with a parabolic airmass model. This is the standard mode for ANDES detectability studies. If `false`, all nights are identical copies of night 1, co-added coherently (S/N scales as √N). See [docs/tutorial.md §5](tutorial.md#tutorial-5-multiple-nights) for the required file naming conventions. |
| `exposure_time_seconds` | float > 0 | Individual exposure time in seconds. |
| `readout_time_seconds` | float ≥ 0 | CCD/detector readout time per exposure (added to exposure time to get cadence). |
| `overhead_time_seconds` | float ≥ 0 | Additional per-exposure overhead (guiding, nodding, etc.). |
| `specific_event` | `true`, `false` | If `true`, set the transit mid-time to `specific_T0_bjd`. Use this when simulating a specific observed transit with a known epoch. |
| `specific_T0_bjd` | BJD float or `null` | Required if `specific_event: true`. The exact BJD of the transit centre. |
| `use_real_data` | `true`, `false` | If `true`, load observed spectra and run their preparation and analysis. Requires per-order observation files `observations_night_{b}_order_{K}.fits` (shape: `n_spectra × n_pixels`) and BERV files `observations_berv_{b}.fits` in the `reference_night/` directory. For `different_nights: false`, the suffix `_{b}` is omitted. Setting `use_real_data: false` (the default) uses only the noise/SNR reference files and generates all spectra synthetically. |
| `noise_scaling_factor` | float > 0 | Global multiplicative scaling applied to the noise. 1.0 = nominal. 0.5 = half the noise (2× SNR). 2.0 = double the noise. |
| `simulate_planet` | `true`, `false` | If `true`, inject the atmospheric signal into the simulated spectra. If `false`, simulate a star-only (null test). |
| `external_planet_model` | `true`, `false` | If `true`, read the planet atmosphere spectrum from an external file. |
| `external_planet_model_file` | path string | Path to the external planet model file. Only used if `external_planet_model: true`. |
| `signal_uses_light_curve` | `true`, `false` | If `true`, modulate the planet signal with the BATMAN transit light curve (depth varies during ingress/egress). If `false`, use a constant depth. |
| `scale_injection` | float > 0 | Multiplicative scale factor applied to the injected planet signal. 1.0 = true amplitude. 0.5 = half the signal. |
| `significant_eccentricity` | `true`, `false` | If `true`, use a full eccentric orbit model for the planet velocity. Set to `false` for circular orbits (eccentricity = 0). |
| `berv_kms` | float | Barycentric Earth Radial Velocity correction in km/s. If 0.0, the simulator computes the BERV from the observation time, coordinates, and observatory location. Set a fixed value to override. |
| `noiseless` | `true`, `false` | If `true`, produce fully noiseless simulated spectra (photon noise = 0). Useful for testing the signal model in isolation. |
| `first_night_noiseless` | `true`, `false` | If `true`, only the first night is noiseless. Subsequent nights have noise. Useful for testing signal injection. |
| `add_throughput_variations` | `true`, `false` | If `true`, add realistic time-varying throughput variations (airmass + instrumental drift) to the simulated spectra. |
| `throughput_seed` | non-negative int or `null` | Optional throughput base seed. Default `null` preserves unseeded draws. The public simulator derives separate streams from the physical order ID and zero-based night index using `SeedSequence([base_seed, order_id, night_index, 1])`. Independent of `noise.noise_seed`. Ignored when throughput is disabled or the run is noiseless. Tutorial 1 sets both base seeds to `12345`. |
| `exposure_time_seconds_per_night` | list of float or `null` | Per-night exposure times in seconds. Only valid when `different_nights: true`. When set, the JD grid for each night is regenerated with that night's own DIT, enabling different cadences per night without requiring separate JD FITS files. Length must equal `n_nights`. `null` (default) uses the same `exposure_time_seconds` for all nights. |
| `mask_v_rotsini` | `true`, `false` | If `true`, mask the CCF velocity range within ±v_rot_sini of Vsys=0 to avoid the stellar line broadening region. |

---

Additional observation controls:

| Field | Meaning |
|---|---|
| `use_accurate_berv` | Calculate barycentric corrections from exposure times, observatory and the planet file's sky coordinates. Check the run messages for any fallback to `berv_kms`. |
| `helium_transmission_spectroscopy` | Select the direct-helium workflow. The corresponding atmosphere and preparation blocks are described in Tutorials 12 and 12b. |

## 5. `noise`

```json
"noise": {
  "noise_choice": "SNR",
  "fixed_snr": null,
  "use_mean_snr": false,
  "snr_correction": 0.0,
  "noise_seed": 12345
}
```

| Field | Valid values | Description |
|---|---|---|
| `noise_choice` | `"SNR"`, `"sig"` | How to determine the per-pixel noise level. `"SNR"`: use the SNR computed from the ETC or instrument model (time-varying, wavelength-dependent). `"sig"`: use the per-pixel uncertainty array from a real-data reference file (only meaningful when `use_real_data: true`). |
| `fixed_snr` | float or `null` | When set to a positive number, override all SNR files and use this constant SNR value for every pixel and every exposure. `null` = use the SNR from the ETC file or instrument model. Typical values: 100 to 300 for bright targets with large telescopes. |
| `use_mean_snr` | `true`, `false` | If `true`, replace the time- and wavelength-varying SNR cube with its mean value. Produces more uniform noise. |
| `snr_correction` | float | Additive offset applied to the SNR at all wavelengths and times. 0.0 = no correction. Positive values increase SNR. |
| `noise_seed` | int or `null` | Integer seed for the Gaussian noise random number generator. Using the same seed reproduces identical noise realisations across runs, essential for reproducibility. Set to `null` to draw a different random noise realisation every run. Default: `12345`. |
| `carmenes_etc_file` | path or `null` | CARMENES ETC response bundle. Selects maximum-per-order anchors with the ANDES analytic blaze and relative exposure-dependent telluric scaling. Default `null` retains the existing S/N-file route. Requires synthetic CARMENES, `noise_choice: "SNR"`, `fixed_snr: null` and `use_mean_snr: false`. |

### CARMENES ETC maximum anchors

Select the ETC input in the simulation JSON:

```json
"noise": {
  "noise_choice": "SNR",
  "fixed_snr": null,
  "use_mean_snr": false,
  "snr_correction": 0.0,
  "noise_seed": 12345,
  "carmenes_etc_file": "inputs/CARMENES_VIS/HD189733b/CARMENES_VIS_ETC_HD189733b.npz"
}
```

EXoPLORE takes the maximum S/N per resolution element and its wavelength from
each downloaded order curve. It converts to pixels using the local native
sampling, {math}`m=\lambda/(R\,\Delta\lambda_{\rm pixel})`, then applies the
ANDES sinc-squared blaze and the exposure's telluric transmission relative
to the anchor wavelength:

```{math}
S_{t,\lambda}=\frac{S_{\mathrm{ETC,max}}}{\sqrt{m_\lambda}}
\sqrt{\frac{B_\lambda}{B_{\mathrm{anchor}}}}
\sqrt{\frac{T_{t,\lambda}}{T_{t,\mathrm{anchor}}}}.
```

For multiple supplied airmasses, the resulting S/N is interpolated between
the ETC conditions for each exposure. The uncertainty follows the existing
ANDES convention, {math}`\sigma=1/S`, in continuum-relative flux units, with the
runner's S/N floor of 1. `observation.noise_scaling_factor` multiplies that
uncertainty and `noise.snr_correction` is added afterward on this route.
Stellar-line scaling and throughput-jitter noise scaling retain the ANDES defaults.

The ETC maximum includes its atmospheric and observing conditions. We adopt
our simulated telluric spectrum for the relative wavelength dependence because
the ETC reference transmission is unavailable. This is the atmospheric-reference
approximation used by this option. The complete ETC curves remain in the bundle
for inspection of broad sensitivity variations.

The same option is available to the CARMENES helium simulation. An incompatible
channel, exposure duration or airmass produces an error. A single-airmass bundle
supports exposures at that airmass. See the
[input instructions](input_files.md#carmenes-etc-maximum-anchor-inputs) for preparing a bundle.

#### Direct sodium transmission uncertainties

The direct sodium experiment uses
`observation/sodium_noise.py:sodium_photon_uncertainty` to derive
photon-dominated errors from the noiseless expected flux. The flux already
contains the stellar lines, planetary transit, tellurics and throughput,
in out-of-transit stellar-continuum units. With the ETC-derived continuum
S/N above and transmission {math}`T`, the effective photon response is:

```{math}
A=\frac{S_{\rm cont}^2}{T},\qquad
\sigma_f=\sqrt{\frac{f}{A}},\qquad
S_{\rm local}=\sqrt{fA}.
```

This counts attenuation once. Dividing the observation by a fitted telluric
transmission also divides its uncertainty by that transmission. Seidel22
then propagates these errors through continuum normalization, the shared
stellar reference and planetary-frame coaddition.

The effective response adopts a photon-dominated approximation to the ETC
budget. Separate background and detector terms require their own measured
or supplied variance budget. The CCF simulator and helium routes retain
their existing uncertainty prescriptions. Their validation is separate from
this sodium experiment.

---

## 6. `atmosphere`

The atmosphere section contains up to six sub-blocks. Each describes a separate atmospheric model computed with petitRADTRANS.

| Sub-block | Purpose |
|---|---|
| `planet_model` | The full planetary atmosphere injected into the simulated spectra |
| `ccf_template` | The template spectrum used for cross-correlation (can differ from the injected model to test template mismatch) |
| `morning_day` | Morning limb, dayside (for limb-asymmetric transit simulations) |
| `morning_night` | Morning limb, nightside |
| `evening_day` | Evening limb, dayside |
| `evening_night` | Evening limb, nightside |

For the `gradual`, `asymmetric` and `simplified_step` modes, the morning and evening models are read from `morning_day` and `evening_day`. The `_night` sub-blocks are unused in these modes.

**Which sub-blocks are used** depends on `limb_asymmetries` and `limb_divisions`:

```json
"limb_asymmetries": true,
"limb_divisions": "gradual",
"cc_with_true_model": false
```

| Field | Valid values | Description |
|---|---|---|
| `limb_asymmetries` | `true`, `false` | If `true`, use distinct morning and evening limb models. If `false`, use only `planet_model` for the whole limb. |
| `limb_divisions` | `"gradual"`, `"asymmetric"`, `"simplified_step"` | How to weight morning vs evening limb contributions over the transit. `"gradual"` (**default**, recommended for most targets): smooth cubic transition, morning limb dominates ingress, equal 50/50 mix at full transit, evening limb dominates egress. `"asymmetric"` (use for an ultra-hot Jupiter like WASP-76 b): asymmetric, morning limb dominates the first quarter of full transit, cubic handover completes at mid-transit, evening limb dominates the second half and egress. Calibrated for planets with extreme day-to-night contrast and poor heat redistribution. `"simplified_step"`: hard step-function, pure morning during ingress, 50/50 during full transit, pure evening during egress. Use for reference tests and simplified scenarios where clean physical interpretation of morning vs evening contributions is needed. |
| `cc_with_true_model` | `true`, `false` | If `true`, use the same model for CCF template as for injection (`planet_model`). If `false`, use the separate `ccf_template` block. |

### 13. Atmosphere model block reference

All six atmosphere sub-blocks share the same structure:

```json
{
  "species": ["H2", "He", "H2O", "CH4", "NH3", "CO", "H2S", "HCN", "Fe", "Ca"],
  "use_easychem": true,
  "metallicity_wrt_solar": 0.53,
  "carbon_to_oxygen_ratio": 0.41,
  "mass_fractions": [0.75, 0.2432, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "mean_molecular_weight": 2.33,
  "reference_pressure_bar": 0.01,
  "cloud_pressure_bar": null,
  "pressure_min_bar": 1e-6,
  "pressure_max_bar": 100.0,
  "pressure_grid_size": 100,
  "isothermal": false,
  "isothermal_temperature_K": null,
  "equilibrium_temperature_K": 1170.0,
  "kappa_ir": 0.01,
  "gamma_guillot": 0.4,
  "two_point": false,
  "two_point_pressures_bar": [1.259, 0.00178],
  "two_point_temperatures_K": [1750.0, 520.0],
  "wind_velocity_kms": 0.0
}
```

| Field | Unit | Description |
|---|---|---|
| `species` |, | List of chemical species to include. Must use petitRADTRANS opacity names. Always include `"H2"` and `"He"` as bulk gas. Common detectable species: `"H2O"`, `"CO"`, `"CH4"`, `"NH3"`, `"H2S"`, `"HCN"`, `"CO2"`, `"C2H2"`. Atomic species: `"Fe"`, `"Ca"`, `"Na"`, `"K"`. The list order must match `mass_fractions`. |
| `use_easychem` |, | If `true`, compute equilibrium chemistry mass fractions using [EasyChem](https://easychem.readthedocs.io) (based on metallicity and C/O ratio). For all species **except H₂ and He**, the `mass_fractions` values in the JSON are overridden by EasyChem output. H₂ and He mass fractions are always taken from the JSON `mass_fractions` list and are not computed by EasyChem. If `false`, use the `mass_fractions` values directly for all species. |
| `metallicity_wrt_solar` | dex | Atmospheric metallicity relative to solar. Only used when `use_easychem: true`. Typical range: -1 to +3. Solar = 0. |
| `carbon_to_oxygen_ratio` |, | Atmospheric C/O ratio (solar ≈ 0.55). Only used when `use_easychem: true`. |
| `mass_fractions` |, | Manual mass fractions for each species in `species` (same order). Must sum to exactly 1. Used when `use_easychem: false`, or for H2 and He even when EasyChem is active. |
| `mean_molecular_weight` | amu | Mean molecular weight of the atmosphere in atomic mass units. Solar H/He dominated: ~2.33. Higher metallicity or heavy species: up to ~3 to 5. |
| `reference_pressure_bar` | bar | Reference pressure level for the planet radius (where R = R_planet from the parameter file). Typically 0.01 bar. |
| `cloud_pressure_bar` | bar or `null` | Pressure of a grey cloud deck. Everything below this pressure is opaque. `null` = cloud-free. |
| `pressure_min_bar` | bar | Top of atmosphere pressure (lowest pressure). Typically 1e-6 bar. |
| `pressure_max_bar` | bar | Bottom of atmosphere pressure (highest pressure). Typically 100 bar. Must be > `pressure_min_bar`. |
| `pressure_grid_size` | int | Number of pressure levels in the log-spaced grid. 100 is standard. Increase for higher accuracy at the cost of speed. |
| `isothermal` | `true`, `false` | If `true`, use a uniform temperature profile at `isothermal_temperature_K`. |
| `isothermal_temperature_K` | K or `null` | Temperature for isothermal profile. Required if `isothermal: true`. |
| `equilibrium_temperature_K` | K | Planet equilibrium temperature used in the Guillot temperature profile. Also used when `two_point: false` and `isothermal: false`. |
| `kappa_ir` | cm²/g | Infrared opacity parameter for the Guillot profile. Typical value: 0.01. |
| `gamma_guillot` |, | Ratio of optical to infrared opacity for Guillot profile. Typical range: 0.1 to 1.0. |
| `two_point` | `true`, `false` | If `true`, interpolate the temperature profile between two (pressure, temperature) pairs. |
| `two_point_pressures_bar` | bar | Two pressure anchor points for the `two_point` profile, as `[P_deep, P_high]`. |
| `two_point_temperatures_K` | K | Temperatures at the two anchor points, as `[T_deep, T_high]`. |
| `wind_velocity_kms` | km/s | Atmospheric wind velocity. Positive = tailwind (redshift), negative = headwind (blueshift). Shifts the planetary spectral lines by a constant offset. 0.0 = no wind. Typical values for hot Jupiters: -5 to +5 km/s. |

**petitRADTRANS species names** (commonly used):

| Species | pRT name | Notes |
|---|---|---|
| Water | `H2O` | Strong NIR absorption |
| Carbon monoxide | `CO` | 2.3 µm bandhead, K-band |
| Methane | `CH4` | Strong in cool planets (< ~1200 K) |
| Ammonia | `NH3` | Moderate. Cool planets |
| Hydrogen sulfide | `H2S` | Moderate NIR absorption |
| Hydrogen cyanide | `HCN` | Detectable in warm C-rich atmospheres |
| Carbon dioxide | `CO2` | Weak in H/He-dominated. Detectable at high C/O |
| Acetylene | `C2H2` | Only significant at very high C/O |
| Iron | `Fe` | Ultra-hot Jupiters (> ~2000 K) |
| Calcium | `Ca` | Ultra-hot Jupiters |
| Sodium | `Na` | Optical. Not NIR |
| Potassium | `K` | Optical. Not NIR |
| Hydrogen (bulk) | `H2` | Always include, bulk gas + CIA opacity |
| Helium (bulk) | `He` | Always include, bulk gas |

---

## 7. `tellurics`

```json
"tellurics": {
  "skycalc_wavelength_limits_um": [0.9, 2.5],
  "include_tellurics": true,
  "use_full_skycalc": false,
  "use_accurate_airmass": true,
  "airmass_evolution": "up_and_down",
  "airmass_limits": [1.4, 1.7],
  "constant_pwv": true,
  "pwv_mm": 10.0,
  "pwv_mm_per_night": null,
  "reference_airmass": 1.0,
  "reference_telluric_file": "inputs/ANDES/HD189733b/tellurics/tell_ref_airmass_1.0.fits",
  "mask_threshold": 0.2,
  "safety_window_pixels": 7
}
```

`tellurics.skycalc_wavelength_limits_um` sets the **SkyCalc download range**
in microns, inside this same simulation JSON. It is unrelated to instrument
convolution. Instrument input files define the wavelength grid. Use a finite
positive pair `[minimum, maximum]` covering all selected orders with margin.
The full YJHK example uses `[0.9, 2.5]`. The UBV example uses `[0.34, 0.65]`.
Missing or `null` uses `[0.95, 2.45]`, which may be too narrow for your
instrument. The field survives configuration
loading/saving. The downloader uses this interval for its SkyCalc query.
EXoPLORE checks that the returned file covers the selected wavelengths. See the {ref}`input guide <skycalc-reference-download>`.

### Three telluric modes

EXoPLORE supports three levels of telluric treatment, selected by combining `include_tellurics` and `use_full_skycalc`. The choice directly affects how realistic the atmospheric systematics are, which spectral regions are masked, and how the pipeline handles the residuals. In the following we describe each mode in turn.

**Mode 1, Airmass-scaled reference** (`include_tellurics: true`, `use_full_skycalc: false`, **default and recommended**): loads a single reference telluric spectrum computed at `reference_airmass` and scales it to each exposure using the Beer-Lambert-Bouguer law (Bouguer 1729, Lambert 1760, Beer 1852):

```
T(X) = T_ref ^ (X / X_ref)
```

where `T_ref` is the reference transmission at airmass `X_ref` and `X` is the airmass at each exposure. This captures the dominant effect, telluric lines deepening as the target moves towards the horizon. PWV is implicitly fixed at the value used when generating the reference file. Requires only one FITS file per instrument/target, and is consequently the appropriate choice for most simulations.

**Mode 2, Full per-exposure telluric evolution** (`include_tellurics: true`, `use_full_skycalc: true`): loads a separate telluric transmission spectrum for each exposure (`tell_spec_{n}.fits`), each computed at its individual airmass **and** PWV. Both quantities vary per exposure, making this the most physically accurate treatment. In particular, this mode is recommended when the telluric evolution during the night is consequential, long transits, high-airmass targets, variable atmospheric conditions. Generate the per-exposure spectra with `scripts/generate_skycalc_inputs.py` (see [docs/input_files.md](input_files.md#4-telluric-reference-spectra)), or supply them from another source, provided they are named `tell_spec_{n}.fits` and placed in the correct directory, the simulator reads whatever FITS files it finds there.

**Mode 3, No tellurics** (`include_tellurics: false`): telluric multiplication is skipped entirely, producing a clean absence of atmospheric contamination. The masking and SNR columns are still computed from the noise model, but far fewer pixels are flagged. This allows us to isolate the pure planetary signal, test the pipeline with no atmospheric baseline, or verify the CCF template matching under ideal conditions.

### Field reference

| Field | Valid values | Description |
|---|---|---|
| `include_tellurics` | `true`, `false` | Master switch. `true` = apply tellurics (Mode 1 or 2 depending on `use_full_skycalc`). `false` = no tellurics at all (Mode 3). |
| `use_full_skycalc` | `true`, `false` | `false` (default) = Mode 1: airmass-scaling from a single reference file. `true` = Mode 2: load per-exposure telluric FITS files (`tell_spec_{n}.fits`), each with individual airmass and PWV. Generate with `python scripts/generate_skycalc_inputs.py` or supply from any source. See [docs/input_files.md](input_files.md#4-telluric-reference-spectra). |
| `use_accurate_airmass` | `true`, `false` | If `true`, compute the airmass from the observation time, target coordinates, and observatory location. If `false`, use the values in `airmass_limits` directly. |
| `airmass_evolution` | `"up_and_down"`, `"up"`, `"down"`, `"constant"` | Shape of the airmass time series when `use_accurate_airmass: false`. `"up_and_down"`: airmass decreases to a minimum at transit midpoint then increases (target transits the meridian). `"up"`: monotonically increasing. `"down"`: monotonically decreasing. `"constant"`: flat at the mean of `airmass_limits`. |
| `airmass_limits` | `[min, max]` | Minimum and maximum airmass during the observation. Used to construct the airmass time series when `use_accurate_airmass: false`. |
| `constant_pwv` | `true`, `false` | Controls PWV handling for both Mode 2 and Mode 3. `true`: all exposures use the same `pwv_mm` value (saved to `Fixed_PWV/pwv_values.fits`). `false`: PWV varies randomly around `pwv_mm` within the [SkyCalc](https://skycalc-ipy.readthedocs.io) grid (saved to `Variable_PWV/pwv_values.fits`). **In Mode 1 (airmass scaling), PWV does not affect the computed telluric spectra**, its value is recorded in the FITS file and used when generating per-exposure files for Mode 2. |
| `pwv_mm` | mm | Precipitable water vapour in mm. Reference value used when `constant_pwv: true` (all exposures), or as the centre of the random draw when `constant_pwv: false`. Typical range: 1 to 30 mm. Dry site (Paranal): 2 to 10 mm. Must be a value on the SkyCalc PWV grid: `[0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0]`. |
| `pwv_mm_per_night` | list of mm or `null` | Per-night PWV values for `different_nights: true` simulations. When set, night `n` uses `pwv_mm_per_night[n]`. Length must equal `n_nights`. Used by the simulator when loading per-night SkyCalc files (Mode 2) and by `scripts/generate_skycalc_inputs.py --night N` when generating those files. Example: `[2.5, 3.5]` for a two-night run with different atmospheric conditions. `null` (default) uses `pwv_mm` for all nights. |
| `reference_airmass` | float | The airmass at which `reference_telluric_file` was computed. Typically 1.0. Used in the Beer-Lambert scaling formula `T(X) = T_ref^(X / X_ref)` for Mode 1. |
| `reference_telluric_file` | path | Path to the FITS file containing the reference telluric transmission spectrum (columns `lam` in nm and `trans` as fractional transmission). Required for Mode 1. If left empty, auto-constructed as `{inputs_dir}tellurics/tell_ref_airmass_{reference_airmass:.1f}.fits`. The standard convention is to place this file in a `tellurics/` subdirectory inside `inputs_dir`, both the bundled ANDES and CARMENES inputs follow this layout. See [docs/input_files.md](input_files.md#4-telluric-reference-spectra). |
| `mask_threshold` | float 0 to 1 | Fractional transmission below which a pixel is flagged as a deep telluric and masked. 0.2 means pixels with < 20% transmission are masked. |
| `safety_window_pixels` | int | Number of pixels around each masked telluric region to also mask (wing buffer). Prevents unmasked contamination near telluric edges. |

---

`tellurics.airmass_limits_per_night` supplies a separate airmass interval for
each night when the selected synthetic observing setup uses those limits.
The [input guide](input_files.md) describes the per-exposure and per-night files.

## 8. `pipeline`

```json
"pipeline": {
  "name": "BL19",
  "sysrem_iterations": 4,
  "snr_mask_threshold": 10.0,
  "prepare_template": true,
  "optimize_sysrem_order_by_order": false,
  "optimize_criterion": "DeltaSigma",
  "sysrem_delta_sigma_threshold": 0.01,
  "kp_vrest_injection": [0.0, 0.0],
  "inject_scale_factor": 1.0,
  "sysrem_robust_halt": false
}
```

Four pipeline recipes are available. They fall into two groups depending on whether they use SYSREM for systematics removal:

**No SYSREM**, `"BL19"`, `"Blain24"`: systematics are removed by telluric correction and spectral fitting, not by iterative SYSREM. The `sysrem_iterations` field is ignored for these pipelines.

**SYSREM-based**, `"ASL19"`, `"Gibson22"`: systematics removal relies on SYSREM iterations. The `sysrem_iterations` field controls how many iterations are applied.

| Field | Valid values | Description |
|---|---|---|
| `name` | `"BL19"`, `"Blain24"`, `"ASL19"`, `"Gibson22"`, `"Seidel22"` | Data reduction pipeline to use. `"BL19"`: [Brogi & Line (2019)](https://doi.org/10.3847/1538-3881/aaffd3)-style pipeline, normalisation, telluric mask, telluric correction, noisy-column masking. No SYSREM. `"Blain24"`: Blain, Sánchez-López & Mollière (2024, AJ, 167, 179), throughput fit removal and telluric fit removal. No SYSREM. `"ASL19"`: [Sánchez-López et al. (2019)](https://doi.org/10.1051/0004-6361/201936084), BL19 normalisation + telluric-window mask + SYSREM iterations. `"Gibson22"`: Gibson et al. (2022), out-of-transit normalisation + SYSREM (blaze correction disabled by default). |
| `sysrem_iterations` | int ≥ 1 | Number of SYSREM iterations. Only used by `"ASL19"` and `"Gibson22"`. Ignored for `"BL19"` and `"Blain24"`. More iterations remove more correlated systematics but also attenuate the planet signal. Typical range: 2 to 8. Start with 4 and optimise. |
| `snr_mask_threshold` | float > 0 | SNR threshold below which columns (wavelength channels) are masked before SYSREM. Pixels with SNR < threshold are excluded. Typical: 10. |
| `prepare_template` | `true`, `false` | If `true`, apply the same filtering to the CCF template as was applied to the data, so that the template correctly represents the distorted planetary signal seen in the residuals. **Always set `true` for rigorous CCF computation and retrievals.** How the template is filtered depends on the pipeline, see the note below. |
| `optimize_sysrem_order_by_order` | `true`, `false` | If `true`, choose the optimal number of SYSREM iterations independently for each spectral order based on `optimize_criterion`. Slower but can improve sensitivity. |
| `optimize_criterion` | `"DeltaSigma"`, `"Maximum"`, `"Max_Diff"` | Criterion for selecting the optimal SYSREM iteration count per order when `optimize_sysrem_order_by_order: true`. `"DeltaSigma"` (**default, recommended**): stop when the fractional decrease in residual standard deviation falls below `sysrem_delta_sigma_threshold`, model-independent and avoids over-subtraction ([Parker et al. 2025](https://doi.org/10.1093/mnras/staf469)). `"Maximum"`: choose the iteration that maximises the CCF S/N. `"Max_Diff"`: choose the iteration that maximises the S/N difference between consecutive iterations. |
| `sysrem_delta_sigma_threshold` | float (default 0.01) | Fractional σ-drop threshold for the `"DeltaSigma"` halting criterion. SYSREM stops when (σ[k-1] - σ[k]) / σ[k-1] < threshold. Only used when `optimize_criterion: "DeltaSigma"`. |
| `kp_vrest_injection` | `[Kp, Vrest]` km/s | Location in Kp-Vsys space at which to inject an additional test signal for cross-injection tests. `[0.0, 0.0]` = no injection. |
| `inject_scale_factor` | float | Scale factor for the additional cross-injection signal. |
| `sysrem_robust_halt` | `true`, `false` | If `true`, stop SYSREM early if the residual variance increases (robust halt criterion). Prevents over-subtraction. |

Additional preparation controls:

| Field | Meaning |
|---|---|
| `continuum_method` | Cheverall26 continuum estimator: `maxima` (default), `cm2024` or `polyfit`. |
| `detrend_method` | `sysrem` (default) or `pca` in the preparation branches that support this switch. Component count is set by `sysrem_iterations`. |
| `czesla2024` | Direct CRIRES+ helium settings, defined in `config/czesla2024.py`. See Tutorial 12. |
| `seidel22` | Stellar-reference transmission settings for `name: "Seidel22"`. Continuum and stellar-core windows use vacuum nm in the stellar frame. See below. |
| `carmenes_helium` | Direct CARMENES helium normalization and averaging settings, defined in `config/carmenes_helium.py`. See Tutorial 12b. |
| `palle2020` | Observed CARMENES input and correction settings used by Tutorial 12b. |
| `oh_treatment` | Explicit OH treatment for the direct-helium routes. Valid choices and required inputs are checked in `config/helium_oh.py`. |

### Seidel22: stellar-reference transmission spectra

Select `pipeline.name: "Seidel22"` to align the stellar spectra, normalise
their continuum and divide by a common out-of-transit reference. Exposures
receive weights proportional to their continuum S/N. The direct array
interface also aligns the transmission spectra to the planet and combines
the selected in-transit exposures. The reference is
[Seidel et al. (2022), MNRAS, 513, L15](https://doi.org/10.1093/mnrasl/slac027).

Continuum levels and exposure S/N are measured after alignment, using the
same wavelength channels on the common stellar-rest-frame grid.

Choose windows for the line interval being analysed. This example illustrates
the fields for a sodium interval. Inspect the continuum and stellar cores
before choosing windows for your data.

```json
"pipeline": {
  "name": "Seidel22",
  "prepare_template": false,
  "seidel22": {
    "continuum_windows_nm": [[589.2, 589.4], [589.8, 589.95]],
    "stellar_mask_windows_nm": [],
    "minimum_continuum_pixels": 10,
    "input_telluric_corrected": false,
    "input_sky_subtracted": false
  }
}
```

All windows use **vacuum nm in the stellar frame**. Supply explicit
`stellar_mask_windows_nm` to exclude low-S/N stellar cores. The direct API
retains these masks as they move through the planet frame. The CCF runner
uses wavelength columns valid in every exposure and therefore takes the
union of the invalid columns after returning to the observer frame.

The current synthetic CCF route requires `tellurics.include_tellurics: false`.
Observed input spectra must already have their telluric absorption and sky
emission corrected. Set both input correction flags to `true` after those
operations. These flags record the state of the supplied data. The direct
sodium runner calls Molecfit before Seidel22. Choose stellar correction through
the `rm_clv` block. Retrieval and SYSREM optimisation are rejected
for this preparation.

For direct sodium or Hα transmission spectra, call
`exoplore.pipelines.seidel22.prepare_transmission()`. Supply the observed
stellar velocity, the planetary velocity relative to the star and explicit
reference/transit exposure indices. Positive velocities indicate recession.
The returned dataclass contains the reference, exposure S/N weights, stellar
and planetary transmission matrices, coadd and coverage per wavelength.
The default uncertainty arrays carry diagonal propagation. The conditional
coadd variance holds the shared reference and continuum estimates fixed.
Set `calculate_covariance=True` in the array call to obtain
`result.coadd_covariance`, a sparse matrix including the shared reference
and both wavelength interpolations. Exclude missing coadd pixels using
`np.isfinite(result.planet_coadd)`. Continuum estimates, S/N weights and any
upstream telluric or RM+CLV model remain fixed in this propagation.

To propagate continuum fitting and exposure-weight variability, also pass
`noise_resamples=500` and an explicit `noise_resample_seed` to the array call.
Each native-pixel Gaussian draw is aligned, normalised, divided by its own
shared reference and combined in the planet frame. The resulting sample
covariance replaces `coadd_covariance` and `lightcurve_covariance`. The
prepared spectrum and light curve remain those of the supplied observation.
`result.covariance_method` identifies the uncertainty method, and
`result.noise_resamples` records the number of draws. Check convergence with
the number of draws for the measurement being reported.
The sample covariance has rank at most `noise_resamples - 1`, which matters
when inverting a covariance for a large wavelength interval.

For a simulation, `noise_expectation` can supply the noiseless flux on the
same input wavelength grid and flux scale. With this argument omitted, the
measured flux supplies the mean for a parametric bootstrap. Supplied errors
and upstream telluric and stellar models remain fixed. Uncertainty in those
models requires separate experiments. Intermediate diagonal variance arrays
retain their conditional interpretation. The resampling options belong to
the direct Python interface. The integrated direct-sodium runner reads
`noise_resamples` and `noise_resample_seed` from `pipeline.seidel22` and
supplies its noiseless expectation for the draws.

Check the preparation transfer function with synthetic injection and recovery
at the relevant S/N, including a control with atmospheric absorption removed.

If interference wiggles are present, `remove_wiggles()` fits a sinusoidal
baseline to explicitly selected continuum windows of the final transmission
spectrum. Supply period bounds in nm and keep planetary absorption outside
the fitting windows. The function returns the fitted baseline for inspection
and errors conditional on that fit. It is an optional separate operation.

The array preparation accepts either line region. Generating a physical
planetary Hα spectrum additionally requires a model for the excited hydrogen
population. Selecting this preparation does not supply that atmospheric model.

### Template preparation and SYSREM distortions

SYSREM removes dominant telluric and stellar structure and also changes the underlying planetary signal as a function of time and wavelength. Preparing the template with the corresponding operator accounts for this distortion when comparing it with the residual data. Use **`prepare_template: true` for the fitted-normalisation and SYSREM pipelines below**. Direct stellar-reference preparation with `Seidel22` requires `prepare_template: false`. The implementation differs by pipeline:

- **`BL19` / `Blain24`**, the template is normalised and continuum-corrected in the same way as the data (polynomial fits, telluric correction). No SYSREM is involved, so the template retains full spectral line structure.

- **`ASL19` / `Gibson22`**, the template must be filtered to match what SYSREM does to the data, but running SYSREM directly on the template would destroy its time-varying in-transit structure. To that end, EXoPLORE uses the fast model-filtering technique of Gibson et al. (2022, MNRAS 512, 4618 to 4638): during the SYSREM run on the data, the time-eigenvectors **U** (one per iteration) are stored. The projection matrix

  **P** = **U**(Λ**U**)†Λ,   Λ = diag(1/σ̂)

  is then precomputed once and applied to the Doppler-shifted template model. This replicates the linear filter that SYSREM applied to the data, accounting for both spectral and temporal distortions, at negligible extra cost compared to rerunning SYSREM. Setting `prepare_template: false` with a SYSREM pipeline bypasses this correction and will degrade CCF performance.

  > Gibson, N. P. et al. (2022). *Relative abundance constraints from high-resolution optical transmission spectroscopy of WASP-121b, and a fast model-filtering technique for accelerating retrievals.* MNRAS, 512, 4618 to 4638. https://doi.org/10.1093/mnras/stac091

---

## 9. `cross_correlation`

```json
"cross_correlation": {
  "velocity_max_kms": 325.0,
  "velocity_step_kms": 1.0,
  "kp_max_kms": 320.0,
  "noise_velocity_max_kms": 250.0,
  "snr_exclude_kms": 25.0,
  "in_trail_left_right": 2,
  "normalized": true,
  "use_inverse_variance_weighting": true,
  "cc_metric": true,
  "ccf_snr": true,
  "welch_ttest": false,
  "all_significance_metrics": false,
  "ssim_metric": false,
  "study_velocity_ranges": false,
  "plot_ccf_xstep": 50.0
}
```

| Field | Unit | Description |
|---|---|---|
| `velocity_max_kms` | km/s | Maximum velocity lag for the CCF computation. The CCF is computed from -`velocity_max_kms` to +`velocity_max_kms` in steps of `velocity_step_kms`. Should be large enough to capture the in-trail signal at all orbital phases. 325 km/s is sufficient for most hot Jupiters. |
| `velocity_step_kms` | km/s | Velocity step size for the CCF grid. 1.0 km/s is standard. Smaller steps increase precision but scale the CCF computation time linearly. |
| `kp_max_kms` | km/s | Maximum Kp for the Kp-Vsys map. The map spans -`kp_max_kms` to +`kp_max_kms`. Should bracket the expected Kp of the target. |
| `noise_velocity_max_kms` | km/s | Maximum velocity used to define the noise region in the CCF (the "out-of-trail" region). The noise baseline is estimated from velocities where the planet signal is absent. |
| `snr_exclude_kms` | km/s | Velocity range around the expected planet signal to exclude when computing the CCF noise baseline. |
| `in_trail_left_right` | int | Number of CCF lags on each side of the peak to include when computing the in-trail signal level. |
| `normalized` | `true`, `false` | If `true`, normalise the CCF by the autocorrelation amplitude (produces values in [-1, +1]). Should generally be `true`. |
| `use_inverse_variance_weighting` | `true`, `false` | If `true`, weight each spectral channel by 1/σ² when computing the CCF. This is the standard inverse-variance weighted CCF. |
| `cc_metric` | `true`, `false` | If `true`, compute and save the standard CCF detection metric and Kp-Vsys map. |
| `ccf_snr` | `true`, `false` | If `true`, compute the CCF S/N map (peak / out-of-trail RMS). This is the main detection statistic reported in the plots. |
| `welch_ttest` | `true`, `false` | If `true`, compute a Welch t-test significance metric comparing in-trail and out-of-trail CCF distributions. |
| `all_significance_metrics` | `true`, `false` | If `true`, compute all available significance metrics (CCF S/N, t-test, SSIM). Increases output file count. |
| `ssim_metric` | `true`, `false` | If `true`, compute the Structural Similarity Index Measure (SSIM) between the CCF matrix and the template trail. |
| `study_velocity_ranges` | `true`, `false` | If `true`, compute the CCF over multiple sub-ranges of the velocity grid for diagnostic purposes. |
| `plot_ccf_xstep` | km/s | X-axis tick spacing for CCF diagnostic plots. |

---

Additional CCF controls:

| Field | Meaning |
|---|---|
| `snr_noise_source` | `peak_row` (default) estimates off-peak scatter within each Kp row. `signal_free_rows` uses a region outside the selected Kp and velocity exclusions. |
| `snr_noise_kp_exclude_kms` | Kp half-width excluded from that signal-free region, default 120 km/s. |
| `plot_velocity_zoom_kms` | Optional symmetric velocity crop for the Earth-frame trail figure. |
| `ccf_kernel` | `normalized` (default) or `matched_filter` for the weighted large-instrument path. |
| `ccf_error_estimate` | `propagated` (default) uses the prepared uncertainties. `mad_residual` estimates each channel's scatter over time from its residuals. |
| `ccf_template_source` | `planet` (default), `telluric` or `stellar`. The latter choices use atmospheric transmission or the PHOENIX stellar spectrum to inspect compatible residual line patterns. |

## 10. `retrieval`

Molecular retrievals require `observation.different_nights: false`. Multiple
nights can share that observing sequence. The current retrieval route has no
per-night order-selection input for different observing sequences.

The C/O–metallicity tutorial uses `retrieval_choice: 1`. The combined-night
C/O–metallicity route (`retrieval_choice: 4`) is currently incomplete.


```json
"retrieval": {
  "enabled": false,
  "sampler": "nested_sampling",
  "log_likelihood": "Blain24",
  "dimensionality": "1D_CtoO_met",
  "retrieval_choice": 1,
  "time_resolution": false,
  "time_resolution_step": 20,
  "live_points": 200,
  "constant_efficiency_mode": false,
  "n_walkers": 10,
  "n_steps": 5000,
  "burnin": 1000,
  "pressure_n_levels": 100,
  "pressure_log_min": -6.0,
  "pressure_log_max": 2.0,
  "use_emulator": false,
  "emulator_path": "",
  "use_likelihood_emulator": false,
  "likelihood_emulator_n_samples": 3000
}
```

| Field | Valid values | Description |
|---|---|---|
| `enabled` | `true`, `false` | Master switch. Set to `false` to skip retrieval entirely (default for exploration runs). When `true`, the retrieval runs at the **end of the same simulation** using the spectral matrices held in memory. Repeating this retrieval starts with a new simulation run. |
| `sampler` | `"nested_sampling"`, `"mcmc"` | Bayesian sampler. `"nested_sampling"`: [MultiNest](https://github.com/JohannesBuchner/PyMultiNest) via [PyMultiNest](https://github.com/JohannesBuchner/PyMultiNest) (faster, better for multimodal posteriors). `"mcmc"`: [emcee](https://emcee.readthedocs.io) ensemble sampler. |
| `log_likelihood` | `"Blain24"`, `"Blain24_beta"`, `"BL19"`, `"Gibson22"` | Log-likelihood function. **Coupled to `dimensionality`, see below.** `"Blain24"`: Blain, Sánchez-López & Mollière (2024, AJ, 167, 179), chi-squared using propagated uncertainties with β fixed to 1. Use with `"1D"`, `"1D_CtoO_met"`, or `"2D"`. `"BL19"`: Brogi & Line (2019), analytic marginalisation over signal scaling. Use with `"1D"`, `"1D_CtoO_met"`, or `"2D"`. `"Blain24_beta"`: Blain24 with one global fitted uncertainty scale, as in Appendix G setup P-19, using `"1D_Blain24_beta"`. `"Gibson22"`: Gibson et al. (2022), chi-squared with free β noise-scaling. Use with `"1D_Gibson22"` or the fixed-template amplitude mode `"1D_alpha_Gibson22"`. Any invalid combination raises a `ValueError` at runtime. |
| `dimensionality` | `"1D"`, `"1D_Gibson22"`, `"1D_Blain24_beta"`, `"1D_CtoO_met"`, `"2D"`, `"1D_alpha"`, `"1D_alpha_Gibson22"` | Retrieval parameter space. **Coupled to `log_likelihood`, see above.** `"1D"`: single VMR (H₂O), Kp, T_eq, wind velocity (4 params), use with BL19 or Blain24. `"1D_Gibson22"`: same plus β noise-scaling parameter (5 params), **use only with Gibson22 likelihood**. `"1D_Blain24_beta"`: the same five coordinates as `"1D_Gibson22"`, paired with `"Blain24_beta"`. `"1D_CtoO_met"`: C/O ratio and metallicity via EasyChem (2 params, Kp and V_wind fixed to planet values), use with BL19 or Blain24. `"1D_alpha"`: Kp, Vrest and amplitude of a fixed template (3 parameters), with BL19 or Blain24. `"1D_alpha_Gibson22"`: those parameters plus β (4 parameters), with Gibson22. `"2D"`: two limb abundances, Kp and two temperature coordinates (5 parameters), with BL19 or Blain24. The current likelihood retains the configured regional temperature profiles. Its sampled temperature coordinates are not propagated into those profiles. |
| `retrieval_choice` | int | Selects the specific retrieval configuration within a dimensionality class. See the retrieval module documentation. |
| `time_resolution` | `true`, `false` | Reserved control for future time-bin selection. The current time-resolved route is selected with `retrieval_choice: 6`. |
| `time_resolution_step` | int | Reserved control for future per-exposure binning. The current time-resolved route uses four transit intervals. |
| `live_points` | int | Number of live points for nested sampling. Higher = more accurate posteriors but slower. Minimum useful: 100. Typical: 200 to 500. |
| `constant_efficiency_mode` | `true`, `false` | MultiNest constant efficiency mode. Faster but less accurate. For exploration only. |
| `n_walkers` | int | Number of walkers for MCMC (emcee). Must be ≥ 2 × number of free parameters. Default: 32. |
| `n_steps` | int | Number of MCMC steps per walker. |
| `burnin` | int | Number of initial MCMC steps to discard as burn-in. |
| `prior_bounds` | dict | Override the default prior bounds per dimensionality. Keys are dimensionality strings (e.g. `"1D_CtoO_met"`). Values are lists of `[low, high]` pairs, one per free parameter in the same order as the parameter labels. If omitted, the built-in defaults are used. |
| `pressure_n_levels` | int (default 100) | Number of atmospheric pressure layers for the retrieval forward model (log-spaced grid). |
| `pressure_log_min` | float (default -6.0) | Log₁₀ of the minimum pressure in bar for the retrieval pressure grid. |
| `pressure_log_max` | float (default 2.0) | Log₁₀ of the maximum pressure in bar for the retrieval pressure grid. |
| `use_emulator` | `true`, `false` | **Experimental.** If `true`, replace the petitRADTRANS call in the retrieval loglike with a pre-trained PCA+MLP spectrum emulator. See `src/exoplore/retrieval/emulator.py` for caveats, benchmarked posteriors are 2 to 3× broader than the true pRT retrieval. Not recommended for scientific use. |
| `emulator_path` | path string | Directory containing the trained emulator files (`mlp_weights.pt`, `pca_components.npy`, etc.). Only used when `use_emulator: true`. |
| `use_likelihood_emulator` | `true`, `false` | **Experimental.** If `true`, train a neural likelihood surrogate (θ → logL) before running MultiNest, then sample the surrogate. See `src/exoplore/retrieval/likelihood_emulator.py` for caveats. Not recommended for scientific use. |
| `likelihood_emulator_n_samples` | int (default 3000) | Number of true pRT loglike evaluations used to train the likelihood surrogate. Only used when `use_likelihood_emulator: true`. |

---

### Blain24 with a fitted uncertainty scale

Select the likelihood and its five-parameter space together:

```json
"retrieval": {
  "enabled": true,
  "log_likelihood": "Blain24_beta",
  "dimensionality": "1D_Blain24_beta"
}
```

The parameters are log10 abundance, Kp, temperature, Vrest and β. Model
preparation remains selected by `pipeline.name`. The likelihood is

```{math}
\ln L=-\frac{1}{2}\sum_i\left(\frac{d_i-m_i}{\beta\sigma_i}\right)^2
       -N\ln\beta.
```

up to terms independent of these fitted parameters. The β-dependent Gaussian
normalization is retained. At β = 1 this gives the fixed-uncertainty Blain24
likelihood. The scalar β multiplies all selected uncertainties.

Set its prior as the last pair in
`retrieval.prior_bounds.1D_Blain24_beta`. The default is `[0.01, 100]`, matching
`1D_Gibson22`. Tutorial 7's noiseless example uses `[0.999, 1.001]`,
matching its restricted Gibson comparison. These are comparison choices. The
paper's P-19 example used `[0.5, 100]`.

Runs using this likelihood include `Blain24_beta` in their output-directory
name so their posterior files remain separate from the fixed-β experiment.

Additional retrieval controls:

| Field | Meaning |
|---|---|
| `error_model` | Cheverall26 likelihood uncertainty choice: `propagated` (default) or `mad_residual`. |
| `model_reprocessing` | `bare` (default) processes the model alone, `embedded` processes it with the retained background, and `projector` removes the data-derived temporal components. The latter two require the corresponding retained arrays. These alternatives are currently restricted to the supported single-night route. |

`1D_extended` and `1D_extended_fast` have schema prior entries, but the current
runner does not define their parameter labels and likelihood selection as
complete retrieval modes.

## 11. `statistics`

```json
"statistics": {
  "enabled": false,
  "noise_study": false,
  "noise_correlations": false,
  "stack_group_size": null,
  "detectability_maps": false
}
```

| Field | Description |
|---|---|
| `enabled` | Reserved statistics flag. The molecular runner selects its executed studies from the CCF, night-stacking and detectability controls. |
| `noise_study` | Reserved flag. The molecular runner does not currently execute a separate noise-only study from this setting. |
| `noise_correlations` | Reserved flag. The molecular runner does not currently generate a noise-correlation matrix from this setting. |
| `stack_group_size` | Number of nights per group in repeated-night stacking studies. `null` selects the ordinary combined analysis. |
| `detectability_maps` | Save the local detection statistic used by the atmospheric sweep. Run `scripts/run_detectability_maps.py` to execute the grid. |

---

## 12. `paths`

```json
"paths": {
  "output_root": "/path/to/outputs",
  "planet_parameter_dir": "planet_params",
  "prt_input_data": "/path/to/petitRADTRANS/input_data/",
  "phoenix_wave_file": "/path/to/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits",
  "phoenix_flux_file": "/path/to/lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
  "skycalc_dir": "",
  "inputs_dir": "/path/to/instrument_inputs/"
}
```

| Field | Description |
|---|---|
| `output_root` | Root directory for all simulation outputs. The full output path for a run is `output_root/PlanetName/Instrument/event_type/simulation_name/` where `simulation_name` is derived from the pipeline, signal flag, number of nights, and noise scaling (e.g. `BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1`). Everything for a single run lives inside that one folder. |
| `planet_parameter_dir` | Directory containing planet JSON files, relative to the repo root. Default: `"planet_params"`. |
| `prt_input_data` | Absolute path to the petitRADTRANS `input_data/` directory. This is where opacity tables, CIA files, and molecular line lists live. Typically `<pRT_install_dir>/petitRADTRANS/input_data/`. |
| `phoenix_wave_file` | Absolute path to the PHOENIX wavelength grid FITS file. Filename: `WAVE_PHOENIX-ACES-AGSS-COND-2011.fits`. Download from the PHOENIX library at https://phoenix.astro.physik.uni-goettingen.de. |
| `phoenix_flux_file` | Absolute path to the PHOENIX model spectrum FITS file for the target star. Filename format: `lte{Teff:05d}-{logg:.2f}-{met:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits`. Select the file matching `stellar_teff_K`, `stellar_logg`, and `stellar_metallicity` from the planet parameter file. |
| `skycalc_dir` | Reserved path field. The current molecular reader and downloader locate per-exposure tellurics under `paths.inputs_dir` using the directory convention in the input guide. |
| `inputs_dir` | Directory containing instrument-specific input files: pre-computed SNR tables, telluric reference spectra, wavelength grids (for instruments like CARMENES where the wavelength grid comes from a calibration file). |

---

## 13. `timing` (top-level)

```json
"timing": false
```

| Field | Valid values | Description |
|---|---|---|
| `timing` | `true`, `false` | If `true`, print a wall-clock timing line after each simulation block completes and write a `timing_report_DD-MM-YYYY_HHhMM.txt` to the run output directory. The report includes a description of each block, the key config choices, and the elapsed time in seconds and minutes. Useful for benchmarking pRT call overhead vs pipeline overhead on a given machine. Default: `false`. |

The per-block breakdown printed to stdout when `timing: true`:

```
[timing] Block 1: X.X s
[timing] Block 2: X.X s
[timing] Blocks 3-6: X.X s  (X.XX min)
[timing] Block 7: X.X s
[timing] Block 8: X.X s
[timing] Block 9 (retrieval only): X.X s  (X.XX min)
[timing] Total wall-clock: X.X s  (X.XX min)
```

---

## 14. `plotting`

Controls diagnostic plots automatically generated at the end of every run.

```json
"plotting": {
  "pipeline_steps_order": null,
  "pipeline_steps_xlim_um": [1.4862, 1.4890],
  "pipeline_steps_sysrem_iterations": [1, 5]
}
```

| Field | Default | Description |
|---|---|---|
| `pipeline_steps_order` | `null` | Absolute spectral-order index to display in the pipeline-steps plot. If the requested order was not processed in the run (e.g. order 44 when the instrument has only 28 orders, or an order excluded by `instrument.order_indices`), the **first processed order** is used. `null` means the first processed order. |
| `pipeline_steps_xlim_um` | `[1.4862, 1.4890]` | Wavelength window [lo, hi] in µm to zoom into. For the SYSREM waterfall every panel uses this window. For the polynomial four-panel it applies to the 1-D panel only (the 2-D panels show the full order). If the window does **not** fall within the displayed order, the **full order** is shown. Set to `null` for the full order. The default is a water-band window at ~1.487 µm. |
| `pipeline_steps_sysrem_iterations` | `[1, 5]` | Which SYSREM iterations to show as panels in the pipeline-steps plot. **Used only for SYSREM pipelines** (ASL19, Gibson22). Ignored by the polynomial pipelines (BL19, Blain24). Each value is a 1-based iteration index. |

### Pipeline-steps diagnostic plot

Every run automatically saves `<run_dir>/plots/pipeline_steps_<sim_name>.pdf`, a stacked diagnostic figure illustrating the data-simulation and preparation steps for the displayed order (set by `pipeline_steps_order`). The top panel is the 1-D in-silico flux at mid-transit, noiseless (black) and noisy (red), zoomed to `pipeline_steps_xlim_um` if set. Below it are time-series matrices (wavelength on the horizontal axis, spectrum number on the vertical), with masked channels in the prepared panels shown as white and ingress/egress marked by red dashed lines. The number of matrix panels depends on the pipeline:

- **Polynomial pipelines (BL19, Blain24)** show three time-series panels below the representative spectrum: noiseless, noisy and prepared residuals. The notebook helper supplies a separate compact preview.
- **SYSREM pipelines (ASL19, Gibson22)** show the systematics peeling away iteration by iteration: the **raw** matrix, then one panel per iteration listed in `pipeline_steps_sysrem_iterations` (for example after iteration 1 and after iteration 5). The intermediate residuals are reconstructed with `apply_sysrem` on the displayed order, using the components saved by the preparation calculation.

Use this plot to inspect dominant structures and residuals. Injection tests and cross-correlation diagnostics help assess how preparation affects the planetary signal.

---

`plotting.pipeline_steps_noise_illustration_factor` optionally adds a labelled
noise-amplified curve to the spectrum panel. `null` keeps only the original
noiseless and noisy spectra. This controls the displayed curve.

## 15. `output`

Selects which per-order spectral matrices are written to disk. Almost all of a run's disk footprint is these matrices. The global masks, velocity grids, Kp-Vsys map, phase, and Julian dates are always written. The calculation uses these matrices in memory. The flags select which arrays are available on disk afterward. If the block is omitted, the defaults below apply.

```json
"output": {
  "save_mat_res": true,
  "save_mat_back": true,
  "save_ccf_store": true,
  "save_propag_noise": true,
  "save_U_sysrem": true,
  "save_mat_cc": false,
  "save_mat_noise": false,
  "save_std_noise": false,
  "save_rm_clv_model": false
}
```

| Field | Default | Needed for | Size (76-order run) |
|---|---|---|---|
| `save_mat_res` | `true` | re-running CCFs with other templates, retrievals | ~334 MB |
| `save_mat_back` | `true` | re-running CCFs, retrievals | ~298 MB |
| `save_ccf_store` | `true` | rebuilding Kp-Vsys maps without re-running the CCF | ~141 MB |
| `save_propag_noise` | `true` | inverse-variance CCF weighting, retrieval log-likelihood | ~416 MB |
| `save_U_sysrem` | `true` | filtering the model for the Gibson22 log-likelihood | ~4 KB |
| `save_mat_cc` | `false` | diagnostic only | ~181 MB |
| `save_mat_noise` | `false` | diagnostic only | ~411 MB |
| `save_std_noise` | `false` | diagnostic only | ~419 MB |
| `save_rm_clv_model` | `false` | stellar distortion and coupled-flux diagnostics when RM/CLV is enabled | depends on orders and exposures |

The default saving flags write ~1.2 GB for a 76-order run, including spectral inputs and uncertainty products used by the CCF and retrieval calculations. Saving every optional matrix uses ~2.2 GB. Set every flag to `false` for a minimal ~few-MB result (the Kp-Vsys map plus metadata). See [outputs.md](outputs.md) for the full matrix descriptions.

---

## 16. Supported instruments

| Config name | Instrument | Coverage (µm) | Orders | R | Recommended mode |
|---|---|---|---|---|---|
| `ANDES_YJHK` | ELT/ANDES, full YJH+K | 0.95 to 2.46 | 76 | 100,000 | A (ETC) |
| `ANDES_YJH` | ELT/ANDES, Y+J+H arm | 0.95 to 1.80 | 55 | 100,000 | A (ETC) |
| `ANDES_K` | ELT/ANDES, K arm | 1.95 to 2.47 | 21 | 100,000 | A (ETC) |
| `ANDES_RIZ` | ELT/ANDES, R+I+Z arm | 0.62 to 0.95 | 34 | 100,000 | A (ETC) |
| `ANDES_UBV` | ELT/ANDES, U+B+V arm | 0.35 to 0.63 | 62 | 100,000 | A (ETC) |
| `CARMENES_NIR` | CAHA/CARMENES NIR | 0.96 to 1.71 | 28 | 80,400 | B (ref-night) |
| `CARMENES_VIS` | CAHA/CARMENES VIS | 0.514 to 0.822 | 44 | 94,600 | B (ref-night) |
| `CRIRES+` | VLT/CRIRES+ | setting-dependent | varies | 100,000 nominal | ETC or reference-night/observed inputs |
| `IGRINS` | IGRINS H+K | 1.43 to 2.52 | input-dependent | 45,000 | Reference-night/observed inputs |

**Modes:**
- **Mode A (ETC-based):** SNR and wavelength grid come from an Exposure Time Calculator file. No real observed data needed. Ideal for future instruments and targets not yet observed. Set `specific_event: false`, `use_real_data: false`.
- **Mode B (reference-night, synthetic):** Uses real JD, airmass, and SNR from an existing observation as the timing and noise model. Stellar and planetary spectra are still synthesised with petitRADTRANS. Set `specific_event: true`, `use_real_data: false`.
- **Mode C (real-data analysis):** Loads real observed spectra and runs the analysis pipeline on them. Set `use_real_data: true`.
- **Fixed SNR (any mode):** Set `noise.fixed_snr` to a positive value to bypass all SNR files and use a single constant SNR for every pixel and exposure.

### ANDES (ELT/ANDES)

All five ANDES bands are supported: `ANDES_YJHK`, `ANDES_YJH`, `ANDES_K`, `ANDES_RIZ`, `ANDES_UBV`.

All ANDES configurations use **Mode A (ETC-based)**. The wavelength grid and SNR per order come from an ETC FITS file placed in `inputs_dir`. For each planet and band, name the file `ANDES_ETC_WAVE_SNR_{BAND}_{Planet}.fits` (e.g. `ANDES_ETC_WAVE_SNR_YJHK_HD189733b.fits`).

The YJHK ETC file must have named extensions: `YJH_WAVE_STARTS`, `YJH_WAVE_MIDS`, `YJH_WAVE_ENDS`, `YJH_SNR_MID`, `K_WAVE_STARTS`, `K_WAVE_MIDS`, `K_WAVE_ENDS`, `K_SNR_MID`. Single-band files (YJH, K, RIZ, UBV) use: `WAVE_STARTS`, `WAVE_MIDS`, `WAVE_ENDS`, `SNR_MID`. See `docs/input_files.md` for details.

### CARMENES NIR (CAHA/CARMENES)

Use `"name": "CARMENES_NIR"`. Coverage 0.96 to 1.71 µm, 28 orders, 4080 pixels/order, R ≈ 80,400.

The wavelength grid is bundled with EXoPLORE (`src/exoplore/instruments/data/wave_CARMENES_NIR.fits`, shape 28 × 4080 µm) and is found automatically by the instrument reader.

**Example order selection:** `configs/hd189733b_carmenes_transit.json`
excludes orders 9, 10, 18, 19 and 20, leaving 23 orders. In the bundled
wavelength grid, orders 9–10 span approximately 1.1204–1.1628 µm and orders
18–20 span 1.3445–1.4332 µm. Inspect the telluric transmission, uncertainties
and residuals for your own observations before adopting that selection.
The grid covers 0.9604–1.7119 µm.

**Mode B (recommended):** Place reference-night files in `inputs_dir/reference_night/`: `julian_date.fits`, `airmass.fits`, `snr.fits`. Set `specific_event: true`.

**Mode A (constant SNR):** Set `noise.fixed_snr` to a positive value and `specific_event: false`. No reference files needed.

### CARMENES VIS (CAHA/CARMENES)

Use `"name": "CARMENES_VIS"`. Coverage 0.514 to 0.822 µm, 44 orders, 4096 pixels/order, R ≈ 94,600.

Bundled wavelength grid: `src/exoplore/instruments/data/wave_CARMENES_VIS.fits` (shape 44 × 4096 µm, averaged from 95 CARMENES VIS calibration frames). Same Mode A/B/C options as CARMENES NIR.

### CRIRES+ (VLT/CRIRES+)

Use `"name": "CRIRES+"`. Coverage and number of orders depend on the wavelength setting. R ≈ 100,000.

CRIRES+ uses **Mode A (ETC-based)**. In order to use it, provide an ETC FITS file with the wavelength grid and SNR for your chosen setting and target. Name the files `CRIRES_ETC_WAVE_{Planet}.fits` and `CRIRES_ETC_SNR_{Planet}.fits` in `inputs_dir`. The number of orders is determined automatically from the shape of the ETC file.

If your setting covers only one detector order, set `"order_indices": [0]` in the config.

---

## 17. Adding a new planet

1. Create `planet_params/YourPlanet.json` following the structure of `HD189733b.json`.
2. Fill in all fields from the literature. Key sources: NASA Exoplanet Archive, TEPCat, original discovery papers.
3. For `kp_kms`, compute from the orbital parameters:
   ```
   Kp = (2π / P) × a × sin(i) / sqrt(1 - e²)
   ```
   or look up the measured value from radial velocity studies.

   For a circular orbit the planet velocity is `Kp·sin(2πφ)`, with `φ = 0` at mid-transit. With `observation.significant_eccentricity` enabled, EXoPLORE solves Kepler's equation for the true anomaly at each phase (`get_V_eccentric` in `src/exoplore/observation/velocity.py`). `eccentricity` and `argument_of_periastron_deg` (in degrees) then also shape the velocity curve.
4. For the PHOENIX stellar model, download the file matching `stellar_teff_K` (rounded to nearest 100 K), `stellar_logg` (nearest 0.5), and `stellar_metallicity` (nearest 0.5) from https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/
5. Create `configs/yourplanet_instrument_event.json` by copying the HD189733b config and updating all planet, instrument, atmosphere, and path fields.
6. Preview: `python scripts/run_exoplore.py configs/yourplanet_instrument_event.json`
7. Run: `python -u scripts/run_exoplore.py configs/yourplanet_instrument_event.json --run 2>&1 | tee run.log`

> **Common mistake:** forgetting `--run`. Without it the simulator prints the summary and exits without computing anything.


## 18. Detectability sweeps and helium configurations

The top-level `detectability` block is read by `scripts/run_detectability_maps.py`.
[Tutorial 11](tutorial.md#tutorial-11-detectability-maps) supplies a worked sweep.

| Field | Meaning |
|---|---|
| `x_variable`, `y_variable` | Atmospheric parameters varied along the two grid axes. |
| `x_values`, `y_values` | Grid values, with empty lists selecting the built-in grids. |
| `molecules` | Species used for the individual search templates. |
| `significance` | `cell_box` selects a local maximum around the injected position. `injected_point` selects its sampled cell. |
| `box_half_vrest_kms`, `box_half_kp_kms` | Half-widths of the local selection box, in km/s. |

`atmosphere.helium` configures the optional helium forward model. Its validated
fields are defined in `config/helium.py`, with Sunbather settings in
`config/sunbather.py`. The top-level `helium_grid` block is defined in
`config/helium_grid.py` and selects a fit to prepared helium spectra.
The direct observed-data recipes are demonstrated in
[Tutorial 12](tutorial.md#tutorial-12-he-i-transmission-spectroscopy-with-crires-wasp-121-b)
and [Tutorial 12b](tutorial.md#tutorial-12b-he-i-transmission-spectroscopy-with-carmenes-gj-3470-b).

## 19. Stellar RM and CLV

Add `rm_clv` to your simulation JSON to select photospheric line distortions
caused by rotation and centre-to-limb variation during a transit. All choices
live in the same simulation config. Existing configs omit this block and keep
both operations disabled.

```json
"rm_clv": {
  "simulate": false,
  "correct": false,
  "effects": "rm_clv",
  "coupling": "factorized",
  "model_file": "",
  "wavelength_medium": "vacuum",
  "projected_obliquity_deg": null,
  "cells_across_diameter": 200,
  "subpixels_per_axis": 1,
  "exposure_subsamples": 3,
  "azimuth_points": 128,
  "stellar_radial_points": 48,
  "planet_radial_points": 24,
  "normalization": "continuum"
}
```

| Field | Meaning |
|---|---|
| `simulate` | Construct synthetic spectra with the selected stellar effects and coupling, before tellurics and noise. |
| `correct` | Apply the fixed opaque-body photospheric correction. Seidel22 divides the individual stellar-frame master-out ratios by the identically prepared model before planetary alignment. Other pipelines divide native flux and uncertainty before preparation. |
| `correction_method` | `matched` uses the injected opaque-body model. `factorized` constructs a separate, simpler photospheric model for Seidel22 while retaining the selected injection coupling. |
| `correction_model_file` | Optional alternative intensity bundle for `correction_method: factorized`. An empty string uses `model_file`. |
| `effects` | `rm_clv` includes both effects. `rm` keeps the disk-centre line shape with continuum limb darkening. `clv` sets surface rotation to zero as a diagnostic control. |
| `coupling` | `factorized` keeps the separate stellar and planetary factors. `spatial` propagates local stellar light through the planetary atmosphere before integrating the flux. |
| `model_file` | NPZ file containing local specific intensities, angular coordinates and synthesis metadata, described below. |
| `wavelength_medium` | `vacuum` or `air`, matching the observing wavelength grid. The bundle must declare the same convention. |
| `projected_obliquity_deg` | Projected angle between stellar spin and orbital angular momentum. Required for `rm` and `rm_clv`. Obtain it for the target system. |
| `cells_across_diameter` | Number of spatial cells across the stellar diameter in factorized mode. Increase it until the spectra converge at the precision needed. |
| `subpixels_per_axis` | Samples along each factorized-mode cell axis to resolve the stellar edge and planetary silhouette. |
| `exposure_subsamples` | Uniform midpoint samples across each exposure. Fluxes are averaged before normalization. |
| `azimuth_points` | Angular quadrature points around each ring in spatial mode. Increase it to check convergence of local line profiles and limb overlap. |
| `stellar_radial_points` | Radial quadrature points for the full stellar disk in spatial mode. |
| `planet_radial_points` | Radial quadrature points for the opaque planetary disk in spatial mode. The atmospheric annulus uses the native pRT radii. |
| `normalization` | `continuum` divides by the integrated local continuum. `band_mean` divides by the mean flux over the selected wavelength interval. |

To study the uncorrected contamination, enable `simulate` alone. To test a
fixed-model correction on a simulation, enable both switches. Seidel22 aligns and
normalizes the atmosphere-free correction model with the data, constructs its
master-out using the same reference exposures and weights, and divides the
individual residual ratios before planetary alignment. Variances are divided
by the squared fixed divisor. Assess uncertainty in the stellar model separately
from these propagated data errors. This follows
the exposure-by-exposure correction discussed by
[Langeveld et al. (2022)](https://academic.oup.com/mnras/article/514/4/5192/6607503).
For measured
data, enable `correct` and supply the appropriate stellar model.

The current integration supports the standard CCF transit runner with circular
orbits and unsplit detector orders. Direct-helium routes, dayside observations,
eccentric orbits and alternative CCF template sources produce an explanatory
error when this feature is enabled. A missing model file also stops the run.

### Local stellar spectra

The input is an NPZ bundle with these arrays and scalar metadata:

| Key | Required content |
|---|---|
| `wavelength_um` | Strictly increasing wavelength vector in microns, with margin for rotation, observer velocity and instrumental convolution. |
| `mu` | Strictly increasing angular coordinates within [0, 1], covering every surface sample. |
| `intensity` | Shape `(n_mu, n_wavelength)`, retaining relative specific intensity between angles. |
| `continuum` | Same shape and units, with spectral lines removed. Required for `continuum` normalization and `rm` mode. |
| `intensity_kind` | Scalar string `specific_intensity`. |
| `intensity_unit` | Scalar string stating the common physical unit. |
| `wavelength_medium` | Scalar string `vacuum` or `air`. |
| `source` | Scalar string identifying the stellar synthesis and its physical parameters. |
| `resolving_power` | Instrumental Gaussian resolution already applied to the synthesis, or `0` for a synthesis without instrumental convolution. Numerical wavelength sampling alone does not determine this value. |

Save arrays exported by your stellar synthesis, for example:

```python
import numpy as np

# These arrays and metadata come from the stellar synthesis for your target.
# Preserve the relative brightness of the different mu spectra.
np.savez_compressed(
    "stellar_intensities.npz",
    wavelength_um=wavelength_um,
    mu=mu,
    intensity=intensity,
    continuum=continuum,
    intensity_kind="specific_intensity",
    intensity_unit=intensity_unit,
    wavelength_medium=wavelength_medium,
    source=synthesis_description,
    resolving_power=applied_instrumental_resolving_power,
)
```

Choose a synthesis appropriate for the target's temperature, gravity,
composition and lines. The loader checks array shapes, coverage and metadata.
The source description records the physical model, whose accuracy must be
validated against the stellar spectrum. The example ATLAS9 FITS files need
their original angular grid, intensity units and synthesis details confirmed
before conversion. A disk-integrated PHOENIX flux file supplies insufficient
angular information for a CLV model.

### Calculation and interpretation

Local spectra are shifted by the projected stellar rotation and integrated
over the visible disk. The opaque planetary silhouette removes the covered
intensities at each exposure sample. The resulting fluxes are averaged,
convolved to the instrumental resolution and sampled in the observing frame.
Only the additional instrumental broadening missing from a Gaussian-convolved
input synthesis is applied.

For continuum normalization, the stellar distortion is

\[
D(\lambda,t)=
\frac{F_{\rm in}(\lambda,t)/F_{c,{\rm in}}(\lambda,t)}
     {F_{\rm out}(\lambda)/F_{c,{\rm out}}(\lambda)}.
\]

With `coupling: factorized`, the continuum transit is removed from this
photospheric factor. The existing planetary transmission, BATMAN white-light
weighting and photospheric distortion are multiplied as separate factors. The configured
noise prescription remains in use. A fixed-model correction propagates
`uncertainty / D`, conditional on that stellar model.

With `output.save_rm_clv_model: true`, the runner writes
`rm_clv_order_<id>_night_<id>.npz` with the distortion,
stellar reference, wavelength, phase, observer-frame stellar velocity and model
source. Enabled experiments have distinct output-directory names containing
their effect and simulation/correction switches.

The method follows the disk-integration construction described by
[Casasayas-Barris et al. (2021), section 3.6](https://arxiv.org/abs/2101.04094).
Strong atomic line cores require particular care with stellar synthesis.
[Canocchi et al. (2024)](https://doi.org/10.1051/0004-6361/202347858)
compare solar Na and K centre-to-limb profiles with 1D and 3D LTE/NLTE models.

(spatial-star-atmosphere-coupling)=
### Spatial star–atmosphere coupling

Choose `coupling: spatial` and `simulate: true` to evaluate the atmosphere
against the stellar spectrum directly behind it. For each exposure sample,
the calculation subtracts the opaque planetary disk and integrates
`I_local × (1 − transmission)` over the atmospheric annulus. The completed
flux is then averaged over the exposure, convolved with the instrument profile
and sampled on the observing wavelength grid.

The native pRT pressure layers determine the atmospheric radii and ray
transmissions. The adapter checks that integrating those rays recovers pRT's
own effective transit radius. This adapter currently requires pRT 3.2.0,
whose opacity conventions have been checked. A different version stops with
an explanatory error. The numerical area check is repeated at runtime.

The new mode includes the continuum transit in the integrated flux. The
BATMAN factor is therefore excluded from the simulated flux in this mode.
The existing analysis windows and CCF reference templates remain in use.
An extended atmosphere can absorb light beyond the nominal continuum transit,
so inspect the exposures selected for the out-of-transit reference.

The stellar synthesis must include continuum intensities, the disk centre
(`mu=1`) and every angle sampled by the disk and atmospheric quadratures.
A small positive minimum `mu` is supported. The calculation checks its rays
and stops if they require angles outside the supplied grid. The
[stellar input guide](stellar_intensities.md) describes synthesis and conversion.
The model must use vacuum wavelengths and declare
`resolving_power: 0`, meaning that instrumental convolution has not yet been
applied. Multiplication by the planetary transmission occurs before the
instrument response.

The current spatial calculation supports a spherical 1D hydrostatic
atmosphere, a circular orbit and the configured bulk wind offset. Orbital
planet and stellar reflex velocities are sampled across the exposure. BERV
and bulk wind retain their exposure values. Set
`atmosphere.limb_asymmetries: false`, `retrieval.enabled: false`,
`observation.scale_injection: 1` and
`observation.signal_uses_light_curve: true`. External effective-radius spectra
cannot supply ray transmissions. Unsupported combinations stop before a run.
Regional atmospheres, spatially varying atmospheric rotation and joint
retrieval trial spectra require further implementation.

Spatial mode uses `stellar_radial_velocity()` with the stellar orbital term
`−K_star sin(2π phase)`, opposite to the planetary orbital term. Factorized
mode uses that same negative stellar orbital term when RM+CLV is active.
Controlled comparisons must also account for the separate treatment of the continuum light curve.

With `observation.simulate_planet: false`, spatial mode retains the opaque
transiting body and removes atmospheric absorption, providing a photospheric
control experiment. With `correct: true`, the fixed correction describes the
opaque-body stellar line distortion and preserves the atmospheric absorption.
Its divisor is independent of the atmospheric line signal. Modelling the full
coupling in atmospheric inference requires joint trial spectra.

Spatial runs have `_spatial` in their output directory. Their per-order/night
stellar archive, when requested with `output.save_rm_clv_model: true`, also
contains `coupled_flux`, `opaque_transmission` and the
atmospheric base and top radii, planetary velocity, pRT version and radial
point count. The base radius is the deepest hydrostatic
radius used by pRT. It can differ from the radius defined at the reference
pressure in the planet configuration.

This construction follows the local-intensity transport described by
[Dethier & Bourrier (2023), equation 4](https://arxiv.org/abs/2304.13759).
Its 1D hydrostatic atmospheric assumptions are those supplied by EXoPLORE's
pRT model.

## Direct synthetic sodium observations

Set `atomic_transmission.enabled: true` to run a CARMENES VIS or ANDES UBV sodium
observation through `ExoploreSimulator.run()`. This route constructs the pRT
atmosphere and stellar observation, multiplies by terrestrial transmission,
applies the instrumental response and native pixel integration, adds the
configured throughput and ETC-based Gaussian noise, fits Molecfit, and
prepares the two sodium transmission spectra with Seidel22.

The atmospheric model is supplied by `atmosphere.planet_model`, the stellar
surface by `rm_clv.model_file`, the observing grid by the instrument file,
and the CARMENES ETC bundle by `noise.carmenes_etc_file`. ANDES reads its
order-centre S/N from the instrument ETC FITS file. The current route supports
one homogeneous isothermal H2/He/Na atmosphere, one native optical order and one
transit, with an explicit fixed BERV. Spatial RM+CLV uses the same planetary
atmosphere as the synthetic observation. With `rm_clv.simulate: false`, a
chromatic BATMAN transit multiplies the disk-integrated stellar spectrum.
The matched stellar correction uses an opaque body without the atmosphere.

`atomic_transmission` supplies these observing and reduction inputs:

| Field | Meaning |
|---|---|
| `enabled` | Select the direct sodium observation. Default `false`. |
| `observing_sequence_file` | NPZ containing increasing `bjd` exposure midpoints and the matching `airmass` array. |
| `wavelength_interval_um` | Vacuum wavelength interval to retain from the selected native order. |
| `model_wavelength_limits_um` | pRT wavelength boundaries covering the observation and Doppler shifts with instrumental padding. |
| `line_wavelengths_nm` | Vacuum Na D2 and D1 wavelengths, in increasing order. |
| `observatory_lon_lat_height` | Longitude and latitude in degrees, height in metres, for the barycentric-time conversion used by Molecfit. |
| `band_half_width_kms` | Half-width of each planetary light-curve and absorption band. Default 20 km/s. |
| `molecfit_workers` | One to sixteen simultaneous external fits. Default 1. |
| `molecfit` | Explicit `CarmenesMolecfitConfig` settings, including molecular anchors, calibrated resolution, fit choices and acceptance gates. |
| `etc_reference_exposure_seconds` | Exposure duration used to generate the ANDES ETC table. Required for ANDES and checked against `observation.exposure_time_seconds`. CARMENES reads the duration from its ETC bundle. |

The telluric reference is selected by
`tellurics.reference_telluric_file`. It must contain a SkyCalc binary table
with `lam` in nm and `trans`, and cover the full modelled observation.
Transmission is scaled by exposure airmass relative to
`tellurics.reference_airmass` before instrumental convolution. The current
route uses one reference transmission with `use_full_skycalc: false`.

For the CARMENES sodium route, uncertainties use the photon-dominated ETC response
in `observation/sodium_noise.py`. The ETC order maximum anchors the continuum
S/N, native sampling converts it to per-pixel S/N, and stellar absorption,
tellurics and throughput determine the expected photon rate. Record the
Gaussian and throughput seeds separately. Set the Seidel22 continuum windows
explicitly and use `pipeline.seidel22.noise_resamples` to propagate continuum
refitting through the final spectral and light-curve covariance.

ANDES uses `observation/andes_sodium_noise.py` with the existing ANDES
order-centre anchor, sinc-squared blaze and relative telluric S/N model.
Its photon response converts the expected flux to Gaussian uncertainties on
the same flux scale. Select one `ANDES_UBV` order covering both lines and the
configured continuum windows. Set `noise.carmenes_etc_file: null` and supply
Molecfit's `calibrated_resolving_power` for ANDES. Each exposure receives an
independent Molecfit fit, with up to sixteen concurrent fits. A 30-second
sequence can therefore require several hours of telluric fitting.

Every execution requires a fresh `paths.output_root`. It writes the resolved
configuration, intrinsic and noisy observation arrays, individual Molecfit
products, and a residual matrix, transmission spectrum and light curve for
each line. `final_summary.json` records the band measurements, uncertainty
method, input hashes, seeds, software version, Git revision and runtime.

## 21. Heterogeneous stellar surface

The optional `stellar_surface` block selects independent local-intensity
components and places them in projected circles or rotating spherical caps.
It works with the current `rm_clv` integration. Components contain spectra,
continua and optional local velocities. Regions reference a component and
supply geometry and overlap priority.

See [the complete field definitions and example](heterogeneous_stellar_surface.md).
Missing, null or empty regions retain the existing quiet-star calculation.
`configs/stellar_surface_example.json` is a lightweight stellar-only example
used by `scripts/example_stellar_components.py`.
