# EXoPLORE Tutorial

This tutorial walks through eight progressively more involved analyses. In particular, by the end the reader will know how to run the default HD 189733 b ANDES simulation, change the target planet, configure the atmospheric forward model, run a CARMENES NIR simulation, combine multiple nights, enable the Bayesian retrieval, assess retrieval pipeline bias, and validate detection significances and retrieval uncertainties. We note that each tutorial builds on the previous one, so we recommend following them in order on a first reading. Readers new to the technique may wish to read the [Concepts primer](concepts.md) first.

---

## Prerequisites

In order to follow this tutorial, EXoPLORE and all dependencies must be installed as described in the [Installation guide](installation.md). Specifically:

```bash
cd EXoPLORE
source .venv/bin/activate
export SDKROOT="$(xcrun --show-sdk-path)"   # macOS only, add to .zshrc / .bashrc
python -m pytest                             # should pass with no failures
```

Furthermore, the PHOENIX stellar model files must be placed on disk and the paths updated in the configuration. If this step has not yet been completed, see [docs/input_files.md](input_files.md#3-stellar-atmosphere-models-phoenix).

---

## Tutorial 1: Run the reference HD 189733 b ANDES simulation

> **Approximate run time:** ~32 min on an Apple Mac Studio M2 Ultra (64 GB RAM) (measured: all 76 ANDES orders, one night, no retrieval; the petitRADTRANS forward model is 95% of the total). The times quoted in these tutorials are approximate and machine-dependent. With `timing: true` in the config, each run writes a `timing_report_<date>.txt` to its output directory with the exact per-block breakdown and total.

This is the reference case. It simulates one ANDES transit of HD 189733 b using a realistic multi-species atmosphere (H₂O, CO, CH₄, NH₃, H₂S, HCN, Fe, Ca) with limb asymmetries enabled, and a H₂O-only CCF template. In the following we describe each step in detail.

:::{note}
**Why does high-resolution cross-correlation spectroscopy work?**

During a primary transit, a hot Jupiter's radial velocity changes by tens of km s<sup>-1</sup> over a few hours as it moves along its orbit, while the telluric absorption from Earth's atmosphere and the stellar spectral lines are essentially stationary in the instrument frame (changing by < 1 km s<sup>-1</sup>). This Doppler separation is the basis of the technique: after removing the quasi-static stellar and telluric components via the preparation pipeline (using, e.g., polynomial fitting of the wavelength- and time-dependent spectral contributions, or SYSREM detrending), the planet signal (Doppler-shifting with the orbital motion) survives as a residual.

The planet signal is extracted by cross-correlating the residual spectra with a model template containing the absorption features of the target molecule. Because the template has thousands of spectral lines, the cross-correlation coherently sums all of them simultaneously, amplifying the planet signal by roughly √N<sub>lines</sub> relative to the noise.

The velocity separation between the planet signal and the quasi-static backgrounds can be quantified precisely. At R = 100,000, each resolution element spans Δv = c/R ≈ 3 km s<sup>-1</sup>. During a transit of duration T<sub>14</sub>, the planet radial velocity evolves as v<sub>P</sub>(φ) = K<sub>P</sub> sin(2πφ), where φ is the orbital phase (zero at mid-transit). The total velocity range swept during transit is therefore Δv<sub>transit</sub> = 2 K<sub>P</sub> sin(π T<sub>14</sub>/P). For HD 189733 b (K<sub>P</sub> = 152.5 km s<sup>-1</sup>, T<sub>14</sub> = 1.94 h, P = 2.219 days), this gives Δv<sub>transit</sub> = 2 × 152.5 × sin(π × 1.94/53.26) ≈ 34.8 km s<sup>-1</sup>, corresponding to ∼12 resolution elements at R = 100,000. Telluric and stellar features, by contrast, drift by < 1 km s<sup>-1</sup> over the same interval. This ∼35× difference in velocity stability allows the preparation pipeline to separate the two contributions.
:::

### Step 1: Inspect the config

Open `configs/hd189733b_andes_transit_clean.json`. The key scientific choices are:

```json
"planet":       { "name": "HD189733b", "parameter_file": "planet_params/HD189733b.json" },
"instrument":   { "name": "ANDES_YJHK" },
"observation":  { "event_type": "transit", "n_nights": 1, "exposure_time_seconds": 30.0 },
"atmosphere":   {
  "limb_asymmetries": true,
  "limb_divisions": "gradual",
  "cc_with_true_model": false,
  "planet_model": {
    "species": ["H2", "He", "H2O", "CH4", "NH3", "CO", "H2S", "HCN", "Fe", "Ca"],
    "use_easychem": true,
    "carbon_to_oxygen_ratio": 0.41,
    "metallicity_wrt_solar": 0.53
  },
  "ccf_template": {
    "species": ["H2", "He", "H2O"],
    "use_easychem": true
  }
},
"pipeline":  { "name": "Blain24" }
```

The atmosphere block defines **two separate petitRADTRANS models**: the injected planet atmosphere (`planet_model` + limb sub-blocks) and the CCF template (`ccf_template`). Because `cc_with_true_model: false`, the simulator uses H₂O-only as the cross-correlation template while injecting the full 10-species atmosphere. This mimics the typical observational strategy: when searching for signal in real data, the exact injected chemistry is not known a priori, and therefore a simpler template is used to cross-correlate. We note that setting `cc_with_true_model: true` would use the full injected model as the template, generally yielding higher S/N but corresponding to an idealised scenario not achievable in practice.

The `"pipeline": "Blain24"` entry selects the Blain24 preparation pipeline (Blain, Sánchez-López & Mollière 2024). Blain24 first removes the instrumental throughput and blaze by dividing each exposure by a second-order polynomial fit over wavelength, and then removes the telluric absorption by fitting a second-order polynomial to the logarithm of the flux as a function of airmass (the log-transmittance of the Earth's atmosphere is, to first order, linear in airmass) and dividing by the resulting fit. Wavelength channels where the fitted telluric transmittance falls below 0.8 are masked, to exclude regions too strongly affected by telluric absorption.

:::{important}
**Preparing pipeline and retrieval consistency**

The preparing pipeline acts on the data matrix in a way that also affects the buried planet signal. For retrieval to be unbiased, the forward model must undergo the same transformation as the data at every likelihood evaluation. For polynomial-based pipelines (BL19, Blain24), the preparation operator is linear and analytically known, so the same polynomial can be applied directly to the Doppler-shifted model; Blain, Sánchez-López & Mollière (2024) formally prove unbiasedness under this condition. For SYSREM-based pipelines (ASL19, Gibson22), Gibson et al. (2022) showed that SYSREM can be represented as a linear projection using the eigenvectors of the data matrix, so the filtered model is obtained efficiently by a single matrix multiplication (M_filt = M - P · M) rather than by iterating SYSREM on every likelihood call.
:::

### Step 2: Update your machine-specific paths

The only fields that need to change on a given machine are under `paths`. Edit these:

```json
"paths": {
  "output_root":       "/wherever/you/want/outputs",
  "prt_input_data":    "/path/to/petitRADTRANS/input_data/",
  "phoenix_wave_file": "/path/to/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits",
  "phoenix_flux_file": "/path/to/lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
}
```

Everything else already points into the repository (`inputs_dir`, telluric file, ETC files).

### Step 3: Preview the simulation (no `--run`)

Running without `--run` prints a summary and exits immediately:

```bash
python scripts/run_exoplore.py configs/hd189733b_andes_transit_clean.json
```

The output is a summary block confirming the planet, instrument, species list, and output root. This is useful to verify the configuration before committing to a long run.

### Step 4: Run the simulation

Add `--run` to actually execute:

```bash
python -u scripts/run_exoplore.py configs/hd189733b_andes_transit_clean.json --run 2>&1 | tee run_log.txt
```

The `-u` flag disables Python's output buffering so every line is flushed immediately; `tee` writes it to both the terminal and `run_log.txt` simultaneously. If you send the process to the background (`... &`), monitor the log with:

```bash
tail -f run_log.txt
```

:::{note}
**Monitoring and warnings**

petitRADTRANS prints many lines of opacity-loading text at the start of the run, which can push important warnings off the screen. EXoPLORE emits two types of warnings inline during the run:

- A `!!!` block if the telluric reference file does not cover the instrument wavelength range (fires within the first order, before any CCF computation).
- A `[mask]` line each time an order is fully masked (no usable pixels after telluric and SNR masking), with a pointer to the warnings folder on the first occurrence.

If running in the background and not watching the terminal, the safest practice after the run completes is to grep the log:

```bash
grep -E "\[mask\]|WARNING|Traceback|Error" run_log.txt
```

And always check the `warnings/` subfolder of the output directory (it contains a text file listing every fully-masked (order, night) pair with their central wavelengths, and is the authoritative post-run record regardless of what was visible on the terminal).
:::

Consecutively, the simulator will:

1. Load planet parameters from `planet_params/HD189733b.json`
2. Build the ANDES wavelength grid (76 spectral orders: 55 YJH + 21 K, R = 100,000, 2048 px/order)
3. Compute equilibrium chemistry with EasyChem (C/O = 0.41, Z = 0.53 dex) for each limb sub-block
4. Run petitRADTRANS for each of the six atmospheric models (planet, CCF template, four limb regions)
5. Build the stellar continuum matrix from the PHOENIX model
6. Simulate the full observing sequence (~390 exposures at 30 s, ~3.3 h baseline + 1.94 h transit)
7. For each of the 76 orders: apply telluric contamination, inject the planetary signal with a limb-weighted BATMAN light curve, add photon noise from the ANDES ETC
8. Apply the Blain24 pipeline (polynomial throughput removal, airmass-based telluric removal, telluric and SNR-column masking)
9. Compute the inverse-variance weighted CCF against the H₂O template
10. Build the Kp-Vsys S/N detection map
11. Save all spectral matrices, mask files, CCF products, and diagnostic plots

:::{figure} figures/tutorial1_ccf_erf_andes.png
:width: 80%
:align: center
CCF matrix in the Earth rest frame, co-added over all 76 ANDES YJHK orders. The x-axis is the velocity in the Earth rest frame and the y-axis is the orbital phase (zero at mid-transit). The bright diagonal streak is the planet signal, which shifts from negative to positive velocity as the planet moves along its orbit during transit. This intermediate product is then co-added in the planet rest frame over a grid of K<sub>P</sub> values to produce the detection map below.
:::

:::{figure} figures/tutorial1_kpvsys_andes.png
:width: 80%
:align: center
Cross-correlation S/N map as a function of the assumed planetary orbital velocity semi-amplitude K<sub>P</sub> and rest-frame planet velocity v<sub>rest</sub>, obtained from the reference HD 189733 b ANDES simulation (Tutorial 1). The colour scale gives the significance of the cross-correlation signal in units of the off-peak standard deviation. The red dashed lines indicate the expected position assuming a circular orbit and no net atmospheric wind (K<sub>P</sub> = 149.4 km s<sup>-1</sup> as configured; literature value 152.5 km s<sup>-1</sup>; v<sub>rest</sub> = 0 km s<sup>-1</sup>). With Blain24 preparation, the peak significance of ~43σ is recovered close to this position (K<sub>P</sub> = 143 km s<sup>-1</sup>, v<sub>rest</sub> = -4 km s<sup>-1</sup>) across the 76 ANDES YJHK orders in a single simulated transit; the small offset reflects noise and the K<sub>P</sub>-v<sub>rest</sub> degeneracy over one transit rather than a physical wind.
:::

:::{note}
**Reading the Kp-Vsys map**

The map is constructed by co-adding the cross-correlation function (CCF) over all in-transit exposures for each assumed value of K<sub>P</sub>. At the correct K<sub>P</sub>, the planet signal stacks coherently and a peak emerges at the planet's rest-frame velocity. Away from the correct K<sub>P</sub>, the co-addition is incoherent and the signal washes out.

The characteristic elongated shape (a vertical streak rather than a point) reflects a partial degeneracy between K<sub>P</sub> and v<sub>rest</sub> over a single transit: a slightly wrong K<sub>P</sub> shifts the peak in v<sub>rest</sub> rather than completely destroying it. This degeneracy can in principle be lifted by combining multiple transits at different orbital phases.

A non-zero v<sub>rest</sub> at the peak indicates that the absorbing gas has a net velocity relative to the planet's Keplerian rest frame. In tidally locked hot Jupiters, a **blueshift** (negative v<sub>rest</sub>) is commonly observed and attributed to day-to-nightside winds: high-altitude gas flows from the hot dayside towards the cooler nightside, and at the terminator this flow has a line-of-sight component pointing towards the observer on both the leading and trailing limbs. For HD 189733 b, multiple studies consistently measure a blueshift of -3.9 to -5.5 km s<sup>-1</sup> (Alonso-Floriano et al. 2019; Sánchez-López et al. 2019; Blain et al. 2024).

A measured v<sub>rest</sub> offset is most commonly interpreted as a day-to-nightside wind, but it is not uniquely diagnostic. Planetary rotation, an offset dayside hotspot, or species-dependent dynamics can produce comparable shifts (Brogi et al. 2023; Smith et al. 2024), as can residual uncertainties in the velocity solution or the wavelength calibration. Such an offset should therefore always be cross-checked against independent indicators before being attributed to winds.
:::

The peak S/N and its position depend on the noise seed (`noise.noise_seed`, default `12345`); changing it draws a different noise realisation. Fully-masked orders (deep telluric bands) are skipped and reported in `<run_name>/warnings/fully_masked_orders_<run_name>.txt`.

### Step 5: Check the outputs

Outputs are written under:
```
output_root/HD189733b/ANDES_YJHK/transit/Blain24_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/
```

Everything for this run lives in that single folder:
- `matrices/`, spectral matrices, masks, CCF data, Kp-Vsys map
- `plots/`, diagnostic PDFs
- `warnings/`, masked-order reports

**Disk footprint:** with the default output settings this run writes about **1.2 GB**, almost all of it the per-order spectral matrices in `matrices/` (one compressed NumPy `.npz` per matrix per order, across the 76 ANDES orders); the diagnostic PDFs in `plots/` add ~40 MB. Which matrices are written is your choice, via the `output` block in the config (see [Outputs](outputs.md)). The defaults keep what is needed to re-run cross-correlations with other templates, run retrievals, and use the Gibson22 likelihood, and drop the pure-diagnostic matrices; saving every matrix would be ~2.2 GB, while a minimal run (just the Kp-Vsys map and metadata) is a few MB.

See [docs/outputs.md](outputs.md) for a complete guide to every output file and code examples to read the Kp-Vsys map.

---

## Tutorial 2: Simulate a different planet (WASP-76 b)

> **Approximate run time:** ~61 min on an Apple Mac Studio M2 Ultra (64 GB RAM), running `configs/wasp76b_andes_ubv_limbasym.json` (Blain24 pipeline, asymmetric limbs, all 62 ANDES UBV orders, one night, no retrieval). The petitRADTRANS forward model dominates: it is run for the four limb atmospheres and convolved with the per-limb wind and rotation kernels, so this case is slower than the single-atmosphere Tutorial 1. Times are approximate and machine-dependent; with `timing: true` the run writes a `timing_report_<date>.txt` with the exact per-block breakdown and total.

WASP-76 b is an ultra-hot Jupiter where the extreme day-to-night temperature contrast produces strong chemical asymmetries between the morning and evening limbs. The key observational signature is a neutral iron signal that strengthens from ingress to egress and progressively blueshifts. At ingress the morning (leading) terminator is in view, where its rotation and the day-to-nightside wind partly compensate and contribute a component near zero rest-frame velocity. As the morning terminator rotates out of view, it stops contributing, because iron has condensed out of the cooler nightside. The near-zero velocity component then vanishes, leaving only the hotter evening (trailing) terminator. There, rotation and wind add towards the observer, giving the net blueshift (Ehrenreich et al. 2020). This tutorial runs the reference WASP-76 b simulation: ANDES UBV (0.35 to 0.63 µm, 62 orders, R = 100,000) with limb asymmetries enabled and a Fe-only CCF template.

A ready-to-run config is provided at `configs/wasp76b_andes_ubv_limbasym.json`. The planet parameter file is at `planet_params/WASP-76b.json`. In the following we describe the key differences from Tutorial 1 and the additional steps required.

### Step 1: Update machine-specific paths

Edit `configs/wasp76b_andes_ubv_limbasym.json` and set your local paths under `paths`:

```json
"paths": {
  "output_root":       "/your/output/directory",
  "prt_input_data":    "/path/to/petitRADTRANS/input_data/",
  "phoenix_wave_file": "/path/to/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits",
  "phoenix_flux_file": "/path/to/lte06300-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
  "inputs_dir":        "inputs/ANDES/WASP76b/"
}
```

WASP-76 is an F7 star (T<sub>eff</sub> ≈ 6329 K). The nearest PHOENIX grid point is `lte06300-4.00-0.0` (6300 K, logg 4.0, [Fe/H] 0.0).

### Step 2: Generate the telluric reference spectrum

ANDES UBV covers the optical range (0.35 to 0.63 µm). This range is not included in the NIR telluric reference file bundled for HD 189733 b. A separate SkyCalc reference spectrum must be generated before running the simulation.

:::{important}
**Why a new telluric file is needed**

The bundled telluric reference covers the NIR (∼0.95 to 2.5 µm). When the simulator loads it for an optical instrument it finds no data at those wavelengths, causing all orders to appear fully masked and the simulation to complete without producing any signal. Whenever you use an instrument arm outside the NIR (such as ANDES UBV, RIZ, or CARMENES VIS) you must generate a telluric reference file covering the appropriate wavelength range.

In the optical (0.35 to 0.63 µm), the dominant telluric absorption is from broad O₃ Chappuis bands (smooth, broadband, mostly removed by the polynomial preparation pipeline) and from weak O₂ features near the band edges. With `mask_threshold: 0.2`, no pixels are masked in this range under typical Paranal conditions.
:::

:::{tip}
If you do not need to tune the precipitable water vapour, the observatory site, or the other SkyCalc parameters for a specific test, a convenient shortcut is to generate a single wide telluric reference covering all the wavelength bands you intend to use (for example the full optical-to-NIR range), and point each instrument config at it. This avoids regenerating a separate reference per instrument arm.
:::

Generate the optical reference file with the `--ref-only` flag, which queries SkyCalc once at the reference airmass (1.0 by default) and writes `tell_ref_airmass_1.0.fits` without generating per-exposure files:

```bash
python scripts/generate_skycalc_inputs.py \
    configs/wasp76b_andes_ubv_limbasym.json \
    --ref-only \
    --mode synthetic
```

This requires an internet connection and the `skycalc_ipy` package (`pip install skycalc_ipy`). The file is written to `inputs/ANDES/WASP76b/tellurics/tell_ref_airmass_1.0.fits`. For the wavelength range to be recognised correctly, the instrument config reads it from `paths.inputs_dir`; the script derives the wavelength limits automatically from the ANDES UBV ETC file.

### Step 3: Run

```bash
python -u scripts/run_exoplore.py configs/wasp76b_andes_ubv_limbasym.json --run
```

The simulation uses 62 UBV orders and a Fe-only CCF template, prepared with the same Blain24 polynomial pipeline as Tutorial 1. Limb asymmetries are enabled with `limb_divisions: asymmetric`, appropriate for the extreme day-to-night contrast of WASP-76 b: the morning limb is modelled at T = 2800 K (isothermal) and the evening limb at T = 3500 K (isothermal), with a day-to-nightside wind of -8 km s<sup>-1</sup> at both limbs (following Ehrenreich et al. 2020; Wardenier et al. 2021; Beltz et al. 2023). Exposure time is 90 s (matching the ANDES ETC files provided). Since telluric features in the optical are broadband and weak, no pixels are expected to be fully masked by the telluric threshold.

:::{figure} figures/tutorial2_ccf_erf_wasp76b.png
:width: 80%
:align: center
CCF matrix in the Earth rest frame for the WASP-76 b ANDES UBV simulation (Tutorial 2), co-added over all 62 spectral orders. The x-axis is the velocity in the Earth rest frame and the y-axis is the orbital phase (zero at mid-transit). The neutral iron signal starts near zero velocity at ingress (where the leading limb rotation and the day-to-nightside wind of -8 km s<sup>-1</sup> approximately cancel) and becomes progressively more blueshifted as the trailing (evening) limb takes over, where planetary rotation and wind add constructively towards the observer. The growing signal depth towards egress reflects the increasing contribution of the iron-bearing evening terminator.
:::

:::{figure} figures/tutorial2_kpvsys_wasp76b.png
:width: 80%
:align: center
Cross-correlation S/N map as a function of the assumed orbital velocity semi-amplitude K<sub>P</sub> and rest-frame planet velocity v<sub>rest</sub>, obtained from the WASP-76 b ANDES UBV simulation (Tutorial 2). The Fe-only CCF template traces neutral iron in the optical (0.35 to 0.63 µm, 62 orders). The red dashed lines mark the expected position for a circular orbit and injected wind velocity (K<sub>P</sub> = 198.3 km s<sup>-1</sup>, v<sub>rest</sub> = -8 km s<sup>-1</sup>). The peak significance of ~29σ is recovered at K<sub>P</sub> = 181 km s<sup>-1</sup>, v<sub>rest</sub> = -9 km s<sup>-1</sup>. The rest-frame velocity matches the injected wind, while the recovered K<sub>P</sub> sits ~17 km s<sup>-1</sup> below the orbital value. This K<sub>P</sub> offset is a physical consequence of the limb asymmetry rather than a fitting error: the time-varying limb weighting and the different net velocities at the leading and trailing terminators (v<sub>rot</sub> + v<sub>wind</sub> at each limb) make the Doppler trail depart from a single sinusoid, biasing the apparent semi-amplitude as the iron-bearing evening terminator comes to dominate towards egress.
:::

Outputs are written under `output_root/WASP76b/ANDES_UBV/transit/Blain24_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/`, with the same `matrices/`, `plots/`, and `warnings/` subfolders as Tutorial 1. **Disk footprint:** with the default output settings this run writes about **0.7 GB** (smaller than Tutorial 1: UBV covers 62 orders and the transit is sampled by fewer exposures), again almost entirely the per-order spectral matrices in `matrices/`. As in Tutorial 1, the set of matrices written is controlled by the `output` block (saving every matrix would be ~1.2 GB; a minimal run is a few MB). See [docs/outputs.md](outputs.md) for the full output guide.

---

## Tutorial 3: Configure the forward model: chemistry, temperature, clouds, and limb structure

> **Approximate run time:** the examples are configuration changes; a full simulation matches Tutorial 1 (~30 min), or ~2 min if restricted to a single order for a quick test.

The forward model is the high-resolution atmospheric transmission spectrum that EXoPLORE injects as the planet signal and cross-correlates against the data. This tutorial is a reference for configuring it. The planet model and the CCF template are two independent blocks in the config and can be set to different compositions. Five examples cover the main configuration options: C/O ratio and chemical composition, temperature structure, cloud deck, limb asymmetry modes, and manual mass fractions.

All examples below are modifications of the reference config. Start from:

```bash
cp configs/hd189733b_andes_transit_clean.json configs/my_custom_atmosphere.json
# Edit the atmosphere block as shown in each example, then run:
python -u scripts/run_exoplore.py configs/my_custom_atmosphere.json --run
```

For Example D (limb asymmetries), a ready-made config for WASP-76 b is provided:

```bash
python -u scripts/run_exoplore.py configs/wasp76b_andes_ubv_limbasym.json --run
```

### Example A: carbon-rich atmosphere (C/O = 0.9)

At C/O = 0.9, CO locks up most of the available oxygen, suppressing H₂O while promoting HCN and C₂H₂. The dominant absorbers shift from H₂O and CO₂ (solar composition, C/O ≈ 0.55) to CO, HCN, and C₂H₂.

```{figure} figures/chemistry_co_ratio.png
:width: 95%
:align: center

The same HD 189733 b atmosphere computed at C/O = 0.55 (blue) and C/O = 0.90 (red), generated with `scripts/illustrate_model_spectrum.py`. At the higher C/O ratio the H₂O bands (for example near 1.1 to 1.5 µm) are noticeably weaker, while carbon-bearing absorbers strengthen towards the red end of the window. The cross-correlation amplitude of an H₂O template would therefore drop for the carbon-rich case.
```

In the config, under `atmosphere.planet_model`:

```json
"carbon_to_oxygen_ratio": 0.9,
"species": ["H2", "He", "H2O", "CO", "HCN", "C2H2"]
```

The CCF template is configured independently via the `ccf_template` block. Setting `cc_with_true_model: false` activates it:

```json
"cc_with_true_model": false,
"ccf_template": {
  "species": ["H2", "He", "H2O", "CO"],
  "use_easychem": true,
  "carbon_to_oxygen_ratio": 0.55,
  "metallicity_wrt_solar": 0.0
}
```

### Example B: isothermal temperature profile

```json
"isothermal": true,
"isothermal_temperature_K": 1200.0
```

This replaces the Guillot profile entirely for that sub-block. The `equilibrium_temperature_K`, `kappa_ir`, and `gamma_guillot` fields are ignored when `isothermal: true`.

### Example C: add a grey cloud deck

To add a grey cloud at 10 mbar (everything below 0.01 bar is opaque):

```json
"cloud_pressure_bar": 0.01
```

Set to `null` for a cloud-free atmosphere. Strong molecular absorption bands at depth are truncated, generally reducing the detection S/N. Physically, the cloud truncates the pressure levels accessible to outgoing radiation: molecular features that form below the cloud top are hidden, reducing the effective scale height of each spectral line and therefore the amplitude of the planet's absorption signal in the cross-correlation. The effect is strongest for species whose opacity peaks at high pressure.

### Example D: choosing a limb-asymmetry mode

:::{note}
**The physical origin of limb asymmetries**

Tidally locked hot Jupiters are expected to develop significant temperature and chemical gradients between their dayside and nightside hemispheres. The morning (leading) limb probes the night-to-day transition of the terminator (gas arriving from the cool nightside), while the evening (trailing) limb probes the day-to-night transition (gas arriving from the hot dayside). These two limbs can therefore exhibit different temperatures, chemistry, and condensate coverage. The degree of asymmetry is expected to depend strongly on the equilibrium temperature, atmospheric composition, and drag regime of each planet (Sánchez-López et al. submitted).

For ultra-hot Jupiters, dayside temperatures can exceed ∼2000 to 3000 K, sufficient to keep refractory species such as neutral iron in the gas phase. On the cooler nightside, these same species may condense, depleting the gas phase. The morning limb, probing gas that has come from the nightside, may therefore carry condensed or depleted species compared to the evening limb, where gas from the hot dayside remains in an atomic or ionic state. This scenario was observationally supported for WASP-76 b: a neutral iron signal grows from ingress to egress and is predominantly blueshifted, interpreted as iron being largely condensed at the morning terminator while remaining in gas phase at the hotter evening terminator, with day-to-night winds contributing an additional blueshift of ∼-5 km s<sup>-1</sup> at the limbs (Ehrenreich et al. 2020). A possible eastward offset of the hot spot towards the evening terminator may further enhance this asymmetry.

In transmission spectroscopy, the leading limb dominates during ingress and the trailing limb during egress. A 1D model that averages over both limbs can bias retrieved molecular abundances and temperatures when significant asymmetry is present.

The **morning limb** is the leading limb (first to cross the stellar disk); the **evening limb** is the trailing limb (last to exit). These terms are used interchangeably throughout the literature and in this documentation.
:::

Three `limb_divisions` modes are available when `limb_asymmetries: true`.

**`"gradual"` (default, recommended):** smooth cubic transition from morning-dominated ingress to evening-dominated egress, with an equal 50/50 mix at full transit. This is appropriate for warm/hot Jupiters where heat redistribution is partial.

```json
"atmosphere": { "limb_asymmetries": true, "limb_divisions": "gradual" }
```

**`"asymmetric"` (an ultra-hot Jupiter like WASP-76 b):** the morning limb dominates the first quarter of full transit, then a cubic handover completes by mid-transit and the evening limb dominates the remainder. This is appropriate for planets with extreme day-to-night temperature contrast and negligible heat redistribution.

```json
"atmosphere": { "limb_asymmetries": true, "limb_divisions": "asymmetric" }
```

**`"simplified_step"` (step-function):** pure morning during ingress, equal mix at full transit, pure evening during egress, with no smooth transition. This is appropriate for reference tests and simplified scenarios where a clean physical interpretation of morning vs evening contributions is needed, free from the complexity of a time-varying transition.

```json
"atmosphere": { "limb_asymmetries": true, "limb_divisions": "simplified_step" }
```

To disable limb asymmetries entirely (approximately 4× faster, as only one pRT call is needed):

```json
"atmosphere": { "limb_asymmetries": false }
```

In this case only the `planet_model` sub-block is used, and the morning/evening blocks are ignored even if present in the JSON.

:::{figure} figures/limb_scaling_factors.png
:width: 80%
:align: center
Time-dependent scaling factors for the leading (morning, red) and trailing (evening, blue) limbs as a function of orbital phase, for the three available <code>limb_divisions</code> modes. The BATMAN transit light curve is shown in black for reference. Vertical dashed lines mark the first and second contacts (T<sub>1</sub>, T<sub>2</sub>), mid-transit (T<sub>c</sub>), third and fourth contacts (T<sub>3</sub>, T<sub>4</sub>). <b>Top:</b> <code>simplified_step</code> mode (the leading limb contributes exclusively during ingress and the trailing limb during egress, with an equal 50/50 mix at full transit and no smooth transition). <b>Middle:</b> <code>gradual</code> mode (smooth cubic transition appropriate for warm and hot Jupiters with partial heat redistribution). <b>Bottom:</b> <code>asymmetric</code> mode (the leading limb dominates the first quarter of full transit before a rapid sigmoid handover to the trailing limb; calibrated for ultra-hot Jupiters such as WASP-76 b where extreme day-to-night temperature contrast produces a strongly asymmetric signal, Ehrenreich et al. 2020).
:::

These weighting curves leave a direct imprint on the data. The figure below shows the time-resolved cross-correlation signal that results, as EXoPLORE simulates it for ANDES.

:::{figure} figures/limb_asymmetry_ccf.png
:width: 100%
:align: center
Time-resolved cross-correlation signal of pseudo-2D atmospheres, simulated for ANDES. <b>Top row:</b> cross-correlation maps in Earth's rest-frame velocity against orbital phase (dashed lines mark the transit contacts T<sub>1</sub>, T<sub>4</sub>). <b>Bottom row:</b> the corresponding 1D cross-correlation functions averaged over ingress (blue) and egress (red), which isolate the leading and trailing limbs respectively. <b>Left:</b> a H₂O signal from HD 189733 b with a modest day-to-nightside wind; the two limbs share nearly the same velocity and merge into a single, slightly asymmetric trail (the leading and trailing limbs enter and leave transit at slightly different times), and the ingress and egress CCFs peak close together. <b>Middle:</b> neutral iron in the ultra-hot Jupiter WASP-76 b, where a day-to-nightside wind and a time-varying limb contribution shift the CCF peak progressively bluewards from ingress to egress (Ehrenreich et al. 2020). <b>Right:</b> a hot Jupiter with a jet-like wind (≈ 7.7 km s<sup>-1</sup> towards the observer on the trailing limb and away from it on the leading limb), of the kind resolved in WASP-127 b by Nortmann et al. (2025). The velocity offset between the limbs now exceeds a resolution element, so the signal splits into two components (a morning component dominating from ingress to mid-transit and an evening component from mid-transit to egress) that cross near the centre of the transit and appear as two separate peaks in the ingress and egress CCFs. This is the observable consequence of the time-dependent weighting shown above, combined with the per-limb winds.
:::

### Example E: manual mass fractions (no EasyChem)

To use fixed mass fractions for all species without calling EasyChem:

```json
"use_easychem": false,
"mass_fractions": [0.74, 0.24, 3e-4, 5e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

The list order must exactly match the `species` list. H₂ and He dominate; trace species have small values. This manual mode is only needed when `use_easychem: false`; when EasyChem is active (the default), it computes mass fractions that automatically sum to 1 and include H₂ and He ([EasyChem docs](https://easychem.readthedocs.io/en/latest/content/notebooks/getting_started.html)), so the user does not need to manage this. With manual fractions, the values must sum to 1: the `Radtrans` class used internally does not normalise or fill the remainder ([petitRADTRANS docs](https://petitradtrans.readthedocs.io/en/latest/content/notebooks/spectral_model.html#Mass-fractions-and-mean-molar-masses)), so fractions that do not sum to 1 will produce incorrect atmospheric opacities without raising an error. Always include H₂ and He explicitly.

---

## Tutorial 4: CARMENES NIR simulation

> **Approximate run time:** ~10 to 15 min (CARMENES NIR, 23 orders, one night, no retrieval).

CARMENES covers the NIR 0.96 to 1.71 µm (Y+J+H, 28 orders) at R ≈ 80,400. In EXoPLORE, the CARMENES NIR wavelength grid is bundled with the repository and found automatically. A ready-to-run config is already provided at `configs/hd189733b_carmenes_transit.json`. In the following we explain every change relative to the ANDES reference config, so that these choices can be adapted to other instruments.

> **Limb asymmetries are disabled in the CARMENES config** (`limb_asymmetries: false`). CARMENES's lower resolving power and 28-order coverage make it less sensitive to signal differences between morning and evening limbs, so using a single 1D model is both faster (~4×) and more appropriate for this instrument. Limb asymmetries can be enabled at any time by adding `"limb_asymmetries": true` and choosing a `limb_divisions` mode.

### Step 1: Create the config (if not using the bundled one)

```bash
cp configs/hd189733b_andes_transit_clean.json configs/hd189733b_carmenes_transit.json
```

### Step 2: Change the instrument block

```json
"instrument": {
  "name": "CARMENES_NIR",
  "observatory": "caha",
  "pixels_per_resolution_element": 3.3,
  "order_indices": [0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,21,22,23,24,25,26,27],
  "split_detectors": false,
  "convolve_to_resolution": true
}
```

The critical change is `"name": "CARMENES_NIR"`. The wavelength grid (`wave_CARMENES_NIR.fits`) is found automatically (it does not need to be copied).

**Excluded orders (9, 10, 18, 19, 20):** this selection follows Alonso-Floriano et al. (2019) for this specific dataset and observing conditions. Orders 18 to 20 (~1.9 µm water band) are consistently opaque from Calar Alto and safely discarded. Orders 9 and 10 (~1.4 µm water band) are more PWV-dependent (under dry conditions they can retain usable signal and may be worth keeping). In general, the optimal selection requires inspecting the telluric transmission in the data being analysed; the 23-order list here is a conservative starting point. All 28 orders can be restored by setting `"order_indices": []`.

### Step 3: Disable limb asymmetries and point to CARMENES inputs

```json
"atmosphere": {
  "limb_asymmetries": false
},
"paths": {
  "inputs_dir": "inputs/CARMENES_NIR/HD189733b/"
},
"tellurics": {
  "reference_telluric_file": "inputs/CARMENES_NIR/HD189733b/tellurics/tell_ref_airmass_1.0.fits"
}
```

### Step 4: Observation settings

The bundled config uses the Alonso-Floriano et al. (2019) reference night (2017-09-07, 45 spectra at 198 s cadence), reproduced with real timestamps and airmass from `reference_night/`:

```json
"observation": {
  "exposure_time_seconds": 198.0,
  "readout_time_seconds": 20.0,
  "specific_event": true,
  "specific_T0_bjd": 2458004.425264
},
"noise": {
  "noise_choice": "SNR"
}
```

:::{note}
A web-based [CARMENES ETC](https://carmenes-etc.lsw.uni-heidelberg.de/) is available and provides per-order S/N estimates for both the VIS and NIR channels as HTML tables (maximum and median S/N per diffraction order, wavelength ranges in Å). EXoPLORE supports a fully ETC-based Mode A for CARMENES (no reference night required) using the same mechanism as ANDES. To use it:

1. Run the CARMENES ETC for your target and copy the per-channel table into a plain text file (one row per order, space-separated).
2. Convert to a FITS file with `scripts/carmenes_etc_to_fits.py`:

```bash
python scripts/carmenes_etc_to_fits.py \
    --input_nir  etc_nir.txt \
    --planet     HD189733b \
    --output_dir inputs/CARMENES_NIR/HD189733b/
```

3. Set `observation.specific_event: false` and leave `noise.fixed_snr` unset. EXoPLORE will find `CARMENES_NIR_ETC_WAVE_SNR_HD189733b.fits` automatically and synthesise the JD grid from the config exposure time.
:::

For a fully synthetic night without a reference night but with a constant SNR across all orders (simpler alternative to the ETC file):

```json
"observation": {
  "specific_event": false
},
"noise": {
  "noise_choice": "SNR",
  "fixed_snr": 150.0
}
```

:::{tip}
**Creating EXoPLORE input files from a real CARMENES night**

If you have your own CARMENES data and want to use it as a reference night, EXoPLORE includes a preparation script that reads the raw per-exposure FITS files produced by the CARACAL pipeline and writes the `julian_date`, `airmass`, `sig`, `snr`, `berv`, and per-order spectra files that the simulator expects.

```bash
python scripts/prepare_carmenes_night.py \
    --night_dir  /path/to/raw/night/ \
    --output_dir inputs/CARMENES_NIR/MyPlanet/reference_night/ \
    --night_index 0 \
    --first car-20170907T20h00m00s-sci-gtoc-nir_A.fits \
    --last  car-20170907T23h59m59s-sci-gtoc-nir_A.fits
```

The script selects exposures between `--first` and `--last`, removes Fabry-Pérot calibration frames, discards exposures with mean S/N below 20 (adjustable with `--snr_threshold`), corrects NaN pixels with the per-exposure median, and writes all output files with the correct shapes and night index suffix. Run `python scripts/prepare_carmenes_night.py --help` for all options.
:::

### Step 5: Run

```bash
python -u scripts/run_exoplore.py configs/hd189733b_carmenes_transit.json --run
```

With the default config (seed = 12345, specific_event reference night, 23 orders), the peak S/N is approximately 7σ at the correct (Kp, Vsys). This is substantially lower than the ANDES result for the same planet, a direct consequence of three compounding limitations of this dataset: CARMENES NIR covers 23 orders versus 76 for ANDES, its resolving power (R ≈ 80,400) is lower than ANDES (R = 100,000), and the reference night provides only 45 spectra compared to roughly 390 for the ANDES simulation. The simulation completes in approximately 5 to 10 minutes on a modern desktop.

---

## Tutorial 5: Multiple nights

> **Approximate run time:** ~5 to 10 min (single order, two nights, no retrieval).

### 5a: Identical nights (any instrument)

To simulate N consecutive transits treated as identical copies of night 1:

```json
"observation": {
  "n_nights": 4,
  "different_nights": false
}
```

All four nights share the same airmass profile, SNR, and noise model. CCF matrices are co-added coherently, and consequently the expected detection S/N scales as √N.

### 5b: Distinct nights with CARMENES_NIR (real data)

Set `different_nights: true` when each night has its own airmass profile, PWV, or cadence. One complete set of reference files is required per night, named with a zero-based night index suffix.

**Required files under `inputs/CARMENES_NIR/<planet>/reference_night/`:**

```
sig_0.fits          sig_1.fits          ...   # per-pixel uncertainties
snr_0.fits          snr_1.fits          ...   # SNR array
julian_date_0.fits  julian_date_1.fits  ...   # BJD timestamps
airmass_0.fits      airmass_1.fits      ...   # airmass time series
```

When `use_real_data: true`, each night `b` may have a different number of spectra `n_spectra_b` (read from its `julian_date_{b}.fits`). The simulator pads all arrays to the maximum night length with NaN and processes each night only on its valid rows. With `use_real_data: false` all nights share the same number of spectra.

**If `use_real_data: true`** (running the analysis pipeline on real observed spectra), also provide per-night BERV and spectra files:

```
observations_berv_0.fits                     # BERV for night 0
observations_berv_1.fits                     # BERV for night 1
observations_night_0_order_K.fits            # Real spectra for night 0, order K
observations_night_1_order_K.fits            # Real spectra for night 1, order K
```

Here `K` is the absolute spectral order number from the instrument's `order_selection` array, not a zero-based counter.

**Config for distinct nights:**

```json
"instrument": { "name": "CARMENES_NIR" },
"observation": {
  "n_nights": 3,
  "different_nights": true,
  "specific_event": true
}
```

Set `specific_event: true` so that per-night Julian dates from the files are used to compute transit phases.

### 5c: Distinct nights with ANDES (fully synthetic)

`different_nights: true` is fully supported for ANDES. Because ANDES uses a theoretical ETC for the SNR model, no JD or airmass files are needed: the simulator synthesises the per-night JD grids automatically, placing successive transit epochs at T₀ + n × P and computing the real airmass from the planet's sky coordinates at those times (when `use_accurate_airmass: true`). Each night thus gets its own airmass evolution naturally. If you prefer to control the airmass manually (for hypothetical scheduling studies), set `use_accurate_airmass: false` and adjust `airmass_limits`, the same `airmass_evolution` shape applies to all nights, though the limits can differ by editing the config before each run.

**What makes each simulated night different** (and why combining them helps):

- **PWV:** set independently per night via `tellurics.pwv_mm_per_night`, e.g. `[2.5, 3.5]`. Different PWV means different telluric depths per order. Orders that are heavily absorbed on a high-PWV night may be partially usable on a lower-PWV night, so the stacked dataset probes more of the spectrum than any single night.
- **JD grid and airmass:** successive transit epochs give a different airmass profile per night (the target may transit at higher or lower airmass depending on the season).
- **Noise realisation:** each night draws an independent noise realisation (`noise_seed` advances per night internally), so noise patterns do not correlate between nights.

Combining N independent nights therefore gives both a √N improvement in S/N and broader effective spectral coverage, as the per-night telluric masks partially complement each other.

**Step 1, Prepare per-night SkyCalc files**

```bash
# Night 0: PWV = 2.5 mm (taken from pwv_mm_per_night[0])
python scripts/generate_skycalc_inputs.py configs/hd189733b_andes_dn_run2.json --night 0

# Night 1: PWV = 3.5 mm (taken from pwv_mm_per_night[1])
python scripts/generate_skycalc_inputs.py configs/hd189733b_andes_dn_run2.json --night 1
```

The script finds the next two observable transits from Paranal, builds the exposure JD grids using the config's `exposure_time_seconds` and `pre_event_hours`, and queries ESO SkyCalc for each exposure. Files are written to:

```
inputs/ANDES/HD189733b/Skycalc_full_event/
  night_0/Fixed_PWV/
    tell_spec_0.fits
    tell_spec_1.fits
    ...
  night_1/Fixed_PWV/
    tell_spec_0.fits
    ...
```

**Step 2, Configure the simulation**

A ready-made two-transit config is provided at `configs/hd189733b_andes_dn_run2.json`. The key settings are:

```json
"observation": {
  "n_nights": 2,
  "different_nights": true,
  "specific_event": false,
  "exposure_time_seconds": 30.0
},
"tellurics": {
  "use_full_skycalc": true,
  "constant_pwv": true,
  "pwv_mm": 2.5,
  "pwv_mm_per_night": [2.5, 3.5],
  "airmass_limits_per_night": [[1.4, 1.7], [1.1, 1.5]]
},
"paths": {
  "inputs_dir": "inputs/ANDES/HD189733b/"
}
```

The simulator constructs the per-night telluric path automatically as `{inputs_dir}Skycalc_{flag_event}/night_{n}/Fixed_PWV/`. No additional configuration is needed beyond `inputs_dir`.

**Step 3, Run**

```bash
python -u scripts/run_exoplore.py configs/hd189733b_andes_dn_run2.json --run 2>&1 | tee run_log.txt
```

> **Validated result:** Night 0 (airmass 1.4 to 1.7, PWV 2.5 mm) gives 33.4σ; night 1 (airmass 1.1 to 1.5, PWV 3.5 mm) gives 30.0σ; combined 47.9σ (76 orders, BL19 pipeline, H₂O template). Night 1 is slightly lower despite the better airmass because the higher PWV masks more orders in the 1.4 µm water band. The combined result is consistent with the expected gain from stacking two independent nights with different observing conditions. For a single-order test use `instrument.order_indices: [35]`.

For a one-transit reference run (single night), use `configs/hd189733b_andes_dn_run1.json`.

The simulator writes diagnostic products both per individual night and for the combined stack. For each night `b` the `plots/` subdirectory contains a CCF Earth-rest-frame matrix, a 1D CCF at the best-K<sub>P</sub> slice, and a Kp-Vsys S/N map labelled `..._night{b}_...`. The combined products appear alongside them. In addition, a single overlay figure showing all individual nights and the combined result is written as `1D_CCF_..._nights_combined.pdf/png`.

:::{figure} figures/tutorial5c_1dccf_threenights.png
:width: 90%
:align: center
1D CCFs (in units of S/N) at the K<sub>P</sub> of maximum significance in their respective maps, co-added over 76 ANDES YJHK spectral orders and the full in-transit sequence, for the two-night HD 189733 b simulation (Tutorial 5c). Night 0 (blue, airmass 1.4 to 1.7, PWV 2.5 mm) and night 1 (red, airmass 1.1 to 1.5, PWV 3.5 mm) are shown individually alongside their combined result (black). The dashed line marks v<sub>rest</sub> = 0. Night 1 is slightly lower in S/N despite the better airmass because the higher PWV increases the telluric opacity in the 1.4 µm water band, masking a larger fraction of the spectral orders. The combination recovers the expected improvement from stacking two independent nights with complementary observing conditions.
:::

---

## Tutorial 6: Enable Bayesian retrieval

> **Approximate run time:** ~28 min on an Apple Mac Studio M2 Ultra (64 GB RAM), running `configs/hd189733b_carmenes_retrieval_blain24_noiseless.json` (CARMENES NIR, order 23, noiseless, Blain24 pipeline and log-likelihood, nested sampling at 200 live points). The retrieval dominates the total; the forward-model blocks that precede it are fast for a single order.

:::{note}
**From detection to atmospheric constraints**

A peak in the Kp-Vsys map establishes that a species is present and yields its approximate orbital and rest-frame velocity. It does not, however, provide a posterior probability distribution over the atmospheric parameters, that is, the abundances, the temperature structure, or the chemical composition ratios. To go from a detection to a constraint of the form log₁₀(X<sub>H₂O</sub>) = -4.2 ± 0.3 we must perform a Bayesian retrieval.

In the retrieval, a petitRADTRANS forward model is computed for a trial set of parameters, Doppler-shifted to each exposure, and processed through the same preparation pipeline as the data. It is then compared to the data through a log-likelihood function, and a nested sampler (MultiNest) or an MCMC (emcee) explores the parameter space to map the posterior. We caution that the forward model must be prepared identically to the data at every likelihood evaluation. This is the pipeline-bias requirement, and it is the subject of Tutorial 7.

Three log-likelihood formulations are available (BL19, Blain24, Gibson22), which differ in how they treat the per-pixel noise and the model amplitude scaling. For a polynomial pipeline with reliable per-pixel uncertainties the Blain24 formulation is a natural choice, although the most suitable formulation depends on the dataset (see the [Concepts primer](concepts.md#5-from-cross-correlation-to-a-likelihood)).
:::

The retrieval is enabled within the **same simulation** by setting `retrieval.enabled: true`. It runs at the end of the simulation, after the matrices have been built and the cross-correlation has been computed, and reads those products directly from memory. Consequently, it requires neither a separate run nor the reloading of saved files. To repeat a retrieval on a past simulation, the full simulation must be re-run, since the intermediate matrices are not reloaded from disk.

The example below uses the Blain24 log-likelihood (Blain, Sánchez-López & Mollière 2024, AJ, 167, 179), which accounts for the per-order noise scaling:

```json
"retrieval": {
  "enabled": true,
  "sampler": "nested_sampling",
  "log_likelihood": "Blain24",
  "dimensionality": "1D",
  "live_points": 200
}
```

The `dimensionality` field selects the free parameters, and the `log_likelihood` must be compatible with it:

| `dimensionality` | `log_likelihood` | Free parameters |
|---|---|---|
| `"1D"` | `"BL19"` or `"Blain24"` | log(VMR), Kp, T_eq, v_rest |
| `"1D_Gibson22"` | `"Gibson22"` **only** | log(VMR), Kp, T_eq, v_rest, β |
| `"1D_CtoO_met"` | `"BL19"` or `"Blain24"` | C/O, metallicity (EasyChem) |
| `"1D_extended"` | `"BL19"` or `"Blain24"` | log(VMR) × 6 species, Kp, T_eq, v_rest |
| `"1D_extended_fast"` | `"BL19"` or `"Blain24"` | log(VMR) × 6 species, T_eq (Kp/v_rest fixed) |
| `"2D"` | `"BL19"` or `"Blain24"` | morning/evening VMR + T_eq |

We note that `1D_Gibson22` must be paired with `Gibson22`, and no other. The β parameter is the noise-scaling degree of freedom that makes the Gibson22 formulation self-consistent; pairing it with Blain24 or BL19 would sample β without ever using it, yielding a meaningless posterior. The simulator raises a `ValueError` on any invalid combination.

The `live_points` field sets the number of active points that MultiNest maintains during nested sampling. A larger value produces a more accurate posterior and a more reliable evidence estimate, at a run time that grows roughly linearly with it. In our experience 200 is adequate for a four-parameter retrieval, and 500 to 1000 is appropriate for publication-quality posteriors. The full run time is typically one to four hours, depending on `live_points`, the number of orders, and the machine.

The retrieval requires three additional dependencies:

```bash
pip install pymultinest emcee corner
```

Here `pymultinest` provides the nested sampler, `emcee` the MCMC sampler, and `corner` the posterior corner plots. PyMultiNest further requires the compiled MultiNest Fortran library (see its [installation guide](https://github.com/JohannesBuchner/PyMultiNest)).

The MultiNest chain files are written to `<run_name>/matrices/matrices_<run_name>/` and the corner plot to `<run_name>/plots/`. See [docs/outputs.md](outputs.md#retrieval-outputs) for how to read and plot the posteriors.

A ready-to-run config is provided at `configs/hd189733b_carmenes_retrieval_blain24_noiseless.json` (CARMENES NIR, order 23, noiseless, Blain24 pipeline and log-likelihood, `1D` dimensionality):

```bash
python -u scripts/run_exoplore.py configs/hd189733b_carmenes_retrieval_blain24_noiseless.json --run
```

```{figure} figures/tutorial6_retrieval_corner.png
:width: 75%
:align: center

Posterior distributions from a `1D` Blain24 retrieval of a single noiseless CARMENES NIR order (order 23) of HD 189733 b. The free parameters are the logarithmic water volume mixing ratio log₁₀(X<sub>H₂O</sub>), the orbital velocity semi-amplitude K<sub>P</sub>, the equilibrium temperature T<sub>eq</sub>, and the rest-frame velocity v<sub>rest</sub>. The diagonal panels show the marginal distribution of each parameter, and the off-diagonal panels the pairwise joint distributions, with contours at the 68 and 95 per cent credible levels. Red lines mark the injected truth values, and dashed lines the 16th, 50th, and 84th posterior percentiles. All four truths are recovered within the credible intervals: log₁₀(X<sub>H₂O</sub>) = -2.9 ± 0.5, K<sub>P</sub> = 150.0 ± 4.0 km s<sup>-1</sup>, T<sub>eq</sub> = 1170 ± 80 K, and v<sub>rest</sub> = 0.0 ± 0.2 km s<sup>-1</sup>.
```

---

## Tutorial 7: Assessing retrieval pipeline bias

> **Approximate run time:** the illustrative scripts run in seconds; each single-order retrieval takes ~10 to 20 min, so the full set of six (three noiseless, three noisy) is ~1 to 3 hours.

Before trusting retrieved atmospheric parameters from real data, it is essential to verify that the pipeline itself does not introduce systematic offsets in the posterior. This is commonly referred to as a **pipeline bias test**, or **unbiasedness test**. EXoPLORE supports a clean, fast version of this test using noiseless simulated data.

### What is pipeline bias?

A retrieval pipeline is *unbiased* if, when the forward model used in the likelihood is the same model used to generate the simulated data, the posterior recovers the injected truth values within the expected uncertainty. Biases can arise from several sources: i) the same masks not being applied in the same order to both data and model; ii) the SYSREM projector not correctly filtering the template; iii) normalisation steps removing or distorting the planetary signal; iv) incorrect Doppler-shift calculation. We note that none of these issues manifest in CCF-based detection maps (they only become apparent when the likelihood function is evaluated).

### Why noiseless data?

In order to isolate pure pipeline systematics, the test should be run with `noiseless: true`, which removes the stochastic noise floor. With perfect noiseless data and the correct forward model, the likelihood should peak sharply at the injected truth. Any offset is consequently pure pipeline bias, not noise. With noisy data, biases can be hidden inside the noise uncertainty and require many injection-recovery realisations to detect.

> **We note that** SYSREM-based pipelines (ASL19, Gibson22) may be problematic with noiseless data, as SYSREM can over-subtract signal in the absence of noise. BL19 and Blain24 are the recommended choices for noiseless bias testing. Gibson22 with a β prior pinned near 1 (see below) is under evaluation.

### Step 1: Configure the test

To keep the retrieval fast (minutes rather than hours), we restrict the simulation to a single spectral order with strong signal from the target species. For H₂O in CARMENES NIR, order 23 (0-indexed) provides good sensitivity. To that end:

```json
{
  "instrument": {
    "name": "CARMENES_NIR",
    "order_indices": [23]
  },
  "observation": {
    "n_nights": 1,
    "noiseless": true
  },
  "atmosphere": {
    "limb_asymmetries": false,
    "planet_model": {
      "species": ["H2", "He", "H2O", "CH4", "NH3", "CO", "H2S", "HCN"],
      "use_easychem": true,
      "carbon_to_oxygen_ratio": 0.41,
      "metallicity_wrt_solar": 0.53
    }
  },
  "pipeline": {
    "name": "BL19",
    "prepare_template": true
  },
  "retrieval": {
    "enabled": true,
    "sampler": "nested_sampling",
    "log_likelihood": "Blain24",
    "dimensionality": "1D",
    "live_points": 200,
    "constant_efficiency_mode": false
  }
}
```

Ready-made configs are provided for each pipeline. Run the three retrievals sequentially:

```bash
# Brogi & Line (2019), BL19 preparation pipeline, BL19 log-likelihood
python -u scripts/run_exoplore.py configs/hd189733b_carmenes_retrieval_bl19_noiseless.json --run

# Blain et al. (2024), Blain24 preparation pipeline, Blain24 log-likelihood
python -u scripts/run_exoplore.py configs/hd189733b_carmenes_retrieval_blain24_noiseless.json --run

# Gibson et al. (2022), Gibson22 preparation pipeline, Gibson22 log-likelihood, β pinned near 1
python -u scripts/run_exoplore.py configs/hd189733b_carmenes_retrieval_gibson22_noiseless.json --run
```

Each run takes a few minutes with 400 live points and one order.

### Step 2: Run

See the terminal commands in Step 1 above. Run all three configs to reproduce the overlay corner plot.

### Step 3: Interpret the results

Open `<run_name>/plots/retrieval_night_0_corner.pdf`. The truth values (plotted as red lines) should lie within the posterior contours. In particular, the following parameters should be checked:

| Parameter | What a bias would look like |
|---|---|
| Kp (km/s) | Posterior offset from the planet's true orbital velocity |
| T_eq (K) | Posterior offset from the simulated equilibrium temperature |
| v_wind (km/s) | Posterior offset from 0 (no wind injected) |
| log₁₀(VMR) | Broader, but centred near the injected abundance |

The log-evidence `ln Z` printed to the terminal should be a meaningful negative number (e.g. -10 to -50). A value near 0 indicates the likelihood is flat and the test has failed (the run log should be checked for error messages).

**Passing criterion:** Kp, T_eq, and v_wind posteriors all cover their truth values within 1 to 2σ. A result like the one obtained during EXoPLORE development:

```
Kp   = 149.9 km/s  (truth 149.4 km/s)   ← <1 km/s offset
T_eq = 1186  K     (truth ~1200 K)       ← within 1σ
v_wind = -0.06 km/s (truth 0)            ← essentially zero
ln Z = -10.8                             ← meaningful evidence
```

confirms no pipeline bias on the dynamical parameters.

:::{figure} figures/tutorial_retrieval_bias_corner.png
:width: 90%
:align: center
Posterior probability distributions for the retrieved atmospheric parameters of HD 189733 b (CARMENES NIR, order 23, noiseless simulation), obtained with the log-likelihood formulations of Brogi & Line (2019; blue), Blain et al. (2024; red), and Gibson et al. (2022; green) using nested sampling (200 live points for Brogi & Line 2019 and Blain et al. 2024; 400 live points for Gibson et al. 2022). Gibson et al. (2022) includes the additional noise-scaling parameter β, shown in the bottom row and column; β is pinned near 1 via an informative prior [0.999, 1.001] (see below). Contours enclose the 68 and 95 per cent credible regions. Black dashed lines mark the injected truth values (log₁₀(X<sub>H₂O</sub>) = -3.0, K<sub>P</sub> = 149.4 km s<sup>-1</sup>, T<sub>eq</sub> = 1170 K, v<sub>wind</sub> = 0 km s<sup>-1</sup>, β = 1). All three formulations recover all truth values within 1 to 2σ.
:::

Once you have run the three retrievals, the overlay corner plot can be generated with:

```bash
python scripts/plot_corner_overlay.py \\
  --output-root /path/to/EXoPLORE_clean_run/HD189733b/CARMENES_NIR/transit \\
  --runs  BL19_withsignal_1nights_SNR_comb1_simdata_noiseless_stdnoisex1 \\
          Blain24_withsignal_1nights_SNR_comb1_simdata_noiseless_stdnoisex1 \\
          Gibson22_withsignal_1nights_SNR_comb1_simdata_noiseless_stdnoisex1 \\
  --labels "Brogi & Line (2019)" "Blain et al. (2024)" "Gibson et al. (2022)" \\
  --truths -3.0 149.4 1170.0 0.0 \\
  --output docs/figures/tutorial_retrieval_bias_corner.png
```

Key optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--n-bins` | 50 | Number of bins for 1D histograms and 2D density maps |
| `--smooth-sigma` | 1.2 | Gaussian smoothing applied to 2D histograms before contouring |
| `--pad` | 0.18 | Fractional padding added to each axis range |
| `--dpi` | 180 | Output resolution |
| `--colors` | blue red green | One hex colour per run |

### Noisy retrieval with ANDES YJHK

The noiseless case above demonstrates bias in the absence of noise. To verify that all three pipelines also recover truth under realistic noise, the same test can be run with a single ANDES YJHK order at J-band (order 35, 1310 to 1326 nm), where H₂O provides strong signal even in one order. The ETC SNR in this order is ≈1256 per resolution element per exposure. The ANDES SNR tables are given per resolution element; EXoPLORE converts them to a per-pixel SNR internally via SNR<sub>pixel</sub> = SNR<sub>resel</sub> / √m, where m = `pixels_per_resolution_element` (2.5 for ANDES), so the per-pixel SNR here is ≈800. Even at this single order the signal is strong enough to make the posteriors tight despite the added noise.

```bash
# BL19, noisy, ANDES YJHK, order 35
python -u scripts/run_exoplore.py configs/hd189733b_andes_retrieval_bl19_noisy.json --run

# Blain24, noisy, ANDES YJHK, order 35
python -u scripts/run_exoplore.py configs/hd189733b_andes_retrieval_blain24_noisy.json --run

# Gibson22, noisy, ANDES YJHK, order 35 (β free; β prior [0.9, 1.1] avoids divergence)
python -u scripts/run_exoplore.py configs/hd189733b_andes_retrieval_gibson22_noisy.json --run
```

With realistic noise the Gibson22 β parameter is identifiable and no prior pinning is needed.

:::{figure} figures/tutorial_retrieval_bias_corner_noisy.png
:width: 90%
:align: center
Posterior probability distributions for the retrieved atmospheric parameters of HD 189733 b (ANDES YJHK, order 35, 1310 to 1326 nm, noisy simulation), obtained with Brogi & Line (2019; blue), Blain et al. (2024; red), and Gibson et al. (2022; green) using nested sampling with 200 live points. The posterior widths differ between formulations for this single noisy order: the Brogi & Line (2019) constraints are broader, while those of Blain et al. (2024) and Gibson et al. (2022) are tighter (Kp to ±2 km s<sup>-1</sup>, T<sub>eq</sub> to ±42 K, v<sub>wind</sub> to ±0.1 km s<sup>-1</sup>). All three formulations recover the truth values within 1 to 2σ, with no evidence of systematic pipeline bias; the difference in width reflects the heteroscedastic noise of a single order (see the text below) rather than a general ordering of the methods. The β parameter of Gibson et al. (2022) is identifiable on noisy data (β = 0.994 ± 0.001) and requires no prior pinning.
:::

The overlay corner plot is generated with:

```bash
python scripts/plot_corner_overlay.py \\
  --output-root /path/to/EXoPLORE_clean_run/HD189733b/ANDES_YJHK/transit \\
  --runs  BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1 \\
          Blain24_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1 \\
          Gibson22_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1 \\
  --labels "Brogi & Line (2019)" "Blain et al. (2024)" "Gibson et al. (2022)" \\
  --truths -3.0 149.4 1170.0 0.0 \\
  --output docs/figures/tutorial_retrieval_bias_corner_noisy.png
```

### Why the posterior widths differ between formulations

The difference in posterior width reflects how each formulation treats the
noise, and it informs the choice of likelihood for a given dataset rather than a
general ranking of the methods. All three assume Gaussian noise and differ in
how they treat the **noise scale** (the full derivation is in the
[Concepts primer](concepts.md#5-from-cross-correlation-to-a-likelihood)).

- Brogi & Line (2019) estimates a **single** global noise level per spectrum
  from the residuals themselves. No per-pixel uncertainty enters its formula.
- Blain et al. (2024) and Gibson et al. (2022) use the **known** per-pixel
  uncertainty, so each pixel is weighted by `1/σ²`.

The practical consequence depends on whether the noise is uniform across
pixels. The script `scripts/illustrate_likelihood_weighting.py` examines this on
a controlled toy spectrum, comparing the two formulations under uniform and
non-uniform noise of **identical total variance**:

```bash
python scripts/illustrate_likelihood_weighting.py \
  --output docs/figures/likelihood_weighting.png
```

```{figure} figures/likelihood_weighting.png
:width: 100%
:align: center

With uniform noise (left) the two formulations are nearly identical. With
heteroscedastic noise of the same total variance (right), concentrated in 15
per cent of the pixels, the Blain et al. (2024) constraint is about five times
tighter in this toy case. The single global noise estimate of Brogi & Line
(2019) is raised by the noisy minority of pixels, which dilutes the contribution
of the clean majority, whereas the per-pixel weighting of Blain et al. (2024)
assigns those noisy pixels little weight.
```

Real spectra are frequently heteroscedastic (telluric cores, blaze edges, and
channels near deep lines carry elevated noise), which may explain why the Blain
et al. (2024) and Gibson et al. (2022) formulations produce tighter posteriors
than Brogi & Line (2019) for this single noisy ANDES order. On noiseless data
the effect vanishes, since there is no noise structure to exploit, which is why
the three converge in the noiseless corner plot earlier in this tutorial. As
noted in the Concepts primer, the Brogi & Line (2019) formulation was developed
and validated on simulated, photon-noise-dominated CRIRES data, where it was
shown to recover statistically correct credibility intervals; its broader
posteriors for this single noisy ANDES order reflect the heteroscedastic noise
of that particular test rather than a general limitation of the method.

### Supported pipeline / likelihood combinations for bias testing

| Preparing pipeline | Log-likelihood | Works noiseless? | Notes |
|---|---|---|---|
| BL19 | BL19 | Yes | Matched filter, noise-independent |
| BL19 | Blain24 | Yes | Chi-squared with propagated noise |
| Blain24 | Blain24 | Yes | Polynomial throughput + telluric removal |
| Gibson22 | Gibson22 (β pinned) | **Yes** | Set `prior_bounds` to pin β ∈ [0.999, 1.001]; see below |
| ASL19 | any | **TBD** | SYSREM may over-subtract noiseless data; not yet evaluated |
| any | Gibson22 (β free) | **No** | β diverges when noise → 0 regardless of preparing pipeline |

For ASL19, `noiseless: false` with realistic noise is the safe choice. Gibson22 with a pinned β prior works correctly on noiseless data, as demonstrated by the corner plot above.

#### Testing Gibson22 noiseless with a pinned β prior

There is a principled way to run a Gibson22 noiseless bias test: use an informative prior that pins β near 1. To do so, set a very narrow prior range in the config:

```json
"retrieval": {
  "dimensionality": "1D_Gibson22",
  "log_likelihood": "Gibson22",
  "prior_bounds": {
    "1D_Gibson22": [[-8.0, 0.0], [85.0, 200.0], [400.0, 1500.0],
               [-25.0, 25.0], [0.999, 1.001]]
  }
}
```

β is still a free parameter (MultiNest is happy) but is effectively fixed to 1, preventing divergence. Although a workaround for illustration purposes, this can also be interpreted as a correctly informative prior encoding the knowledge that noiseless data has perfectly calibrated uncertainties. The recovered T_eq, Kp, and v_wind posteriors consequently isolate purely pipeline-induced biases, with no noise contribution. If an offset survives, it is real; if it disappears, it was a noise fluctuation.

---

## Tutorial 8: Detection significance and statistical validation

> **Approximate run time:** the illustrative scripts (`illustrate_significance_sampling.py`, `illustrate_pp_calibration.py`) run in seconds; the full statistical study is ~20 to 30 min per velocity step; the real p-p calibration set (30 single-order retrievals) takes several hours.

The significance of a detection has more than one defensible definition (see the [Concepts primer](concepts.md#6-detection-significance-three-complementary-measures)), and a careful analysis reports more than one. This tutorial covers the three significance metrics EXoPLORE provides, a subtle bias in the Welch t-test that depends on the cross-correlation velocity step, and the p-p (coverage) plot, which validates whether the uncertainties returned by a retrieval can be trusted.

### The significance metrics

Significance can be assessed along two distinct routes, and they answer slightly different questions. The first is cross-correlation based: it asks whether a template matches the data at the planet's velocity, and quantifies the match. The second is retrieval based: it compares the Bayesian evidence of a model containing the species with that of a null model, and so accounts for the full parameter space and prior volume. The two are complementary and should not be conflated.

The cross-correlation metrics are selected in the `cross_correlation` block:

| Metric | Config | What it measures |
|---|---|---|
| Cross-correlation S/N | `"ccf_snr": true` | Peak CCF value divided by the standard deviation of the CCF away from the planet trail |
| Welch's t-test | `"welch_ttest": true` | Statistical separation between the in-trail and out-of-trail distributions of CCF values |

Setting `"all_significance_metrics": true` computes both in the same run, writing both Kp-Vsys maps. These metrics involve only the cross-correlation and do not invoke the retrieval.

The retrieval-based metric is the **Bayesian evidence**, obtained from the retrieval (Tutorial 6). Running the retrieval for the model and for a null model without the species yields two evidences whose ratio is a Bayes factor, which can be mapped to a frequentist sigma. This is a separate analysis from the cross-correlation, governed by the `retrieval` block, and is considerably more expensive.

### The velocity-sampling bias of the Welch t-test

The S/N and the Welch t-test do not always agree, and where they disagree there is a reason worth understanding. A cross-correlation function is a smooth, matched-filter response, so its values are correlated on the scale of the resolution element. The Welch t-test assumes the in-trail and out-of-trail values it compares are independent. When the CCF is sampled more finely than the resolution element (oversampling, which happens whenever a small velocity step is used), the in-trail window holds more points than there are independent ones, and the t-test treats correlated points as independent. This inflates the Welch significance. The S/N, which depends only on the peak value and the noise scatter, is unaffected.

The script `scripts/illustrate_significance_sampling.py` demonstrates this on a controlled toy cross-correlation, drawing many noise realisations of the same injected peak and scanning the velocity step:

```bash
python scripts/illustrate_significance_sampling.py \
  --output docs/figures/significance_sampling.png
```

```{figure} figures/significance_sampling.png
:width: 80%
:align: center

Median significance of a fixed injected peak across many noise realisations, as a function of the cross-correlation velocity step. The S/N (blue) is flat: it does not depend on how finely the CCF is sampled. The Welch t-value (red) rises steeply once the step falls below the resolution element (dashed line), because the in-trail CCF values become correlated and the t-test counts them as independent. Shaded regions are the 16 to 84 percentile spread over realisations.
```

To first order, the offset between the Welch and S/N significances is approximately √(Resolution / Sampling), the square root of the number of correlated points per resolution element. The exact factor depends on the correlation structure of the CCF (in the toy above the inflation is somewhat steeper, since correlated samples also reduce the in-trail variance estimate), so the relation should be read as a guide rather than a correction to be applied blindly. The practical consequence is that Welch significances are not directly comparable across analyses that use different velocity steps; the S/N is the more transportable metric, and any quoted significance should state the velocity step and in-trail window used.

**Running the full statistical study.** The toy isolates the mechanism. To reproduce the effect with the full forward model, run the simulator with many noise realisations of the same injected atmosphere: set `observation.n_nights` to the number of realisations, `observation.different_nights: false` (so the conditions are fixed and only the noise changes), `statistics.enabled: true`, and `cross_correlation.all_significance_metrics: true`. Keep `retrieval.enabled: false`: the statistical study and the `all_significance_metrics` flag concern only the cross-correlation, and leaving the retrieval enabled would run a full Bayesian retrieval on every realisation, which is unnecessary here and very slow. EXoPLORE then records, for each realisation, the S/N, the Welch significance, and the recovered Kp and Vrest, writing them into the run's `matrices` directory (the S/N statistics as `stats_*.npz` and the Welch t-test statistics as `stats_welch_*.npz`). Running the study at two velocity steps (`cross_correlation.velocity_step_kms`, for example 1.3 and 3.0 km/s) isolates the Welch inflation.

Each run already writes a single-case corner plot automatically (`Corner_plot_<run_name>.png`, one per metric when `all_significance_metrics: true`), produced by `exoplore.analysis.stats.plot_stats`. To overlay several cases (here the two metrics at the two velocity steps) in a single corner, the driver `scripts/plot_significance_study.py` reads the saved arrays:

```bash
python scripts/plot_significance_study.py \
  --oversampled-root /path/to/sig_study_oversampled \
  --critical-root    /path/to/sig_study_critical \
  --kp-truth 149.4 --vrest-truth 0.0 \
  --output docs/figures/significance_study_corner.png
```

```{figure} figures/significance_study_corner.png
:width: 85%
:align: center

Distributions of the recovered significance, K<sub>P</sub>, and V<sub>rest</sub> over 100 noise realisations of the same injected HD 189733 b atmosphere (ANDES, single order), for the S/N (blue) and Welch (red) metrics at the oversampled (solid) and critical (dashed) velocity steps. The S/N distributions for the two steps coincide (the metric is insensitive to the sampling), while the Welch distribution shifts to higher significance at the oversampled step (median 7.4 against 5.5). Dotted lines mark the injected truth values; the recovered K<sub>P</sub> and V<sub>rest</sub> scatter because a single order yields a modest per-order significance.
```

A single ANDES order carries enough signal for a per-order detection of a strong absorber such as H₂O even with realistic noise (here a median S/N of about 3.7), although at this modest per-order significance the recovered peak position varies appreciably from one noise realisation to the next, as the scatter in the figure shows. Co-adding orders (or nights) raises the significance and tightens the recovered K<sub>P</sub> and V<sub>rest</sub>.

### p-p plots: validating retrieval uncertainties

A retrieval can return a tight, confident-looking posterior whose credible intervals are nonetheless wrong. A p-p plot (probability-probability plot, also called a coverage plot) tests whether the credible intervals mean what they claim. The construction repeats an injection-recovery experiment many times: for each realisation with a known truth, one records the posterior percentile at which the truth lies, and from those percentiles builds a coverage curve, the fraction of realisations in which the truth falls inside the central credible interval of each nominal level. A calibrated retrieval follows the diagonal; a curve below the diagonal indicates posteriors that are too narrow (over-confident, under-covering), and a curve above indicates posteriors that are too wide.

The script `scripts/illustrate_pp_calibration.py` shows both cases on a fast linear-Gaussian toy, where the posterior is known analytically:

```bash
python scripts/illustrate_pp_calibration.py --output docs/figures/pp_calibration.png
```

```{figure} figures/pp_calibration.png
:width: 100%
:align: center

Coverage (p-p) plots from 300 simulations. Left: a well-specified inference (the noise assumed by the posterior matches the data) follows the diagonal. Right: an over-confident inference (the posterior underestimates the noise) sits below the diagonal, the signature of credible intervals that are too tight. The grey bands are the 1σ and 2σ binomial confidence regions for a finite number of simulations.
```

This diagnostic is the calibration counterpart of the precision-versus-accuracy problem (Concepts primer, [Section 7](concepts.md#7-precision-is-not-accuracy)): a 1D retrieval of an inhomogeneous atmosphere can produce tight posteriors that do not cover the truth at the stated rate, which a p-p plot exposes directly.

**The real retrieval version.** A true p-p plot requires running the retrieval on every realisation, so it is expensive. The example here uses the well-specified case: a one-dimensional atmosphere is injected and a one-dimensional Blain24 retrieval is performed (`configs/hd189733b_andes_retrieval_blain24_noisy.json`, a single ANDES order). Because the retrieval model matches the injected truth, the coverage curve should follow the diagonal, confirming that the Blain24 one-dimensional retrieval returns statistically calibrated uncertainties. We stress that this is a check of the inference machinery, not a scientific result: when the model matches the truth by construction, calibration is expected, and a diagonal coverage curve confirms only that the sampler and likelihood are statistically self-consistent. The scientifically relevant miscalibration arises from model mismatch, in particular from fitting a one-dimensional model to an atmosphere that is in reality multi-dimensional, and is not captured by this test. The well-specified p-p plot serves as a reference: it shows what a calibrated coverage curve looks like on real retrievals, and provides the baseline against which a genuine miscalibration would stand out.

A p-p plot requires a single truth per parameter to compute the coverage. An inhomogeneous (pseudo-2D) injection has no single truth for, say, the H₂O abundance, since the two limbs differ, so the coverage of a one-dimensional retrieval against such an injection is not well defined. The bias of a one-dimensional retrieval on an inhomogeneous atmosphere is therefore better shown by comparing the retrieved value with the injected limb values directly (in the manner of the corner plots in Tutorial 7), rather than through a p-p plot.

The driver `scripts/run_pp_calibration.py` automates the calibration test: it runs N independent noisy retrievals, each with its own noise seed (`base_seed + i`) and its own output directory (`pp_real_000`, `pp_real_001`, ...) so that nothing is overwritten, then computes the coverage curve for each parameter. It is **resumable**, so it can be left running and re-plotted at any time:

```bash
python scripts/run_pp_calibration.py \
  --config configs/hd189733b_andes_retrieval_blain24_noisy.json \
  --n 30 --base-seed 1000 --live-points 100 \
  --output-root /path/to/pp_calibration \
  --truths -3.0 149.4 1170.0 0.0 \
  --param-names "log10(X_H2O)" "Kp" "T_eq" "v_wind" \
  --figure docs/figures/pp_calibration_real.png
```

To rebuild the figure from whatever realisations have finished, without running anything further, add `--plot-only`. Thirty realisations at 100 live points on a single order are enough for a usable coverage curve; more realisations tighten it.

```{figure} figures/pp_calibration_real.png
:width: 75%
:align: center

Coverage (p-p) plot from 30 real Blain24 retrievals of HD 189733 b (ANDES, single order), one per independent noise realisation. The dynamical and thermal parameters (K_P, T_eq) lie close to the diagonal, while the H₂O abundance runs below it (over-confident) and the wind velocity slightly above it. The grey bands are the 1σ and 2σ ranges expected for 30 simulations. The curves are stepped because only 30 realisations were used.
```

This result is instructive precisely because it is **not** a clean diagonal. The example was set up to look well specified (a one-dimensional retrieval of a one-dimensional injection), but injection and retrieval are not in fact the same model, and the p-p plot, which is far more sensitive to model mismatch than a corner plot, exposes it. The atmosphere is injected with EasyChem equilibrium chemistry and the full set of opacity species, whereas the retrieval fits a single, pressure-independent H₂O abundance. A free constant abundance cannot reproduce an EasyChem profile, for several reasons that act together:

- The injection contains additional absorbers (CH₄, NH₃, CO, H₂S, HCN, and others) whose lines remain in the data but are absent from the H₂O-only retrieval model, acting as nuisance opacity.
- EasyChem produces a pressure-dependent abundance, while the retrieval fits a single constant value, so the retrieved number is an effective abundance.
- The mean molecular weight in EasyChem follows the full composition, whereas an H₂O-dominated retrieval assumes a different value, changing the scale height.

The dynamical and thermal parameters are recovered with calibrated uncertainties because the retrieval can match their injected behaviour (the temperature structure shares the same Guillot form and K_P is fixed by line positions), but the abundance is biased and over-confident. Read positively, the abundance miscalibration is a signature that the data carry information the constant-abundance model cannot absorb, including a sensitivity to the non-uniform vertical profile. Isolating that sensitivity cleanly would require a controlled experiment in which a single species is injected with an EasyChem profile and retrieved as a free constant, so that the nuisance-species and mean-molecular-weight confounders are removed.

---

## Troubleshooting

**Missing `--run` flag**, Running without `--run` only prints the simulation summary and exits. The `--run` flag must be added to actually execute the simulation.

**`FileNotFoundError` for a PHOENIX file**, `paths.phoenix_wave_file` and `paths.phoenix_flux_file` must be set to the absolute paths on the local machine. These files are ∼500 MB each and are not included in the repository. Download from https://phoenix.astro.physik.uni-goettingen.de.

**`FileNotFoundError` for petitRADTRANS opacity tables**, Set `paths.prt_input_data` to the `input_data/` directory inside the petitRADTRANS installation. Run `python -c "import petitRADTRANS; print(petitRADTRANS.__file__)"` to find the installation path.

**macOS `ImportError` for petitRADTRANS**, Add `export SDKROOT="$(xcrun --show-sdk-path)"` to the shell profile. This sets the Fortran compiler SDK path required by petitRADTRANS on macOS.

**Species not in pRT opacity database**, Not all species are available for all wavelength ranges. Check the [petitRADTRANS documentation](https://petitradtrans.readthedocs.io) for the list of available line lists. Unavailable species should be removed from the `species` list.

**Kp-Vsys map peak at wrong location**, Check `planet_params/YourPlanet.json`: `kp_kms` should match the literature value for the planet's orbital velocity semi-amplitude. Furthermore, check `systemic_velocity_kms` sign: negative = blueshifted system. Also verify `atmosphere.planet_model.wind_velocity_kms`: a non-zero wind shifts the detected Vsys by that amount.

**Slow simulation**, In order to speed up test runs, use `instrument.order_indices: [0, 5, 10, 20]` to run only a subset of orders, disable limb asymmetries (`limb_asymmetries: false`, which avoids the additional pRT calls for the morning and evening limbs), and use an isothermal profile while debugging.

**Fully masked order warnings**, If spectral orders in deep telluric bands are fully masked (all pixels fail SNR or negative-value checks), the simulator continues gracefully and writes a report to `<run_name>/warnings/fully_masked_orders_{run_name}.txt`. These orders are automatically excluded from CCF co-adding. This is expected behaviour and not an error.

**Simulation running all orders when you only want one**, `cross_correlation.order_selection` is a CCF-only filter. To restrict the entire simulation (Blocks 3 to 9) to a specific set of orders, use `instrument.order_indices`. For a single-order retrieval test: `"instrument": { "order_indices": [23] }`. We caution that forgetting this will run all orders through the forward model and make the retrieval far slower than necessary.

**Retrieval posterior is completely flat / log Z ≈ 0**, This means the log-likelihood function is returning the same value for every parameter combination. The run log should be checked for Python `Traceback` or `KeyError` lines printed before the MultiNest output, these are exceptions silently swallowed inside the Fortran/Python interface that cause pymultinest to return a default constant value. Common causes are: i) a missing key in the configuration dict passed to `preparing_pipeline` or `call_pRT` (check that `instrument.order_indices` is set and the config is otherwise valid); ii) using a SYSREM-based pipeline with `noiseless: true` without a pinned β prior (over-subtraction may render data identically zero for all models, use BL19 or Blain24, or Gibson22 with `prior_bounds` pinning β near 1).

**Retrieval corner plot shows `Too few points to create valid contours`**, The posterior is effectively flat (all 200 equal-weight samples uniformly cover the prior). See the point above. We caution that when using the Gibson22 likelihood with `noiseless: true`, the β hyperparameter diverges and the posterior cannot be constrained, switching to BL19 or Blain24 resolves this.
