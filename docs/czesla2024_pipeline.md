# Direct helium transmission with the `czesla2024` preparation pipeline

Choose `pipeline.name: "czesla2024"` to prepare an observed CRIRES+ transit
time series for direct He I transmission spectroscopy. The normal EXoPLORE
JSON configuration and runner select this workflow. Its products are a
stellar-frame transmission map, a planetary-frame spectrum and helium light
curves in both frames.

The scientific method and WASP-121 b benchmark come from **Czesla et al.
(2024), “The overflowing atmosphere of WASP-121 b: High-resolution He I
λ10833 transmission spectroscopy with VLT/CRIRES+,” A&A 692, A230**.
Read and cite the [published paper](https://doi.org/10.1051/0004-6361/202451003)
([open PDF](https://www.aanda.org/articles/aa/pdf/2024/12/aa51003-24.pdf)),
especially Sections 2.1 and 3.4. This option implements the stated direct
transmission procedure. The exact unpublished molecfit windows and continuum
recipe are not available; the corresponding EXoPLORE choices are explicit.

This preparation option is separate from the molecular cross-correlation
analysis. Selecting it produces direct line products without loading pRT
opacities, computing Kp–Vsys maps or running SYSREM. The existing preparation
options continue to use their existing execution paths.

## 1. Understand the measurement

In this tutorial we will start from a reduced CRIRES+ night and follow the
helium signal from individual spectra to a transmission map, an averaged
spectrum and a light curve. We use WASP-121 b to make the steps concrete;
afterwards, the same preparation choice can be configured for another target.
Keep the complete example configuration open as you work through the steps.

The quantity we measure is the spectrum during transit divided by a reference
spectrum of the star outside transit. A value of one means that the spectra
agree. A value of 0.98 means two per cent additional absorption relative to
that reference. This is a spectral transmission measurement, so the light
curves below should not be read as ordinary broadband transit photometry.

Telluric absorption comes from Earth's atmosphere. Stellar absorption stays
nearly fixed after shifting each spectrum into the stellar frame. Planetary
absorption moves through the stellar spectrum as the planet's radial velocity
changes. We first correct the telluric absorption, align the stellar spectra,
and divide each spectrum by a weighted out-of-transit stellar reference. We
then shift the resulting transmission spectra into the planetary frame before
averaging the chosen in-transit exposures.

A trail in the map is a useful diagnostic. It is not by itself a detection
probability, and residual sky or stellar structure can also appear in these
products. The workflow preserves the observations needed to investigate those
possibilities rather than replacing a failed correction with a flat spectrum.

## 2. Obtain and reduce a CRIRES+ transit

Obtain science frames and associated calibrations from the
[ESO archive](https://archive.eso.org/). Follow the
[CRIRES+ reduction guide](crires_reduction.md) for the cr2res dark, flat,
trace/wavelength and AB nodding extraction stages. Work in a fresh reduction
directory; keep original compressed and decompressed observations intact.
The new preparation step consumes the extracted products, not raw images.

The upstream reduction should provide:

- individual `cr2res_obs_nodding_extractedA.fits` and `extractedB.fits` files;
- `timeseries_manifest.txt`, pairing each individual exposure's UTC start MJD
  with its extracted file path;
- the original raw FITS files referenced by the extracted-product provenance.

Use the physical segment identifier, such as `CHIP1.INT1_02`, rather than an
order index that may change when orders are sorted. For the WASP-121 benchmark,
this segment covers approximately 1077.34–1084.58 nm.

The extracted A/B files can share a primary header inherited from one input
frame. Therefore the individual midpoint is reconstructed from the manifest
time, nod and raw-file provenance. It is not taken blindly from the extracted
header. The exposure-duration keyword is explicitly configured. For the
benchmark, `ESO DET SEQ1 EXPTIME` yields 451.4352425 s; using the nominal 450 s
value is a different timing choice.

## 3. Configure the target and selections

Start from `configs/wasp121b_crires_czesla2024.json`. Its principal choices are:

```json
{
  "planet": {"name": "WASP121b"},
  "instrument": {"name": "CRIRES+"},
  "observation": {
    "event_type": "transit",
    "use_real_data": true,
    "simulate_planet": false,
    "n_nights": 1,
    "exposure_time_seconds": 451.4352425
  },
  "pipeline": {
    "name": "czesla2024",
    "sysrem_iterations": 0,
    "prepare_template": false,
    "czesla2024": {
      "input_path": "inputs/CRIRES_PLUS/WASP121b/czesla2024_corrected",
      "wavelength_frame": "topocentric_vacuum_nm",
      "order_segment": "CHIP1.INT1_02",
      "oh_mode": "baseline"
    }
  },
  "paths": {"output_root": "outputs/wasp121_czesla2024"}
}
```

This excerpt shows the structure; **use the complete example file**, since
scientific settings in `pipeline.czesla2024` are required rather than inferred.
The typed `Czesla2024Config` describes all fields.

Running numbers are one-based and chronological. The night analysed by Czesla et al. (2024) uses
1–10 and 36–40 for the reference and 16–32 for the second-to-third-contact
coadd. **For another dataset, supply its own selections.** The pipeline does
not require 40 exposures or quietly reuse the benchmark selections. Reference
and in-transit selections must be disjoint, unique and within the dataset.

For a new target, also supply its coordinates, observatory longitude/latitude/
height, systemic velocity, ephemeris, Kp and contact phases. The current orbital
velocity calculation is circular, `v_planet = Kp sin(2π phase)`. Eccentric
orbits require an appropriate velocity extension before using this recipe.
Run separate nights with separate reference selections and output roots.

Preview with the standard runner:

```bash
python scripts/run_exoplore.py configs/wasp121b_crires_czesla2024.json
```

Without `--run`, this prints the selected recipe and exits. The direct branch
requires observed transit data, zero SYSREM iterations and disabled molecular
retrieval. It does not silently reinterpret a conflicting configuration.

## 4. Validate telluric correction on one exposure

Install `esorex`, `cr2res` and `molecfit` as described in the reduction guide.
The new wrapper resolves `esorex` on PATH, or uses the explicit executable in
`configs/crires_czesla2024_molecfit.json`. It fits only the selected segment.

Preview a one-exposure pilot:

```bash
python scripts/prepare_crires_czesla2024.py \
  configs/wasp121b_crires_czesla2024.json \
  mynight/reduced/timeseries_manifest.txt \
  mynight/raw \
  configs/crires_czesla2024_molecfit.json \
  mynight/new_helium_pilot \
  --pilot-exposure 1
```

Add `--run` to execute it. A fit typically costs minutes per exposure;
the example limits each recipe to 240 s. The pilot's manifest sidecar and
output directory must be new. It does not change the original manifest.

The example settings specify water fitting, O2 at a fixed relative column,
separate constant continua for the anchor windows and a fitted Gaussian
instrumental kernel. Wavelength refinement starts with `WLC_CONST=0` and
`WLC_N=1`. Box and Lorentzian widths are explicitly zero. These settings avoid
the installed recipe's large starting wavelength displacement and nonzero
fixed-kernel widths; they are local validated settings, not claimed verbatim
author settings.

The example anchor windows, in native vacuum nm, are 1077.54–1077.62,
1080.20–1080.31, 1081.32–1081.43, 1083.65–1083.74 and 1084.04–1084.16.
An anchor is removed for a given exposure when it overlaps a predicted
observer-frame He component within the configured 50 km/s guard. All anchors
must remain inside the selected segment. No unrelated order is fitted.

Inspect the fitted residuals, kernel width and wavelength adjustment before
continuing. Czesla et al. (2024) report adjustments of order 0.1 km/s. A fit displaced by
20–100 km/s is a calibration failure to investigate, not an acceptable
correction because a plot looks convincing. The example's 1 km/s rejection
gate is a failure threshold, not a claim of 0.1 km/s calibration accuracy.

For orientation, the first-exposure test with this wrapper on the benchmark
night returned **+0.176 km/s**. Its model fit took about one minute on the
local test machine. These are measured pilot results, not values that your
dataset is expected to reproduce. The important question is whether a small
wavelength refinement brings the telluric anchors into agreement without
distorting the helium region. Inspect A and B nods separately as well: a
consistent nod-dependent offset deserves investigation even when every fit
passes the numerical gate.

The wrapper runs official `molecfit_calctrans` after an accepted model fit.
Full-grid `mlambda` provides refined vacuum wavelengths, and dimensionless
`mtrans` provides absorption transmission. `mflux`, which includes continuum
scaling, is not a telluric divisor. Failed status, width, displacement or
full-grid checks stop preparation; there is no unity or unrelated-order fallback.

## 5. Prepare the full corrected night

Once the pilot is credible, remove `--pilot-exposure`, specify a **new** output
directory matching `pipeline.czesla2024.input_path`, and pass `--run`:

```bash
python scripts/prepare_crires_czesla2024.py \
  configs/wasp121b_crires_czesla2024.json \
  mynight/reduced/timeseries_manifest.txt \
  mynight/raw \
  configs/crires_czesla2024_molecfit.json \
  inputs/CRIRES_PLUS/WASP121b/czesla2024_corrected \
  --run
```

This is a serial per-exposure fit and can take roughly an hour for a 40-frame
night, depending on the system and convergence. A fresh fit is not needed
when credible, provenance-verified per-exposure corrections already exist.
The old exploratory nod-wide caches are not compatible inputs to this recipe.

Outputs include a `night_report.json` and each exposure's recipe commands,
logs, quality checks and `correction.npz`. Input raw/extracted files are
read-only. The transmission stage verifies extraction hashes, individual raw
header hashes, correction hashes and the wavelength gate again.

## 6. Run direct transmission preparation

```bash
python scripts/run_exoplore.py configs/wasp121b_crires_czesla2024.json --run
```

For each exposure, the pipeline:

1. Divides the extracted flux and error by the validated absorption model.
2. Applies the configured native OH mask, if requested, before interpolation.
3. Applies the barycentric and systemic corrections to reach the stellar frame.
4. Normalizes consistently in the explicit stellar-frame continuum windows.
5. Builds the inverse-variance out-of-transit reference in the stellar frame.
6. Divides all spectra by that same reference.
7. Aligns the transmission spectra to the planet and coadds the selected frames.
8. Extracts the configured planetary- and stellar-frame helium light curves.
9. Propagates native noise through repeated normalization, shared reference
   construction, interpolation and coaddition.

The optical Doppler convention is explicit:

```text
lambda_star   = lambda_top * (1 + BERV/c) / (1 + gamma/c)
lambda_planet = lambda_star / (1 + v_planet/c)
```

Velocities are km/s, positive for recession; BERV is Astropy's additive
barycentric correction. Native cr2res wavelengths are vacuum. Do not apply an
additional air-to-vacuum conversion. All arrays and configured windows use nm.

The example's first-order continuum bands are an explicit implementation
choice inspired by [Allart et al. (2023)](https://arxiv.org/abs/2307.05580),
and is distinct from the normalization described by Czesla et al. (2024). Their degree and minimum valid
pixel count are configurable. No cosmic-ray replacement, frame dropping or
polynomial detrending against airmass is silently added.

## 7. Read the map, spectrum and curves

Now we can inspect what the preparation has actually produced. The figure
below comes from running this EXoPLORE option on the original WASP-121 b
benchmark observations. It is an example of the diagnostic output, rather
than a copy of a figure from the paper.

```{figure} figures/czesla2024_wasp121_diagnostic.png
:alt: Stellar-frame transmission map, planetary-frame helium spectrum, and two helium light curves from the WASP-121 b benchmark observations.
:width: 100%

Direct helium preparation of the WASP-121 b benchmark night. From top to
bottom: stellar-frame transmission map, planetary-frame coadded spectrum,
planetary-frame light curve and stellar-frame light curve. Red in the map
means lower transmission; blue means higher transmission. Magenta lines mark
the vacuum helium triplet, red tracks mark the predicted orbital motion,
and black dashed lines mark optical transit contacts. The spectrum and curve
uncertainties here use 128 native-noise realizations; the distributed example
configuration requests 512. These uncertainties are conditional on the adopted
correction and preparation choices, as discussed in Step 9.
```

### Follow the moving absorption in the stellar frame

Read the map from bottom to top: each row is one exposure, and time zero is
mid-transit. We have aligned the star, so a residual tied to a stellar line
would remain at approximately the same wavelength. Absorption travelling
with the planet instead follows a sloping path, moving from shorter to longer
wavelengths across transit. The red tracks show where the three helium
components would lie for the configured orbital velocity. They are predictions
from the ephemeris, not fits to the coloured structure.

We can see reduced transmission near the stronger pair of helium tracks in
this example. The useful next question is whether that structure survives
changes in reference selection, OH treatment and nod subset. A visually
prominent trail alone cannot establish its physical origin. Pay particular
attention to structure that continues well outside transit or follows the
terrestrial OH positions rather than the predicted planetary motion.

### Bring the planet to rest before averaging

The second panel answers a different question: what line profile remains when
we remove the planet's orbital motion? Each transmission spectrum is shifted
first, and only then are exposures 16–32 averaged. Averaging in the stellar
frame would spread a moving line over its orbital velocity range and change
its apparent width and depth.

The two stronger triplet components are close together and can form one
blended feature; the weaker component lies to their blue side. The dip near
the stronger pair is visible in the example. To measure a residual velocity,
fit all three components with a shared shift as in Step 8. Do not estimate a
wind speed from the deepest pixel: noise, blending and the intrinsic profile
all affect that pixel. Agreement with the paper should be assessed using the
same selections, wavelength convention and definition of width.

### Ask when the absorption changes

The last two panels retain the individual exposure times. In the planetary
frame, a fixed wavelength band follows the predicted moving line. In the
stellar frame, a fixed band stays on the star. They consequently sample
different parts of a travelling feature and need not have the same shape.
The example also uses different band widths in the two frames, so their
depths are not directly interchangeable.

The dashed contact markers help us compare the spectral absorption with the
optical transit. A dip extending beyond those markers motivates checks of an
extended atmosphere, but also of the reference and residual backgrounds.
Keep every exposure in view; the pipeline does not force the light curve to
follow an optical transit model.

Products are written under:

```text
<paths.output_root>/<planet.name>/czesla2024/
    summary.json
    transmission.npz
    transmission_diagnostic.png
```

An existing output directory is refused, preserving all prior results.
The map is in the **stellar frame**: magenta markers show the vacuum triplet,
red tracks show the expected planetary motion, and gold tracks show terrestrial
OH transformed into that frame. The spectrum beneath it is in the **planetary
frame**. Its coadd uses only the explicit in-transit selection. Both curves
use fixed configured bands; a missing or masked band is reported as missing,
not estimated by bridging a gap.

The precise triplet wavelengths in the example are 1083.2057472,
1083.3216751 and 1083.3306444 nm. Oscillator strengths and their source are
recorded explicitly; see [Geach et al., Table 1](https://doi.org/10.1029/2024GL112885)
for the Drake–Morton atomic values used here. Rounded wavelengths can matter
when interpreting kilometre-per-second shifts.

Read the numerical products directly:

```python
import numpy as np
from pathlib import Path

directory = Path("outputs/wasp121_czesla2024/WASP121b/czesla2024")
with np.load(directory / "transmission.npz") as products:
    wavelength = products["planet_wave_nm"]
    transmission = products["planet_coadd"]
    statistical_error = products["planet_coadd_error"]
    stellar_map = products["stellar_transmission"]
    planet_curve = products["planet_lightcurve"]
```

## 8. Fit a documented helium multiplet when appropriate

The public `fit_helium_multiplet` function implements the slab model of Czesla et al. (2024),
`T = 1-f + f exp(-sum(tau_j))`, with shared velocity and intrinsic Gaussian
width for all three oscillator-weighted components. The baseline is fixed at
one. Filling factor, effective resolution, starting values and bounds are
explicit function arguments; fitting is not silently added to preparation.

```python
from exoplore.config import SimulationConfig
from exoplore.pipelines.czesla2024 import fit_helium_multiplet

cfg = SimulationConfig.from_json("configs/wasp121b_crires_czesla2024.json")
he = cfg.pipeline.czesla2024
selected = (wavelength >= 1082.95) & (wavelength <= 1083.48)
fit = fit_helium_multiplet(
    wavelength[selected], transmission[selected], statistical_error[selected],
    np.asarray(he.helium_vacuum_lines_nm),
    np.asarray(he.helium_oscillator_strengths),
    filling_factor=0.05,
    effective_resolution=61600,
    initial=[11.7, 2.6, 10.0],
    bounds=[[9.0, -30.0, 1.0], [15.0, 30.0, 40.0]],
)
```

The filling factor 0.05 and effective R=61600 are the WASP-121 coadd choices
from Czesla et al. (2024), and should be adjusted for other datasets. The latter already includes exposure
smearing; do not add that smearing again. The reported Gaussian width is
sigma; the Doppler parameter used by Czesla et al. (2024) is `b = sqrt(2) sigma`.

This convenience fit uses marginal error weights and returns a diagnostic
point estimate. It does not reproduce the paper's full 43-parameter time
model or its MCMC, and is not a replacement for a justified correlated
likelihood. Column, covering fraction and width can be degenerate.

## 9. Compare OH modes and assess uncertainty

Baseline mode retains the stated AB sky-subtraction treatment and marks OH.
Molecfit corrects absorption; this workflow does not claim it removes OH
airglow emission. Mask mode is a separately declared sensitivity test using
explicit native topocentric windows. Change `oh_mode` to `mask` in a new
configuration and use a new output root. No dedicated OH model is applied.

The example OH markers are 1083.2103, 1083.2412, 1083.4241 and 1083.43338 nm,
with +/-0.015 nm windows based on Allart et al. (2023) and the markers shown by Czesla et al. (2024).
Interpolation requires valid native contributors and does not bridge masks.
Equivalent width is not reported if its integration window has missing pixels.

Noise realizations share a rebuilt reference across all exposures. They therefore
propagate the statistical correlations produced by reference division,
normalization and interpolation into coadd/curve samples. Optional molecfit
RMS inflation affects the draws, not the native inverse-variance reference
weights. Random seed, configuration and input hashes are saved.

These are conditional noise uncertainties. Stellar activity, photospheric
RM/CLV, physical AB extraction correlations, fitted telluric-model uncertainty,
calibration and ephemeris errors are not all marginalized. No automatic
detection significance or physical upper limit is issued. Check nod subsets,
out/out controls, OH residuals and normalization sensitivity before physical
interpretation. A moving feature and a numerically close published fit are
useful evidence to investigate, not substitutes for those checks.

## References and attribution

- **Czesla et al. (2024), A&A 692, A230**:
  [published article](https://doi.org/10.1051/0004-6361/202451003).
  Credit this work for the original WASP-121 b observing and direct helium
  analysis procedure used as the methodological benchmark here.
- Allart et al. (2023): [paper](https://arxiv.org/abs/2307.05580), supporting
  continuum-window and OH-line choices.
- ESO [CRIRES+ pipeline](https://www.eso.org/sci/software/pipelines/cr2res/)
  and [molecfit documentation](https://www.eso.org/sci/software/pipelines/molecfit/).
  EXoPLORE wraps their reduction and absorption-model products.
- Smette et al. (2015), A&A 576, A77, and Kausch et al. (2015), A&A 576,
  A78: molecfit; cite the underlying tools as well as the scientific method.

Use the existing [EXoPLORE citation guidance](citations.md) for the framework.
