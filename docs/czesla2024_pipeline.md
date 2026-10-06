# Tutorial: Direct He I transmission spectroscopy with CRIRES+

This tutorial describes how to prepare an observed CRIRES+ transit for direct
He I λ10833 transmission spectroscopy. Specifically, we use the WASP-121 b
observations analysed by [Czesla et al. (2024)](https://doi.org/10.1051/0004-6361/202451003)
to illustrate the reduction, telluric correction, construction of the stellar
reference, and extraction of the planetary transmission spectrum. The
`czesla2024` preparation choice selects this procedure through the usual
EXoPLORE configuration and runner. In the following, we describe each step
and the resulting transmission map, spectrum, and helium light curves.

The observing and analysis methodology is presented in Sections 2.1 and 3.4
of Czesla et al. (2024), *The overflowing atmosphere of WASP-121 b:
High-resolution He I λ10833 transmission spectroscopy with VLT/CRIRES+*
([open PDF](https://www.aanda.org/articles/aa/pdf/2024/12/aa51003-24.pdf)).
The example uses the same observations as that study. Consequently, it
illustrates a reanalysis of the published dataset, rather than an independent
observation of the system. The fit windows and continuum prescription adopted
here are specified below, since the publication does not provide every
setting needed to reproduce those operations exactly.

## Step 1: Understand the transmission measurement

During a primary transit, absorption in the planetary atmosphere modifies
the stellar spectrum over a narrow wavelength interval. In order to measure
this additional absorption, we divide each corrected, normalised spectrum by
a reference constructed from observations outside transit. The resulting
transmission is approximately one away from variable spectral features;
for example, a value of 0.98 corresponds to two per cent additional
absorption relative to the reference.

The planet's radial velocity changes during transit, while the stellar lines
remain approximately stationary once the spectra are aligned in the stellar
rest frame. Following Czesla et al. (2024), we construct the out-of-transit
reference in this frame and divide every spectrum by it. We subsequently
remove the planet's orbital velocity from the transmission spectra before
co-adding the selected in-transit exposures. This sequence allows the stellar
reference and the planetary average to be constructed in their respective
rest frames.

Here we analyse the resolved helium triplet directly. This preparation choice
therefore produces spectral transmission products without the molecular
cross-correlation, SYSREM, or atmospheric retrieval stages used in other
EXoPLORE tutorials.

## Step 2: Reduce the CRIRES+ observations

Obtain science frames and associated calibrations from the
[ESO archive](https://archive.eso.org/). Follow the
[CRIRES+ reduction guide](crires_reduction.md) for the cr2res dark, flat,
trace/wavelength and AB nodding extraction stages. Work in a fresh reduction
directory; keep original compressed and decompressed observations intact.
The helium preparation takes the extracted one-dimensional spectra as input.

The upstream reduction should provide:

- individual `cr2res_obs_nodding_extractedA.fits` and `extractedB.fits` files;
- `timeseries_manifest.txt`, pairing each individual exposure's UTC start MJD
  with its extracted file path;
- the original raw FITS files referenced by the extracted-product provenance.

For WASP-121 b, the helium triplet lies in `CHIP1.INT1_02`, which covers
approximately 1077.34–1084.58 nm. We identify this segment by its physical
name because its numerical index can change when the extracted orders are
sorted.

The extracted A/B files can share a primary header inherited from one input
frame. Therefore the individual midpoint is reconstructed from the manifest
time, nod position, and corresponding raw-file header. This ensures that each
spectrum receives its own exposure midpoint for the barycentric and orbital
velocity calculations. In this example, the configured keyword
`ESO DET SEQ1 EXPTIME` gives an exposure duration of 451.4352425 s.

## Step 3: Inspect the configuration and select the exposures

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

The excerpt above shows the principal choices. Use the complete example file,
which also specifies the ephemeris, velocities, continuum windows, helium
line data, and uncertainty calculation. Set `input_path` to the corrected
spectra prepared in Step 5, and choose an unused `output_root` for this run.

Running numbers start at one and follow chronological order. For the night
analysed by Czesla et al. (2024), we use exposures
1–10 and 36–40 for the reference and 16–32 for the second-to-third-contact
co-addition. These selections are entered in `reference_running_numbers`
and `in_transit_running_numbers`, respectively. For another dataset, select
the appropriate out-of-transit baseline and in-transit interval from its own
observing sequence. The two selections must be disjoint.

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
retrieval, as shown in the example configuration.

## Step 4: Test the telluric correction on one exposure

Install `esorex`, `cr2res` and `molecfit` as described in the reduction guide.
The preparation script resolves `esorex` on PATH, or uses the executable in
`configs/crires_czesla2024_molecfit.json`. It fits only the selected segment.

In order to assess the telluric correction before processing a full night,
first preview a single-exposure run. Replace `mynight` with your data directory:

```bash
python scripts/prepare_crires_czesla2024.py \
  configs/wasp121b_crires_czesla2024.json \
  mynight/reduced/timeseries_manifest.txt \
  mynight/raw \
  configs/crires_czesla2024_molecfit.json \
  mynight/new_helium_pilot \
  --pilot-exposure 1
```

Add `--run` to execute the test. A fit typically takes minutes per exposure;
the example limits each recipe to 240 s. Use a new output directory. The
script creates a separate manifest for this test and preserves the original
night manifest.

The example settings specify water fitting, O2 at a fixed relative column,
separate constant continua for the anchor windows and a fitted Gaussian
instrumental kernel. Wavelength refinement starts with `WLC_CONST=0` and
`WLC_N=1`, with box and Lorentzian widths set to zero. These are the
EXoPLORE settings tested for this example; the exact recipe settings used by
Czesla et al. (2024) are not fully specified in the publication.

The example anchor windows, in native vacuum nm, are 1077.54–1077.62,
1080.20–1080.31, 1081.32–1081.43, 1083.65–1083.74 and 1084.04–1084.16.
An anchor is removed for a given exposure when it overlaps a predicted
observer-frame He component within the configured 50 km/s guard. All anchors
must remain inside the selected segment. This keeps the telluric fit within
the helium order while excluding the expected planetary absorption.

Inspect the fitted residuals, kernel width and wavelength adjustment before
continuing. Czesla et al. (2024) report wavelength adjustments of order
0.1 km/s. The first-exposure test for this example returned +0.176 km/s;
however, the appropriate refinement must be assessed for each observation.
Displacements of 20–100 km/s indicate a calibration problem that must be
resolved before constructing transmission spectra. The example rejects
adjustments larger than 1 km/s, although passing this threshold alone does
not establish the calibration precision. Examine the residuals and the
adjustments for the A and B nod positions separately.

The wrapper runs official `molecfit_calctrans` after an accepted model fit.
Full-grid `mlambda` provides refined vacuum wavelengths, and dimensionless
`mtrans` provides absorption transmission. `mflux`, which includes continuum
scaling, should therefore not be used to divide the observed spectra.
Preparation stops if the fitted model or wavelength solution fails its
configured checks.

## Step 5: Correct the full observing sequence

Once the single-exposure correction has been assessed, omit `--pilot-exposure`, specify a **new** output
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
night, depending on the system and convergence. If validated per-exposure
corrections from this preparation already exist, they can be reused directly.

Outputs include a `night_report.json` and each exposure's recipe commands,
logs, quality checks and `correction.npz`. Input raw/extracted files are
read-only. The transmission stage verifies extraction hashes, individual raw
header hashes, correction hashes and the wavelength gate again.

## Step 6: Construct the transmission spectra

We now apply the direct preparation to the corrected observing sequence:

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

In order to normalise the spectra consistently, the example fits a
first-order polynomial in the stellar-frame bands 1082.3–1082.6 nm and
1083.9–1084.2 nm. This choice is informed by
[Allart et al. (2023)](https://arxiv.org/abs/2307.05580).
It is specified explicitly because the complete continuum prescription of
Czesla et al. (2024) is unavailable. The bands, polynomial degree, and minimum
number of valid pixels can be adjusted in the configuration. Assess their
sensitivity for the target under study before interpreting the line profile.

## Step 7: Inspect the transmission map, spectrum, and light curves

The following figure shows the products obtained with this preparation
choice for the WASP-121 b observations analysed by Czesla et al. (2024).
We describe the physical quantities in each panel below.

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

### Stellar-frame transmission map

The top panel shows spectral transmission as a function of wavelength and
time relative to mid-transit. Each row corresponds to one exposure, with
time increasing upwards. Since the spectra are aligned in the stellar frame,
residual stellar features remain at approximately fixed wavelengths. In
contrast, absorption moving with the planet changes wavelength as its
line-of-sight orbital velocity evolves during transit. The red tracks show
the expected positions of the three helium components for the adopted
ephemeris and orbital velocity.

Reduced transmission is visible near the stronger pair of helium tracks.
To assess this structure, compare its time and wavelength dependence with
the planetary tracks and terrestrial OH positions. In particular, repeat
the preparation with different reference selections and nod subsets to
establish how these choices affect the recovered feature.

### Planetary-frame transmission spectrum

The second panel shows the average transmission spectrum after removal of
the planet's orbital motion. Specifically, we shift each transmission
spectrum to the planetary rest frame and subsequently co-add exposures
16–32, following the selection of Czesla et al. (2024). This alignment
preserves a feature travelling with the planet; averaging in the stellar
frame would broaden it over the range of orbital velocities sampled during
the selected interval.

The two stronger triplet components are closely spaced and appear as a
blended absorption feature, whereas the weaker component lies at shorter
wavelengths. In order to measure a residual velocity or line width, we fit
the three components jointly as described in Step 8. This accounts for
their relative positions and strengths. A quantitative comparison with
Czesla et al. (2024) requires consistent exposure selections, wavelength
conventions, and definitions of the fitted width.

### Helium light curves

The last two panels show the mean spectral transmission within a configured
helium band for each exposure. The planetary-frame band follows the expected
orbital motion of the line, while the stellar-frame band remains stationary
relative to the star. Consequently, the two curves sample different parts
of the moving absorption profile. Their band widths also differ in this
example, and must be considered when comparing the amplitudes.

The dashed lines indicate the optical transit contacts. These allow the
timing of the helium absorption to be compared with the passage of the
planetary disc across the star. Absorption outside this interval may provide
information about extended material; its interpretation also requires
assessment of the out-of-transit reference and residual stellar and sky
contributions. The curves are extracted directly from the spectra, without
imposing an optical transit shape.

Products are written under:

```text
<paths.output_root>/<planet.name>/czesla2024/
    summary.json
    transmission.npz
    transmission_diagnostic.png
```

Use a new output directory for each comparison, since existing results are
preserved. The plotted OH positions are transformed from the topocentric
frame into the stellar frame; some lie outside the displayed interval.
If masking leaves incomplete coverage of a light-curve band, the affected
measurement is reported as missing.

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

## Step 8: Fit the helium triplet

The public `fit_helium_multiplet` function implements the slab model of Czesla et al. (2024),
`T = 1-f + f exp(-sum(tau_j))`, with shared velocity and intrinsic Gaussian
width for all three components, with their relative optical depths set by
the oscillator strengths. Here, `f` is the fraction of the stellar disc
covered by the absorbing material and `tau_j` is the optical depth of
component `j`. The continuum baseline is fixed at one. The following
optional fit uses the arrays loaded in Step 7 and requires the covering
fraction, effective resolving power, initial parameters, and bounds:

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

The filling factor of 0.05 and effective resolving power of 61,600 follow
the WASP-121 b co-added spectrum analysis of Czesla et al. (2024). The
effective resolution already accounts for exposure smearing. For another
dataset, supply values appropriate to its geometry and instrumental and
exposure broadening. The fitted parameters are the logarithmic helium
column density, velocity shift, and Gaussian width `sigma`; the Doppler
parameter used by Czesla et al. (2024) is `b = sqrt(2) sigma`.

This fit provides a point estimate using the marginal uncertainties of
the co-added spectrum. It covers the averaged triplet profile only; the
full 43-parameter time-dependent model and MCMC analysis of Czesla et al.
(2024) are outside the present preparation workflow. In particular, the
column density, covering fraction, and width can be degenerate, and their
joint uncertainties require an appropriate likelihood and posterior analysis.

## Step 9: Assess OH residuals and uncertainties

The `baseline` mode retains the AB sky-subtraction treatment described by
Czesla et al. (2024). Residual OH emission should be inspected separately,
since molecfit models telluric absorption. To assess the sensitivity to OH,
set `oh_mode` to `mask` in a separate configuration and select a new output
root. This excludes the configured OH windows in the native topocentric
frame before interpolation. Both modes retain the same stellar-reference
and planetary-frame preparation sequence; neither fits an OH emission model.

The example OH markers are 1083.2103, 1083.2412, 1083.4241 and 1083.43338 nm,
with +/-0.015 nm windows based on Allart et al. (2023) and the markers shown by Czesla et al. (2024).
Interpolation requires valid native contributors and does not bridge masks.
Equivalent width is not reported if its integration window has missing pixels.

Noise realizations share a rebuilt reference across all exposures. They therefore
propagate the statistical correlations produced by reference division,
normalization and interpolation into coadd/curve samples. Optional molecfit
RMS inflation affects the draws, not the native inverse-variance reference
weights. Random seed, configuration and input hashes are saved.

The resulting uncertainties describe the propagation of the adopted noise
model through the preparation. Stellar variability, Rossiter–McLaughlin and
centre-to-limb effects, correlations introduced by the physical AB extraction,
and uncertainty in the telluric model, calibration, and ephemeris are not all
included. Consequently, physical interpretation requires additional controls:
compare the nod subsets, divide independent out-of-transit subsets by one
another, and assess the sensitivity to OH masks and continuum choices. The
workflow provides the spectral products and conditional uncertainties needed
for this assessment; a detection significance or physical upper limit requires
a separate statistical analysis that accounts for the relevant backgrounds.

## References

- **Czesla et al. (2024), A&A 692, A230**:
  [published article](https://doi.org/10.1051/0004-6361/202451003).
  Original WASP-121 b observations and direct helium analysis methodology.
- Allart et al. (2023): [paper](https://arxiv.org/abs/2307.05580), supporting
  continuum-window and OH-line choices.
- ESO [CRIRES+ pipeline](https://www.eso.org/sci/software/pipelines/cr2res/)
  and [molecfit documentation](https://www.eso.org/sci/software/pipelines/molecfit/).
  EXoPLORE wraps their reduction and absorption-model products.
- Smette et al. (2015), A&A 576, A77, and Kausch et al. (2015), A&A 576,
  A78: molecfit; cite the underlying tools as well as the scientific method.

Use the existing [EXoPLORE citation guidance](citations.md) for the framework.
