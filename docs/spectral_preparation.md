# Spectral preparation

Preparation removes the dominant stellar, telluric and instrumental structure
before we measure the planetary spectrum. The choice depends on the data and
on the observable we want to recover. Molecular cross-correlation combines
many weak lines after detrending. Resolved-line transmission spectroscopy
divides by a stellar reference and measures the remaining line profile.

Select the recipe through `pipeline.name` in the JSON. The
{ref}`configuration reference <pipeline-settings>` lists the controls.
The recipes implemented in EXoPLORE are summarised below.

| `pipeline.name` | Preparation | Worked example |
|---|---|---|
| `BL19` | Bright-pixel normalisation and polynomial telluric fits | [Tutorial 7](tutorial.md#tutorial-7-assessing-retrieval-pipeline-bias) |
| `Blain24` | Polynomial throughput removal and log-flux fits against airmass | [Tutorial 1](tutorial.md#tutorial-1-run-the-reference-hd-189733-b-andes-simulation) |
| `ASL19` | Pseudo-continuum normalisation, telluric-window masking and SYSREM | See the controls below |
| `Gibson22` | Reference normalisation and SYSREM | [Tutorial 7](tutorial.md#tutorial-7-assessing-retrieval-pipeline-bias) |
| `Nortmann26` | Common-blaze normalisation and division SYSREM | [Tutorial 9](tutorial.md#tutorial-9-analyse-real-crires-data-wasp-127-b) |
| `Cheverall26` | Continuum normalisation and PCA or SYSREM | [Tutorial 10](tutorial.md#tutorial-10-analyse-real-igrins-data-l-98-59-d) |
| `czesla2024` | CRIRES+ helium stellar-reference division and planetary-frame averaging | [Tutorial 12](tutorial.md#tutorial-12-he-i-transmission-spectroscopy-with-crires-wasp-121-b) |
| `Palle2020` | Observed CARMENES helium correction and arithmetic-reference transmission | [Tutorial 12b](tutorial.md#tutorial-12b-he-i-transmission-spectroscopy-with-carmenes-gj-3470-b) |
| `carmenes_helium` | Synthetic CARMENES helium observations with the shared arithmetic-reference preparation | See the helium settings below |
| `Seidel22` | Stellar-reference transmission with an optional stellar-transit correction | {ref}`Tutorial 13 <tutorial-13-sodium>` |

## Molecular cross-correlation

### BL19

The [Brogi & Line (2019)](https://doi.org/10.3847/1538-3881/aaffd3)
preparation normalises each exposure by the mean of its 300 brightest retained
pixels. The implementation also accepts a pseudo-continuum fit through its
normalisation helper. It masks deep telluric channels, fits the spectral and
temporal telluric structure, and masks noisy columns. This branch uses
polynomial fits without SYSREM.

Select `pipeline.name: "BL19"`. The telluric mask is controlled by
`tellurics.mask_threshold`. See [model preparation](config_reference.md#template-preparation-and-sysrem-distortions)
when using these residuals for a CCF or retrieval.

### Blain24

[Blain, Sánchez-López & Mollière (2024)](https://doi.org/10.3847/1538-3881/ad2c8b)
remove throughput with a polynomial over wavelength, then fit the logarithm
of the throughput-corrected flux against airmass in each channel. EXoPLORE
uses degree-two fits for both steps and propagates the supplied uncertainties
through the divisions. The telluric-fit stage also produces the channel mask.

Select `pipeline.name: "Blain24"`. Tutorial 1 shows the preparation figure
and the resulting CCF. The retrieval likelihood is selected separately through
`retrieval.log_likelihood`, including the fixed-uncertainty `Blain24` and fitted
noise-scale `Blain24_beta` choices.

### ASL19

The [Sánchez-López et al. (2019)](https://doi.org/10.1051/0004-6361/201936084)
branch divides each exposure by a second-order polynomial fitted to maxima
in 80 wavelength bins. It masks deep telluric absorption, expands the mask by
`tellurics.safety_window_pixels`, and removes temporal components with SYSREM.

Select `pipeline.name: "ASL19"`. `pipeline.sysrem_iterations` controls the
component count. The order-by-order optimisation settings can select that
count from residual variance or injection recovery. Inspect how the planetary
signal changes with the chosen count before applying the recipe to a new night.

### Gibson22

For observed data, this branch divides the exposures by the wavelength-dependent
out-of-transit median spectrum. In the standard synthetic route, it divides each
exposure by its own spectral median. It then applies the telluric mask and
SYSREM, retaining the fitted time basis for model filtering following
[Gibson et al. (2022)](https://doi.org/10.1093/mnras/stac091).

Select `pipeline.name: "Gibson22"` and set `pipeline.sysrem_iterations`.
Tutorial 7 compares preparations and retrieval likelihoods. The preparation
name and the likelihood name are separate configuration choices.

### Nortmann26

Select `pipeline.name: "Nortmann26"` for common-blaze normalisation, column
and deep-telluric masking, followed by division SYSREM on CRIRES+ segments.
`pipeline.sysrem_iterations` sets the component count. The fitted time basis
is saved for division-form model filtering. [Tutorial 9](tutorial.md#tutorial-9-analyse-real-crires-data-wasp-127-b)
explains the preparation, upstream reduction and implementation details.

### Cheverall26

Select `pipeline.name: "Cheverall26"` for per-exposure rescaling, continuum
normalisation, masking and PCA or SYSREM detrending on IGRINS orders.
`pipeline.continuum_method` selects the continuum estimator,
`pipeline.detrend_method` selects the detrending method, and
`pipeline.sysrem_iterations` sets the component count.
[Tutorial 10](tutorial.md#tutorial-10-analyse-real-igrins-data-l-98-59-d)
explains the preparation, noise convention and injection-recovery settings.

### Preparing the model

Detrending also changes the planetary signal. A model compared with the
prepared spectra needs the corresponding transformation. Polynomial recipes
apply their normalisation and fits to the model. SYSREM recipes use the
stored data-derived basis to filter it. Division SYSREM has its own division
form of that filter.

Use the [template-preparation settings](config_reference.md#template-preparation-and-sysrem-distortions)
for the selected branch. Component counts and masking affect signal recovery,
so test them with injections at the cadence and noise level of the observations.

## Resolved-line transmission

These routes form a stellar reference and average the transmission spectra in
the planetary frame. Their continuum windows, reference exposures and velocity
conventions are part of the measurement. Telluric correction and sky-emission
treatment are configured in the corresponding observing workflow.

### Czesla2024: CRIRES+ helium

Select `pipeline.name: "czesla2024"` and supply `pipeline.czesla2024`.
Following [Czesla et al. (2024)](https://doi.org/10.1051/0004-6361/202451003),
the preparation aligns spectra with the star, forms an out-of-transit reference,
divides by that reference and combines the planet-aligned transmission spectra.
The JSON supplies continuum intervals and exposure membership. The workflow
also configures Molecfit and the treatment of OH emission.

[Tutorial 12](tutorial.md#tutorial-12-he-i-transmission-spectroscopy-with-crires-wasp-121-b)
shows these steps on WASP-121 b. Its uncertainty calculation can repeat the
preparation with perturbed spectra, including the shared stellar reference.

### Palle2020 and the CARMENES helium simulation

Select `pipeline.name: "Palle2020"` for the observed CARMENES route described
in [Tutorial 12b](tutorial.md#tutorial-12b-he-i-transmission-spectroscopy-with-carmenes-gj-3470-b).
`pipeline.palle2020` supplies the observed inputs and telluric/OH settings.
The shared preparation normalises the spectra, aligns them with the star,
divides by an arithmetic-mean reference and averages the planet-aligned
full-transit spectra. The reference is
[Pallé et al. (2020)](https://doi.org/10.1051/0004-6361/202037719).

`pipeline.name: "carmenes_helium"` selects the synthetic helium route and
requires `pipeline.carmenes_helium`. It uses the same arithmetic-reference
preparation with synthetic extracted spectra. The helium atmosphere and
observational inputs are supplied through the corresponding configuration
blocks. The observed and synthetic routes have different input requirements.
Their saved diagonal errors have a defined conditional scope, so consult the
helium uncertainty settings before assigning a detection significance.

### Seidel22: resolved atomic lines

Select `pipeline.name: "Seidel22"` and supply `pipeline.seidel22`.
The preparation aligns spectra with the star, normalises them in the configured
continuum windows and forms a continuum-S/N-weighted out-of-transit reference.
After reference division, it can apply the configured opaque-body stellar
correction. It then aligns the transmission spectra with the planet and
combines the selected exposures. This follows the stellar-reference method of
[Seidel et al. (2022)](https://doi.org/10.1093/mnrasl/slac027).

{ref}`Tutorial 13 <tutorial-13-sodium>`
shows Molecfit, RM+CLV correction and the resolved sodium profiles.
[Tutorial 14](tutorial.md#tutorial-14-stellar-spots-and-faculae-in-na-d2-transmission)
uses the same preparation to compare stellar-surface corrections. The
[Seidel22 configuration](config_reference.md#seidel22-stellar-reference-transmission-spectra)
defines continuum windows in vacuum nm and the noise-resampling controls.
