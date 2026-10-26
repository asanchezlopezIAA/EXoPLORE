# Installation

## Requirements

EXoPLORE requires Python 3.10 or later and a UNIX-like system (Linux or macOS, Windows is not tested). In addition, approximately 10 GB of free disk space are needed to accommodate the [petitRADTRANS](https://petitradtrans.readthedocs.io) opacity tables and PHOENIX stellar model files.

## 1. Clone the repository

```bash
git clone https://github.com/asanchezlopezIAA/EXoPLORE.git
cd EXoPLORE
```

## 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install EXoPLORE and its dependencies

petitRADTRANS and EasyChem builds require a Fortran compiler. On macOS, set the
SDK path before installation if your compiler needs it:

```bash
export SDKROOT="$(xcrun --show-sdk-path)"
```

Install the package and the molecular-tutorial dependencies:

```bash
python -m pip install -e ".[prt,batman]"
```

This installs EXoPLORE in editable mode, its core numerical dependencies, and the
`prt` and `batman` extras declared in `pyproject.toml`. These extras supply
petitRADTRANS, EasyChem and the transit light-curve package used by the molecular
tutorials. Run the tutorial commands from the repository root so that relative
configuration and input paths resolve.

## 4. Set up petitRADTRANS opacity data

EXoPLORE uses [petitRADTRANS](https://petitradtrans.readthedocs.io) to compute the planetary transmission and emission spectra that serve as the forward model. The `prt` extra above installs petitRADTRANS and EasyChem. The current
radiative-transfer interface uses petitRADTRANS 3. Scientific opacity tables
are downloaded separately.

Download the opacity tables following the [petitRADTRANS documentation](https://petitradtrans.readthedocs.io/en/latest/content/installation.html). In particular, the line-by-line opacity tables required for high-resolution retrieval amount to several GB. Note the absolute path to the resulting `input_data/` directory, as it must be set in the simulation config under `paths.prt_input_data`.

## 5. Install retrieval dependencies (optional)

These packages are only required when `retrieval.enabled: true`:

```bash
python -m pip install -e ".[retrieval]"
```

`pymultinest` wraps the [MultiNest](https://github.com/JohannesBuchner/PyMultiNest) nested sampling algorithm and requires the compiled MultiNest Fortran library to be installed separately. We refer to the [PyMultiNest installation guide](https://github.com/JohannesBuchner/PyMultiNest) for platform-specific instructions.

## 6. Download PHOENIX stellar model files

EXoPLORE uses two PHOENIX files for the target star: a shared wavelength grid and a model spectrum selected to match the stellar effective temperature, surface gravity, and metallicity. Both are available from the [PHOENIX library](https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/):

```
WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits   ← example for Teff = 5000 K
```

The absolute paths to these files must be set in the simulation config under `paths.phoenix_wave_file` and `paths.phoenix_flux_file`.

## 7. Download CARMENES reference-night files (CARMENES tutorials only)

The CARMENES NIR tutorials require per-pixel SNR, uncertainty, and observed spectra
for the HD 189733 b reference night. These files are deposited on Zenodo:

**DOI: [10.5281/zenodo.20613621](https://doi.org/10.5281/zenodo.20613621)**

Download the 30 FITS files from Zenodo and place them inside the cloned repository.
The directory already exists after cloning, but if needed you can create it with:

```bash
mkdir -p inputs/CARMENES_NIR/HD189733b/reference_night/
```

Then move the downloaded files there:

```bash
mv ~/Downloads/snr_0.fits \
   ~/Downloads/sig_0.fits \
   ~/Downloads/observations_night_0_order_*.fits \
   inputs/CARMENES_NIR/HD189733b/reference_night/
```

All paths above are relative to the repository root (`EXoPLORE/`). Run these
commands from inside the cloned repository directory.

The reference ANDES wavelength, ETC and telluric files are included in the
checkout. Its atmospheric opacity tables and PHOENIX spectra are still required.

## 8. Set up instrument input files

The instrument and observing mode select the required wavelength, S/N and
telluric inputs. The CARMENES reference-night S/N and uncertainty arrays come
from the Zenodo deposit above. See [Input files](input_files.md) for the bundled
ANDES inputs and the files needed for another instrument or observing night.

To generate new SkyCalc spectra, install the downloader dependencies:

```bash
python -m pip install -e ".[skycalc]"
```

Set `tellurics.skycalc_wavelength_limits_um` in the simulation JSON before
running the {ref}`SkyCalc downloader <skycalc-reference-download>`.
The real-data tutorials also describe their external reduction and telluric-fit
tools. The [interactive notebook](tutorial.md#tutorial-1-run-the-reference-hd-189733-b-andes-simulation)
requires a Jupyter environment with this project environment selected as its kernel.

## 9. Verify the installation

Install the test dependencies and run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

All tests should pass. To confirm that the simulation chain is functional, preview the reference HD 189733 b configuration:

```bash
python scripts/run_exoplore.py configs/hd189733b_carmenes_transit.json
```

This prints the simulation summary without executing anything. To run the full simulation, add `--run`.
