# Installation

## Requirements

EXoPLORE requires Python 3.10 or later and a UNIX-like system (Linux or macOS; Windows is not tested). In addition, approximately 10 GB of free disk space are needed to accommodate the [petitRADTRANS](https://petitradtrans.readthedocs.io) opacity tables and PHOENIX stellar model files.

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

```bash
pip install -e .
```

This installs the `exoplore` package in editable mode together with all core dependencies (NumPy, SciPy, Astropy, [matplotlib](https://matplotlib.org), and others) declared in `pyproject.toml`.

## 4. Install petitRADTRANS

EXoPLORE uses [petitRADTRANS](https://petitradtrans.readthedocs.io) to compute the planetary transmission and emission spectra that serve as the forward model. To install it:

```bash
pip install petitRADTRANS
```

On macOS, it is often necessary to set the Fortran compiler SDK path before installation:

```bash
export SDKROOT="$(xcrun --show-sdk-path)"
```

We recommend adding this line to `~/.zshrc` or `~/.bashrc` to avoid repeating it in future sessions. After installing, download the opacity tables following the [petitRADTRANS documentation](https://petitradtrans.readthedocs.io/en/latest/content/installation.html). In particular, the line-by-line opacity tables required for high-resolution retrieval amount to several GB. Note the absolute path to the resulting `input_data/` directory, as it must be set in the simulation config under `paths.prt_input_data`.

## 5. Install retrieval dependencies (optional)

These packages are only required when `retrieval.enabled: true`:

```bash
pip install pymultinest emcee corner
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

The ANDES tutorials do not require any additional downloads.

## 8. Set up instrument input files

Each instrument requires pre-computed signal-to-noise tables and a telluric reference spectrum. For CARMENES NIR these are provided via the Zenodo deposit above. For other instruments see [Input files](input_files.md) for the expected directory layout and file formats.

## 9. Verify the installation

```bash
python -m pytest
```

All tests should pass. To confirm that the simulation chain is functional, preview the reference HD 189733 b configuration:

```bash
python scripts/run_exoplore.py configs/hd189733b_carmenes_transit.json
```

This prints the simulation summary without executing anything. To run the full simulation, add `--run`.
