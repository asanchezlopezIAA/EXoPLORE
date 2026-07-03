# Data acknowledgements

## CARMENES reference night, HD 189733 b (2017-09-07)

The reference-night files in `inputs/CARMENES_NIR/HD189733b/reference_night/`
contain CARMENES NIR observations of one transit of HD 189733 b on the night of
7 to 8 September 2017 (45 spectra, 28 orders, 0.96 to 1.71 µm, R ≈ 80,400). The files
were obtained from the **CAHA Public Archive** and are deposited on Zenodo for
direct download:

**Zenodo DOI: [10.5281/zenodo.20613621](https://doi.org/10.5281/zenodo.20613621)**

The deposit contains:
- `snr_0.fits`, per-pixel SNR array (45 × 28 × 4080)
- `sig_0.fits`, per-pixel uncertainty array (45 × 28 × 4080)
- `observations_night_0_order_0.fits` … `observations_night_0_order_27.fits`, normalised observed spectra per order

The files `julian_date_0.fits`, `airmass_0.fits`, and `observations_berv_0.fits`
are included in the git repository and do not need to be downloaded separately.

These data were first published in:

> Alonso-Floriano, F. J., Sánchez-López, A., Snellen, I. A. G., et al. (2019).
> *Multiple water band detections in the CARMENES near-infrared transmission
> spectrum of HD 189733 b.*
> A&A, 621, A74.
> [doi:10.1051/0004-6361/201834339](https://doi.org/10.1051/0004-6361/201834339)

and subsequently used in Sánchez-López et al. (2019, A&A, 630, A53) and
Blain, Sánchez-López & Mollière (2024, AJ, 167, 179).

**If you use these files in published work, please consider citing Alonso-Floriano et al.
(2019), Sánchez-López et al. (2019), and Blain, Sánchez-López & Mollière (2024), and including the following acknowledgement:**

> Based on data from the CAHA Archive at CAB (INTA-CSIC). The CAHA Archive is
> part of the Spanish Virtual Observatory project funded by
> MCIN/AEI/10.13039/501100011033 through grant PID2023-146210NB-I00.

---

## [SkyCalc](https://skycalc-ipy.readthedocs.io) telluric model

Telluric transmission spectra generated with `scripts/generate_skycalc_inputs.py`
use the ESO SkyCalc model via the `skycalc_ipy` Python package.  Please cite:

> Noll, S., Kausch, W., Barden, M., et al. (2012).
> *An atmospheric radiation model for Cerro Paranal.*
> A&A, 543, A92.
> [doi:10.1051/0004-6361/201219040](https://doi.org/10.1051/0004-6361/201219040)

> Jones, A., Noll, S., Kausch, W., Szyszka, C., & Kimeswenger, S. (2013).
> *An advanced scattered moonlight model for Cerro Paranal.*
> A&A, 560, A91.
> [doi:10.1051/0004-6361/201322433](https://doi.org/10.1051/0004-6361/201322433)

The `--mode astro` airmass computation uses [astroplan](https://astroplan.readthedocs.io) (Morris et al. 2018,
AJ 155, 128) following the methodology of Dash et al.
(arXiv:2602.22830, 2026).

---

## PHOENIX stellar models

Stellar spectra use the PHOENIX-ACES-AGSS-COND-2011 library:

> Husser, T.-O., Wende-von Berg, S., Dreizler, S., et al. (2013).
> *A new extensive library of PHOENIX stellar atmospheres and synthetic spectra.*
> A&A, 553, A6.
> [doi:10.1051/0004-6361/201219058](https://doi.org/10.1051/0004-6361/201219058)
