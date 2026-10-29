# HD 189733 b sodium reference inputs

Tutorial 13 uses a synthetic ANDES observing sequence with 30 s exposures
and 6 s readout. `observing_sequence.csv` contains 388 BJD_TDB exposure
midpoints and geometric airmasses. Mid-transit is BJD_TDB 2458321.68158481.
The site coordinates are longitude −70.4042°, latitude −24.6272°, height
2635 m (Paranal). The airmass range is 1.4771–1.9501. BERV is fixed to
zero in the simulation JSON as an idealized velocity convention.

`paranal_reference.fits` contains an ESO SkyCalc reference at airmass 1
and PWV 10 mm. It retains original rows between 585.98 and 601.02 nm,
without interpolation or instrumental convolution. Column `lam` uses vacuum
nm and `trans` is dimensionless. The source grid has resolving power
150,000. The full SkyCalc query parameters are in `skycalc_parameters.json`.
The simulation uses the transmission column and scales it by exposure
airmass before convolution onto the ANDES grid.

SkyCalc references: Noll et al. (2012), A&A 543, A92,
https://doi.org/10.1051/0004-6361/201219040, and Jones et al. (2013),
A&A 560, A91, https://doi.org/10.1051/0004-6361/201322433.

The local stellar-intensity bundle must be synthesized separately using
`docs/stellar_intensities.md` and placed here as
`stellar_Na_intensities.npz`. The published figures use an intrinsic
MARCS/Turbospectrum Na-NLTE synthesis with Teff 5052 K, log g 4.587,
[Fe/H] −0.03, microturbulence 1 km/s and A(Na) 6.14. Its 71 angular
samples span mu = 1e−10 to 1. Conversion to vacuum retains the local
continuum. These parameters describe the figure's model. Exact numerical
recovery also requires the same synthesis inputs and opacity tables.

Reference FITS SHA256:
`d4cee93f5981d95458a7d9adfc4a6f160b30a819f9b4a867c443474de12666b6`

Validated stellar bundle SHA256:
`d6a0756d76fda1c5a74fa1e71869497d3d5104ffb38ff603374f08e078059a0c`

The sequence NPZ is generated from the CSV by the short code block in
Tutorial 13. Output arrays and Molecfit working directories are generated
by a run and belong under the configured output directory.
