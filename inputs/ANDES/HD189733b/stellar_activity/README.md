# Controlled stellar components for Tutorial 14

Generate the component NPZ files from the Tutorial 13 quiet intensity bundle:

```bash
python scripts/run_stellar_activity_tutorial.py --prepare
```

The script reads `../sodium/stellar_Na_intensities.npz` relative to this input
folder. It writes `spot.npz` and `facula.npz` here, retaining all original
wavelength and mu samples and the specific-intensity unit.

The spot continuum is 0.60 times the quiet continuum. Its normalized Na D2
intensity is multiplied by `1 - 0.40*g(v)`, where `g` is a Gaussian centred
at 589.158326 nm in vacuum with sigma 6 km/s.

The facular continuum ratio is `1.10 + 0.20*(1-mu)`. Its normalized spectrum
is `q + 0.40*(1-q)*g(v)`, with `q` the quiet normalized intensity. The core
filling and brightness contrast are controlled inputs. They provide an
example of stellar-model incompleteness, with no temperature calibration
or empirical active-region fit for HD 189733.

The component metadata records these coefficients and the quiet-file SHA256.
The displayed experiment uses the Tutorial 13 quiet bundle with SHA256:

`d6a0756d76fda1c5a74fa1e71869497d3d5104ffb38ff603374f08e078059a0c`

The generated binary inputs are local products. Their generation script and
prescriptions allow the same spectra to be reconstructed from the quiet grid.
