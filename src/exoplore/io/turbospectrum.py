"""Read plane-parallel Turbospectrum intensity exports for RM and CLV.

Turbospectrum's intensity output contains an angular header, three flux
columns, then absolute and continuum-normalised intensity for each mu.
The absolute columns preserve the relative surface brightness across the star.
This reader follows the v20 documentation and keeps synthesis separate from
EXoPLORE's rotation, transit integration and instrumental convolution.


scripts/prepare_stellar_intensities.py calls this reader and bundle writer.
The output is the rm_clv.model_file input read by io/stellar_surface.py.
Keep the absolute brightness variation between synthesis angles throughout
conversion, because it sets the amount of light occulted during transit.
"""

from pathlib import Path
import hashlib
import json

import numpy as np

from exoplore.stellar.surface import LocalIntensityGrid


def _read_export(path):
    """Read the documented ASCII layout and its own angular coordinates."""
    # The first ASCII line records the viewing-angle cosines used by the synthesis.
    with Path(path).open() as stream:
        header = stream.readline().split()
    # Require the documented angular header before interpreting the intensity columns.
    if header[:2] != ["#", "mu-points"]:
        raise ValueError("Expected a plane-parallel Turbospectrum '# mu-points' header")
    try:
        # Read the angular coordinates as numerical values.
        mu = np.asarray(header[2:], dtype=float)
    except ValueError as error:
        raise ValueError("Invalid Turbospectrum angular header") from error
    if len(mu) < 2:
        raise ValueError("Export at least two angular intensities, including mu=1")
    # Read all wavelength rows, retaining two dimensions for shape checks.
    table = np.loadtxt(path, ndmin=2)
    # Each row has three initial columns and two intensity columns per viewing angle.
    if table.shape[1] != 3+2*len(mu):
        raise ValueError("Turbospectrum columns must be wavelength, two fluxes and intensity pairs")
    if not np.isfinite(table).all():
        raise ValueError("Turbospectrum output contains nonfinite values")
    # Each angular pair is absolute intensity followed by I/I_continuum.
    return table[:, 0], mu, table[:, 3::2].T, table[:, 4::2].T


def _vacuum_coordinates(wavelength_angstrom):
    """Standard dry-air Edlen coordinates and their spectral-density Jacobian.

    The same refractive-index expression is used by the helium data reader.
    Here its derivative also converts per-Angstrom intensities, preserving
    I_lambda d_lambda when the wavelength coordinate changes.
    """
    # Use the input air wavelengths in Angstrom for the refractive-index expression.
    wave = np.asarray(wavelength_angstrom, dtype=float)
    if np.any((wave < 2000) | (wave > 20000)):
        raise ValueError("This air-to-vacuum conversion supports 2000–20000 Angstrom")
    # Express inverse wavelength in inverse micrometres, then square it for the dispersion law.
    squared_frequency = (10000/wave)**2
    # Evaluate the dry-air refractive index at each wavelength.
    index = 1+1e-8*(8342.13+2406030/(130-squared_frequency)
                   +15997/(38.9-squared_frequency))
    # Differentiate that index with respect to the input air wavelength.
    derivative = (-2*squared_frequency/wave)*1e-8* (
        2406030/(130-squared_frequency)**2+15997/(38.9-squared_frequency)**2)
    # Return vacuum wavelength and d(lambda_vacuum)/d(lambda_air),
    # which converts the associated spectral intensity density.
    return wave*index, index+wave*derivative


def read_turbospectrum_intensities(path, *, input_medium, output_medium,
                                  continuum_path=None):
    """Return a local intensity grid in micron wavelength coordinates.

    Both wavelength conventions must be supplied explicitly. Intensities
    retain per-Angstrom units in the chosen output convention.

    A separate export from the same atmosphere with no line lists supplies
    the continuum directly. Without it, I_cont is recovered from I/(I/I_cont).
    The latter inherits rounding from Turbospectrum's normalised columns and
    fails if a printed normalised intensity is zero.
    """
    # Require explicit air/vacuum conventions before converting wavelengths.
    if input_medium not in ("air", "vacuum") or output_medium not in ("air", "vacuum"):
        raise ValueError("Wavelength conventions must be 'air' or 'vacuum'")
    if input_medium == "vacuum" and output_medium == "air":
        raise ValueError("Export vacuum coordinates directly or choose vacuum output")
    # Read the common wavelength axis and each angle's absolute and normalized intensities.
    wave, mu, intensity, normalised = _read_export(path)
    # Recover the continuum as I divided by I/I_cont when only one export is supplied.
    if continuum_path is None:
        if np.any(normalised <= 0):
            raise ValueError("Rounded zero intensities require a separate continuum export")
        # Keep the recovered continuum on the same absolute scale as the line intensities.
        continuum = intensity/normalised
    else:
        # A separate line-free synthesis supplies the continuum directly.
        continuum_wave, continuum_mu, continuum, continuum_normalised = _read_export(continuum_path)
        # Require line and continuum exports to describe identical wavelengths and angles.
        if not np.array_equal(wave, continuum_wave) or not np.array_equal(mu, continuum_mu):
            raise ValueError("Line and continuum exports must have identical wavelength and mu grids")
        if not np.allclose(continuum_normalised, 1., atol=1e-5, rtol=0):
            raise ValueError("The continuum export still contains spectral lines")
    # Convert both wavelength coordinates and per-Angstrom spectral densities to vacuum.
    if input_medium == "air" and output_medium == "vacuum":
        wave, jacobian = _vacuum_coordinates(wave)
        # Scale intensity density to preserve I_lambda d_lambda under the coordinate change.
        intensity = intensity/jacobian
        # Apply the same density conversion to the continuum.
        continuum = continuum/jacobian
    # Convert the wavelength coordinate from Angstrom to micrometres for EXoPLORE.
    # The intensity density retains its per-Angstrom physical units.
    return LocalIntensityGrid(wave*1e-4, mu, intensity, continuum)


def save_turbospectrum_bundle(path, destination, *, source, input_medium,
                              output_medium="vacuum", continuum_path=None,
                              resolving_power=0., additional_exports=()):
    """Write the numeric NPZ input consumed by rm_clv.model_file.

    source identifies the atmosphere, abundances and synthesis used. The
    additional_exports contains (intensity_path, continuum_path) pairs from
    the same synthesis at further mu values. This supports programs that limit
    the number of output angles per run. Overlapping angles must agree.
    The archive also records hashes of the exports. Existing files are protected
    so preparation cannot silently replace a validated stellar model.
    """
    # Require a synthesis description and a meaningful prior instrumental resolution.
    if not source.strip():
        raise ValueError("Describe the synthesis source and stellar parameters")
    if not np.isfinite(resolving_power) or resolving_power < 0:
        raise ValueError("resolving_power must be zero or finite and positive")
    # Convert the first angular export and its optional continuum.
    local = read_turbospectrum_intensities(path, input_medium=input_medium,
        output_medium=output_medium, continuum_path=continuum_path)
    # Track the input exports for the saved provenance record.
    inputs = [Path(path)] + ([] if continuum_path is None else [Path(continuum_path)])
    # Merge additional angular exports from the same stellar synthesis.
    for intensity_path, extra_continuum_path in additional_exports:
        # Use the same wavelength convention and continuum rules for each additional export.
        extra = read_turbospectrum_intensities(intensity_path,
            input_medium=input_medium, output_medium=output_medium,
            continuum_path=extra_continuum_path)
        if not np.array_equal(local.wavelength_um, extra.wavelength_um):
            raise ValueError("Additional angular exports must share the wavelength grid")
        # Compare repeated angular samples before keeping one copy.
        for value in np.intersect1d(local.mu, extra.mu):
            # Find the matching angular rows in the existing and additional grids.
            first = np.flatnonzero(local.mu == value)[0]
            second = np.flatnonzero(extra.mu == value)[0]
            # Require line and continuum intensities to agree at shared angles.
            if not (np.allclose(local.intensity[first], extra.intensity[second], rtol=2e-5, atol=0)
                    and np.allclose(local.continuum[first], extra.continuum[second], rtol=2e-5, atol=0)):
                raise ValueError(f"Overlapping mu={value} intensities differ between exports")
        # Combine the angular coordinates from both exports.
        angles = np.concatenate((local.mu, extra.mu))
        # Sort the combined angles and retain one row for each distinct mu.
        angles, selected = np.unique(angles, return_index=True)
        # Merge the line and continuum rows using the same selected angular indices.
        local = LocalIntensityGrid(local.wavelength_um, angles,
            np.concatenate((local.intensity, extra.intensity))[selected],
            np.concatenate((local.continuum, extra.continuum))[selected])
        inputs.append(Path(intensity_path))
        if extra_continuum_path is not None:
            inputs.append(Path(extra_continuum_path))
    # Record the wavelength conversion, continuum source and hashes of every export.
    provenance = dict(input_medium=input_medium, output_medium=output_medium,
        continuum_method=("separate line-free synthesis" if continuum_path else "absolute/normalised intensity"),
        input_files=[dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                     for p in inputs])
    # Exclusive creation protects existing files and uses exactly the requested path.
    with Path(destination).open("xb") as stream:
        # Write numerical arrays and scalar metadata in the bundle format used by RM+CLV.
        np.savez_compressed(stream, wavelength_um=local.wavelength_um, mu=local.mu,
            intensity=local.intensity, continuum=local.continuum,
            intensity_kind="specific_intensity", wavelength_medium=output_medium,
            intensity_unit="erg s^-1 cm^-2 Angstrom^-1 sr^-1", source=source,
            resolving_power=resolving_power, provenance_json=json.dumps(provenance))
    # Return the converted grid so the command-line tool can report its dimensions.
    return local
