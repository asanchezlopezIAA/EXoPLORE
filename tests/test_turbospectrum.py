"""Check angular intensities, continuum handling and wavelength conventions."""

import numpy as np
import pytest
from pathlib import Path
import os
import subprocess
import sys

from exoplore.io.stellar_surface import load_local_intensities
from exoplore.io.turbospectrum import (
    read_turbospectrum_intensities, save_turbospectrum_bundle, _vacuum_coordinates)


def export(tmp_path, *, continuum=False, zero=False):
    """Small documented ASCII layout with known angular surface brightness."""
    path = tmp_path/("continuum.spec" if continuum else "lines.spec")
    norm = 1. if continuum else .5
    table = np.array([[5890., 1., 10., 2*norm, norm, 4*norm, norm],
                      [5891., 1., 10., 3*norm, norm, 6*norm, norm]])
    if zero:
        table[0, 4] = 0.
    np.savetxt(path, table, header="mu-points 0.0001 1.0")
    return path


def test_export_preserves_angular_brightness_and_recovers_continuum(tmp_path):
    local = read_turbospectrum_intensities(export(tmp_path), input_medium="air", output_medium="air")
    np.testing.assert_array_equal(local.mu, [1e-4, 1.])
    np.testing.assert_allclose(local.intensity, [[1., 1.5], [2., 3.]])
    np.testing.assert_allclose(local.continuum, [[2., 3.], [4., 6.]])


def test_separate_continuum_handles_zero_printed_normalised_intensity(tmp_path):
    path = export(tmp_path, zero=True)
    with pytest.raises(ValueError, match="separate continuum"):
        read_turbospectrum_intensities(path, input_medium="air", output_medium="air")
    local = read_turbospectrum_intensities(path, input_medium="air", output_medium="air",
        continuum_path=export(tmp_path, continuum=True))
    np.testing.assert_allclose(local.continuum, [[2., 3.], [4., 6.]])


def test_air_conversion_preserves_spectral_density(tmp_path):
    path = export(tmp_path)
    air = read_turbospectrum_intensities(path, input_medium="air", output_medium="air")
    vacuum = read_turbospectrum_intensities(path, input_medium="air", output_medium="vacuum")
    wave, jacobian = _vacuum_coordinates(air.wavelength_um*1e4)
    # Independently differentiate the coordinate conversion to check the density factor.
    plus, _ = _vacuum_coordinates(air.wavelength_um*1e4+.01)
    minus, _ = _vacuum_coordinates(air.wavelength_um*1e4-.01)
    np.testing.assert_allclose((plus-minus)/.02, jacobian, rtol=1e-9)
    np.testing.assert_allclose(vacuum.wavelength_um*1e4, wave)
    np.testing.assert_allclose(vacuum.intensity*jacobian, air.intensity)
    np.testing.assert_allclose(vacuum.intensity/vacuum.continuum, air.intensity/air.continuum)


def test_bundle_loads_and_existing_output_is_protected(tmp_path):
    path = export(tmp_path)
    destination = tmp_path/'local.npz'
    save_turbospectrum_bundle(path, destination, source="known test atmosphere",
                             input_medium="air")
    local, resolution, metadata = load_local_intensities(destination, wavelength_medium="vacuum")
    assert resolution == 0
    assert metadata['source'] == "known test atmosphere"
    assert local.continuum is not None
    with pytest.raises(FileExistsError):
        save_turbospectrum_bundle(path, destination, source="other", input_medium="air")


def test_continuum_grid_mismatch_is_rejected(tmp_path):
    path = export(tmp_path)
    continuum = export(tmp_path, continuum=True)
    text = continuum.read_text().replace('1.0', '0.9', 1)
    continuum.write_text(text)
    with pytest.raises(ValueError, match="identical"):
        read_turbospectrum_intensities(path, input_medium="air", output_medium="air",
                                      continuum_path=continuum)


def test_unrecognised_header_is_rejected(tmp_path):
    path = tmp_path/'flux.spec'
    path.write_text('5890 0.5 10\n5891 0.6 11\n')
    with pytest.raises(ValueError, match="mu-points"):
        read_turbospectrum_intensities(path, input_medium="air", output_medium="vacuum")


def test_additional_angles_merge_without_repeating_disk_centre(tmp_path):
    path = export(tmp_path)
    extra = tmp_path/'extra.spec'
    extra.write_text(path.read_text().replace('0.0001', '0.5'))
    local = save_turbospectrum_bundle(path, tmp_path/'combined.npz', source="same model",
        input_medium="air", additional_exports=[(extra, None)])
    np.testing.assert_allclose(local.mu, [1e-4, .5, 1.])
    assert local.intensity.shape == (3, 2)


def test_different_syntheses_cannot_be_merged_at_a_common_angle(tmp_path):
    path = export(tmp_path)
    extra = tmp_path/'different.spec'
    table = np.loadtxt(path)
    table[:, 5] *= 2
    np.savetxt(extra, table, header="mu-points 0.5 1.0")
    with pytest.raises(ValueError, match="Overlapping"):
        save_turbospectrum_bundle(path, tmp_path/'combined.npz', source="different model",
            input_medium="air", additional_exports=[(extra, None)])


def test_preparation_script_runs_from_checkout_without_pythonpath(tmp_path):
    path = export(tmp_path)
    destination = tmp_path/'script_output.npz'
    script = Path(__file__).resolve().parents[1]/'scripts/prepare_stellar_intensities.py'
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(script), str(path), str(destination),
        '--input-medium', 'air', '--source', 'test atmosphere'],
        cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    local, _, _ = load_local_intensities(destination, wavelength_medium='vacuum')
    assert local.intensity.shape == (2, 2)
