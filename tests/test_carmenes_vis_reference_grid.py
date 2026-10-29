"""Guard the repaired VIS header and the original wavelength payload.

Only the malformed first FITS block was repaired. The payload digest was
captured before that repair; changing the wavelength data requires a separate
calibration review, not regeneration of this reference to make a test pass.
"""

import hashlib
from importlib.resources import files
import warnings

from astropy.io import fits
import numpy as np

from exoplore.instruments.carmenes_vis import get_instrument_info
from exoplore.instruments.wavegrid import get_WaveGrid


def test_vis_header_is_valid_and_wavelength_payload_unchanged():
    reference = files("exoplore.instruments").joinpath("data/wave_CARMENES_VIS.fits")
    raw = reference.read_bytes()
    # A single 2880-byte header block precedes the original big-endian doubles.
    assert hashlib.sha256(raw[2880:]).hexdigest() == (
        "9db11ad642b16955b9360fe3e09bdf26289f586606d5f2a9491c801a8402c033"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with fits.open(str(reference)) as hdul:
            hdul.verify("exception")
            wavelength = hdul[0].data
            assert wavelength.shape == (44, 4096)
            assert hdul[0].header["BITPIX"] == -64
            assert np.isfinite(wavelength).all()


def test_exoplore_loads_bundled_vis_grid_without_altering_values(tmp_path):
    # Empty inputs force the instrument's existing bundled-grid fallback.
    from exoplore.config.models import InstrumentConfig, ObservationConfig
    from exoplore.instruments.base import InstrumentInputs
    observation = ObservationConfig()
    info = get_instrument_info(InstrumentInputs(
        inputs_dir=str(tmp_path), planet_name="HD189733b",
        observation=observation, fixed_snr=100.,
    ))
    wave, n_pixels, *optional_inputs = get_WaveGrid(
        InstrumentConfig(name="CARMENES_VIS"), observation, info, 44,
    )
    # Preserve the existing loader convention: pixel axis first at this stage.
    expected = np.frombuffer(
        files("exoplore.instruments").joinpath("data/wave_CARMENES_VIS.fits").read_bytes(),
        dtype=">f8", count=44 * 4096, offset=2880,
    ).reshape(44, 4096)
    assert n_pixels == 4096
    np.testing.assert_array_equal(wave, expected.T)
    assert all(item is None for item in optional_inputs)
