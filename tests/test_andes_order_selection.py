"""ANDES ETC noise must follow physical orders through every selection."""
import numpy as np
import pytest

from exoplore.instruments.wavegrid import pixel_snr_for_andes_order, pixel_snr_one_order


@pytest.fixture
def instrument():
    # Distinct wavelengths, ETC anchors and off-centre midpoints make an
    # accidental selected-list index observable in both the blaze and S/N.
    wavelength = np.array([np.linspace(start, start+.012, 64)
                           for start in [.95, 1.04, 1.31, 1.72]])
    midpoint_nm = np.array([954., 1048., 1317., 1725.])
    snr = np.array([400., 600., 1200., 900.])
    return wavelength, midpoint_nm, snr


@pytest.mark.parametrize('selected', [[0, 1, 2, 3], [0, 1], [2], [1, 3], [3, 0, 2]])
@pytest.mark.parametrize('exposures', [1, 5])
def test_selected_order_matches_complete_instrument(instrument, selected, exposures):
    wave, mid, anchor = instrument
    x = np.linspace(-1., 1., wave.shape[1])
    # Each exposure sees a different telluric depth at the same physical pixels.
    tellurics = 1-.6*np.linspace(.5, 1., exposures)[:, None]*np.exp(-((x-.2)/.1)**2)
    complete = [pixel_snr_for_andes_order(wave, order, tellurics, anchor, mid,
                                        px_per_resel=2.5) for order in range(len(wave))]
    for order in selected:
        actual = pixel_snr_for_andes_order(wave, order, tellurics, anchor, mid,
                                          px_per_resel=2.5)
        np.testing.assert_array_equal(actual, complete[order])


def test_complete_order_sequence_preserves_existing_calculation(instrument):
    wave, mid, anchor = instrument
    tellurics = np.linspace(.3, 1., 3*wave.shape[1]).reshape(3, -1)
    for order in range(len(wave)):
        # In the previous full-order loop, h was the physical order ID.
        centre = int(np.argmin(abs(wave[order]-mid[order]*1e-3)))
        previous = pixel_snr_one_order(wave[order], tellurics, anchor[order],
                                      px_per_resel=2.5, center_pix=centre)
        actual = pixel_snr_for_andes_order(wave, order, tellurics, anchor, mid,
                                          px_per_resel=2.5)
        np.testing.assert_array_equal(actual, previous)


def test_etc_anchor_and_resel_conversion_are_preserved(instrument):
    wave, mid, anchor = instrument
    for order in range(len(wave)):
        snr = pixel_snr_for_andes_order(wave, order, np.ones((2, wave.shape[1])),
                                       anchor, mid, px_per_resel=2.5)
        centre = np.argmin(abs(wave[order]-mid[order]*1e-3))
        np.testing.assert_allclose(snr[:, centre], anchor[order]/np.sqrt(2.5))


def test_separate_nights_use_their_own_tellurics(instrument):
    wave, mid, anchor = instrument
    dry = np.ones((3, wave.shape[1]))
    wet = dry.copy()
    wet[:, 10:20] = .25
    a = pixel_snr_for_andes_order(wave, 2, dry, anchor, mid, px_per_resel=2.5)
    b = pixel_snr_for_andes_order(wave, 2, wet, anchor, mid, px_per_resel=2.5)
    np.testing.assert_allclose(b[:, 10:20], .5*a[:, 10:20])
    np.testing.assert_array_equal(a[:, 30:], b[:, 30:])
