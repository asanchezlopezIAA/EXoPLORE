"""Match Welch velocity labels and samples on the original CCF grid."""
import numpy as np
import pytest
from scipy.stats import ttest_ind
from exoplore.ccf.statistics import Welch_ttest_map


@pytest.mark.parametrize('half_width', [0, 1, 2])
def test_every_window_and_background_match_direct_selection(half_width):
    rng = np.random.default_rng(61)
    velocity = np.arange(-10., 11.)
    ccf = rng.normal(size=(len(velocity), 7, 2))
    result = Welch_ttest_map(
        ccf, velocity, np.array([100., 150.]),
        in_trail_left_right=half_width, snr_exclude_kms=3.,
        velocity_step_kms=1.)
    centres = np.arange(half_width, len(velocity) - half_width)
    np.testing.assert_array_equal(result[3], velocity[centres])
    for row, centre in enumerate(centres):
        inside = np.arange(centre-half_width, centre+half_width+1)
        outside = np.arange(len(velocity))
        outside = outside[(outside < inside[0]-3) | (outside > inside[-1]+3)]
        for column in range(2):
            expected = ttest_ind(ccf[inside, :, column].ravel(),
                                 ccf[outside, :, column].ravel(), equal_var=False)
            np.testing.assert_allclose(result[1][row, column], expected.statistic)
            np.testing.assert_allclose(result[2][row, column], expected.pvalue)


@pytest.mark.parametrize('step', [1.3, 3.])
def test_zero_velocity_signal_has_zero_reported_velocity(step):
    rng = np.random.default_rng(431)
    velocity = np.arange(-40, 41)*step
    ccf = np.broadcast_to(rng.normal(size=(1, 60, 1)),
                          (len(velocity), 60, 1)).copy()
    ccf += 2*np.exp(-.5*(velocity/3)**2)[:, None, None]
    result = Welch_ttest_map(
        ccf, velocity, np.array([150.]), in_trail_left_right=2,
        snr_exclude_kms=15., velocity_step_kms=step)
    peak = np.argmax(result[1][:, 0])
    assert result[3][peak] == 0.
