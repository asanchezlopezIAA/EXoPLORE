"""Nonfinite aggregate diagnostics must leave the statistical study executable."""
import numpy as np
import pytest
from exoplore.plotting.kpvsys import plot_Kp_Vrest, plot_1D_CCF
from exoplore.plotting.models import DetectionPlotSettings


@pytest.mark.parametrize('value', [np.inf, np.nan])
@pytest.mark.parametrize('kind', ['map', 'slice'])
def test_nonfinite_plot_is_reported_without_mutating_statistics(tmp_path, value, kind):
    settings = DetectionPlotSettings('test', str(tmp_path)+'/', 149., 320., 250.,
                                     50., 4, False, 'transit')
    statistic = np.zeros((3, 3));statistic[1, 1] = value
    original = statistic.copy()
    with pytest.warns(RuntimeWarning, match='nonfinite values'):
        if kind == 'map':
            plot_Kp_Vrest(settings, np.array([-1., 0., 1.]), statistic,
                          np.array([-1., 0., 1.]), save_plot=True)
        else:
            plot_1D_CCF(settings, np.array([-1., 0., 1.]), statistic,
                        1, value, 1, 0., [-1., 1.], save_plot=True)
    np.testing.assert_array_equal(statistic, original)
    assert not list(tmp_path.iterdir())
