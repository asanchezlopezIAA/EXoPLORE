"""Diagnostic curves must display the actual simulated flux, without noise inflation."""
import numpy as np


def test_pipeline_steps_displays_unscaled_noisy_flux(monkeypatch, tmp_path):
    from matplotlib.axes import Axes
    from exoplore.plotting.steps import plot_pipeline_steps

    wave = np.linspace(1.0, 1.01, 20)
    phase = np.linspace(-0.02, 0.02, 5)
    noiseless = np.ones((5, 20))
    noisy = noiseless + np.linspace(-0.01, 0.01, 100).reshape(5, 20)
    original_noisy = noisy.copy()
    plotted = {}
    original_plot = Axes.plot

    def capture_plot(self, *args, **kwargs):
        if kwargs.get('label') in ('Noisy', 'Noiseless'):
            plotted[kwargs['label']] = np.asarray(args[1]).copy()
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, 'plot', capture_plot)
    plot_pipeline_steps(
        'noise_scale_test', str(tmp_path) + '/', wave, phase,
        np.array([1, 2, 3]), np.array([0, 4]), np.arange(20),
        noiseless, noisy, noisy - 1, spec_idx=2,
        sysrem_stages={'Corrected (residual)': noisy - 1},
        save_plot=False, show_plot=False,
    )
    np.testing.assert_array_equal(plotted['Noisy'], noisy[2])
    np.testing.assert_array_equal(plotted['Noiseless'], noiseless[2])
    np.testing.assert_array_equal(noisy, original_noisy)


def test_pipeline_steps_marks_masked_residual_channels_white(monkeypatch, tmp_path):
    from matplotlib.axes import Axes
    from exoplore.plotting.steps import plot_pipeline_steps

    wave = np.linspace(1.0, 1.01, 20)
    flux = np.ones((5, 20))
    residual = np.zeros_like(flux)
    residual[:, 4] = np.nan
    meshes = []
    original = Axes.pcolormesh

    def capture_mesh(self, *args, **kwargs):
        mesh = original(self, *args, **kwargs)
        meshes.append(mesh)
        return mesh

    monkeypatch.setattr(Axes, 'pcolormesh', capture_mesh)
    plot_pipeline_steps(
        'white_mask_test', str(tmp_path) + '/', wave, np.linspace(-0.02, 0.02, 5),
        np.array([1, 2, 3]), np.array([0, 4]), np.arange(20),
        flux, flux, residual, spec_idx=2,
        sysrem_stages={'Corrected (residual)': residual},
        save_plot=False, show_plot=False,
    )
    np.testing.assert_array_equal(meshes[-1].cmap.get_bad(), [1, 1, 1, 1])
    assert np.ma.getmaskarray(meshes[-1].get_array()).sum() == 5


def test_pipeline_steps_adds_separate_tenfold_noise_illustration(monkeypatch, tmp_path):
    from matplotlib.axes import Axes
    from exoplore.plotting.steps import plot_pipeline_steps

    wave = np.linspace(0.953, 0.9545, 20)
    noiseless = np.ones((5, 20))
    noisy = noiseless + np.linspace(-0.01, 0.01, 100).reshape(5, 20)
    original_noisy = noisy.copy()
    curves = {}
    original = Axes.plot

    def capture_plot(self, *args, **kwargs):
        if kwargs.get('label'):
            curves[kwargs['label']] = np.asarray(args[1]).copy()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, 'plot', capture_plot)
    plot_pipeline_steps(
        'tenfold_noise_test', str(tmp_path) + '/', wave, np.linspace(-0.02, 0.02, 5),
        np.array([1, 2, 3]), np.array([0, 4]), np.arange(20),
        noiseless, noisy, noisy - 1, spec_idx=2, xlim_1d=(0.953, 0.9545),
        sysrem_stages={'Corrected (residual)': noisy - 1},
        noise_illustration_factor=10, save_plot=False, show_plot=False,
    )
    np.testing.assert_array_equal(curves['Noisy'], noisy[2])
    np.testing.assert_array_equal(curves['Noiseless'], noiseless[2])
    np.testing.assert_allclose(curves['10× noise deviations (illustration only)'],
                               noiseless[2] + 10 * (noisy[2] - noiseless[2]))
    np.testing.assert_array_equal(noisy, original_noisy)


def test_noise_illustration_omits_excluded_channels(monkeypatch, tmp_path):
    from matplotlib.axes import Axes
    from exoplore.plotting.steps import plot_pipeline_steps

    wave = np.linspace(0.953, 0.9545, 20)
    baseline = np.full((5, 20), 0.8)
    noisy = baseline + 0.001
    noisy[:, 4] = 1.0  # Fill value for a masked pixel.
    good = np.delete(np.arange(20), 4)
    curves = {}
    handles = {}
    original = Axes.plot

    def capture(self, *args, **kwargs):
        lines = original(self, *args, **kwargs)
        if kwargs.get('label'):
            curves[kwargs['label']] = np.asarray(args[1]).copy()
            handles[kwargs['label']] = lines[0]
        return lines

    monkeypatch.setattr(Axes, 'plot', capture)
    plot_pipeline_steps(
        'masked_noise', str(tmp_path) + '/', wave, np.linspace(-0.02, 0.02, 5),
        np.array([1, 2, 3]), np.array([0, 4]), good,
        baseline, noisy, noisy - baseline, spec_idx=2,
        sysrem_stages={'Corrected (residual)': noisy - baseline},
        noise_illustration_factor=10, save_plot=False, show_plot=False,
    )
    for name in ['Noiseless', 'Noisy', '10× noise deviations (illustration only)']:
        assert np.isnan(curves[name][4])
    np.testing.assert_allclose(curves['Noisy'][good] - curves['Noiseless'][good], 0.001)
    np.testing.assert_allclose(curves['10× noise deviations (illustration only)'][good]
                               - curves['Noiseless'][good], 0.01)

    assert handles['Noisy'].get_zorder() > handles['Noiseless'].get_zorder()
    assert handles['Noisy'].get_zorder() > handles['10× noise deviations (illustration only)'].get_zorder()
    assert handles['Noisy'].get_alpha() == 0.8
    assert handles['Noiseless'].get_alpha() == 0.65
    assert handles['10× noise deviations (illustration only)'].get_alpha() == 0.6
    assert not handles['Noisy'].axes.child_axes
