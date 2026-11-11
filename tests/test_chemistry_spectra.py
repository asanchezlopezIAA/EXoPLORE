"""Check chemistry-to-spectrum bookkeeping and the illustrated spectral resolution."""
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import exoplore.analysis.chemistry_spectra as spectra
from exoplore.atmosphere.prt import convolve
from exoplore.plotting.chemistry import plot_chemistry_transmission


def test_chunks_cover_range_and_include_convolution_margins():
    chunks = spectra.wavelength_chunks(.6, 5., 3500.)
    assert chunks[0][0] == .6
    assert chunks[-1][1] == 5.
    margin = 8/(2*np.sqrt(2*np.log(2))*3500)
    for i, (left, right, padded_left, padded_right) in enumerate(chunks):
        assert padded_left < left < right < padded_right
        assert np.log(left/padded_left) == pytest.approx(margin)
        assert np.log(padded_right/right) == pytest.approx(margin)
        if i:
            assert left == chunks[i-1][1]


@pytest.mark.parametrize('low, high, resolution', [
    (0., 5., 3500.), (5., .6, 3500.), (.6, 5., 0.), (.6, 5., np.nan),
])
def test_invalid_spectral_settings_fail(low, high, resolution):
    with pytest.raises(ValueError):
        spectra.wavelength_chunks(low, high, resolution)


def test_chunked_spectra_match_full_convolution_and_reuse_cache(tmp_path, monkeypatch):
    p = np.array([1e-8, 1., 100.])
    t = np.full(3, 1000.)
    region = SimpleNamespace(reference_pressure_bar=.01, cloud_pressure_bar=None)
    planet = SimpleNamespace(name='controlled planet', R_pl=1e10, R_star=1e11)
    columns = {name: ({'H2O':np.full(3, factor)}, np.full(3, 2.3))
               for name, factor in zip(spectra.CASES, (1., 1., 2., 3.))}
    monkeypatch.setattr(spectra, 'prepare_spectral_gases',
        lambda *args: (region, planet, p, t, 1000., columns))
    (tmp_path/'opacities').mkdir()
    calls = []
    native = np.geomspace(.59, .64, 85000)
    # Put a spectral line near a chunk boundary to check the overlap treatment.
    feature = 1e-3*np.exp(-.5*(np.log(native/.615)/1e-4)**2)

    def radtrans(pressures, low, high, input_data):
        calls.append((low, high))
        return (native >= low) & (native <= high)

    def transit(mask, temperatures, mf, mmw, gravity, rp, rs, p0, **kwargs):
        np.testing.assert_array_equal(temperatures, t)
        np.testing.assert_array_equal(mmw, np.full(3, 2.3))
        return native[mask], (.02 + mf['H2O'][0]*feature)[mask]

    monkeypatch.setattr(spectra, '_radtrans', radtrans)
    monkeypatch.setattr(spectra, 'compute_transmission_spectrum', transit)
    config = SimpleNamespace(paths=SimpleNamespace(prt_input_data=str(tmp_path)))
    kwargs = dict(low=.6, high=.63, resolving_power=3500., log_width=.015)
    arrays, report = spectra.compare_transmission_spectra(config, 'morning_day', tmp_path, **kwargs)
    expected = np.interp(arrays['wavelength_um'], native, convolve(native, .02+feature, 3500.))
    np.testing.assert_allclose(arrays['easychem'], expected, atol=1e-9, rtol=0)
    assert report['thermal_control_max_abs_ppm'] == 0.
    assert report['peak_abs_difference_ppm']['mixing'] > 0.
    assert report['fresh_chunks'] == len(calls)
    original_calls = len(calls)
    cached, second = spectra.compare_transmission_spectra(config, 'morning_day', tmp_path, **kwargs)
    assert len(calls) == original_calls
    assert second['fresh_chunks'] == 0
    assert second['reused_chunks'] == original_calls
    np.testing.assert_array_equal(cached['photochemistry'], arrays['photochemistry'])


def test_transmission_figure_converts_depths_and_residuals_correctly():
    wave = np.linspace(.6, 5., 10)
    arrays = dict(wavelength_um=wave, easychem=np.full(10,.02),
                  mixing=np.full(10,.0201), photochemistry=np.full(10,.0202))
    fig = plot_chemistry_transmission(arrays, 'HD 189733 b', 3500.)
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), 2.)
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), 100.)
    np.testing.assert_allclose(fig.axes[1].lines[1].get_ydata(), 200.)
    assert '3500' in fig.axes[0].get_title()
    plt.close(fig)


@pytest.fixture
def accepted_gases(tmp_path, monkeypatch):
    """Build small accepted-column records to test profile reuse independently."""
    from dataclasses import asdict
    import hashlib
    import json
    from exoplore.analysis.chemistry import SPECIES, comparison_settings
    from exoplore.config.chemistry import VulcanConfig, EddyDiffusionConfig

    settings = VulcanConfig(network_file='thermo/NCHO_photo_network.txt',
        elemental_abundances={'H':1., 'He':.08}, photochemistry=True,
        stellar_flux_file='uv.txt', orbit_radius_au=.031,
        zenith_angle_deg=48., diurnal_factor=1.,
        eddy_diffusion=EddyDiffusionConfig(mode='constant', value_cm2_s=1e9))
    region = SimpleNamespace(chemistry=SimpleNamespace(vulcan=settings), reference_pressure_bar=.01)
    planet = SimpleNamespace(M_pl=1e30, R_pl=1e10, R_star=1e11)
    p = np.array([1e-8,1.,100.])
    t = np.full(3,1000.)
    arrays = {'species':np.array(SPECIES), 'easychem':np.full((9,3),.1)}
    cases = comparison_settings(settings)
    report = {'cases':{name:{'cache_keys':[name]} for name in cases}}
    config = SimpleNamespace(paths=SimpleNamespace(vulcan_cache_dir=str(tmp_path),
                                                   vulcan_directory=str(tmp_path)))
    names = np.array([*SPECIES, 'He'])
    gas = dict(pressure_bar=p, temperature_K=t, species=names,
        mole_fractions=np.full((3,10),.1), molecular_masses=np.ones(10))
    for name, case in cases.items():
        folder = tmp_path/name
        folder.mkdir()
        np.savez(folder/'gas_profile.npz', **gas)
        arrays[name] = np.full((9,3),.1)
        record = dict(request={'settings':asdict(case), 'physical':{'stellar_radius_cm':planet.R_star}},
            gas_profile_sha256=hashlib.sha256((folder/'gas_profile.npz').read_bytes()).hexdigest(),
            end_case=1, atom_loss={'H':0.})
        (folder/'manifest.json').write_text(json.dumps(record))
        (folder/'radius_alignment_test.json').write_text(json.dumps(dict(
            reference_pressure_bar=.01, reference_radius_cm=planet.R_pl, planet_mass_g=planet.M_pl)))
    monkeypatch.setattr(spectra, '_read_column',
        lambda *args: (region, cases, planet, p, t, 1000., arrays, report))
    monkeypatch.setattr(spectra, 'equilibrium_reference',
        lambda *args, **kwargs: (arrays['easychem'], {}, gas))
    return config, tmp_path


def test_accepted_gases_use_complete_mixture_without_running_vulcan(accepted_gases):
    config, output = accepted_gases
    *_, columns = spectra.prepare_spectral_gases(config, 'morning_day', output)
    assert set(columns) == set(spectra.CASES)
    for mf, mu in columns.values():
        np.testing.assert_allclose(mu, 1.)
        assert set(mf) == {'H2','He',*spectra.LINE_SPECIES}
        # H remains in the mass denominator although it has no line table here.
        np.testing.assert_allclose(sum(mf.values()), .9)


@pytest.mark.parametrize('change', ['settings', 'gas_hash', 'radius', 'stellar_radius', 'convergence'])
def test_altered_chemistry_or_system_parameters_fail(accepted_gases, change):
    import json
    config, output = accepted_gases
    folder = output/'photochemistry'
    path = folder/'manifest.json'
    record = json.loads(path.read_text())
    if change == 'settings':
        record['request']['settings']['diurnal_factor'] = .5
    elif change == 'gas_hash':
        record['gas_profile_sha256'] = 'altered'
    elif change == 'radius':
        # A mismatching radius record should fail before any spectrum is calculated.
        (folder/'radius_alignment_test.json').write_text(json.dumps(dict(
            reference_pressure_bar=.01, reference_radius_cm=2e10, planet_mass_g=1e30)))
    elif change == 'stellar_radius':
        record['request']['physical']['stellar_radius_cm'] = 2e11
    else:
        record['end_case'] = 3
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        spectra.prepare_spectral_gases(config, 'morning_day', output)
