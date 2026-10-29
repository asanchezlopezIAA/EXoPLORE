"""Check direct sodium route selection and explicit input configuration."""
from dataclasses import asdict
import pytest
from exoplore.config import SimulationConfig
from exoplore.config.atomic_transmission import AtomicTransmissionConfig
from exoplore.core.spectroscopy_mode import spectroscopy_route


def atomic(**changes):
    values = dict(enabled=True,observing_sequence_file='night.npz',
                  wavelength_interval_um=[.5885,.5929],model_wavelength_limits_um=[.5882,.5932],
                  line_wavelengths_nm=[589.158326,589.755814],
                  observatory_lon_lat_height=[-2.5463,37.2236,2168.],molecfit={'esorex':'esorex'})
    values.update(changes)
    return AtomicTransmissionConfig(**values)


def test_default_and_opted_in_routes():
    c = SimulationConfig()
    assert spectroscopy_route(c) == 'standard'
    c.pipeline.name = 'Seidel22'
    c.atomic_transmission = atomic()
    assert spectroscopy_route(c) == 'atomic_simulation'
    c.atomic_transmission.enabled = False
    assert spectroscopy_route(c) == 'standard'


def test_config_round_trip_keeps_atomic_and_uncertainty_inputs():
    c = SimulationConfig.from_dict(dict(atomic_transmission=asdict(atomic()),
        pipeline=dict(name='Seidel22',seidel22=dict(continuum_windows_nm=[[590.1,590.2]],
                                                  noise_resamples=600,noise_resample_seed=123))))
    restored = SimulationConfig.from_dict(c.to_dict())
    assert restored.to_dict() == c.to_dict()
    assert restored.pipeline.seidel22.noise_resamples == 600


def test_simulator_dispatches_the_direct_runner(monkeypatch):
    from exoplore.core import atomic_simulation
    from exoplore.core import ExoploreSimulator
    calls = []
    monkeypatch.setattr(atomic_simulation,'validate_atomic_run',lambda c: calls.append('validate'))
    monkeypatch.setattr(atomic_simulation,'run_atomic_simulation',lambda c: calls.append('run'))
    c = SimulationConfig.from_dict(dict(atomic_transmission=asdict(atomic()),
        pipeline=dict(name='Seidel22',prepare_template=False,
                      seidel22=dict(continuum_windows_nm=[[590.1,590.2]]))))
    simulator = ExoploreSimulator(c)
    simulator.summary()
    assert calls == ['validate']
    simulator.run()
    assert calls == ['validate','run']


@pytest.mark.parametrize('changes',[dict(molecfit_workers=17),dict(molecfit_workers=True),
    dict(line_wavelengths_nm=[]),dict(wavelength_interval_um=[.6,.5]),dict(observing_sequence_file='')])
def test_rejects_incomplete_inputs(changes):
    with pytest.raises(ValueError):atomic(**changes)


def test_andes_requires_explicit_matching_etc_exposure(tmp_path):
    from exoplore.core.atomic_simulation import validate_atomic_run
    c = SimulationConfig.from_dict(dict(atomic_transmission=asdict(atomic()),
        instrument=dict(name='ANDES_UBV',order_indices=[56]),
        observation=dict(exposure_time_seconds=30.,throughput_seed=12345,use_accurate_berv=False),
        atmosphere=dict(planet_model=dict(isothermal=True,use_easychem=False,
            species=['H2','He','Na'],mass_fractions=[.74,.259,.001])),
        pipeline=dict(name='Seidel22',prepare_template=False,
                      seidel22=dict(continuum_windows_nm=[[590.1,590.2]]))))
    with pytest.raises(ValueError,match='ETC reference exposure'):
        validate_atomic_run(c)
    c.atomic_transmission.etc_reference_exposure_seconds=90.
    with pytest.raises(ValueError,match='ETC reference exposure'):
        validate_atomic_run(c)


def test_andes_reference_exposure_round_trip():
    value = atomic(etc_reference_exposure_seconds=30., molecfit_workers=16)
    c = SimulationConfig.from_dict(dict(atomic_transmission=asdict(value)))
    restored = SimulationConfig.from_dict(c.to_dict()).atomic_transmission
    assert restored.etc_reference_exposure_seconds == 30.
    assert restored.molecfit_workers == 16
