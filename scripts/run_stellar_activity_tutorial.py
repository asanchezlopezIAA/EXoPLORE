"""Run Tutorial 14 with Tutorial 13's atmosphere, observation and preparation.

From the repository root:
    PYTHONPATH=src python scripts/run_stellar_activity_tutorial.py --prepare
    PYTHONPATH=src python scripts/run_stellar_activity_tutorial.py --run --prt-input-data YOUR_INPUT_DATA

Each surface gets its own observation and Molecfit fits. Matched and quiet
corrections share those exact data and fits. Existing completed cases can be
resumed after their saved configuration and observation hashes are checked.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from exoplore.config import SimulationConfig, SurfaceRegionConfig
from exoplore.atmosphere.transmission import ChordTransmission
from exoplore.core.atomic_simulation import (build_atomic_observation, calculate_atomic_atmosphere,
    fit_atomic_tellurics, reduce_atomic_observation, telluric_input_hash, validate_atomic_run)
from exoplore.instruments import load_instrument
from exoplore.planets import load_planet
from exoplore.stellar.coupled import SpatialTransitModel

CASES = ('quiet', 'unocculted_spot', 'crossed_spot', 'crossed_facula')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(values):
    """Include array coordinates and shape when recording numerical identities."""
    h = hashlib.sha256()
    for key in sorted(values):
        value = np.ascontiguousarray(values[key])
        h.update(key.encode())
        h.update(str((value.shape, value.dtype.str)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def make_components(config):
    """Modify the supplied quiet intensities on their original wavelength/mu grids."""
    quiet = Path(config.rm_clv.model_file)
    parent_hash = file_hash(quiet)
    source = load_arrays(quiet)
    wave, mu = source['wavelength_um'], source['mu']
    centre = config.atomic_transmission.line_wavelengths_nm[0]/1000
    velocity = 299792.458*(wave/centre-1)
    gaussian = np.exp(-.5*(velocity/6.)**2)[None, :]
    shape = source['intensity']/source['continuum']
    for name in ('spot', 'facula'):
        path = Path(config.stellar_surface.components[name].intensity_model_file)
        if name == 'spot':
            contrast = np.full((len(mu), 1), .60)
            normalized = shape*(1-.40*gaussian)
        else:
            contrast = 1.10+.20*(1-mu[:, None])
            normalized = shape+.40*(1-shape)*gaussian
        metadata = dict(parent_quiet_sha256=parent_hash,
            model='controlled modification of Tutorial 13 local intensities',
            component=name, gaussian_sigma_kms=6., profile_coefficient=.40,
            continuum_contrast='.60' if name=='spot' else '1.10 + .20*(1-mu)',
            physical_calibration='unvalidated demonstration spectra')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            stored = load_arrays(path)
            if json.loads(str(stored['metadata_json'])) != metadata:
                raise ValueError(f'Existing component has different settings: {path}')
            continue
        payload = {k: source[k] for k in ('wavelength_um','mu','intensity_kind',
                    'wavelength_medium','intensity_unit','resolving_power')}
        continuum = source['continuum']*contrast
        np.savez_compressed(path, **payload, intensity=continuum*normalized,
            continuum=continuum, source='Tutorial 14 controlled '+name,
            metadata_json=json.dumps(metadata, sort_keys=True))


def case_config(base, name):
    """Change the stellar surface and output folder, retaining every observing choice."""
    config = SimulationConfig.from_dict(base.to_dict())
    config.paths.output_root = str(Path(base.paths.output_root)/name)
    if name == 'quiet':
        config.stellar_surface = None
    else:
        region = SurfaceRegionConfig('facula' if name=='crossed_facula' else 'spot',
            x=-.35, y=.55 if name=='unocculted_spot' else -.65, radius=.25)
        config.stellar_surface = replace(config.stellar_surface, regions=[region], correction_surface='quiet')
    return config


def fixed_settings(config):
    """Fields that must agree across surfaces and with the Tutorial 13 reference."""
    value = config.to_dict()
    result = {k: value[k] for k in ('instrument','observation','noise','pipeline','rm_clv','atomic_transmission','tellurics')}
    result['planet'] = dict(name=config.planet.name, parameter_sha256=file_hash(config.planet.parameter_file))
    result['retrieval_enabled'] = config.retrieval.enabled
    result['atmosphere'] = dict(planet_model=value['atmosphere']['planet_model'],
                               limb_asymmetries=value['atmosphere']['limb_asymmetries'])
    result['rm_clv'].pop('model_file')
    result['atomic_transmission'].pop('observing_sequence_file')
    result['tellurics'].pop('reference_telluric_file')
    return result


def shared_atmosphere(config, root, planet):
    """Keep the same pRT ray table in all four forward calculations."""
    fingerprint = hashlib.sha256(json.dumps(dict(
        atmosphere=asdict(config.atmosphere.planet_model),
        planet=file_hash(config.planet.parameter_file),
        boundaries=config.atomic_transmission.model_wavelength_limits_um),sort_keys=True).encode()).hexdigest()
    path = root/'planetary_rays.npz'
    if path.exists():
        data = load_arrays(path)
        if str(data['settings_sha256']) != fingerprint:
            raise ValueError('The saved planetary atmosphere belongs to different scientific settings')
        rays = ChordTransmission(data['wavelength_um'],data['impact_parameter_cm'],data['transmission'],str(data['prt_version']))
    else:
        rays = calculate_atomic_atmosphere(config, planet)
        np.savez_compressed(path, wavelength_um=rays.wavelength_um,
            impact_parameter_cm=rays.impact_parameter_cm, transmission=rays.transmission,
            prt_version=rays.prt_version,settings_sha256=fingerprint)
    return rays


def reuse_quiet(config, previous, output, rays, planet, resolution):
    """Check a completed quiet observation before reusing its full-night products."""
    old = SimulationConfig.from_json(previous/'resolved.json')
    if fixed_settings(config) != fixed_settings(old):
        raise ValueError('Quiet reference has different observing or atmospheric settings')
    for now, before in [(config.rm_clv.model_file,old.rm_clv.model_file),
                        (config.tellurics.reference_telluric_file,old.tellurics.reference_telluric_file),
                        (config.planet.parameter_file,old.planet.parameter_file)]:
        if file_hash(now) != file_hash(before):
            raise ValueError('Quiet reference scientific input hash differs')
    data = load_arrays(previous/'observations.npz')
    sequence = load_arrays(config.atomic_transmission.observing_sequence_file)
    for key in ('bjd','airmass'):
        np.testing.assert_array_equal(data[key],sequence[key])
    # Re-evaluate complete atmospheric transport at widely separated saved phases.
    intrinsic = load_arrays(previous/'intrinsic_spectra.npz')
    indices = np.array([0, len(data['phase'])//3, len(data['phase'])//2, len(data['phase'])-1])
    result = SpatialTransitModel(config.rm_clv,planet,resolution).evaluate(
        intrinsic['native_wave_um'],data['phase'][indices],intrinsic['stellar_velocity_kms'][indices],
        config.observation.exposure_time_seconds,chords=rays,
        planet_velocity_kms=intrinsic['planet_velocity_kms'][indices],return_intrinsic=True)
    np.testing.assert_array_equal(result.intrinsic.wavelength_um,intrinsic['wavelength_um'])
    differences = {}
    for key, attribute in [('flux','flux'),('opaque_flux','opaque_flux'),('stellar_flux','stellar_flux'),('continuum','continuum')]:
        actual = getattr(result.intrinsic,attribute)
        expected = intrinsic[key][indices]
        difference = float(np.max(abs(actual-expected))/np.max(abs(expected)))
        if difference > 1e-9:
            raise ValueError(f'Quiet replay differs for {key}: {difference}')
        differences[key] = difference
    output.mkdir()
    for name in ('observations.npz','intrinsic_spectra.npz'):
        shutil.copy2(previous/name,output/name)
    # Each accepted correction must contain the exact raw exposure being reused.
    corrected = dict(wave_nm=[],flux=[],error=[],transmittance=[])
    for n in range(len(data['bjd'])):
        directory = previous/'molecfit'/f'exposure_{n+1:03d}'
        if not json.loads((directory/'report.json').read_text())['accepted']:
            raise ValueError('An archived telluric correction was rejected')
        entry = load_arrays(directory/'correction.npz')
        for field, raw in [('flux','raw_flux'),('error','raw_error'),('native_wave_nm','wave_nm')]:
            np.testing.assert_array_equal(entry[field],data[raw][n])
        corrected['wave_nm'].append(entry['refined_wave_nm'])
        corrected['flux'].append(entry['flux']/entry['transmittance'])
        corrected['error'].append(entry['error']/entry['transmittance'])
        corrected['transmittance'].append(entry['transmittance'])
    corrected = {k:np.asarray(v) for k,v in corrected.items()}
    corrected['input_sha256'] = np.array(telluric_input_hash(config,data))
    np.savez_compressed(output/'corrected_observations.npz',**corrected)
    config.to_json(output/'resolved.json')
    original = json.loads((previous/'final_summary.json').read_text())
    record = dict(reused_from=str(previous),relative_replay_errors=differences,
        original_physical_seconds=original['physical_seconds'],original_reduction_seconds=original['seconds'])
    (output/'reference_reuse.json').write_text(json.dumps(record,indent=2))


def run_case(config, output, rays, planet, resolution, reuse=None):
    """Calculate a surface once, then prepare its identical observation under two models."""
    started = time.monotonic()
    if not output.exists():
        if reuse is not None:
            reuse_quiet(config,reuse,output,rays,planet,resolution)
        else:
            output.mkdir()
            config.to_json(output/'resolved.json')
            build_atomic_observation(config,output,planetary_rays=rays)
        (output/'forward_runtime.json').write_text(json.dumps(dict(seconds=time.monotonic()-started)))
    else:
        saved = SimulationConfig.from_json(output/'resolved.json')
        if saved.to_dict()!=config.to_dict():
            raise ValueError('Existing case has different configuration')
    data = load_arrays(output/'observations.npz')
    spectral = load_arrays(output/'intrinsic_spectra.npz')
    velocities = SimpleNamespace(stellar_velocity_kms=spectral['stellar_velocity_kms'],planet_velocity_kms=spectral['planet_velocity_kms'])
    path = output/'corrected_observations.npz'
    if path.exists():
        corrections = load_arrays(path)
    else:
        corrections = fit_atomic_tellurics(config,output,data,planet,resolution)
    if str(corrections['input_sha256']) != telluric_input_hash(config,data):
        raise ValueError('Saved telluric corrections do not match this observation')
    forward = json.loads((output/'forward_runtime.json').read_text())['seconds']
    identity = array_hash({k:data[k] for k in ('raw_flux','raw_error','wave_nm','bjd')})
    for assumed in ('matched','quiet') if config.stellar_surface else ('matched',):
        destination = output/assumed
        if (destination/'final_summary.json').exists():
            continue
        destination.mkdir(exist_ok=True)
        analysis = SimulationConfig.from_dict(config.to_dict())
        if analysis.stellar_surface:
            analysis.stellar_surface.correction_surface = assumed
        analysis.paths.output_root = str(destination)
        analysis.to_json(destination/'resolved.json')
        observation = dict(data)
        if assumed=='matched':
            observation['bare_model'] = data.get('matched_bare_model',data['bare_model'])
        # Both calls use the same flux, uncertainty, reference exposures and telluric fit.
        reduce_atomic_observation(analysis,destination,observation,velocities,planet,resolution,
            time.monotonic(),forward,corrections=corrections)
        (destination/'paired_observation.json').write_text(json.dumps(dict(
            observation_sha256=identity,correction=assumed,planetary_rays_sha256=file_hash(output.parent/'planetary_rays.npz')),indent=2))
    print('Completed surface:',output.name,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/hd189733b_andes_stellar_activity.json'))
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--prt-input-data')
    parser.add_argument('--reuse-quiet',type=Path)
    args=parser.parse_args()
    config=SimulationConfig.from_json(args.config)
    if args.prt_input_data:
        config.paths.prt_input_data=args.prt_input_data
    make_components(config)
    if args.prepare or not args.run:
        return
    validate_atomic_run(config)
    root=Path(config.paths.output_root)
    root.mkdir(parents=True,exist_ok=True)
    campaign_file=root/'campaign.json'
    if campaign_file.exists():
        if SimulationConfig.from_json(campaign_file).to_dict()!=config.to_dict():
            raise ValueError('The campaign directory belongs to a different configuration')
    else:
        config.to_json(campaign_file)
    start_file=root/'campaign_started.json'
    if not start_file.exists():
        start_file.write_text(json.dumps(dict(unix_seconds=campaign_file.stat().st_mtime)))
    original_start=json.loads(start_file.read_text())['unix_seconds']
    planet=load_planet(config.planet.parameter_file)
    resolution=load_instrument(config).res
    start=time.monotonic()
    rays=shared_atmosphere(config,root,planet)
    fingerprint=array_hash(dict(wavelength_um=rays.wavelength_um,impact_parameter_cm=rays.impact_parameter_cm,transmission=rays.transmission))
    for name in CASES:
        current=case_config(config,name)
        if fixed_settings(current)!=fixed_settings(config):
            raise AssertionError('A comparison changed non-stellar scientific settings')
        print('Starting surface:',name,flush=True)
        run_case(current,Path(current.paths.output_root),rays,planet,resolution,args.reuse_quiet if name=='quiet' else None)
        if array_hash(dict(wavelength_um=rays.wavelength_um,impact_parameter_cm=rays.impact_parameter_cm,transmission=rays.transmission))!=fingerprint:
            raise AssertionError('The shared planetary atmosphere changed')
    (root/'campaign_runtime.json').write_text(json.dumps(dict(seconds=time.time()-original_start,
        current_session_seconds=time.monotonic()-start,
        planetary_array_sha256=fingerprint,cases=list(CASES)),indent=2))


if __name__=='__main__':
    main()
