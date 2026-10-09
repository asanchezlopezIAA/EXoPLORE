"""EXoPLORE helium simulation inputs for the observed-spectrum molecfit route.

Generated files contain one synthetic extracted order, not raw detector data.
No nod subtraction, airglow emission, or wavelength error is fabricated.
"""
from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import Event

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils import iers

from exoplore.instruments.crires_czesla2024 import (
    HeliumMolecfitConfig, prepare_crires_czesla2024,
)
from exoplore.pipelines.czesla2024 import exposure_metadata, read_corrected_night


def utc_midpoints(bjd_tdb: np.ndarray, science) -> Time:
    """Invert barycentric arrival times to UTC at the supplied observatory."""
    location = EarthLocation.from_geodetic(*science.observatory_lon_lat_height[:2],
                                         height=science.observatory_lon_lat_height[2]*u.m)
    target = SkyCoord(science.target_ra_deg*u.deg, science.target_dec_deg*u.deg)
    arrival = np.asarray(bjd_tdb, float)
    time = Time(arrival, format='jd', scale='tdb', location=location)
    with iers.conf.set_temp('auto_download', False), iers.conf.set_temp('auto_max_age', None):
        for _ in range(4):
            time = Time(arrival-time.light_travel_time(target).to_value(u.day),
                        format='jd', scale='tdb', location=location)
    return time.utc


def fit_synthetic_night(output: Path, arrays: dict, raw: np.ndarray,
                        error: np.ndarray, settings, science, duration: float) -> dict:
    """Save the entire synthetic matrix, then fit every exposure with molecfit.

    ABBA is an observing schedule label here. Synthetic one-dimensional fluxes
    already represent extracted spectra; this function does not simulate the
    detector-level extraction or subtract an invented sky image.
    """
    directory = output/'synthetic_spectra'
    directory.mkdir(exist_ok=False)
    arrival = science.t0_bjd_tdb + np.asarray(settings.phase_midpoints)*science.period_days
    times = utc_midpoints(arrival, science)
    extension, order = science.order_segment.rsplit('_', 1)
    manifest = directory/'timeseries_manifest.txt'
    with manifest.open('x') as stream:
        for index, time in enumerate(times):
            nod = 'ABBA'[index % 4]
            header = fits.Header()
            header['SYNTHET'] = True
            header['ORIGIN'] = 'EXOPLORE_SIMULATION'
            header['OBJECT'] = 'HD209458b synthetic helium transit'
            header['MJD-OBS'] = float(time.mjd)-duration/2/86400
            header['DATE-OBS'] = Time(header['MJD-OBS'], format='mjd', scale='utc').isot
            header[science.exposure_time_keyword] = duration
            header['EXPTIME'] = duration
            header['UTC'] = float((header['MJD-OBS'] % 1)*86400)
            header['HIERARCH ESO SEQ NODPOS'] = nod
            airmass = settings.airmass[index] if settings.airmass is not None else 1.
            header['AIRMASS'] = airmass
            header['HIERARCH ESO TEL ALT'] = float(np.rad2deg(np.arcsin(1/airmass)))
            # Adopted site conditions, not measurements of this hypothetical night.
            header['HIERARCH ESO TEL AMBI RHUM'] = 15.
            header['HIERARCH ESO TEL AMBI PRES START'] = 750.
            header['HIERARCH ESO TEL AMBI TEMP'] = 15.
            header['HIERARCH ESO TEL TH M1 TEMP'] = 15.
            header['HIERARCH ESO TEL GEOLON'] = science.observatory_lon_lat_height[0]
            header['HIERARCH ESO TEL GEOLAT'] = science.observatory_lon_lat_height[1]
            header['HIERARCH ESO TEL GEOELEV'] = float(science.observatory_lon_lat_height[2])
            header['HIERARCH ESO INS SLIT1 WID'] = .2
            header['WFRAME'] = 'TOPOCENTRIC VACUUM'
            header['SPECRES'] = settings.instrumental_resolving_power
            table = fits.BinTableHDU.from_columns([
                fits.Column(name=f'{order}_01_WL', format='D', unit='nm', array=arrays['wave_nm'][index]),
                fits.Column(name=f'{order}_01_SPEC', format='D', array=raw[index]),
                fits.Column(name=f'{order}_01_ERR', format='D', array=error[index]),
            ], name=extension)
            path = directory/f'simulation_{index+1:04d}_extracted{nod}.fits'
            fits.HDUList([fits.PrimaryHDU(header=header), table]).writeto(path, overwrite=False)
            bjd, berv, _ = exposure_metadata(path, science)
            if abs(bjd-arrival[index])*86400 > .001 or abs(berv-settings.berv_kms[index]) > 1e-5:
                raise ValueError('Synthetic FITS timing/BERV differs from injected spectrum')
            stream.write(f'{header["MJD-OBS"]:.12f} {path.resolve()}\n')
    fit_settings = HeliumMolecfitConfig(**json.loads(Path(settings.molecfit_config_path).read_text()))
    corrected = output/'molecfit'
    if settings.molecfit_workers == 1 and not settings.molecfit_reuse_path:
        prepare_crires_czesla2024(manifest, directory, science, fit_settings, corrected, synthetic_input=True)
    else:
        corrected.mkdir(exist_ok=False)
        jobs = corrected/'independent_jobs'
        jobs.mkdir()
        reuse = Path(settings.molecfit_reuse_path) if settings.molecfit_reuse_path else None
        if reuse is not None:
            previous = json.loads((reuse/'configuration.json').read_text())
            from dataclasses import asdict
            if previous['science'] != asdict(science) or previous['molecfit'] != asdict(fit_settings):
                raise ValueError('Completed fits may only be reused with identical scientific settings')
        rows = manifest.read_text().splitlines()
        stop = Event()

        def fit_one(index: int) -> dict:
            """Fit one exposure with the unchanged ESO wrapper in its own directory."""
            if stop.is_set():
                raise RuntimeError('Another exposure failed; pending fits cancelled')
            number, nod = index+1, 'ABBA'[index % 4]
            prior = reuse/f'exposure_{number:02d}_{nod}' if reuse else None
            try:
                if prior and (prior/'report.json').exists() and json.loads((prior/'report.json').read_text())['accepted']:
                    source = prior
                    report = json.loads((prior/'report.json').read_text())
                    spectrum = Path(rows[index].split(maxsplit=1)[1])
                    if hashlib.sha256(spectrum.read_bytes()).hexdigest() != report['extracted_sha256']:
                        raise ValueError('Reusable fit does not belong to the identical generated spectrum')
                    report['reused_fit_report'] = str((prior/'report.json').resolve())
                else:
                    single_manifest = jobs/f'exposure_{number:04d}_manifest.txt'
                    with single_manifest.open('x') as stream:
                        stream.write(rows[index]+'\n')
                    job = jobs/f'exposure_{number:04d}'
                    prepare_crires_czesla2024(single_manifest, directory, science, fit_settings,
                                             job, synthetic_input=True)
                    source = job/f'exposure_01_{nod}'
                    report = json.loads((source/'report.json').read_text())
                    report['independent_job_report'] = str((source/'report.json').resolve())
                    report['exposure'] = number
                payload = (source/'correction.npz').read_bytes()
                if hashlib.sha256(payload).hexdigest() != report['correction_sha256']:
                    raise ValueError('Molecfit correction hash differs from its accepted report')
                target = corrected/f'exposure_{number:02d}_{nod}'
                target.mkdir()
                with (target/'correction.npz').open('xb') as stream:
                    stream.write(payload)
                with (target/'report.json').open('x') as stream:
                    json.dump(report, stream, indent=2)
                print(f"  Full-night molecfit: exposure {number}/{len(rows)}, shift {report['he_shift_kms']:+.3f} km/s", flush=True)
                return report
            except Exception:
                stop.set()
                raise

        with ThreadPoolExecutor(max_workers=settings.molecfit_workers) as executor:
            reports = list(executor.map(fit_one, range(len(rows))))
        with (corrected/'night_report.json').open('x') as stream:
            json.dump(reports, stream, indent=2)
    return read_corrected_night(replace(science, input_path=str(corrected)))
