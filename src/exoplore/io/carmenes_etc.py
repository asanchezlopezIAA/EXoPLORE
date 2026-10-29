"""Read full wavelength-dependent S/N curves from CARMENES ETC responses.

The public ETC embeds its numerical tables in Bokeh data sources. Wavelengths
are converted from Angstrom to nm. Bundles retain S/N per resolution element
and observing conditions for observation/carmenes_noise.py.


scripts/carmenes_etc_to_fits.py calls these functions to prepare a response
bundle. observation/carmenes_noise.py subsequently selects each order maximum
and converts its S/N to native-pixel uncertainties for the chosen experiment.
"""

import base64
import gzip
import json
import re
import hashlib
from pathlib import Path

import numpy as np


def extract_etc_arrays(response_html, channel):
    """Return the full VIS or NIR order arrays, preserving pixel-dependent S/N.

    VIS exports 61 orders of 4096 samples. NIR exports two 2040-pixel sections
    for each of 28 orders. The bundled EXoPLORE VIS grid covers the first 44
    ETC orders, so order selection is performed after reading these arrays.
    """
    if channel not in ('VIS', 'NIR'):
        raise ValueError('Choose ETC channel VIS or NIR')
    # Collect the numerical plot data embedded in the ETC HTML response.
    sources = []

    # Visit nested Bokeh JSON entries and retain every ColumnDataSource.
    def walk(value):
        if isinstance(value, dict):
            if value.get('name') == 'ColumnDataSource':
                # Retain this Bokeh source for its wavelength and S/N arrays.
                sources.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    # Decode either a JSON numerical list or a binary Bokeh array.
    def decode(value):
        array = value['array']
        if isinstance(array, list):
            return np.asarray(array, float)
        # Binary plot data use base64 encoding and can also carry gzip compression.
        payload = base64.b64decode(array['data'])
        if payload[:2] == b'\x1f\x8b':
            payload = gzip.decompress(payload)
        # Interpret the decoded bytes with the numerical dtype recorded by Bokeh.
        return np.frombuffer(payload, dtype=value['dtype'])

    # Read the application/json script blocks that contain the ETC plotting data.
    for block in re.findall(r'<script type="application/json"[^>]*>(.*?)</script>',
                            response_html, re.S):
        walk(json.loads(block))
    # Use the expected detector-section length to identify response curves.
    length = 4096 if channel == 'VIS' else 2040
    sections = []
    # Read each plotted wavelength/S/N pair.
    for source in sources:
        data = dict(source['attributes']['data']['entries'])
        # Decode the x wavelength and y S/N coordinates together.
        wave, snr = decode(data['x']), decode(data['y'])
        # Retain curves with the expected detector sampling and increasing wavelengths.
        if len(wave) == length:
            if wave.shape != snr.shape or np.any(np.diff(wave) <= 0):
                raise ValueError('ETC wavelength and S/N arrays have invalid shapes or ordering')
            # Convert Angstrom wavelengths to nm while preserving the supplied S/N samples.
            sections.append((wave/10, snr))
    # VIS has 61 full-order curves. NIR has two sections for each of 28 orders.
    expected = 61 if channel == 'VIS' else 56
    if len(sections) != expected:
        raise ValueError(f'Expected {expected} ETC {channel} curves, found {len(sections)}')
    # Return one wavelength/S/N row per VIS order.
    if channel == 'VIS':
        return tuple(np.asarray([row[j] for row in sections]) for j in (0, 1))
    # Join the two NIR detector sections for each physical order.
    return tuple(np.asarray([np.r_[sections[2*i][j], sections[2*i+1][j]]
                             for i in range(28)]) for j in (0, 1))


def write_etc_bundle(responses, channel, exposure_seconds, output_path):
    """Save downloaded HTML curves as a simulation input with their conditions.

    responses is a sequence of (airmass, HTML path) pairs. Enter the exposure
    duration used in those ETC requests. All responses must use the same target,
    exposure duration, seeing and extinction, varying only the airmass.
    """
    # Sort the downloaded HTML files by their explicitly supplied airmass.
    conditions = sorted((float(x), Path(p)) for x, p in responses)
    if not conditions or not np.isfinite(exposure_seconds) or exposure_seconds <= 0:
        raise ValueError('Supply ETC responses and a positive exposure duration')
    # Build the increasing airmass axis saved with the response curves.
    airmass = np.array([x for x, _ in conditions])
    if (not np.isfinite(airmass).all() or np.any(airmass < 1)
            or np.any(np.diff(airmass) <= 0)):
        raise ValueError('ETC response airmasses must be distinct and at least one')
    # Collect S/N curves, source-file hashes and one common wavelength grid.
    curves, sources, wave = [], [], None
    # Read the response for every supplied observing condition.
    for x, path in conditions:
        # Keep the exact HTML bytes for its provenance hash.
        payload = path.read_bytes()
        # Extract the full requested instrument channel.
        w, snr = extract_etc_arrays(payload.decode(), channel)
        # Require the responses to share identical wavelength sampling.
        if wave is not None and not np.array_equal(wave, w):
            raise ValueError('ETC responses use different wavelength grids')
        wave = w
        curves.append(snr)
        # Record which original response supplied each airmass model.
        sources.append(dict(airmass=x, file=path.name,
                            sha256=hashlib.sha256(payload).hexdigest()))
    # Exclusive creation protects prior scientific input files.
    with Path(output_path).open('xb') as stream:
        # Save the shared wavelength grid, stacked S/N curves, conditions and provenance.
        np.savez_compressed(stream, wave_nm=wave,
                            snr_per_resel=np.asarray(curves), airmass=airmass,
                            exposure_seconds=float(exposure_seconds), channel=channel,
                            source_records=json.dumps(sources))
