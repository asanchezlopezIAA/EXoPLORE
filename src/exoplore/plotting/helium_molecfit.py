"""Common per-exposure telluric-fit diagnostics for direct He I spectroscopy.

Read accepted molecfit reports without refitting spectra. CRIRES+ nods and
CARMENES detector sections are distinguished explicitly. Kernel widths are
reported in the fitting grid's pixels, with fitted/fixed status from the recipe.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


def read_molecfit_diagnostics(directory: Path, instrument: str) -> list[dict]:
    """Read chronological reports, including a reused first-exposure pilot.

    ``directory`` may be the correction directory or its parent night directory.
    Only accepted reports contribute; absent or rejected reports raise an error.
    """
    if instrument not in ('CRIRES+', 'CARMENES_NIR'):
        raise ValueError('Helium molecfit diagnostics support CRIRES+ and CARMENES_NIR')
    directory = Path(directory)
    root = directory / 'molecfit' if (directory / 'molecfit').is_dir() else directory
    if (root / 'night_report.json').exists():
        reports = json.loads((root / 'night_report.json').read_text())
        entries = [(report['exposure'], report.get('nod', ''), report) for report in reports]
    else:
        entries = []
        for path in sorted(root.glob('exposure_*/**/report.json')):
            exposure = int(re.search(r'exposure_(\d+)', str(path.relative_to(root))).group(1))
            section = path.parent.name if path.parent.name.startswith('detector_') else ''
            entries.append((exposure, section, json.loads(path.read_text())))
        reused = directory / 'reused_first_exposure.json'
        if reused.exists() and not any(number == 1 for number, _, _ in entries):
            report_path = Path(json.loads(reused.read_text())['report'])
            entries.append((1, '', json.loads(report_path.read_text())))
    if not entries:
        raise ValueError(f'No molecfit exposure reports in {root}')
    rows = []
    for exposure, section, report in sorted(entries, key=lambda entry: (entry[0], entry[1])):
        if not report.get('accepted'):
            raise ValueError(f'Exposure {exposure} has no accepted molecfit correction')
        parameters = report['parameters']
        flags = dict(option[2:].split('=', 1) for option in report['command']
                     if option.startswith('--') and '=' in option)
        shift = report.get('wavelength_shift_kms', report.get('he_shift_kms'))
        if shift is None:
            raise ValueError(f'Exposure {exposure} lacks a wavelength-adjustment diagnostic')
        group = f"Nod {report['nod']}" if instrument == 'CRIRES+' else section.replace('_', ' ').capitalize() or 'Science fibre A'
        rows.append(dict(exposure=exposure, group=group, shift=float(shift),
                         gaussian=float(parameters['gaussfwhm']),
                         lorentzian=float(parameters.get('lorentzfwhm', 0)),
                         gaussian_fitted=flags.get('FIT_RES_GAUSS') == 'TRUE',
                         lorentzian_fitted=flags.get('FIT_RES_LORENTZ') == 'TRUE',
                         reduced_chi2=float(parameters['reduced_chi2'])))
    return rows


def plot_molecfit_diagnostics(directory: Path, output: Path, instrument: str) -> None:
    """Write the standard three-panel calibration plot to a new output file."""
    rows = read_molecfit_diagnostics(directory, instrument)
    with plt.rc_context({'font.size': 16, 'axes.labelsize': 18, 'legend.fontsize': 14}):
        figure, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
        for group in dict.fromkeys(row['group'] for row in rows):
            selected = [row for row in rows if row['group'] == group]
            x = [row['exposure'] for row in selected]
            line, = axes[0].plot(x, [row['shift'] for row in selected], 'o-', label=group)
            colour = line.get_color()
            for field, flag, style in (('gaussian', 'gaussian_fitted', 'o-'),
                                       ('lorentzian', 'lorentzian_fitted', 's--')):
                if field == 'lorentzian' and not any(row[field] > 0 for row in selected):
                    continue
                status = 'fitted' if all(row[flag] for row in selected) else 'fixed' if not any(row[flag] for row in selected) else 'fitted/fixed'
                axes[1].plot(x, [row[field] for row in selected], style, color=colour,
                             label=f'{group}: {field.capitalize()} ({status})')
            axes[2].plot(x, [row['reduced_chi2'] for row in selected], 'o-', color=colour)
        axes[0].axhline(0, color='.5', ls=':')
        axes[0].set_ylabel('Wavelength adjustment\n(km/s)')
        axes[1].set_ylabel('Kernel FWHM\n(fit-grid pixels)')
        axes[2].set_ylabel(r'$\chi_{\mathrm{R}}^2$')
        axes[2].set_xlabel('Chronological exposure number')
        axes[0].legend(); axes[1].legend()
        for axis in axes:
            axis.grid(alpha=.2)
            axis.tick_params(length=6, width=1.2)
        axes[-1].set_xticks(np.arange(1, max(row['exposure'] for row in rows) + 1, 5))
        figure.tight_layout()
        try:
            with Path(output).open('xb') as stream:
                figure.savefig(stream, format=Path(output).suffix.lstrip('.'), dpi=180,
                               bbox_inches='tight')
        finally:
            plt.close(figure)
