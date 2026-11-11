"""Check the native VULCAN initial state before time integration.

The adapter copies this file into the native working directory. Its command-line
entry point runs in the user's VULCAN environment using NumPy and the standard
library. The external checkout remains unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def check_initial_state(variable, atmosphere, species, composition, elemental_abundances,
                        non_gas_species=(), tolerance=1e-3):
    """Validate densities and elemental ratios for the configured gas column.

    Elemental abundances are number ratios to H nuclei. This check precedes
    chemistry and transport, so a finite but incorrect equilibrium result is
    rejected alongside non-finite values. Every network species contributes.
    """
    names = list(species)
    y = np.asarray(variable.y, float)
    p = np.asarray(atmosphere.pco, float)
    t = np.asarray(atmosphere.Tco, float)
    if y.shape != (len(p), len(names)) or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("VULCAN initial number densities must be finite, non-negative and match the grid")
    if t.shape != p.shape or not np.isfinite(t).all() or np.any(t <= 0):
        raise ValueError("VULCAN initial temperatures must be finite and positive")
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("VULCAN initial pressures must be finite and positive")
    gas = [i for i, name in enumerate(names) if name not in non_gas_species]
    total_gas = y[:, gas].sum(axis=1)
    if not gas or np.any(total_gas <= 0):
        raise ValueError("VULCAN initialisation contains an empty gas layer")
    records = {str(row['species']): row for row in np.atleast_1d(composition)}
    if any(name not in records for name in names):
        raise ValueError("The native composition table lacks an initial species")
    hydrogen = y @ np.array([records[name]['H'] for name in names], float)
    if np.any(hydrogen <= 0):
        raise ValueError("VULCAN initialisation has no hydrogen nuclei in a layer")
    errors = {}
    for element, expected in elemental_abundances.items():
        if element not in composition.dtype.names:
            raise ValueError(f"Native composition table lacks element {element}")
        actual = (y @ np.array([records[name][element] for name in names], float)) / hydrogen
        relative = np.abs(actual / expected - 1)
        errors[element] = float(relative.max())
        if errors[element] > tolerance:
            layer = int(np.argmax(relative))
            raise ValueError(
                f"VULCAN initial {element}/H={actual[layer]:.8g} at layer {layer} "
                f"differs from configured {expected:.8g} by {relative[layer]:.3g} "
                f"(allowed relative error {tolerance:g}). Check initialisation, "
                "network coverage and the native FastChem diagnostics.")
    masses = np.array([records[names[i]]['mass'] for i in gas], float)
    gas_mu = (y[:, gas] @ masses) / total_gas
    return {'layers': len(p), 'status': 'valid', 'max_relative_element_errors': errors,
            'gas_mmw_range': [float(gas_mu.min()), float(gas_mu.max())]}


def run_checked_native():
    """Run the native solver with its initial state checked before integration."""
    import importlib
    import runpy
    import subprocess
    import sys

    settings = json.loads(Path('exoplore_initialisation_inputs.json').read_text())
    report_path = Path('initialisation_checks.json')
    # Generate this column's network with the selected interpreter. A failed
    # generator stops the run before an older chem_funs.py can be imported.
    # Argument lists also preserve executable paths containing spaces.
    if Path('make_chem_funs.py').is_file():
        subprocess.run([sys.executable, 'make_chem_funs.py'], check=True)
    # Follow the native import order after network generation. The wrapper
    # adds validation to the initialiser and returns its arrays unchanged.
    if Path('store.py').is_file():
        importlib.import_module('store')
    native = importlib.import_module('build_atm')
    original = native.InitialAbun.ini_y
    checked = False

    def checked_initialisation(self, variable, atmosphere, *args, **kwargs):
        nonlocal checked
        result = original(self, variable, atmosphere, *args, **kwargs)
        try:
            report = check_initial_state(
                result, atmosphere, native.species, native.compo,
                settings['elemental_abundances'], settings['non_gas_species'],
                settings['tolerance'])
        except ValueError as error:
            report_path.write_text(json.dumps({'status': 'rejected', 'error': str(error)}, indent=2))
            raise
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        checked = True
        return result

    native.InitialAbun.ini_y = checked_initialisation
    try:
        # The native -n option uses the network generated and checked above.
        sys.argv = ['vulcan.py', '-n']
        runpy.run_path('vulcan.py', run_name='__main__')
        if not checked:
            raise ValueError("The native VULCAN initialisation check was not executed")
    finally:
        native.InitialAbun.ini_y = original


if __name__ == '__main__':
    run_checked_native()
