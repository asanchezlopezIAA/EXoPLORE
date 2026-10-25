#!/usr/bin/env python3
"""Preview a CARMENES helium night or fit its first exposure with Palle2020."""

# Role in the calculation
# Prepare observed CARMENES helium spectra. Preview the inputs and fit settings, or run
# the configured external telluric fits with --run.

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    """Run a pilot only when explicitly requested with --run."""
    from exoplore.config.models import SimulationConfig
    from exoplore.core.simulator import ExoploreSimulator
    from exoplore.pipelines.palle2020 import run_palle2020
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--pilot-label',default='pilot')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    config=SimulationConfig.from_json(args.config)
    if config.pipeline.name!='Palle2020' or not config.observation.use_real_data:
        parser.error('Use an observed Palle2020 configuration')
    print(ExoploreSimulator(config).summary())
    if args.run:
        print(run_palle2020(config,pilot=True,pilot_label=args.pilot_label))
    else:
        print('Add --run to fit the first exposure. This can take minutes; the output directory must be new.')


if __name__=='__main__':
    main()
