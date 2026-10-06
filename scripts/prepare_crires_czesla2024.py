#!/usr/bin/env python3
"""Preview or execute target-order molecfit preparation for czesla2024.

All outputs must be new. Raw cr2res extraction is an upstream stage.
The optional pilot uses one manifest exposure and never changes the manifest.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exoplore.config import SimulationConfig
from exoplore.instruments.crires_czesla2024 import HeliumMolecfitConfig,prepare_crires_czesla2024


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('raw_directory',type=Path)
    parser.add_argument('molecfit_config',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--pilot-exposure',type=int)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    config=SimulationConfig.from_json(args.config)
    if config.pipeline.name!='czesla2024':parser.error('Select pipeline.name=czesla2024')
    settings=HeliumMolecfitConfig(**json.loads(args.molecfit_config.read_text()))
    rows=sorted(args.manifest.read_text().splitlines(),key=lambda line:float(line.split()[0]))
    if args.pilot_exposure is not None:
        if not 1<=args.pilot_exposure<=len(rows):parser.error('Pilot exposure outside manifest')
        rows=[rows[args.pilot_exposure-1]]
    print(f'czesla2024: {len(rows)} exposures, segment {config.pipeline.czesla2024.order_segment}')
    print(f'New correction directory: {args.output}')
    print('Cost: typically minutes per exposure; run a single-exposure pilot first.')
    if not args.run:
        print('Preview only. Pass --run to execute the ESO recipes.');return
    manifest=args.manifest
    if args.pilot_exposure is not None:
        # Exclusive sidecar preserves the original manifest and every prior pilot.
        manifest=args.output.with_name(args.output.name+'_pilot_manifest.txt')
        with manifest.open('x') as stream:stream.write(rows[0]+'\n')
    prepare_crires_czesla2024(manifest,args.raw_directory,config.pipeline.czesla2024,settings,args.output)


if __name__=='__main__':main()
