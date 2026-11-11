"""Compare chemistry profiles and transit spectra from one simulation JSON."""
import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exoplore.config import SimulationConfig
from exoplore.analysis.chemistry import compare_profiles
from exoplore.plotting.chemistry import plot_chemistry_profiles, plot_chemistry_transmission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--region',default='morning_day',choices=(
        'planet_model','ccf_template','morning_day','morning_night','evening_day','evening_night'))
    parser.add_argument('--output',type=Path,default=Path('outputs/tutorial15'))
    parser.add_argument('--spectra', action='store_true',
                        help='Calculate transit spectra from the completed abundance comparison')
    parser.add_argument('--prt-input-data', type=Path,
                        help='Override paths.prt_input_data for the opacity tables')
    parser.add_argument('--resolving-power', type=float, default=3500.)
    parser.add_argument('--wavelength-min-um', type=float, default=.6)
    parser.add_argument('--wavelength-max-um', type=float, default=5.)
    args=parser.parse_args()
    if args.spectra:
        from exoplore.analysis.chemistry_spectra import compare_transmission_spectra
        arrays, report = compare_transmission_spectra(
            SimulationConfig.from_json(args.config), args.region, args.output,
            input_data=args.prt_input_data, low=args.wavelength_min_um,
            high=args.wavelength_max_um, resolving_power=args.resolving_power)
        fig = plot_chemistry_transmission(arrays, report['planet'], report['resolving_power'])
        fig.savefig(args.output/'transmission_spectra.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
        print('Transit spectra and timings:', args.output/'transmission_spectra.json')
        return
    arrays,report=compare_profiles(SimulationConfig.from_json(args.config),args.region,args.output)
    fig=plot_chemistry_profiles(arrays,report['planet'])
    fig.savefig(args.output/'abundance_profiles.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Profiles and timings:',args.output/'comparison.json')


if __name__=='__main__':
    main()
