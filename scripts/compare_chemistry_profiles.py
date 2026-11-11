"""Plot equilibrium and disequilibrium profiles from one simulation JSON."""
import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exoplore.config import SimulationConfig
from exoplore.analysis.chemistry import compare_profiles
from exoplore.plotting.chemistry import plot_chemistry_profiles


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--region',default='morning_day',choices=(
        'planet_model','ccf_template','morning_day','morning_night','evening_day','evening_night'))
    parser.add_argument('--output',type=Path,default=Path('outputs/tutorial15'))
    args=parser.parse_args()
    arrays,report=compare_profiles(SimulationConfig.from_json(args.config),args.region,args.output)
    fig=plot_chemistry_profiles(arrays,report['planet'])
    fig.savefig(args.output/'abundance_profiles.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Profiles and timings:',args.output/'comparison.json')


if __name__=='__main__':
    main()
