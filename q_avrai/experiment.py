# Runs the standard-vs-opponent experiment of the 2020 paper: half of the nations
# (one side of a bicolouring of the Rochester coupling map) are static opponents,
# alternating between the two halves from one run to the next.
#
#   python experiment.py --civs 7 9 11 --runs 10 --war motte

import argparse
import os

from game import run, L
from layout import get_points

parser = argparse.ArgumentParser()
parser.add_argument('--civs', type=int, nargs='+', default=[7, 9, 11])
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--years', type=int, default=20)
parser.add_argument('--war', choices=['cz', 'motte'], default='motte')
parser.add_argument('--rules', choices=['feb2020', 'final', 'final_gain'], default='final')
parser.add_argument('--war_fraction', type=float, default=1, help='fraction for set_relationship (motte only)')
parser.add_argument('--coupling', choices=['all', 'rochester'], default='all')
parser.add_argument('--aer_method', default=None, help="e.g. 'matrix_product_state' for many nations")
parser.add_argument('--skirmish', choices=['const', 'scaled'], default=None, help='skirmishes (an extension); recommended: scaled')
parser.add_argument('--skirmish_fraction', type=float, default=0.5)
parser.add_argument('--spare_opponents', action='store_true', help='skip skirmishes involving a frozen opponent (recommended)')
parser.add_argument('--opponent_type', default='static', help="'static', or 'none' for a standard run")
parser.add_argument('--first_seed', type=int, default=0)
parser.add_argument('--out', default='runs')
args = parser.parse_args()

_, _, half = get_points(L)
opponent_type = None if args.opponent_type=='none' else args.opponent_type

for num_civs in args.civs:
    h = 0
    for r in range(args.runs):
        seed = args.first_seed + r
        if opponent_type:
            h = (h+1)%2
            opponent = half[h]
        else:
            opponent = []
        war = args.war if (args.war=='cz' or args.war_fraction==1) else f'motte{args.war_fraction:g}'
        tag = '' if args.coupling=='all' else f'_{args.coupling}'
        if args.skirmish:
            tag += f'_sk{args.skirmish}{args.skirmish_fraction:g}' + ('spare' if args.spare_opponents else '')
        folder = os.path.join(args.out, f'{num_civs}_{args.rules}_{war}{tag}_{opponent_type}_{seed}')
        if os.path.exists(os.path.join(folder, 'maps.npz')):
            print('skipping', folder)
            continue
        print(folder)
        run(num_civs, folder, years=args.years, opponent_type=opponent_type, opponent=opponent, war=args.war, rules=args.rules, war_fraction=args.war_fraction, coupling=args.coupling, aer_method=args.aer_method,
            skirmish=args.skirmish, skirmish_fraction=args.skirmish_fraction, skirmish_spare_opponents=args.spare_opponents, seed=seed)
