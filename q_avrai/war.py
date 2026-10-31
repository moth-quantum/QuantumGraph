# War behaviour: after a nation loses a city, does it turn on the nation that took it?
# Extends the `war` count in the 2020 analysis notebook.
#
# An event is a city transfer in year y from a standard (non-opponent) loser to a winner.
# The war step is applied at the start of year y, before the moves of that year, so:
#   response: the loser's move in year y targets the winner (defence or attack against it)
#   baseline: the loser's move in year y-1 targeted the winner
# The same is reported for the winner targeting the loser. 'first' restricts to transfers
# in which neither nation had been involved in an earlier transfer, as in 2020.
#
#   python war.py --sources 2020 feb2020_cz feb2020_motte0 ... feb2020_motte

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis import load_runs, GRID, MUTED, INK

DEFAULT = ['2020', 'final_cz', 'final_motte']


def events(runs):
    out = []
    for run in runs:
        involved = set()
        for year in range(1, len(run)):
            dump = run[year]
            for loser, winner in dump['transfers'].items():
                first = loser not in involved and winner not in involved
                if loser not in dump['opponent']:
                    target = lambda civ, y: run[y]['moves'][civ][1]
                    out.append({
                        'response': target(loser, year)==winner,
                        'baseline': target(loser, year-1)==winner,
                        'winner_response': target(winner, year)==loser,
                        'winner_baseline': target(winner, year-1)==loser,
                        'first': first,
                    })
            for loser, winner in dump['transfers'].items():
                involved |= {loser, winner}
    return out


def rate(evs, key):
    k, n = sum(e[key] for e in evs), len(evs)
    p = k/n if n else np.nan
    return p, np.sqrt(p*(1-p)/n) if n else np.nan, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--out', default='figures')
    parser.add_argument('--civs', type=int, nargs='+', default=[7, 9, 11])
    parser.add_argument('--sources', nargs='+', default=DEFAULT)
    parser.add_argument('--name', default='war_response')
    args = parser.parse_args()

    rows = []
    print(f'{"source":>20} {"events":>6} {"loser: response":>16} {"baseline":>9} {"winner: response":>17} {"baseline":>9} {"first-war resp.":>16}')
    for source in args.sources:
        evs = [e for n in args.civs for e in events(load_runs(source, n, args.runs))]
        r = {key: rate(evs, key) for key in ['response', 'baseline', 'winner_response', 'winner_baseline']}
        fr = rate([e for e in evs if e['first']], 'response')
        rows.append((source, r))
        print(f'{source:>20} {len(evs):>6} {r["response"][0]:>10.2f}±{r["response"][1]:.2f} {r["baseline"][0]:>9.2f}'
              f' {r["winner_response"][0]:>11.2f}±{r["winner_response"][1]:.2f} {r["winner_baseline"][0]:>9.2f}'
              f' {fr[0]:>10.2f} (n={fr[2]})')

    labels = {'2020': '2020 data', 'feb2020_cz': 'cz', 'feb2020_motte': '1', 'final_cz': 'cz', 'final_motte': 'set_relationship'}
    names = [labels.get(s, s.replace('feb2020_motte', '')) for s, _ in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for key, colour, dx, label in [('baseline', '#9ec5f4', -0.18, 'loser, year before'),
                                   ('response', '#1c5cab', 0.18, 'loser, year after')]:
        p = [r[key][0] for _, r in rows]
        e = [r[key][1] for _, r in rows]
        ax.bar(x+dx, p, width=0.34, color=colour, label=label, yerr=e, capsize=2.5,
               error_kw={'elinewidth': 1, 'ecolor': INK})
    ax.set_xticks(x, names)
    ax.set_xlabel('War step: 2020 data, 2020 cz method, or set_relationship with this fraction')
    ax.set_ylabel('Share of moves targeting\nthe nation that took the city')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.name+'.png')
    fig.savefig(path, dpi=200)
    print('saved', path)


if __name__ == '__main__':
    main()
