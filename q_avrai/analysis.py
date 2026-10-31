# Standard nations against static opponents: mean area per year, as in the
# results figure of the 2020 paper. Compares the 2020 data (data_2020/) with
# new runs using the 2020 cz method and the Motte (set_relationship) method.
#
#   python analysis.py --runs runs --out figures

import argparse
import ast
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

STANDARD = '#2a78d6'
OPPONENT = '#eb6834'
INK = '#1f1f1e'
MUTED = '#6b6a64'
GRID = '#e4e2dc'

TITLES = {
    '2020': '2020 data',
    'feb2020_cz': 'Port, 2020 method (cz)',
    'feb2020_motte': 'Port, set_relationship',
    'final_cz': 'Port, final 2020 code (cz)',
    'final_motte': 'Port, final 2020 code, set_relationship',
    'final_gain_cz': 'Port, final code with gain fix (cz)',
    'final_gain_motte': 'Port, final code with gain fix, set_relationship',
}


def load_runs(source, num_civs, runs_folder):
    '''Returns a list of runs, each a list of year dicts with 'area' and 'opponent'.'''
    if source=='2020':
        files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'data_2020', f'{num_civs}_simulator_static_*.txt')))
    else:
        files = sorted(glob.glob(os.path.join(runs_folder, f'{num_civs}_{source}_static_*', 'data.txt')))
    runs = []
    for file in files:
        with open(file) as f:
            runs.append([ast.literal_eval(line) for line in f if line.strip()])
    return runs


def areas(runs, years):
    '''Areas per year for standard and opponent nations, and the per-run final gap.'''
    standard = [[] for _ in range(years)]
    opponent = [[] for _ in range(years)]
    gaps = []
    for run in runs:
        for year, dump in enumerate(run[:years]):
            for civ, a in dump['area'].items():
                (opponent if civ in dump['opponent'] else standard)[year].append(a)
        last = run[years-1]
        s = [a for c, a in last['area'].items() if c not in last['opponent']]
        o = [a for c, a in last['area'].items() if c in last['opponent']]
        gaps.append(np.mean(s) - np.mean(o))
    return standard, opponent, gaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--out', default='figures')
    parser.add_argument('--civs', type=int, nargs='+', default=[7, 9, 11])
    parser.add_argument('--years', type=int, default=20)
    parser.add_argument('--sources', nargs='+', default=['2020', 'final_cz', 'final_motte'], choices=list(TITLES))
    parser.add_argument('--name', default='standard_vs_opponent')
    parser.add_argument('--sweep', type=float, nargs='*', help='plot the final gap against these war fractions instead')
    args = parser.parse_args()
    if args.sweep:
        return sweep(args)
    SOURCES = [(s, TITLES[s]) for s in args.sources]
    os.makedirs(args.out, exist_ok=True)

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})

    fig, axes = plt.subplots(len(args.civs), len(SOURCES), figsize=(4.2*len(SOURCES), 3.2*len(args.civs)),
                             sharex=True, sharey='row', squeeze=False)
    summary = []
    x = np.arange(args.years)
    for r, num_civs in enumerate(args.civs):
        for c, (source, title) in enumerate(SOURCES):
            ax = axes[r][c]
            runs = load_runs(source, num_civs, args.runs)
            ax.grid(axis='y', color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ['top', 'right']:
                ax.spines[side].set_visible(False)
            if r==0:
                ax.set_title(title, color=INK, fontsize=11)
            if c==0:
                ax.set_ylabel(f'{num_civs} nations\nmean area', color=INK)
            if r==len(args.civs)-1:
                ax.set_xlabel('Year')
            if not runs:
                ax.text(0.5, 0.5, 'no runs', transform=ax.transAxes, ha='center', color=MUTED)
                continue
            standard, opponent, gaps = areas(runs, args.years)
            # bars are +-std/2, so that their full length is the standard deviation, as in 2020
            for data, colour, label, dx in [(opponent, OPPONENT, 'opponent', -0.12), (standard, STANDARD, 'standard', 0.12)]:
                mean = np.array([np.mean(d) for d in data])
                err = np.array([np.std(d)/2 for d in data])
                ax.errorbar(x+dx, mean, yerr=err, fmt='o-', color=colour, markersize=4, linewidth=1.5,
                            elinewidth=1, capsize=2.5, markeredgecolor='white', markeredgewidth=0.6, label=label)
            ax.text(0.02, 0.97, f'{len(runs)} runs', transform=ax.transAxes, va='top', color=MUTED, fontsize=8.5)
            t = stats.ttest_1samp(gaps, 0) if len(gaps)>1 else None
            summary.append((num_civs, source, len(runs), np.mean(gaps), np.std(gaps, ddof=1)/np.sqrt(len(gaps)) if len(gaps)>1 else np.nan,
                            sum(g>0 for g in gaps), t.pvalue if t else np.nan))

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[::-1], labels[::-1], loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(args.out, args.name+'.png')
    fig.savefig(path, dpi=200)
    print('saved', path)

    print()
    print(f'Final-year gap (mean area of standard minus opponent nations), per run, year {args.years-1}')
    print(f'{"nations":>7} {"source":>13} {"runs":>4} {"mean gap":>9} {"sem":>7} {"runs won":>8} {"p (t-test)":>10}')
    for num_civs, source, n, mean, sem, wins, p in summary:
        print(f'{num_civs:>7} {source:>13} {n:>4} {mean:>9.1f} {sem:>7.1f} {wins:>5}/{n:<2} {p:>10.3f}')


def sweep(args):
    '''Final-year gap against the fraction used in set_relationship, with the cz method as a reference.'''
    fractions = sorted(args.sweep)
    shades = ['#86b6ef', '#3987e5', '#1c5cab']  # one blue ramp: 7, 9, 11 nations
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    print(f'{"nations":>7} {"fraction":>8} {"runs":>4} {"mean gap":>9} {"sem":>7}')
    for num_civs, shade in zip(args.civs, shades):
        means, sems = [], []
        for f in fractions:
            source = 'feb2020_motte' if f==1 else f'feb2020_motte{f:g}'
            _, _, gaps = areas(load_runs(source, num_civs, args.runs), args.years)
            means.append(np.mean(gaps))
            sems.append(np.std(gaps, ddof=1)/np.sqrt(len(gaps)))
            print(f'{num_civs:>7} {f:>8g} {len(gaps):>4} {means[-1]:>9.1f} {sems[-1]:>7.1f}')
        ax.errorbar(fractions, means, yerr=sems, fmt='o-', color=shade, linewidth=1.5, markersize=5,
                    capsize=3, markeredgecolor='white', markeredgewidth=0.6, label=f'{num_civs} nations')
        _, _, gaps = areas(load_runs('feb2020_cz', num_civs, args.runs), args.years)
        ax.axhline(np.mean(gaps), color=shade, linestyle=':', linewidth=1.2)
        print(f'{num_civs:>7} {"cz":>8} {len(gaps):>4} {np.mean(gaps):>9.1f} {np.std(gaps, ddof=1)/np.sqrt(len(gaps)):>7.1f}')
    ax.set_xlabel('Fraction of the set_relationship step (0: no war step)')
    ax.set_ylabel(f'Final gap in mean area\n(standard minus opponent)')
    ax.text(1.0, 1.02, 'dotted: 2020 cz method', transform=ax.transAxes, ha='right', color=MUTED, fontsize=8.5)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    path = os.path.join(args.out, 'war_fraction_sweep.png')
    fig.savefig(path, dpi=200)
    print('saved', path)


if __name__ == '__main__':
    main()
