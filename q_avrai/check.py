# Compares new runs with the 2020 data, and prints PASS or FAIL for each criterion.
#
#   python check.py                      # checks final_cz and final_motte in runs/
#   python check.py --sources final_cz
#
# Criteria, for each number of nations, with at least 10 runs:
#   1. Standard nations beat their opponents: the per-run final gap has p < 0.01.
#   2. Standard nations win at least 75% of runs.
#   3. The final mean areas of standard and of opponent nations are each within 15% of 2020.
#   4. At least 95% of standard nations explore in year 0 (2020: all of them).
# Criteria 1-3 fail for the feb2020 rules but not for the wrong (set_bloch) rotation rule,
# whose end-of-run numbers are close to 2020. Criterion 4 catches the rotation rule.
# All pass for 20 seeded runs of the port as it stands (31 Oct 2026).

import argparse

import numpy as np
from scipy import stats

from analysis import load_runs, areas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--sources', nargs='+', default=['final_cz', 'final_motte'])
    parser.add_argument('--civs', type=int, nargs='+', default=[7, 9, 11])
    parser.add_argument('--years', type=int, default=20)
    parser.add_argument('--tolerance', type=float, default=0.15)
    args = parser.parse_args()

    failures = 0
    for source in args.sources:
        for n in args.civs:
            runs = load_runs(source, n, args.runs)
            ref_s, ref_o, _ = areas(load_runs('2020', n, args.runs), args.years)
            if len(runs)<10:
                print(f'{source:>12} {n:>2} nations: FAIL, only {len(runs)} runs (need 10)')
                failures += 1
                continue
            s, o, gaps = areas(runs, args.years)
            p = stats.ttest_1samp(gaps, 0, alternative='greater').pvalue
            wins = np.mean(np.array(gaps)>0)
            rel_s = np.mean(s[-1])/np.mean(ref_s[-1]) - 1
            rel_o = np.mean(o[-1])/np.mean(ref_o[-1]) - 1
            year0 = [m[0]=='Y' for run in runs for c, m in run[0]['moves'].items() if c not in run[0]['opponent']]
            explore = np.mean(year0)
            checks = [('gap p<0.01', p<0.01, f'p={p:.1e}'),
                      ('year-0 explore>=95%', explore>=0.95, f'{explore:.0%}'),
                      ('wins>=75%', wins>=0.75, f'{wins:.0%}'),
                      ('standard area', abs(rel_s)<=args.tolerance, f'{rel_s:+.0%}'),
                      ('opponent area', abs(rel_o)<=args.tolerance, f'{rel_o:+.0%}')]
            ok = all(c[1] for c in checks)
            failures += not ok
            detail = ', '.join(f'{name} {value}{"" if good else " (FAIL)"}' for name, good, value in checks)
            print(f'{source:>12} {n:>2} nations, {len(runs)} runs: {"PASS" if ok else "FAIL"}  [{detail}]')
    print('ALL PASS' if not failures else f'{failures} FAILED')
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
