#!/bin/sh
# Regenerates the main results from scratch and checks them against the 2020 data.
# About 30-40 minutes on a laptop. Run from this folder.
set -e
export PYTHONPATH=..
python -m pytest -q test_reproduction.py
python experiment.py --civs 7 9 11 --runs 20 --war cz
python experiment.py --civs 7 9 11 --runs 20 --war motte
python analysis.py --sources 2020 final_cz final_motte
python war.py --sources 2020 final_cz final_motte
python check.py
