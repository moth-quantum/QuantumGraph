# Checks that the port still behaves like the 2020 code that produced the paper's data.
#
#   PYTHONPATH=.. python -m pytest test_reproduction.py

import glob
import os
import random

import numpy as np
import pytest

from game import Game
from layout import get_points
from replay import replay

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('num_civs', [7, 9, 11])
def test_map_rules_replay_2020_logs_exactly(num_civs):
    '''Fed the cities placed in a 2020 run, the map rules give the 2020 areas and transfers.'''
    path = sorted(glob.glob(os.path.join(HERE, 'data_2020', f'{num_civs}_simulator_static_*.txt')))[0]
    assert replay(path)==[]


def test_2020_rotation_matches_set_bloch_for_full_steps():
    '''The 2020 rotation rule differs from set_bloch only in its phases, which only matter for fraction<1.'''
    rng = np.random.default_rng(0)
    for _ in range(5):
        target = dict(zip('XYZ', rng.standard_normal(3)))
        states = []
        for single in ['2020', 'quantumgraph']:
            game = Game(2, single=single, seed=1)
            game.set_state({'X':1, 'Y':0, 'Z':0}, 0, update=True)
            game.set_state(target, 0, update=True)
            states.append(np.array([game.ai.get_bloch(0)[p] for p in 'XYZ']))
        assert np.allclose(states[0], states[1], atol=0.05)


def test_standard_nations_all_explore_in_year_0():
    '''From the balanced start, the 2020 quarter step towards exploration leaves every
    standard nation exploring in year 0, as in the 2020 data. With the phase convention
    of set_bloch this fails, which is what first broke the reproduction.'''
    _, _, half = get_points(256)
    random.seed(0)
    np.random.seed(0)
    game = Game(7, opponent=half[1], opponent_type='static', rules='final', war='motte', seed=0)
    game.area = {civ: 0 for civ in range(7)}
    game.border = [{} for _ in range(7)]
    game.update()
    tactics = [game.make_move(civ)[0] for civ in range(7) if civ not in half[1]]
    assert tactics and all(t=='Y' for t in tactics)
