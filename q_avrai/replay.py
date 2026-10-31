# Replays a logged run through the map rules of game.py, using the logged cities, and
# reports any year in which the areas or city transfers differ from the log. Applied to
# the 2020 logs in data_2020/, this checks that the map rules are exactly those of 2020.
#
#   python replay.py data_2020/7_simulator_static_1582897636.txt

import ast
import sys

from game import Game, L, get_points


class _NoAI:
    coupling_map = []

    def update_tomography(self):
        pass


class Replay(Game):
    '''The map rules of Game, with no quantum process: moves come from a log.'''

    def __init__(self, num_civs):
        self.num_civs = num_civs
        self.opponent, self.opponent_type = [], None
        # these only affect the quantum updates, which are skipped here
        self.war, self.rules, self.war_fraction, self.single = 'cz', 'final', 1, '2020'
        self.points, _, _ = get_points(L)
        self.height = {(x, y): 1 for x in range(L) for y in range(L)}
        self.ai = _NoAI()
        self.owner = {(x, y): None for x in range(L) for y in range(L)}
        self.influence = {(x, y): {} for x in range(L) for y in range(L)}
        self.cities = [[] for _ in range(num_civs)]
        self.city_owners, self.ruins = {}, []
        for civ in range(num_civs):
            self.add_city(self.points[civ], civ)
        self.loss, self.gain = [0]*num_civs, [0]*num_civs
        self.transfers, self.area, self.border = {}, {}, []

    def set_state(self, *args, **kwargs):
        pass


def replay(path):
    '''Returns the years in which the replay differs from the log at `path`.'''
    with open(path) as f:
        log = [ast.literal_eval(line) for line in f if line.strip()]
    n = len(log[0]['area'])
    game = Replay(n)
    mismatched = []
    for year, dump in enumerate(log):
        game.transfers = {}
        game.area = {civ: 0 for civ in range(n)}
        game.border = [{} for _ in range(n)]
        game.update()
        if game.area!=dump['area'] or game.transfers!=dump['transfers']:
            mismatched.append(year)
        # as in Game.make_move, but with the city taken from the log
        for civ in range(n):
            city = dump['moves'][civ][2]
            surplus = game.get_surplus(civ)
            if surplus<1:
                old_city = game.choose_city(civ, 'remove')
                if old_city:
                    game.remove_city(old_city, civ)
                    surplus += 1
            if surplus>0 and city is not None:
                game.add_city(tuple(city), civ)
    return mismatched


if __name__ == '__main__':
    for path in sys.argv[1:]:
        bad = replay(path)
        print(path, 'matches' if not bad else f'differs in years {bad}')
