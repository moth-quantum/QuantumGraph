# Q_Avrai: geopolitical map and history generation driven by QuantumGraph.
#
# Ported from main.py on the CoG branch of qiskit-community/QuantumGraph, as used for
# "A quantum procedure for map generation" (J. R. Wootton, IEEE CoG 2020, arXiv:2005.10327).
# The game rules are kept exactly as in the 2020 code, including its quirks. Two versions
# of the 2020 rules are available through `rules`:
#   'feb2020': the code as of 28 Feb 2020 (commit 11b9aaa), which produced the emulated
#              7, 9 and 11 nation results in the paper.
#   'final':   the final code on the CoG branch (from 9 Mar 2020), used for the Rochester runs.
#   'final_gain': as 'final', but with the aggression fraction based on gained area.
# They differ in the initial states, the cz step for a city transfer, and in whether the
# fraction of the rotation towards aggression uses gained area ('feb2020') or, as a bug,
# lost area ('final').
#
# The only intended change from 2020 is the `war` option: 'cz' reproduces the 2020 method for a city transfer
# (rotate both nations to <X>=1, then apply a cz), while 'motte' instead uses
# set_relationship to target <XZ>=<ZX>=1 directly, as the paper wanted to do.
#
# Drawing has been removed from the game loop. Instead, the state of the map is saved
# each year so that it can be rendered afterwards (see render.py).

import json
import os
import random
from random import choice
from random import random as rnd

import numpy as np
from qiskit_aer import AerSimulator
from qiskit.synthesis import OneQubitEulerDecomposer
from scipy.linalg import fractional_matrix_power as pwr

from quantumgraph import QuantumGraph
from layout import get_points

L = 256
radius = 5


def get_influence(d):
    # return influence at distance `d` from a city
    return (d<2*radius)*( (d<radius) + 1/max(1,d)**2 )


class Game():

    def __init__(self, num_civs, opponent=None, opponent_type=None, war='cz', rules='final', war_fraction=1, single='2020', coupling='all', aer_method=None, skirmish=None, skirmish_fraction=0.5, skirmish_spare_opponents=False, seed=None):

        self.num_civs = num_civs
        self.opponent = opponent if opponent else []
        self.opponent_type = opponent_type
        self.war = war
        self.rules = rules
        self.war_fraction = war_fraction
        self.single = single
        # skirmishes (an extension, not in the 2020 game): a small step towards the war
        # state for every pair of nations between which territory changed hands this round
        #   None: off;  'const': fraction skirmish_fraction per pair;
        #   'scaled': skirmish_fraction * (area moved)/(pi r^2), capped at 1 (recommended,
        #   with skirmish_fraction=0.5 and skirmish_spare_opponents=True)
        # skirmish_spare_opponents skips skirmishes involving a frozen opponent on either
        # side, so that opponents stay static (a two-qubit step changes both qubits)
        self.skirmish = skirmish
        self.skirmish_fraction = skirmish_fraction
        self.skirmish_spare_opponents = skirmish_spare_opponents
        self.flow = {}

        self.points, _, _ = get_points(L)

        # the height map is flat, as in 2020
        self.height = {}
        for x in range(L):
            for y in range(L):
                self.height[x,y] = 1

        # all-to-all coupling, as for the 2020 emulated runs, or that of Rochester, as for
        # the 2020 hardware runs (pairs off the coupling map then get no tomography)
        options = {}
        if seed is not None:
            options['seed_simulator'] = seed
        if aer_method:
            options['method'] = aer_method
        backend = AerSimulator(**options) if options else None
        if coupling=='rochester':
            _, rochester, _ = get_points(L)
            coupling_map = [(a, b) for a, b in rochester if a<num_civs and b<num_civs]
        else:
            coupling_map = None
        self.ai = QuantumGraph(num_civs, coupling_map=coupling_map, backend=backend)  # 8192 shots, as in 2020
        for civ in range(num_civs):
            if rules=='feb2020':
                if civ in self.opponent:
                    self.set_state({'X':rnd(),'Y':0,'Z':rnd()}, civ, update=False)
                else:
                    self.set_state({'X':0,'Y':1,'Z':0}, civ, update=False) # initially set to explore
            else:
                self.set_state({'X':1,'Y':1,'Z':1}, civ, update=False)

        self.owner = {}
        self.influence = {}
        for x in range(L):
            for y in range(L):
                self.owner[x,y] = None
                self.influence[x,y] = {}

        self.cities = [[] for _ in range(num_civs)]
        self.city_owners = {}
        self.ruins = []
        for civ in range(num_civs):
            self.add_city(self.points[civ],civ)

        self.loss = [0 for _ in range(num_civs)]
        self.gain = [0 for _ in range(num_civs)]
        self.transfers = {}
        self.area = {}
        self.border = []

    def set_state(self, target_expect, qubit, fraction=1, update=False):
        '''
        Rotates a qubit towards a target state. With `single='2020'` this uses the
        `set_state` of the 2020 QuantumGraph, with its own choice of eigenvector phases.
        The phases matter for partial rotations (fraction<1): the 2020 choice and that of
        the current `set_bloch` give different paths towards the target.
        '''
        if self.single!='2020':
            self.ai.set_bloch(target_expect, qubit, fraction=fraction, update=update)
            return

        def normalize(expect):
            R = np.sqrt( expect['X']**2 + expect['Y']**2 + expect['Z']**2 )
            return {pauli:expect[pauli]/R for pauli in expect}

        def get_basis(expect):
            normalized_expect = normalize(expect)
            theta = np.arccos(normalized_expect['Z'])
            phi = np.arctan2(normalized_expect['Y'],normalized_expect['X'])
            state0 = [np.cos(theta/2),np.exp(1j*phi)*np.sin(theta/2)]
            state1 = [np.conj(state0[1]),-np.conj(state0[0])]
            return [state0,state1]

        current_basis = get_basis(self.ai.get_bloch(qubit))
        target_basis = get_basis(target_expect)
        U = np.zeros((2,2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[j][k] += target_basis[i][j]*np.conj(current_basis[i][k])
        if fraction!=1:
            U = pwr(U, fraction)
        the,phi,lam = OneQubitEulerDecomposer().angles(U)
        self.ai.qc.u(the,phi,lam,qubit)
        if update:
            self.ai.update_tomography()

    def add_city(self, city, civ, remove=False):
        # update the following when a `city` is added to `civ` (or removed for `remove=True`):
        # `cities`, `city_owners`, `ruins`, `influence`
        (x,y) = city
        if remove:
            self.cities[civ].remove(city)
            self.city_owners.pop(city)
            self.ruins.append(city)
        else:
            self.cities[civ].append(city)
            self.city_owners[city] = civ
        for dx in range(-2*radius,2*radius+1):
            for dy in range(-2*radius,2*radius+1):
                (xx,yy) = (x+dx,y+dy)
                d = np.sqrt(dx**2+dy**2)
                inf = get_influence(d)
                if d<2*radius:
                    if (xx,yy) in self.influence:
                        if civ in self.influence[xx,yy]:
                            if remove:
                                self.influence[xx,yy][civ] -= inf
                            else:
                                self.influence[xx,yy][civ] += inf
                        else:
                            self.influence[xx,yy][civ] = inf

    def remove_city(self, city, civ):
        # runs `add_city` for `remove=True`
        self.add_city(city, civ, remove=True)

    def coupled(self, civ0, civ1):
        return (min(civ0,civ1), max(civ0,civ1)) in self.ai.coupling_map

    def update(self):

        num_civs = self.num_civs
        owner, influence, city_owners, cities = self.owner, self.influence, self.city_owners, self.cities
        area, border, loss, gain, transfers = self.area, self.border, self.loss, self.gain, self.transfers

        for civ in range(num_civs):
            area[civ] = 0
            border[civ] = {}
            loss[civ] = 0
            gain[civ] = 0

        for x in range(L):
            for y in range(L):
                infs = {civ:influence[x,y][civ] for civ in influence[x,y]}
                if infs:
                    civ = max(infs, key=infs.get)
                    if owner[x,y]!=civ and owner[x,y]!=None:
                        loss[owner[x,y]] += 1
                        gain[civ] += 1
                        self.flow[owner[x,y], civ] = self.flow.get((owner[x,y], civ), 0) + 1
                    owner[x,y] = civ
                    if (x,y) in city_owners:
                        if city_owners[x,y]!=civ:
                            # city is protected from changing hands
                            if (x,y)==cities[city_owners[x,y]][0]:
                                owner[x,y] = city_owners[x,y]
                            else: # city changes hands
                                transfers[city_owners[x,y]] = civ
                                self.remove_city((x,y),city_owners[x,y])
                                self.add_city((x,y),civ)

        for civ in range(num_civs):
            area[civ] = 0
            border[civ] = {}

        for x in range(L):
            for y in range(L):
                civ = owner[x,y]
                if civ != None:
                    area[civ] += 1
                    for (dx,dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        if (x+dx, y+dy) in owner:
                            neighbour = owner[x+dx, y+dy]
                            if neighbour!=civ:
                                if neighbour in border[civ]:
                                    if (x,y) not in border[civ][neighbour]:
                                        border[civ][neighbour].append((x,y))
                                else:
                                    border[civ][neighbour] = [(x,y)]

        ai = self.ai
        for civ in range(num_civs):
            # not on static or random opponent civs
            if (civ not in self.opponent) or (self.opponent_type not in ['static', 'random']):
                if civ in transfers:
                    winner = transfers[civ]
                    if self.war=='motte' and self.coupled(civ, winner):
                        # target <XZ>=<ZX>=1 between loser and winner directly
                        ai.set_relationship({'XZ':1,'ZX':1}, civ, winner, fraction=self.war_fraction, update=False)
                    else:
                        # set losing civ to max defense
                        self.set_state({'X':1,'Y':0,'Z':0}, civ, update=False)
                        if self.coupled(civ, winner):
                            if self.rules=='feb2020':
                                # if possible, do a cz to swap it in for defense specifically against the winner
                                ai.qc.cz(winner,civ)
                            else:
                                # as much as possible, set up a <XZ>=<ZX>=1 state between loser and winner
                                self.set_state({'X':1,'Y':0,'Z':0}, winner, update=False)
                                ai.qc.cz(winner,civ)
                else:
                    if None in border[civ]:
                        frontier = len(border[civ][None])
                    else:
                        frontier = 0
                    if frontier>(loss[civ]+gain[civ]): # when frontiers are dominant, increase exploration
                        self.set_state({'X':0,'Y':1,'Z':0}, civ, fraction=1/4, update=False)
                    else:
                        if loss[civ]>gain[civ]:  # when losses are dominant, increase defense
                            self.set_state({'X':1,'Y':0,'Z':0}, civ, fraction=min(1, loss[civ]/(np.pi*radius**2)), update=False)
                        else: # when gains are dominant, increase aggression
                            # the final 2020 code uses `loss` rather than `gain` here
                            amount = gain[civ] if self.rules in ['feb2020', 'final_gain'] else loss[civ]
                            self.set_state({'X':0,'Y':0,'Z':1}, civ, fraction=min(1, amount/(np.pi*radius**2)), update=False)
        if self.skirmish:
            frozen = self.opponent if self.opponent_type in ['static', 'random'] else []
            for (loser, winner), area_moved in sorted(self.flow.items()):
                if loser in frozen:
                    continue  # frozen opponents get no updates, as for the war step
                if self.skirmish_spare_opponents and winner in frozen:
                    continue
                if transfers.get(loser)==winner or not self.coupled(loser, winner):
                    continue  # a full war step was taken, or the pair cannot be targeted
                fraction = self.skirmish_fraction
                if self.skirmish=='scaled':
                    fraction = min(1, self.skirmish_fraction*area_moved/(np.pi*radius**2))
                ai.set_relationship({'XZ':1,'ZX':1}, loser, winner, fraction=fraction, update=False)
        self.flow = {}

        ai.update_tomography()

    def get_surplus(self, civ):
        # for a `civ`, return the number of cities that can be added
        if len(self.cities[civ])==1:
            return 1
        else:
            return int( self.area[civ]/(np.pi*radius**2) - len(self.cities[civ]) )

    def choose_city(self, civ, tactic, neighbour=None):

        border, height, city_owners, ruins, influence = self.border, self.height, self.city_owners, self.ruins, self.influence

        if tactic in ['Y','X']: # defend ('X') or explore ('Y')

            minf = np.inf
            min_city = []
            if neighbour in border[civ]:
                for pos in border[civ][neighbour]:
                    if (height[pos]!=0) and (pos not in city_owners) and (pos not in ruins):
                        inf = influence[pos][civ]
                        if inf<minf:
                            minf = inf
                            min_city = [pos]
                        elif inf==minf:
                            min_city.append(pos)
                if min_city:
                    city = choice(min_city)
                else:
                    city = None
            else:
                city = None

        elif tactic=='Z': # attack ('Z')

            maxf = 0
            max_city = []
            if neighbour in border[civ]:
                for pos in border[civ][neighbour]:
                    if (height[pos]!=0) and (pos not in city_owners) and (pos not in ruins):
                        if neighbour in influence[pos]:
                            inf = influence[pos][neighbour]
                            if inf>maxf and inf<=get_influence(radius/2):
                                maxf = inf
                                max_city = [pos]
                            elif inf==maxf:
                                max_city.append(pos)
            if max_city:
                city = choice(max_city)
            else:
                city = None

        elif tactic=='remove':

            maxf = 0
            city = None
            for pos in self.cities[civ][1:]:
                inf = influence[pos][civ]
                if inf>maxf:
                    maxf = inf
                    city = pos

        return city

    def get_tactic(self, civ):

        # We need to look at the single qubit pauli expectation values
        # for the civ itself, as well as all the two qubit ones with
        # its neighbours.
        # For this we calculate expect[neighbour][`pauli`], the sum
        # of the single qubit expectation value for each `pauli` with
        # the two qubit expectation values with the given neighbour
        # for which the support on civ is `pauli`.

        border = self.border

        if (civ in self.opponent) and (self.opponent=='random'):

            best_tactic = choice(['X','Y','Z'])
            target_neighbour = choice(list(border[civ].keys()))

        else:

            rho_civ = self.ai.get_bloch(civ)

            expect = {}
            for neighbour in border[civ]:
                if neighbour!=None:
                    rho_both = self.ai.get_relationship(civ,neighbour)
                    expect[neighbour] = {}
                    for pauli_civ in ['X','Y','Z']:
                        expect[neighbour][pauli_civ] = rho_civ[pauli_civ]
                        for pauli_ngbhr in ['X','Y','Z']:
                            expect[neighbour][pauli_civ] += rho_both[pauli_civ+pauli_ngbhr]

            # Of all these, the maximum is value is determined, as well as the
            # corresponding tactic and neighbour.
            best_tactic,target_neighbour = 'Y',None
            max_expect = -1
            for neighbour in expect:
                for tactic in expect[neighbour]:
                    if expect[neighbour][tactic]>max_expect:
                        max_expect = expect[neighbour][tactic]
                        best_tactic,target_neighbour = tactic, neighbour

        if best_tactic=='Y':
            target_neighbour = None

        return best_tactic,target_neighbour

    def make_move(self, civ):

        # determine a tactic and a target neighbour, based on civ state
        tactic,neighbour = self.get_tactic(civ)
        city = self.choose_city(civ,tactic,neighbour)

        # see if the civ has capacity for a new city
        surplus = self.get_surplus(civ)

        # if there is no capacity, remove a city
        if (surplus<1) and (tactic in ['X','Y','Z']):
            old_city = self.choose_city(civ,'remove')
            if old_city:
                self.remove_city(old_city,civ)
                surplus += 1

        # if there is now capacity...
        if surplus>0:
            # and the proposed city is valid, add it
            if (city!=None):
                self.add_city(city,civ)
            # otherwise, attempt to explore
            else:
                tactic = 'Y'
                neighbour = None
                city = self.choose_city(civ,tactic,neighbour)
                if city:
                    self.add_city(city,civ)

        return tactic,neighbour,city

    def snapshot(self):
        # the information needed to draw the map, as it was drawn in 2020:
        # territory after `update`, cities after the moves
        owner = np.full((L,L), -1, dtype=np.int16)
        for (x,y),civ in self.owner.items():
            if civ is not None:
                owner[x,y] = civ
        return {
            'owner': owner,
            'cities': [[list(c) for c in cs] for cs in self.cities],
            'ruins': [list(r) for r in self.ruins],
        }


def run(num_civs, folder, years=20, opponent_type=None, opponent=None, war='cz', rules='final', war_fraction=1, single='2020', coupling='all', aer_method=None, skirmish=None, skirmish_fraction=0.5, skirmish_spare_opponents=False, seed=None, verbose=True):
    '''
    Runs a game and saves it in `folder`: `data.txt` holds one line per year in the 2020
    format (moves, transfers, area, opponent), `data.json` holds the same as JSON, and
    `maps.npz` plus `cities.json` hold what is needed to draw each year.
    '''

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(folder, exist_ok=True)

    game = Game(num_civs, opponent=opponent, opponent_type=opponent_type, war=war, rules=rules, war_fraction=war_fraction, single=single, coupling=coupling, aer_method=aer_method, skirmish=skirmish, skirmish_fraction=skirmish_fraction, skirmish_spare_opponents=skirmish_spare_opponents, seed=seed)

    owners, cities, log = [], [], []
    for year in range(years):

        if verbose:
            print(year, end=' ', flush=True)

        game.transfers = {}

        game.area = {civ:0 for civ in range(num_civs)}
        game.border = [{} for _ in range(num_civs)]
        game.update()

        moves = {}
        for civ in range(num_civs):
            moves[civ] = game.make_move(civ)

        snap = game.snapshot()
        owners.append(snap['owner'])
        cities.append({'cities':snap['cities'], 'ruins':snap['ruins']})

        dump = {'moves':moves, 'transfers':dict(game.transfers), 'area':dict(game.area), 'opponent':game.opponent}
        log.append(dump)
        with open(os.path.join(folder,'data.txt'), 'a') as file:
            file.write(str(dump)+'\n')

    if verbose:
        print()

    np.savez_compressed(os.path.join(folder,'maps.npz'), owner=np.array(owners))
    meta = {'num_civs':num_civs, 'years':years, 'opponent_type':opponent_type, 'war':war, 'rules':rules, 'war_fraction':war_fraction, 'single':single, 'coupling':coupling, 'skirmish':skirmish, 'skirmish_fraction':skirmish_fraction, 'skirmish_spare_opponents':skirmish_spare_opponents, 'seed':seed,
            'L':L, 'radius':radius, 'points':game.points[:num_civs]}
    with open(os.path.join(folder,'cities.json'), 'w') as file:
        json.dump({'meta':meta, 'years':cities}, file)
    with open(os.path.join(folder,'data.json'), 'w') as file:
        json.dump([{k:(v if k!='moves' else {c:list(m) for c,m in v.items()}) for k,v in d.items()} for d in log], file)

    return game
