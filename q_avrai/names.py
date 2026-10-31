# Place names, ported from hacks/names.py on the CoG branch (2020). Names are spliced
# together from places and people related to IBM, to quantum computing, and to JWO.
# The 2020 version also removed English words using nltk; that dependency is dropped
# here, with a small list of exclusions instead.

import random

SEED_CITIES = [
    # places, people and things related to IBM
    ('York','town'), ('Rüschli','kon'), ('Zür',''), ('Ro','chester'), ('Johannes','burg'),
    ('Alma','den'), ('Böb','lingen'), ('Lon','don'), ('Burling','ton'), ('Ar','monk'),
    ('Dub','lin'), ('Wat',''), ('Holler','ith'), ('Denk',''), ('Blue',''),
    # places related to JWO
    ('','ly'), ('Sut','ton'), ('Kirk','stall'), ('Bas','el'),
    # places and people related to quantum
    ('Aros','a'), ('Copen','hagen'), ('Cam','bridge'), ('Ber','lin'), ('Ulm',''), ('Bern',''),
    ('Planck',''), ('Ein','stein'), ('Bohr',''), ('Shor',''), ('Heisen','berg'), ('Pauli',''),
    ('Comp','ton'), ('Sommer','feld'), ('Grover',''), ('Bell',''),
    # fantasy sounding placename elements
    ('','helm'), ('','rule'), ('','fell'),
]

EXCLUDE = {'bluely', 'bellhelm', 'bluehelm', 'londonly'}


def city_names(seed=0):
    firsts, lasts = [], []
    for first, last in SEED_CITIES:
        if first not in firsts+['']:
            firsts.append(first)
        if last not in lasts+['']:
            lasts.append(last)
    names = []
    for first in firsts:
        for last in lasts:
            if (first[-1]!=last[0] or last[0] in ['t','s']) \
            and (first+last not in ['sc','tk','ei']) \
            and ((first,last) not in SEED_CITIES) \
            and (first+last).lower() not in EXCLUDE:
                names.append(first+last)
    random.Random(seed).shuffle(names)
    return names
