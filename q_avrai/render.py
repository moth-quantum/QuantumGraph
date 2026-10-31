# Draws the maps saved by game.run in the style of an old hand-drawn map:
# parchment, watercolour territories with darker edges, smoothed ink borders,
# capitals, towns and ruins.
#
#   python render.py runs/11_motte_static_0                 # final year
#   python render.py runs/11_motte_static_0 --years 4 9 14 19 --scale 8

import argparse
import json
import os

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

PARCHMENT = '#efe4c8'
INK = '#3b2f24'
RUIN = '#8a7d6b'

# muted watercolours, spread around the hue circle so that neighbours stay distinct
WASHES = ['#d6ad62', '#9fbb84', '#d48b73', '#86a9cc', '#b897c9', '#7fb5a8',
          '#d29aae', '#c4c27a', '#a3a0d6', '#c9a184', '#8fc1d4', '#e0b98f']


def lab(rgb):
    '''CIELAB coordinates for an sRGB colour, so that distances match perceived differences.'''
    c = np.where(rgb > 0.04045, ((rgb + 0.055)/1.055)**2.4, rgb/12.92)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ c
    xyz /= np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787*xyz + 16/116)
    return np.array([116*f[1] - 16, 500*(f[0] - f[1]), 200*(f[1] - f[2])])


def load(folder):
    owner = np.load(os.path.join(folder, 'maps.npz'))['owner']
    with open(os.path.join(folder, 'cities.json')) as f:
        cities = json.load(f)
    return owner, cities


def neighbours(owner):
    '''Pairs of nations that share a border in any year.'''
    pairs = set()
    for grid in owner:
        for a, b in [(grid[1:, :], grid[:-1, :]), (grid[:, 1:], grid[:, :-1])]:
            m = (a!=b) & (a>=0) & (b>=0)
            for p, q in zip(a[m], b[m]):
                pairs.add((min(p, q), max(p, q)))
    return pairs


def assign_colours(num_civs, pairs, seed=0):
    '''Greedy colouring: each nation gets the wash furthest from those of its neighbours.'''
    rng = np.random.default_rng(seed)
    adj = {c: set() for c in range(num_civs)}
    for p, q in pairs:
        adj[p].add(q)
        adj[q].add(p)
    rgb = [np.array(to_rgb(w)) for w in WASHES]
    lab_ = [lab(c) for c in rgb]
    colour = {}
    for civ in sorted(adj, key=lambda c: -len(adj[c])):
        used = [colour[n] for n in adj[civ] if n in colour]
        scores = [min([np.linalg.norm(lab_[k]-lab_[u]) for u in used], default=100) + 0.5*rng.random() for k in range(len(WASHES))]
        colour[civ] = int(np.argmax(scores))
    return {civ: rgb[k] for civ, k in colour.items()}


def smooth_owner(grid, scale, sigma):
    '''Upscales the ownership grid, smoothing the borders.'''
    best = np.zeros((grid.shape[0]*scale, grid.shape[1]*scale))
    out = np.full(best.shape, -1, dtype=np.int16)
    for civ in [-1] + sorted(set(np.unique(grid)) - {-1}):
        mask = ndimage.gaussian_filter((grid==civ).astype(float), sigma)
        mask = ndimage.zoom(mask, scale, order=1)
        better = mask > best
        best[better] = mask[better]
        out[better] = civ
    return out


def paper(shape, seed=0):
    '''Parchment texture: low-frequency mottling and a soft vignette.'''
    rng = np.random.default_rng(seed)
    h, w = shape
    mottle = sum(ndimage.zoom(ndimage.gaussian_filter(rng.standard_normal((h//s+1, w//s+1)), 1.5), s, order=1)[:h, :w] * a
                 for s, a in [(64, 0.05), (16, 0.025), (4, 0.012)])
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx-w/2)/(w/2), (yy-h/2)/(h/2))
    vignette = 1 - 0.22*np.clip(r-0.55, 0, None)**1.6
    base = np.array(to_rgb(PARCHMENT))
    return np.clip(base[None, None, :] * (1+mottle)[..., None] * vignette[..., None], 0, 1)


def draw(owner, cities, colours, scale=8, sigma=0.9, seed=0, title=None, crop=None, frame=True, hatch=False):
    hi = smooth_owner(owner, scale, sigma)
    img = paper(hi.shape, seed)

    # unclaimed land: faint diagonal hatching, as for terra incognita
    if hatch:
        yy, xx = np.mgrid[0:hi.shape[0], 0:hi.shape[1]]
        lines = ((xx + yy) % (2*scale)) < max(1, scale//5)
        m = (hi==-1) & lines
        img[m] = img[m]*0.86

    # territories: a translucent wash, deepening towards the edges
    border = np.zeros(hi.shape, dtype=bool)
    border[1:, :] |= hi[1:, :]!=hi[:-1, :]
    border[:-1, :] |= hi[1:, :]!=hi[:-1, :]
    border[:, 1:] |= hi[:, 1:]!=hi[:, :-1]
    border[:, :-1] |= hi[:, 1:]!=hi[:, :-1]
    dist = ndimage.distance_transform_edt(~border)
    edge = np.exp(-dist/(1.6*scale))
    for civ, rgb in colours.items():
        m = hi==civ
        if not m.any():
            continue
        alpha = (0.55 + 0.35*edge[m])[:, None]
        tint = rgb*(1 - 0.25*edge[m])[:, None]
        img[m] = img[m]*(1-alpha) + tint*alpha

    # ink borders: firmer along the frontier with unclaimed land
    ink = np.array(to_rgb(INK))
    line = ndimage.gaussian_filter(border.astype(float), 0.35*scale/4+0.4)
    line = np.clip(line*2.2, 0, 1)
    frontier = np.zeros(hi.shape, dtype=bool)
    unclaimed = hi==-1
    frontier |= border & ndimage.binary_dilation(unclaimed, iterations=max(1, scale//4))
    line = np.maximum(line*0.75, ndimage.gaussian_filter(frontier.astype(float), scale/6)*2.5)
    line = np.clip(line, 0, 0.9)[..., None]
    img = img*(1-line) + ink*line

    # crop to the land that has ever been claimed, with a margin
    if crop is None:
        crop = (0, owner.shape[0], 0, owner.shape[1])
    r0, r1, c0, c1 = [v*scale for v in crop]
    img = img[r0:r1, c0:c1]
    h, w = img.shape[:2]

    fig = plt.figure(figsize=(w/200, h/200), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, interpolation='lanczos', extent=(c0, c1, r1, r0))
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.axis('off')

    # a double-line frame
    if frame:
        for inset, lw in [(2.2*scale, scale*0.35), (3.4*scale, scale*0.12)]:
            ax.add_patch(plt.Rectangle((c0+inset, r0+inset), w-2*inset, h-2*inset, fill=False,
                                       edgecolor=INK, linewidth=lw, alpha=0.85))

    # cities are stored as (x, y) map coordinates, with x horizontal and y downwards
    # as in the 2020 pygame output
    def to_px(c):
        return (c[0]+0.5)*scale, (c[1]+0.5)*scale

    for ruin in cities['ruins']:
        x, y = to_px(ruin)
        ax.plot(x, y, marker='x', markersize=scale*0.55, color=RUIN, markeredgewidth=scale*0.14, alpha=0.8)
    for civ_cities in cities['cities']:
        for k, city in enumerate(civ_cities):
            x, y = to_px(city)
            if k==0:  # capital
                ax.plot(x, y, marker='*', markersize=scale*2.1, color='#f6efdc', markeredgecolor=INK, markeredgewidth=scale*0.16)
            else:
                ax.plot(x, y, marker='o', markersize=scale*0.85, color=INK, markeredgecolor='#f6efdc', markeredgewidth=scale*0.08)

    if title:
        ax.text(c1-5.5*scale, r1-5.5*scale, title, ha='right', va='bottom', color=INK, fontsize=scale*1.9,
                family='serif', style='italic', alpha=0.85)
    return fig


def extent(owner, margin=12):
    '''Rows and columns (of the transposed grids) covering all land ever claimed, plus a margin.'''
    claimed = (owner>=0).any(axis=0).T
    rows, cols = np.where(claimed)
    n = claimed.shape[0]
    return (max(0, rows.min()-margin), min(n, rows.max()+1+margin),
            max(0, cols.min()-margin), min(n, cols.max()+1+margin))


def transpose(owner):
    # owner grids are indexed [x, y]; images are [row=y, column=x]
    return owner.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--years', type=int, nargs='*', help='years to draw (default: last)')
    parser.add_argument('--scale', type=int, default=8, help='pixels per map cell')
    parser.add_argument('--out', help='output folder (default: the run folder)')
    parser.add_argument('--no_title', action='store_true')
    args = parser.parse_args()

    owner, cities = load(args.folder)
    years = args.years if args.years else [len(owner)-1]
    colours = assign_colours(cities['meta']['num_civs'], neighbours(owner))
    out = args.out or args.folder
    os.makedirs(out, exist_ok=True)
    for year in years:
        fig = draw(transpose(owner[year]), cities['years'][year], colours, scale=args.scale,
                   title=None if args.no_title else f'Year {year+1}', crop=extent(owner))
        path = os.path.join(out, f'map_year_{year+1:02d}.png')
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print('saved', path)


if __name__ == '__main__':
    main()
