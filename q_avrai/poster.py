# A poster version of a map: the map from render.py, with a parchment margin and a
# checkered border, a title cartouche, a compass rose, and names for the nations.
#
#   python poster.py showcase/53_final_motte_rochester_None_0 --year 30

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, Circle, Rectangle

from render import load, neighbours, assign_colours, draw, transpose, paper, INK, PARCHMENT
from names import city_names

TITLE_FONT = ['Luminari', 'Trattatello', 'Palatino', 'DejaVu Serif']
TEXT_FONT = ['Hoefler Text', 'Iowan Old Style', 'Palatino', 'Georgia', 'DejaVu Serif']
PAPER_LIGHT = '#f6efdc'
PT = 72/200  # points per pixel, at the 200 dpi used throughout


def compass(ax, x, y, r, scale):
    '''An eight-pointed compass rose centred at (x, y), in pixels, with y downwards.'''
    lw = scale*0.12
    ax.add_patch(Circle((x, y), r*0.62, fill=False, edgecolor=INK, linewidth=lw, zorder=5))
    ax.add_patch(Circle((x, y), r*0.55, fill=False, edgecolor=INK, linewidth=lw*0.6, zorder=5))
    for k in range(8):
        a = np.pi/2 - k*np.pi/4
        long = k%2==0
        tip = r if long else r*0.62
        half = r*(0.13 if long else 0.1)
        d = np.array([np.cos(a), -np.sin(a)])
        n = np.array([-d[1], d[0]])
        p_tip = np.array([x, y]) + tip*d
        for side, shade in [(1, INK), (-1, PAPER_LIGHT)]:
            ax.add_patch(Polygon([(x, y), p_tip, np.array([x, y]) + half*(d + side*n)*0.9],
                                 closed=True, facecolor=shade, edgecolor=INK, linewidth=lw*0.7,
                                 zorder=6 if long else 5.5))
    ax.text(x, y - r*1.12, 'N', ha='center', va='bottom', color=INK, fontsize=3.2*scale*PT,
            family=TEXT_FONT, zorder=6)


def neatline(ax, x0, y0, x1, y1, width, segment, scale):
    '''A checkered border band between two rectangles, as on old maps.'''
    lw = scale*0.12
    for inset in [0, width]:
        ax.add_patch(Rectangle((x0+inset, y0+inset), x1-x0-2*inset, y1-y0-2*inset, fill=False,
                               edgecolor=INK, linewidth=lw*(2 if inset==0 else 1), zorder=3))
    for (ax0, ay0, ax1, ay1, horizontal) in [(x0+width, y0, x1-width, y0+width, True),
                                             (x0+width, y1-width, x1-width, y1, True),
                                             (x0, y0+width, x0+width, y1-width, False),
                                             (x1-width, y0+width, x1, y1-width, False)]:
        length = (ax1-ax0) if horizontal else (ay1-ay0)
        n = max(1, int(round(length/segment)))
        step = length/n
        for k in range(0, n, 2):
            if horizontal:
                ax.add_patch(Rectangle((ax0+k*step, ay0+width*0.3), step, width*0.4, color=INK, zorder=3))
            else:
                ax.add_patch(Rectangle((ax0+width*0.3, ay0+k*step), width*0.4, step, color=INK, zorder=3))
        ax.add_patch(Rectangle((ax0, ay0+width*0.3) if horizontal else (ax0+width*0.3, ay0),
                               (length if horizontal else width*0.4), (width*0.4 if horizontal else length),
                               fill=False, edgecolor=INK, linewidth=lw*0.6, zorder=3))


def poster(folder, year=None, scale=10, title='Q_Avrai', subtitle=None, name_seed=0):
    owner, cities = load(folder)
    meta = cities['meta']
    year = len(owner)-1 if year is None else year-1
    colours = assign_colours(meta['num_civs'], neighbours(owner))
    grid = transpose(owner[year])
    state = cities['years'][year]

    fig = draw(grid, state, colours, scale=scale, frame=False, hatch=True)
    ax = fig.axes[0]
    h, w = grid.shape[0]*scale, grid.shape[1]*scale

    # nation names at the capitals, sized to fit the nation, placed largest first and
    # shrunk (or dropped) where they would collide with a name already placed
    names = city_names(name_seed)
    areas = np.bincount(grid[grid>=0].ravel(), minlength=meta['num_civs'])
    renderer = fig.canvas.get_renderer()
    placed = []
    for civ in np.argsort(-areas):
        civ_cities = state['cities'][civ]
        if not civ_cities or areas[civ]==0:
            continue
        x, y = (civ_cities[0][0]+0.5)*scale, (civ_cities[0][1]+0.5)*scale
        label = '\u2009'.join(names[civ].upper())
        px = min(2.6*scale, 1.1*np.sqrt(areas[civ])*scale/(0.75*len(label)))
        while px >= 0.9*scale:
            text = ax.text(x, y - 1.6*scale, label, ha='center', va='bottom', color=INK, fontsize=px*PT,
                           family=TEXT_FONT, zorder=7,
                           path_effects=[pe.withStroke(linewidth=0.35*scale*PT*2, foreground=PAPER_LIGHT)])
            box = text.get_window_extent(renderer).expanded(1.08, 1.25)
            if not any(box.overlaps(b) for b in placed):
                placed.append(box)
                break
            text.remove()
            px *= 0.85

    compass(ax, w - 24*scale, 24*scale, 13*scale, scale)

    # enlarge the figure to make room for a margin and a title band, with the map inset
    margin, band = 16*scale, 30*scale
    W, H = w + 2*margin, h + 2*margin + band
    fig.set_size_inches(W/200, H/200)
    ax.set_position((margin/W, (margin+band)/H, w/W, h/H))

    back = fig.add_axes((0, 0, 1, 1), zorder=-1)
    back.imshow(paper((H, W), seed=7), extent=(0, W, H, 0))
    back.set_xlim(0, W)
    back.set_ylim(H, 0)
    back.axis('off')

    neatline(back, margin*0.35, margin*0.35, W-margin*0.35, H-band-margin*0.35+margin*0.0, margin*0.5,
             8*scale, scale)
    back.add_patch(Rectangle((margin, margin), w, h, fill=False, edgecolor=INK, linewidth=scale*0.25, zorder=4))

    # title cartouche in the band below the map
    cy = H - band/2 - margin*0.15
    if subtitle is None:
        subtitle = (f'{meta["num_civs"]} nations after {year+1} years, '
                    'each governed by a qubit of a simulated quantum computer')
    t1 = back.text(W/2, cy - 2.2*scale, title, ha='center', va='center', color=INK, fontsize=11*scale*PT,
                   family=TITLE_FONT, zorder=5)
    t2 = back.text(W/2, cy + 6.2*scale, subtitle, ha='center', va='center', color=INK, fontsize=2.4*scale*PT,
                   family=TEXT_FONT, style='italic', zorder=5)
    renderer = fig.canvas.get_renderer()
    half = max(t.get_window_extent(renderer).width for t in [t1, t2])/2 + 6*scale
    for pad, lw, fill in [(0, 0.3, True), (1.5*scale, 0.1, False)]:
        back.add_patch(Rectangle((W/2 - half + pad, cy - 11*scale + pad), 2*(half - pad), 22*scale - 2*pad,
                                 facecolor=PAPER_LIGHT if fill else 'none', edgecolor=INK,
                                 linewidth=lw*scale*PT*2, zorder=4))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--year', type=int, help='year to draw, from 1 (default: last)')
    parser.add_argument('--scale', type=int, default=10, help='pixels per map cell')
    parser.add_argument('--title', default='Q_Avrai')
    parser.add_argument('--subtitle')
    parser.add_argument('--name_seed', type=int, default=0)
    parser.add_argument('--out', help='output file (default: poster.png in the run folder)')
    args = parser.parse_args()

    fig = poster(args.folder, args.year, args.scale, args.title, args.subtitle, args.name_seed)
    out = args.out or os.path.join(args.folder, 'poster.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
