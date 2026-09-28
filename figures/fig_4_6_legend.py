"""
The shared key for Figure 4.5, the DoublePPI-against-PPI++ panel.

Two keys in one strip rather than eight parallel entries, as in Figure 2.2,
each laid out over two rows so that the labels can be set larger than the
panel's tick labels:
the tuned second-order weight under one heading and the classical mean under
another, so that each member is named once and the colours carry only the
correlation. The classical mean appears at two correlations only; at rho = 0
it is PPI++'s own limit and its curve would say nothing the zero line does
not already say.

Drawn once and placed above the panel.

Writes: output/pdf/legend_gain_ratio.pdf
Include with \\includegraphics[width=0.88\\textwidth]{...}.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style

NAME = "legend_gain_ratio"
WIDTH_FRACTION = 0.1948

#: Same mapping from correlation to colour, and the same dash, as the panel.
COLORS = list(style.sequential_colors(len(panels.GAIN_RHOS)))[::-1]
DOTTED = panels.GAIN_CLASSICAL_DASH


def _color(rho):
    return COLORS[panels.GAIN_RHOS.index(rho)]


def build():
    style.use_thesis_style()
    width = WIDTH_FRACTION * style.TEXTWIDTH_IN / style.RENDER_SCALE
    # Generous, because the two keys and their titles are laid out by the
    # renderer and a canvas that is too short clips the title rather than
    # growing; `bbox_inches="tight"` removes whatever is left over.
    fig = plt.figure(figsize=(width, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Three rows of two rather than two of three: the strip is included at
    # the width of the panel below it, so its ink has to be about as wide as
    # the panel's, or the type arrives on the page smaller than the panel's
    # own tick labels.
    size = style._LEGEND_SIZE * 1.08
    common = dict(loc="center left", frameon=False, handlelength=1.6,
                  columnspacing=1.4, labelspacing=0.7, fontsize=size,
                  title_fontproperties={"size": size})

    left = ax.legend(
        handles=[Line2D([], [], color=_color(r)) for r in panels.GAIN_RHOS],
        labels=[r"$\rho=%g$" % r for r in panels.GAIN_RHOS],
        title=r"$\mathrm{D}_{\!\!+\!}$", bbox_to_anchor=(0.0, 0.5),
        ncol=2, **common)
    ax.add_artist(left)

    # Place the second key just after the first rather than at a guessed
    # fraction: the width of six correlation entries depends on the font and
    # would have to be re-tuned by hand every time either changes.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x1 = left.get_window_extent(r).transformed(fig.transFigure.inverted()).x1

    ax.legend(
        handles=[Line2D([], [], color=_color(rho), linestyle=DOTTED)
                 for rho in panels.GAIN_CLASSICAL_RHOS],
        labels=[r"$\rho=%g$" % rho for rho in panels.GAIN_CLASSICAL_RHOS],
        title=r"$\mathrm{C}$", bbox_to_anchor=(x1 + 0.06, 0.5),
        ncol=1, **common)

    # The canvas was made generous so the titles would not be clipped; now it
    # is pulled back onto the two keys. They are anchored at the middle of the
    # axes, which fills the figure, so they stay centred as it shrinks and the
    # saved file carries no band of white above and below.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()
    tops, bots = [], []
    for art in list(ax.artists) + ([ax.get_legend()] if ax.get_legend() else []):
        b = art.get_window_extent(r).transformed(inv)
        tops.append(b.y1)
        bots.append(b.y0)
    fig.set_size_inches(fig.get_size_inches()[0], max(tops) - min(bots) + 0.12)
    return fig


if __name__ == "__main__":
    style.save(build(), NAME)
