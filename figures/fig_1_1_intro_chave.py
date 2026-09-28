"""
Figure 1.1 -- what the two corrections are for, on one real population.

The mean aboveground biomass of a tropical tree, estimated from n felled trees
and a published allometric equation applied to the rest. The equation is the
dry-forest one of Chave et al. (2005) used on every forest type, which is a
domain transfer of the kind that happens in practice; it misses the largest
trees and therefore has lambda* = 2.398 and t(20) = 0.34.

The root mean squared error rather than a ratio, so that the figure can be
read before any of the notation of Chapter 3 has been introduced, and rooted
rather than squared so that the axis carries kilograms: the number is the
typical error of the estimated mean, against a true mean of 1134 kg. The
vertical axis is linear, which is what makes the inset worth having -- it
magnifies the range from n = 400, where every curve of the main panel has
collapsed onto the floor.

Writes: output/pdf/intro_chave.pdf
Include with \\includegraphics[width=0.85\\textwidth]{...}.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import populations, style

NAME = "intro_chave"
WIDTH_FRACTION = 0.85
DATASET = "agb_chave05_dry"

#: Spelled out, since the introduction has not defined any symbol yet. The
#: fixed rule s = 1/2 is left out: it is a variant of the guard, not a
#: separate idea, and the panel is crowded enough.
#: The last field is a line-width multiplier. DoublePPI carries the extra
#: weight: it is the estimator the thesis proposes, and the introduction
#: should let the eye find it before it reads the key.
EMPHASIS = 1.55

SERIES = [("C", "Classical mean", style.CLA, "-", 1.0),
          ("P0", "PPI", style.PPI, "-", 1.0),
          ("P+", "PPI++", style.PPP, "-", 1.0),
          ("SEl_P+", "Guarded PPI++", style.tint(style.PPP), (0, (6, 2)), 1.0),
          ("Dnu", "DoublePPI", style.DOL, "-", EMPHASIS),
          ("SEl_Both", "Guarded DoublePPI", style.tint(style.DOL),
           (0, (6, 2)), 1.0)]

#: A 1-2-5 ladder, so the labels are a regular sequence rather than whichever
#: grid points happen to fall inside the range.
X_TICKS = [10, 20, 50, 100, 200, 500]
ZOOM_TICKS = [400, 600, 800]


def curves():
    """Absolute RMSE, from the stored Delta and the exact variance of the mean.

    Under sampling without replacement Var(Ybar) = (S^2/n)(1 - n/M) exactly,
    so the ratios the populations stores can be turned back into an absolute scale
    without a second simulation.
    """
    d = populations.dataset(DATASET)
    Y = populations.load(DATASET)[0]
    M, S2 = len(Y), float(np.var(Y, ddof=1))
    n = d["n"]
    mse_c = (S2 / n) * (1.0 - n / M)
    out = {"C": np.sqrt(mse_c)}
    out.update({k: np.sqrt(mse_c * (1.0 + d[k] / 100.0))
                for k, *_ in SERIES if k != "C"})
    return n, out, float(Y.mean())


def build():
    n, m, theta = curves()
    fig, ax = style.new_figure(width_fraction=WIDTH_FRACTION)

    base_lw = plt.rcParams["lines.linewidth"]
    for key, _, color, dash, weight in SERIES:
        ax.plot(n, m[key], color=color, linestyle=dash,
                linewidth=base_lw * weight, zorder=3 + (weight > 1.0))

    ax.set_xscale("log")
    # Cut at 3000 kg: PPI++ starts at 8629 and leaves the frame, which is
    # the point of it. Holding the whole excursion would flatten every
    # other curve onto the axis.
    ax.set_ylim(0.0, 3000.0)
    ax.set_xlim(n.min(), n.max())
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(["%d" % t for t in X_TICKS])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"RMSE of $\hth$ [kg]" if False else
                  r"RMSE of $\widehat{\theta}$ [kg]")

    # The key goes under the panel rather than beside it: six spelled-out
    # names in a column would take a third of the width from the figure.
    handles = [Line2D([], [], color=c, linestyle=d,
                      linewidth=plt.rcParams["lines.linewidth"] * w)
               for _, _, c, d, w in SERIES]
    fig.tight_layout()
    ax.legend(handles=handles, labels=[lab for _, lab, _, _, _ in SERIES],
              loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              frameon=False, handlelength=1.6, columnspacing=1.6)

    # The corner where the curves of the main panel run together.
    # Low and to the right, just above the region it magnifies, and allowed
    # to overhang the edge of the panel by a little: the saved bounding box
    # grows with it, and the connectors stay short enough to follow.
    box = ax.get_position()
    axin = fig.add_axes([box.x0 + 0.72 * box.width, box.y0 + 0.31 * box.height,
                         0.42 * box.width, 0.40 * box.height])
    axin.set_zorder(10)
    axin.patch.set_facecolor("white")
    for key, _, color, dash, weight in SERIES:
        axin.plot(n, m[key], color=color, linestyle=dash,
                  linewidth=base_lw * weight, zorder=3 + (weight > 1.0))
    # Linear in n here: over a range of two hundred either side a
    # logarithmic axis would space the labels unevenly for no gain.
    axin.set_xlim(400, n.max())
    axin.set_ylim(0.0, 200.0)
    axin.xaxis.set_major_locator(mticker.FixedLocator(ZOOM_TICKS))
    axin.xaxis.set_major_formatter(
        mticker.FixedFormatter(["%d" % t for t in ZOOM_TICKS]))
    axin.xaxis.set_minor_locator(mticker.NullLocator())
    axin.yaxis.set_major_locator(mticker.FixedLocator([0, 100, 200]))
    axin.yaxis.set_major_formatter(mticker.FixedFormatter(["0", "100", "200"]))
    axin.tick_params(labelsize=style._TICK_SIZE * 0.62)
    axin.grid(True, which="major", zorder=0)
    axin.set_axisbelow(True)
    for spine in axin.spines.values():
        spine.set_linewidth(0.9)
    ax.indicate_inset_zoom(axin, edgecolor=style.GREY_TEXT, linewidth=1.1,
                           alpha=1.0)
    return fig


if __name__ == "__main__":
    style.save(build(), NAME)
