"""
Appendix B, Figure B.3 (b) -- Price over the whole correlation range.
Same simulation, rules and bootstrap t-hat as panel (a).

The whole correlation range at n = 20, where the fixed rule's asymmetry is
the headline: it gives up more than fifteen points at rho = 0.98 and does so
at every correlation, because a constant multiplier cannot tell a lambda that
deserves keeping from one that does not.

The inset carries the other half. Between rho = 0.3 and 0.5 the three curves
are within a few points of one another and the full scale cannot separate
them, yet that is the region where the rules stop protecting and start
costing. The inset sits in the lower left because nothing is drawn there.

No clipping of any kind: lambda-hat is used exactly as Definition 3.2 defines
it.

Writes: output/pdf/shrink_price.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.

Suggested subcaption
--------------------
\\caption{Price: the whole range, $n = 20$}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ppi import experiments, panels, gaussian, style

NAME = "shrink_price"
WIDTH_FRACTION = 0.49

#: Where the inset sits, in axes coordinates. Every curve falls away to the
#: right, so the lower left is empty -- but only well below them: the box is
#: kept small enough that its upper right corner stays clear of the bundle.
INSET_BOX = [0.14, 0.07, 0.40, 0.32]


def build():
    fig, ax = style.new_figure(width_fraction=WIDTH_FRACTION)
    d = experiments.shrink_panel_price()
    delta = gaussian.delta({k: d[k] for k in experiments.PANEL_RULES})
    panels.shrink_curves(ax, d["rho"], delta)

    style.set_rho_axis(ax, axis="x")
    ax.set_xlabel(r"$\rho$")
    # The axis title stands in panel (a), which shares this scale.

    # The transition zone, in its own box where the panel is empty.
    z = experiments.shrink_price_zoom()
    zdelta = gaussian.delta({k: z[k] for k in experiments.PANEL_RULES})
    axins = ax.inset_axes(INSET_BOX)
    panels.shrink_curves(axins, z["rho"], zdelta)
    axins.set_xlim(*experiments.SHRINK_ZOOM_RANGE)
    axins.set_ylim(zdelta["S0"].min() - 0.6, zdelta["P+"].max() + 0.6)
    axins.set_xticks(list(experiments.SHRINK_ZOOM_RANGE))
    axins.tick_params(labelsize=13)
    axins.grid(True, which="major", zorder=0)
    axins.set_axisbelow(True)
    # Closed on all four sides, so it reads as a box rather than as a second
    # set of axes floating in the panel.
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
    axins.set_facecolor("white")
    ax.indicate_inset_zoom(axins, edgecolor=style.GREY_TEXT, alpha=0.8,
                           linewidth=1.2)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    style.save(build(), NAME)
