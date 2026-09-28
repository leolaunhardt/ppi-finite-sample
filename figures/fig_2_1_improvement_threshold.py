"""
Figure 2.1 -- The improvement threshold under normality.
Section 3.4, Corollary 3.2 (``cor:thres-gauss``).

PPI++ has the smaller variance only above the curve rho = 1/sqrt(n - 2). The
curve is therefore a boundary rather than a quantity to be read off, so the
plane is coloured by the member that wins on each side of it and the regions
carry their own names. That is how the maps of Section 4 are drawn, and it
spares the reader a key that names a line rather than a region.

Single estimator only: the cross-fit variant (CFP) has been dropped from this
figure, and the region is labelled PPI++ rather than SSP, since the surviving
text no longer draws the single-sample/cross-fit distinction here.

The sample-size axis is now logarithmic, n = 6 to 1000, replacing the earlier
linear-then-compressed scale: the curve falls smoothly with no join to mark.
The vertical axis is tightened to the curve's own range instead of the full
unit interval, since PPI++ never needs a correlation above 0.5 on this grid.

Writes: output/pdf/improvement_threshold_gaussian.pdf
Include with \\includegraphics[width=0.6\\textwidth]{...}.

Suggested caption
-----------------
\\caption[The improvement threshold under normality]{\\textbf{The improvement
threshold under normality.} The correlation PPI\\texttt{++} needs in order to
beat the classical mean, $\\rho = 1/\\sqrt{n-2}$ of
Corollary~\\ref{cor:thres-gauss}. Above the boundary
\\textcolor{ppp}{$\\mathrm{P}_{\\!+}$} has the smaller variance, below it
\\textcolor{cla}{$\\mathrm{C}$} does; at $n=6$ the predictor still needs
$\\rho=0.5$. The sample-size axis is logarithmic.}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.ticker as mticker

from ppi import style, theory

NAME = "improvement_threshold_gaussian"

#: Include the result with \includegraphics[width=0.6\textwidth]{...}.
#: Change this and the canvas follows, so the labels keep their size on paper.
WIDTH_FRACTION = 0.6

N_MIN, N_MAX = 6, 1000
X_TICKS = [6, 10, 20, 50, 100, 200, 500, 1000]

#: Tightened to the curve's own range, rho(6) = 0.5, rather than [0, 1].
Y_MAX = 0.55
Y_TICKS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

#: Where the region names sit, in axis coordinates. The tuned region takes its
#: area centroid; the classical one is a wedge that narrows to the right, so
#: its centroid would put the right end of a label that is wider than it is
#: tall up against the boundary. It is therefore set down and to the left of
#: the centroid (0.295, 0.227), which is where the whole box clears the curve.
GREEN_LABEL = (0.583, 0.610)
RED_LABEL = (0.235, 0.165)


def build():
    fig, ax = style.new_figure(width_fraction=WIDTH_FRACTION, grid=False)

    n = np.geomspace(N_MIN, N_MAX, 800)
    thr = theory.gaussian_improvement_threshold(n, c=theory.C_SINGLE)

    # The two regions, in the colours their winners carry everywhere else.
    ax.fill_between(n, 0.0, thr, color=style.CLA, zorder=1)
    ax.fill_between(n, thr, Y_MAX, color=style.PPP, zorder=1)
    # The boundary itself, dashed and white: it separates two filled regions
    # rather than carrying a reading of its own, exactly as the threshold
    # rules of Figures 4.1 and 4.2 do.
    ax.plot(n, thr, color="white", linewidth=2.0, linestyle=(0, (5, 3)),
            zorder=3)

    # Region names inside the regions, so the panel needs no key. Each sits at
    # the area centroid of its own region, computed in axis coordinates rather
    # than in n: the horizontal axis is logarithmic, so the centroid of the
    # drawn shape is what the eye reads as the middle, not the centroid in n.
    ax.text(GREEN_LABEL[0], GREEN_LABEL[1],
            r"$\mathrm{P}_{\!\!+\!}\succ\mathrm{C}$",
            transform=ax.transAxes, color="white", fontsize=22,
            ha="center", va="center", zorder=4)
    ax.text(RED_LABEL[0], RED_LABEL[1],
            r"$\mathrm{C}\succ\mathrm{P}_{\!\!+\!}$",
            transform=ax.transAxes, color="white", fontsize=22,
            ha="center", va="center", zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(N_MIN, N_MAX)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([str(t) for t in X_TICKS])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_ylim(0.0, Y_MAX)
    ax.set_yticks(Y_TICKS)

    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\rho$", rotation=0, labelpad=20, va="center")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    style.save(build(), NAME)
