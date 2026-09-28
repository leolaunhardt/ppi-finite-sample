"""
Figure 2.2 (b) -- Error of PPI++ against classical, Gaussian case: relative.
Section 3.4, Proposition 3.1 in the limit N -> infinity.

The same three correlations as panel (a) -- a bad predictor (rho=0.1), a
middling one (rho=0.5) and a good one (rho=0.9) -- now as Delta_C of Chapter
6, equation ``eq:delta-C``:

    Delta_C = 100 [ (1 + rho^2)/(n - 1) - rho^2 ]   at c = 1,

so the classical mean is the flat line at zero and the three curves that were
almost indistinguishable in panel (a) separate cleanly. In closed form, not
simulated.

Colours match panel (a): light to dark green with the correlation, red for
the classical mean. At rho=0.1 the predictor never pays for itself inside the
range shown -- the threshold n=2+1/rho^2 is 102 -- at rho=0.5 it crosses
almost exactly at the left edge, n=6, and at rho=0.9 it is already well
negative there.

Writes: output/pdf/ssp_vs_classical_correlations.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.

Suggested subcaption
--------------------
\\caption{Relative}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from ppi import panels, style, theory

NAME = "ssp_vs_classical_correlations"
WIDTH_FRACTION = 0.49
N_MAX = 50

#: Bad, middling and good predictor -- matches panel (a) exactly.
RHOS = [0.1, 0.5, 0.9]

#: Light to dark with the correlation, matching panel (a); the plain colour
#: is kept for rho = 0.9 (Table A.1's PPI++).
GREEN_TINTS = [30, 60, 100]

#: Matches panel (a).
CLASSICAL_WEIGHT = 1.45


def rho_color(rho):
    pct = GREEN_TINTS[RHOS.index(rho)]
    return style.PPP if pct == 100 else style.tint(style.PPP, pct)


def build():
    fig, ax = style.new_figure(width_fraction=WIDTH_FRACTION)
    n = style.n_grid(n_max=N_MAX)

    for rho in RHOS:
        ax.plot(n, theory.ppp_relative_mse_gaussian(n, rho, theory.C_SINGLE),
                color=rho_color(rho), label=r"PPI++, $\rho=%g$" % rho,
                zorder=4)
    # The classical mean is the baseline of Delta_C, so it is the zero line,
    # drawn last and heavier to match panel (a).
    ax.axhline(0.0, color=style.CLA, zorder=5, label="C",
               linewidth=plt.rcParams["lines.linewidth"] * CLASSICAL_WEIGHT)

    style.set_linear_n_axis(ax, n_max=N_MAX)

    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\Delta_{\mathrm{C}}$ in %")

    # The same reduced type as the multi-panel figures of Section 5, so that
    # two half-width panels do not shout next to them.
    ax.xaxis.label.set_size(panels.ATLAS_LABEL_SIZE)
    ax.yaxis.label.set_size(panels.ATLAS_LABEL_SIZE)
    ax.tick_params(labelsize=panels.ATLAS_TICK_SIZE)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    style.save(build(), NAME)
