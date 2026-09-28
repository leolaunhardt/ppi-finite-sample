"""
Figure 2.2 (a) -- Error of PPI++ against classical, Gaussian case: nominal.
Section 3.4, Proposition 3.1 in the limit N -> infinity.

Mean squared error of the classical mean and of PPI++ against the labelled
sample size, at three correlations: a bad predictor (rho=0.1), a middling one
(rho=0.5) and a good one (rho=0.9). Computed in closed form, not simulated:
dividing Proposition 3.1 by the classical risk sigma_y^2/n leaves

    MSE / MSE(Ybar) = 1 - rho^2 + (1 + rho^2)/(n - 1),

and the classical mean is at 1. The scale sigma_y^2/sigma_f^2 cancels, so the
picture depends on (n, rho) alone; sigma_y = 1 fixes the vertical unit.

Mean squared error rather than variance, because PPI++ is biased at order 1/n
in general -- though under joint normality with mu_f = 0 that bias vanishes,
which is what makes the closed form available here.

The three PPI++ curves are shaded from light to dark green with the
correlation, so the darkest curve is the best predictor; the classical mean is
a single red curve, chosen to stand apart from the green family, drawn last
and a little heavier so that it reads as the reference it is. The key is not
in the panel: it is shared with the companion panel and set above both. On the
nominal scale the four lines are hard to tell apart at small n -- which is the
point of the companion panel, where the same expressions are shown relative
to the classical mean and the gaps become readable.

Writes: output/pdf/cf_vs_ss_nominal.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.

Suggested subcaption
--------------------
\\caption{Nominal}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from ppi import panels, style, theory

NAME = "cf_vs_ss_nominal"
WIDTH_FRACTION = 0.49
N_MAX = 50

#: Bad, middling and good predictor. Also used by panel (b), so the two
#: figures share both the correlations and the colours.
RHOS = [0.1, 0.5, 0.9]

#: Light to dark with the correlation, so the strongest predictor is the most
#: saturated line; the plain colour is kept for rho = 0.9 (Table A.1's PPI++).
GREEN_TINTS = [30, 60, 100]

#: The classical mean carries extra weight in both panels of the figure: it is
#: the reference, not one series among four.
CLASSICAL_WEIGHT = 1.45


def rho_color(rho):
    pct = GREEN_TINTS[RHOS.index(rho)]
    return style.PPP if pct == 100 else style.tint(style.PPP, pct)


def build():
    fig, ax = style.new_figure(width_fraction=WIDTH_FRACTION)
    n = style.n_grid(n_max=N_MAX)

    for rho in RHOS:
        ratio = 1.0 + theory.ppp_relative_mse_gaussian(
            n, rho, theory.C_SINGLE) / 100.0
        ax.plot(n, ratio / n, color=rho_color(rho),
                label=r"PPI++, $\rho=%g$" % rho, zorder=3)
    # Drawn last and heavier: it is the baseline every green curve is read
    # against, and on the nominal scale it runs inside the family.
    ax.plot(n, 1.0 / n, color=style.CLA, label="C", zorder=5,
            linewidth=plt.rcParams["lines.linewidth"] * CLASSICAL_WEIGHT)

    style.set_linear_n_axis(ax, n_max=N_MAX)
    ax.set_ylim(0.0, None)

    ax.set_xlabel("$n$")
    ax.set_ylabel("MSE")

    # The same reduced type as the multi-panel figures of Section 5, so that
    # two half-width panels do not shout next to them.
    ax.xaxis.label.set_size(panels.ATLAS_LABEL_SIZE)
    ax.yaxis.label.set_size(panels.ATLAS_LABEL_SIZE)
    ax.tick_params(labelsize=panels.ATLAS_TICK_SIZE)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    style.save(build(), NAME)
