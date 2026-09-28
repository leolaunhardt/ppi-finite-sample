"""
Appendix B, Figure B.3 (a) -- Protection over the belief region.
Gaussian simulation with N = 10^4, 2e5 replicates, bootstrap t-hat (B = 200).

Relative mean squared error against the labelled sample size, at rho = 0.2.
The correlation is chosen so that n spans the belief region: with
t_lambda = (n-1) rho^2/(1+rho^2) = (n-1)/26 here, the range n = 6 to 79 is
t = 0.19 to 3.00, and the improvement threshold t = 1 of Corollary 2.1 falls
at n = 27, inside the frame. The upper axis gives t directly.

Drawn are unguarded PPI++, the fixed half-guard and the two rules of
eq:two-rules, liberal (4, 1) and conservative (8, 3/2), with t-hat from the
paired bootstrap, the input they were selected on in ppi/hyper.py.

No clipping of any kind: lambda-hat is used exactly as Definition 3.2 defines
it.

Writes: output/pdf/shrink_protection.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.

Suggested subcaption
--------------------
\\caption{Protection: the belief region $\\tC \\le 3$}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ppi import experiments, panels, gaussian, style

NAME = "shrink_protection"
WIDTH_FRACTION = 0.49
N_MAX = 80

#: Capped: unshrunken PPI++ reaches +17 at the smallest n, and letting that
#: set the scale flattens everything the panel is about. It simply leaves.
YLIM = (-4.0, 5.0)


def build():
    fig, ax = style.new_figure(width_fraction=WIDTH_FRACTION)
    d = experiments.shrink_protection_n()
    delta = gaussian.delta({k: d[k] for k in experiments.PANEL_RULES})

    panels.shrink_curves(ax, d["n"], delta)

    ax.set_xlim(0.0, N_MAX)
    ax.set_xticks(np.arange(0, N_MAX + 1, 20))
    ax.set_xticks(np.arange(0, N_MAX + 1, 10), minor=True)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\Delta_{\mathrm{C}}$ in %")

    # A second scale on top, so the belief region can be read off directly.
    rho2 = experiments.SHRINK_PROT_RHO ** 2
    top = ax.secondary_xaxis(
        "top", functions=(lambda n: (n - 1.0) * rho2 / (1.0 + rho2),
                          lambda t: 1.0 + t * (1.0 + rho2) / rho2))
    top.set_xticks([0, 1, 2, 3])
    top.set_xlabel(r"$t_{\mathrm{C}}$", labelpad=8)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    style.save(build(), NAME)
