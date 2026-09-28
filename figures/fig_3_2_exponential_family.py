"""
Figure 3.2 -- The two selected members of the exponential family.
Section 3.2, equation ``eq:expfamily``.

    alpha_E(h, kappa)(t) = 2^(-(h/t)^kappa)

drawn for the two parametrisations Section 5.1 selects, against the oracle
alpha* = t/(1+t) they approximate. The calibration family over h and kappa
that the earlier version of this figure showed has been dropped: those curves
were the search, not the result, and the point here is how the two chosen
rules stand to the oracle.

The t axis is that of Figure 3.1, logarithmic from 1/20 to 20. The family is a
shift in log t -- the half-value point enters only through log h -- so on this
axis the two curves are one shape translated, each crossing one half at
t = h.

Writes: output/pdf/exponential_family.pdf
Include with \\includegraphics[width=0.6\\textwidth]{...}.

Suggested caption
-----------------
\\caption[Guarded Tuning: the two selected rules]{\\textbf{The two selected
members of the exponential family \\eqref{eq:expfamily}}, against the oracle
$t/(1+t)$ in black. The \\emph{liberal} rule $\\haE(4,1)$ is dashed and the
\\emph{conservative} rule $\\haE(8,\\tfrac32)$ dotted, in the colours they
carry in Section~\\ref{sec:atlas}; each crosses one half at its own half-value
point $\\widehat t = h$. Both are smooth throughout and attain neither
endpoint. The axis is that of Figure~\\ref{fig:shrink-windows}.}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style

NAME = "exponential_family"
WIDTH_FRACTION = 0.6

#: (h, kappa, legend label, style key) -- the two rules of Section 5.1.
RULES = [(4.0, 1.0, r"$\widehat{\alpha}_{\mathrm{E}}(4,1)$", "SEl"),
         (8.0, 1.5, r"$\widehat{\alpha}_{\mathrm{E}}(8,\frac{3}{2})$", "SEc")]


def build():
    return panels.exponential_family_panel(RULES,
                                           width_fraction=WIDTH_FRACTION)


if __name__ == "__main__":
    style.save(build(), NAME)
