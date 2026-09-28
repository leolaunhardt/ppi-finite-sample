"""
Figure 4.6 (b) -- The tuned exponential family: curvature via kappa.
Section 3.2, equation ``eq:expfamily``.

The same family at a fixed h = 2, for four steepnesses. As in the plug-in
family, every curve meets 1/2 at t = h whatever kappa is; unlike it, none of
them ever reaches zero or one exactly, and none of them has the kink the
truncation gives the plug-in family. Read against Figure 4.2 (b).

Writes: output/pdf/exponential_family_kappa.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.

Suggested subcaption
--------------------
\\caption{Curvature via $\\kappa$}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style, theory

NAME = "exponential_family_kappa"
KAPPA_VALUES = [0.5, 1.0, 2.0, 4.0]
H = panels.DP_H

#: The two parametrisations of Section 5.1, drawn on top of the family and
#: heavier than it: the liberal rule dashed, the conservative one dotted, as
#: in the panels of Section 5.3.
HIGHLIGHT = [({"h": 4.0, "kappa": 1.0},
              r"$\widehat{\alpha}_{\mathrm{E}}(4,1)$", "SEl"),
             ({"h": 8.0, "kappa": 1.5},
              r"$\widehat{\alpha}_{\mathrm{E}}(8,\frac{3}{2})$", "SEc")]

SHOW_DP_REFERENCE = False


def build():
    return panels.shrinkage_family_panel(
        theory.s_exponential_family, "kappa", KAPPA_VALUES, {"h": H},
        label=r"$\kappa=%g$", highlight=HIGHLIGHT, ylabel=False)


if __name__ == "__main__":
    style.save(build(), NAME)
