"""
Figure 4.6 (a) -- The tuned exponential family: shift by h.
Section 3.2, equation ``eq:expfamily``.

    s_E(h, kappa) = 2^(-(h/t)^kappa)

at kappa = 1, for the same four half-value points as Figure 4.2 (a), so the
two families are directly comparable through h: read this panel against
Figure 4.2 (a) at the same h. Below h the exponential rule rises considerably
more slowly and never truncates; above it, it approaches 1 like
1 - h ln2 / t.

Set SHOW_DP_REFERENCE to add the direct plug-in rule as a grey dashed curve.

Writes: output/pdf/exponential_family_h.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.

Suggested subcaption
--------------------
\\caption{Shift by $h$}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style, theory

NAME = "exponential_family_h"
H_VALUES = [1.0, 2.0, 3.0, 4.0]
KAPPA = 1.0

#: The two parametrisations of Section 5.1, drawn on top of the family and
#: heavier than it: the liberal rule dashed, the conservative one dotted, as
#: in the panels of Section 5.3.
HIGHLIGHT = [({"h": 4.0, "kappa": 1.0},
              r"$\widehat{\alpha}_{\mathrm{E}}(4,1)$", "SEl"),
             ({"h": 8.0, "kappa": 1.5},
              r"$\widehat{\alpha}_{\mathrm{E}}(8,\frac{3}{2})$", "SEc")]


def build():
    return panels.shrinkage_family_panel(
        theory.s_exponential_family, "h", H_VALUES, {"kappa": KAPPA},
        label="$h=%g$", highlight=HIGHLIGHT)


if __name__ == "__main__":
    style.save(build(), NAME)
