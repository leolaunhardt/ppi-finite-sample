"""
Figure 4.1 -- the Gaussian map before any guarding.

The best of the five unguarded members C, P_0, P_+, D_0, D_+ over
(rho, lambda*) at n = 50 and lambda_0 = nu_0 = 1. Drawn on the same axes,
with the same t-scale on top, as the ten-member map of Figure 4.2, so that
the two can be read as a before and after.

Writes: output/pdf/regions_basic.pdf
Include with \\includegraphics[width=0.92\\textwidth]{...}.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style, unified

NAME = "regions_basic"
WIDTH_FRACTION = 0.8355
N = 50

#: Identical to the ten-member map, so the two share a horizontal scale.
#: At n = 50 the scale runs to t = (n-1)/2 = 24.5 at rho = 1, so plain integers
#: carry it across the whole frame instead of stopping short of the half-way
#: mark as the old set did. Anything below t = 1 crowds against it: the two
#: are only 0.04 apart in rho.
T_TICKS = [1, 2, 5, 10, 20]

#: The five unguarded members of Table 4.1; Duv is the second-order guard.
ESTIMATORS = [nm for nm in unified.UNSHRUNKEN_ALL if nm != "Duv"]


def build():
    rules = panels.switch_points(ESTIMATORS, N)
    return panels.regions_panel(ESTIMATORS, N, width_fraction=WIDTH_FRACTION,
                                lam_max=2.0, rules=rules, legend="right",
                                legend_cols=1, t_axis=T_TICKS)


if __name__ == "__main__":
    style.save(build(), NAME)
