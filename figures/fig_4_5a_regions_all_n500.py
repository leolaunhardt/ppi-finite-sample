"""
The best member at n = 500.
Section 5.3, from the unified framework of Table A.1 in its Gaussian form.

The region map repeated at n = 500; same conventions throughout.

Writes: output/pdf/best_member_regions_all_n500.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style, unified

NAME = "best_member_regions_all_n500"
WIDTH_FRACTION = 0.5844
N = 500

#: Identical in role to Figures 4.1 and 4.2, but at n = 500 the scale runs to
#: t_C = (n-1)/2 = 249.5 at rho = 1, so the round values are an order of
#: magnitude larger. Spread as evenly over the frame as round numbers allow.
T_TICKS = [5, 20, 50, 100, 200]

ESTIMATORS = unified.ALL_TEN


def build():
    rules = panels.switch_points(ESTIMATORS, N)
    return panels.regions_panel(ESTIMATORS, N, width_fraction=WIDTH_FRACTION,
                                lam_max=2.0, rules=rules, legend=None,
                                t_axis=T_TICKS)


if __name__ == "__main__":
    style.save(build(), NAME)
