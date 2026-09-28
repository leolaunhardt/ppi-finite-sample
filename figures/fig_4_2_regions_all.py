"""
The best member at n = 50, all ten estimators.
Section 5.3, from the unified framework of Table A.1 in its Gaussian form.

Each point of the (rho, lambda*) plane is coloured by the member of Table A.1 with the smallest relative mean squared error, with lambda_0 = nu_0 = 1 and s = u = 1/2. The ten members include the double-shrunken one, which shrinks both orders at one half.

Writes: output/pdf/best_member_regions_all.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.8 subfigure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style, unified

NAME = "best_member_regions_all"
WIDTH_FRACTION = 0.7649
N = 50

#: The second scale on top, in the statistic the thresholds are stated in.
#: Identical to Figure 4.1, so the two can be read as a before and after.
T_TICKS = [1, 2, 5, 10, 20]

ESTIMATORS = unified.ALL_TEN


def build():
    rules = panels.switch_points(ESTIMATORS, N)
    return panels.regions_panel(ESTIMATORS, N, width_fraction=WIDTH_FRACTION,
                                lam_max=2.0, rules=rules, legend="right",
                                legend_cols=2, legend_all=True,
                                outline="P+", t_axis=T_TICKS)


if __name__ == "__main__":
    style.save(build(), NAME)
