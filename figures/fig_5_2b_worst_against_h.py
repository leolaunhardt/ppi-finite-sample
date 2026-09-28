"""
Figure 5.2) Worst case -- Worst case
Section 5.1, selecting (h, kappa) for the tuned exponential family.

Worst case Delta_C against h, one curve per curvature, over the all regime.

The trade-off panel opened out along h, which it hides. The dashed green line
is unshrunken PPI++. Reading the two panels together gives the usable range:
the mean is flat up to about h = 4 and collapses beyond it, while the worst
case is still falling there, so h just below the collapse is what buys
protection most cheaply.

Simulated: the rule is a nonlinear function of the estimated t-hat, which is
here the paired bootstrap of eq:boot with B = 200, the default of
Section 3.3. See ppi/hyper.py for the sufficient-statistic path that makes the
41 x 24 candidate grid affordable.

Writes: output/pdf/hyper_worst_h.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.

Suggested subcaption
--------------------
\\caption{Worst case}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style

NAME = "hyper_worst_h"
WIDTH_FRACTION = 0.49

#: The two members named in the text, marked as in Figure 5.1.
MARKS = [(8.0, 1.5, r"$\widehat{\alpha}_{\mathrm{E}}(8,\frac{3}{2})$"),
         (4.0, 1.0, r"$\widehat{\alpha}_{\mathrm{E}}(4,1)$")]
OFFSETS = [(-26, 62), (-104, -24)]


def build():
    return panels.hyper_breakdown_panel(
        "all", "worst", width_fraction=WIDTH_FRACTION,
        mark=MARKS, offsets=OFFSETS)


if __name__ == "__main__":
    style.save(build(), NAME)
