"""
Figure 5.1) No belief -- No belief
Section 5.1, selecting (h, kappa) for the tuned exponential family.

Trade-off between the mean and the worst case over the all regime.

Motivated by Figure 4.4, but zoomed into the exponential family itself: there
the classical mean at the origin compresses the whole family into a narrow
band, and the point here is precisely to resolve that band. Candidates more
conservative than the fixed rule leave the frame.

Each curve is one curvature traced over h; the heavy grey line is the
non-dominated front. The green dot is unshrunken PPI++, the brown dot the
fixed rule at s = 1/2.

Simulated: the rule is a nonlinear function of the estimated t-hat, which is
here the paired bootstrap of eq:boot with B = 200, the default of
Section 3.3. See ppi/hyper.py for the sufficient-statistic path that makes the
41 x 24 candidate grid affordable.

Writes: output/pdf/hyper_tradeoff_all.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.

Suggested subcaption
--------------------
\\caption{No belief}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style

NAME = "hyper_tradeoff_all"
WIDTH_FRACTION = 0.49

#: The two members named in the text, annotated in place.
MARKS = [(4.0, 1.0, r"$\widehat{\alpha}_{\mathrm{E}}(4,1)$"),
         (8.0, 1.5, r"$\widehat{\alpha}_{\mathrm{E}}(8,\frac{3}{2})$")]
OFFSETS = [(10, 12), (12, 10)]


def build():
    return panels.hyper_tradeoff_panel(
        "all", width_fraction=WIDTH_FRACTION, mark=MARKS, offsets=OFFSETS, ylabel=False)


if __name__ == "__main__":
    style.save(build(), NAME)
