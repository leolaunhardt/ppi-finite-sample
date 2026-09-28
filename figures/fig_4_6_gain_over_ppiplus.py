"""
Figure 4.6 -- Double-PPI against PPI++ over the sample size.

One panel instead of two: the comparison is drawn Delta against PPI++,
evaluated from the closed form of eq:gain-rel so that rho = 0 can be drawn as
well. The vertical axis is symmetrically logarithmic about a linear band of
one percent, and parity is the line at zero in PPI++'s own colour. The
magnified corner has been dropped: the inverse hyperbolic sine already
resolves the approach to zero, and the inset cost the panel its right margin.

The key is a strip of its own, placed above the panel in LaTeX, so that the
classical mean is named once and the colours carry only the correlation, as
in Figure 2.2. Build it with figures/fig_4_6_legend.py.

Two red boxes mark the strips on which PPI++ beats both curves of the same
correlation at once, that is where Delta(D_+) and Delta(C) are both positive.
They run from n = 18 to 32 at rho = 1/4 and from n = 6 to 6.6 at rho = 1/2,
the interval between the crossing of C at n = 1 + (1+rho^2)/rho^2 and the
crossing of D_+ at K_n = rho^2/(1-rho^2).

Writes: output/pdf/gain_ratio.pdf
Include with \\includegraphics[width=0.88\\textwidth]{...}.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style

NAME = "gain_ratio"
WIDTH_FRACTION = 0.7098


def build():
    return panels.gain_delta_panel(width_fraction=WIDTH_FRACTION,
                                   legend=False, inset=False,
                                   strips=panels.GAIN_STRIPS)


if __name__ == "__main__":
    style.save(build(), NAME)
