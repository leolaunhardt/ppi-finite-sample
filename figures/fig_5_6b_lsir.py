"""
Figure 5.6: lsi_lowbr.

Delta against the classical mean over n. Green: PPI++ solid, its guard under
the estimated rule (4, 3/2) dashed and under the fixed half-guard dotted.
Purple: the tuned Double-PPI weight solid and the two-level guard dashed.
t-hat from the paired bootstrap, B = 500.

Writes: output/pdf/atlas_s_lsi_lowbr.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.

Suggested subcaption
--------------------
\\caption{LSI-R recidivism}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style

NAME = "atlas_s_lsi_lowbr"
WIDTH_FRACTION = 0.49
DATASET = "lsi_lowbr"


def build():
    return panels.atlas_panel(DATASET, baseline="C", ylabel=False,
                              width_fraction=WIDTH_FRACTION)


if __name__ == "__main__":
    style.save(build(), NAME)
