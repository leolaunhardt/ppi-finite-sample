"""
Figure 5.4: pas_galaxy_spiral.

Delta against PPI at the fixed weight lambda_0 = 1 over n, with PPI at a fixed weight, both Double-PPI
weights and the two-level guard G^{C,P+}_+(4, 3/2). Simulated through
ppi/populations.py: the labelled sample is drawn without replacement, the
unlabelled block is its complement, and the grid is capped at a fifth of the
population.

Writes: output/pdf/atlas_b_pas_galaxy_spiral.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.

Suggested subcaption
--------------------
\\caption{Spiral galaxies}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style

NAME = "atlas_b_pas_galaxy_spiral"
WIDTH_FRACTION = 0.49
DATASET = "pas_galaxy_spiral"


def build():
    return panels.atlas_panel(DATASET, baseline="P0", ylabel=True,
                              width_fraction=WIDTH_FRACTION,
                              order=panels.ATLAS_ORDER_P0,
                              reference=panels.ATLAS_REFERENCE_P0,
                              yscale="linear")


if __name__ == "__main__":
    style.save(build(), NAME)
