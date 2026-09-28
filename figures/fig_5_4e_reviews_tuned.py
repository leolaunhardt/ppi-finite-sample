"""
Figure 5.4: amazon_tuned.

Delta against PPI at the fixed weight lambda_0 = 1 over n, with PPI++, both
Double-PPI weights, the two-level guard G^{C,P+}_+(4, 3/2) and the classical
mean. C is drawn but excluded from the reference set that sets the vertical
frame, so it appears where it falls and never sets the scale.

Simulated through ppi/populations.py: the labelled sample is drawn without
replacement, the unlabelled block is its complement, and the grid is capped at
a fifth of the population.

Writes: output/pdf/atlas_a_amazon_tuned.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.32 subfigure.

Suggested subcaption
--------------------
\\caption{Reviews, BERT fine-tuned}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style

NAME = "atlas_a_amazon_tuned"
WIDTH_FRACTION = 0.32
DATASET = "amazon_tuned"


def build():
    return panels.atlas_panel(DATASET, baseline="P0", ylabel=False,
                              width_fraction=WIDTH_FRACTION,
                              order=panels.ATLAS_ORDER_P0,
                              reference=panels.ATLAS_REFERENCE_P0,
                              yscale="symlog", xscale="log",
                              linewidth=panels.ATLAS_LINEWIDTH)


if __name__ == "__main__":
    style.save(build(), NAME)
