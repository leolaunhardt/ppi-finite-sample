"""
Figure 5.4: agb_chave05_dry.

Delta against PPI at the fixed weight lambda_0 = 1 over n, with PPI++, both
Double-PPI weights, the two-level guard G^{C,P+}_+(4, 3/2) and the classical
mean. C is drawn but excluded from the reference set that sets the vertical
frame, so it appears where it falls and never sets the scale.

Simulated through ppi/populations.py: the labelled sample is drawn without
replacement, the unlabelled block is its complement, and the grid is capped at
a fifth of the population.

Writes: output/pdf/atlas_e_agb_chave05_dry.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.32 subfigure.

Suggested subcaption
--------------------
\\caption{Tree biomass}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style

NAME = "atlas_e_agb_chave05_dry"
WIDTH_FRACTION = 0.32
DATASET = "agb_chave05_dry"


def build():
    return panels.atlas_panel(DATASET, baseline="P0", ylabel=True,
                              width_fraction=WIDTH_FRACTION,
                              order=panels.ATLAS_ORDER_P0,
                              reference=["Dnu", "SEl_Both", "SEc_Both"],
                              yscale="linear", xscale="log",
                              linewidth=panels.ATLAS_LINEWIDTH)


if __name__ == "__main__":
    style.save(build(), NAME)
