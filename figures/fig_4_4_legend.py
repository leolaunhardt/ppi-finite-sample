"""
The shared key for the ten members of Table A.1.

Drawn once and placed above Figure 4.2, where two panels share it and neither
has room for a key of its own.

Writes: output/pdf/legend_members_all.pdf
Include with \\includegraphics[width=0.95\\textwidth]{...}.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style, unified

NAME = "legend_members_all"


def build():
    return panels.members_legend(cols=5, width_fraction=0.95)


if __name__ == "__main__":
    style.save(build(), NAME)
