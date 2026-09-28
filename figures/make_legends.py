"""
The shared legend strips.

Several figures in Chapters 5 and 6 are grids of subfigures that all use the
same colours. Putting the key inside one of them makes that panel taller than
its neighbours and the grid lopsided, so each group gets its key as a strip of
its own, included as a row between the panels and the caption.

Writes five files into output/pdf/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ppi import panels, style

SOLID, DASH, DOT = "-", (0, (5, 2.5)), (0, (1, 2))
def shrink():
    """Appendix B, the selected rules on the risk scale."""
    entries = [("line", style.CLA, SOLID, "C")]
    for key in panels.SHRINK_ORDER:
        color, dash = panels.SHRINK_STYLE[key]
        # The half-guard carries its argument here, as it does in the weak
        # predictor key, so the two keys read alike.
        label = style.ESTIMATOR_LABEL["S0_P+" if key == "S0" else key]
        entries.append(("line", color, dash, label))
    return panels.legend_strip(entries, ncol=5)
def _entry(key, lw=None):
    color, dash = panels.ATLAS_STYLE.get(
        key, (style.ESTIMATOR_COLOR.get(key, style.GREY_TEXT), SOLID))
    return ("line", color, dash, style.ESTIMATOR_LABEL[key], lw)


def _oracle(lw=None):
    """The oracle floor, which is a rule on every panel and so in every key."""
    o = panels.ORACLE_STYLE
    return ("line", o["color"], o["linestyle"], panels.ORACLE_LABEL,
            o["linewidth"] if lw is None else lw)


def atlas_strong():
    """Section 5.3, strong predictors: the key for the free cell of the grid.

    Two columns of five, filled down the left column first, which is how
    matplotlib fills an `ncol` legend. The first order stands in the left
    column and the second in the right, so each row pairs a weight with its
    counterpart one order up and the last row carries what belongs to
    neither.
    """
    left = ["P0", "P+", "SEl_P+", "SEc_P+", "C"]
    right = ["D0", "Dnu", "SEl_Both", "SEc_Both"]
    entries = [_entry(k, lw=panels.ATLAS_LINEWIDTH) for k in left]
    entries += [_entry(k, lw=panels.ATLAS_LINEWIDTH) for k in right]
    entries.append(_oracle(lw=panels.ATLAS_LINEWIDTH))
    return panels.legend_block(entries, ncol=2, width_fraction=0.32,
                               handlelength=1.5)


def atlas_binary():
    """The binary grid, where PPI++ is the baseline.

    Three rows: the first order, the second order, and what belongs to
    neither. matplotlib fills column by column, so the entries are listed in
    that order and the two cells the last row does not need are left blank.
    """
    rows = [["P0", "P+", "SEl_P+", "SEc_P+"],
            ["D0", "Dnu", "SEl_Both", "SEc_Both"],
            ["C", "__oracle__", None, None]]
    entries = []
    for col in range(4):
        for row in rows:
            key = row[col]
            if key is None:
                entries.append(("blank", None, None, None))
            elif key == "__oracle__":
                entries.append(_oracle())
            else:
                entries.append(_entry(key, lw=None))
    return panels.legend_strip(entries, ncol=4)


def atlas_weak_block():
    """Section 5.3, weak predictors.

    Two columns: the classical mean and the floor share the top row, the
    first-order weights run down the left column and the second-order ones
    down the right.
    """
    left = ["C", "P+", "SEl_P+", "SEc_P+", "S0_P+"]
    right = ["Dnu", "SEl_Both", "SEc_Both"]
    entries = [_entry("C", lw=None)]
    entries += [_entry(k, lw=None) for k in left[1:]]
    entries = [_entry(left[0], lw=None)] + [_entry(k, lw=None) for k in left[1:]]
    entries += [_oracle()] + [_entry(k, lw=None) for k in right]
    return panels.legend_block(entries, ncol=2, width_fraction=0.49)


BUILDERS = {
    "legend_shrink": shrink,
    "legend_atlas_strong": atlas_strong,
    "legend_atlas_binary": atlas_binary,
    "legend_atlas_weak_block": atlas_weak_block,
}

if __name__ == "__main__":
    for name, builder in BUILDERS.items():
        style.save(builder(), name, preview=True)
