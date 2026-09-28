"""
Excess risk at lambda* = 0.25, n = 50.
Section 5.3, from the unified framework of Table A.1 in its Gaussian form.

The same section at lambda* = 0.25, below the belief line, where the map no longer depends on the calibration.

Writes: output/pdf/excess_all_badly_calibrated.pdf
Include with \\includegraphics[width=\\textwidth]{...} inside a 0.49 subfigure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import panels, style, unified

NAME = "excess_all_badly_calibrated"
WIDTH_FRACTION = 0.49
N = 50
LAMBDA_STAR = 0.25
ESTIMATORS = unified.ALL_TEN


def build():
    return panels.excess_log_panel(ESTIMATORS, N, LAMBDA_STAR, s=0.5,
                                   ylabel=False)


if __name__ == "__main__":
    style.save(build(), NAME)
