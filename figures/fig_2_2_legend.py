"""
The shared key for Figure 2.2.

Two keys in one strip rather than four parallel entries: the classical mean on
its own, and the three PPI++ curves under a single heading, so that "PPI++" is
written once and the colours carry only the correlation. Drawn once and placed
above the two panels.

Writes: output/pdf/legend_gauss_mse.pdf
Include with \\includegraphics[width=0.8\\textwidth]{...}.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import style
from figures.fig_2_2a_nominal_mse import (CLASSICAL_WEIGHT, RHOS,
                                               rho_color)

NAME = "legend_gauss_mse"
WIDTH_FRACTION = 0.8


def build():
    style.use_thesis_style()
    base = plt.rcParams["lines.linewidth"]
    width = WIDTH_FRACTION * style.TEXTWIDTH_IN / style.RENDER_SCALE
    fig = plt.figure(figsize=(width, 0.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    left = ax.legend(
        handles=[Line2D([], [], color=style.CLA, linewidth=base * CLASSICAL_WEIGHT)],
        labels=["Classical"], loc="center left", bbox_to_anchor=(0.04, 0.5),
        frameon=False, handlelength=1.6, fontsize=style._LEGEND_SIZE)
    ax.add_artist(left)

    ax.legend(handles=[Line2D([], [], color=rho_color(r), linewidth=base)
                       for r in RHOS],
              labels=[r"$\rho=%g$" % r for r in RHOS],
              title="PPI++", loc="center left", bbox_to_anchor=(0.34, 0.5),
              ncol=3, frameon=False, handlelength=1.6, columnspacing=1.6,
              fontsize=style._LEGEND_SIZE,
              title_fontproperties={"size": style._LEGEND_SIZE})
    return fig


if __name__ == "__main__":
    style.save(build(), NAME)
