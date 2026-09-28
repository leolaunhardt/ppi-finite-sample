"""
Figure 4.6 -- the guard window.

A guard with factor alpha beats the untuned member from t > alpha/(2-alpha)
and beats the tuned one until t < (1+alpha)/(1-alpha). Read as statements
about alpha at a given t, the two invert to

    alpha < 2t/(1+t)        still better than the untuned member
    alpha > (t-1)/(t+1)     still better than the tuned one

so the guard is the best of the three exactly inside that band. The picture
is read twice: with t = t_C(W) and alpha for the first-order guard,
where the untuned member is the classical mean and the tuned one is W, and
with t = t_{P_+}(D_+) and beta for the second-order guard, where they
are P_+ and D_+.

The oracle factor alpha* = t/(1+t) shares its denominator with both edges and
its numerator lies between theirs, so it is strictly inside the band at every
t. The half-guard alpha = 1/2 is the horizontal chord whose window is exactly
(1/3, 3).

Writes: output/pdf/guard_window.pdf
Include with \\includegraphics[width=0.5\\textwidth]{graphics/guard_window}.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import style

NAME = "guard_window"
#: Matches the \includegraphics width in 2chapter34.tex, so the type comes
#: out the same size on the page as in every other figure.
WIDTH_FRACTION = 0.5

T_MIN, T_MAX = 0.05, 20.0
G_HALF = 0.5
#: The plane is partitioned by which member wins, and each region carries the
#: colour of its winner, as the maps of Section 4 do. The picture is drawn for
#: the first-order reading, where the untuned side is the classical mean and
#: the tuned side PPI++; the second-order reading swaps in P_+ and D_+ and is
#: stated in the caption rather than coloured a second time.
LEFT_COLOR = style.CLA                 # C beats the guard
RIGHT_COLOR = style.PPP                # P_+ beats the guard
BAND_COLOR = style.tint(style.PPP)     # the guard beats both

T_TICKS = [0.05, 1.0 / 3.0, 1.0, 3.0, 10.0, 20.0]
T_LABELS = [r"$\frac{1}{20}$", r"$\frac{1}{3}$", "1", "3", "10", "20"]

#: Region names, in axis coordinates.
LEFT_LABEL = (0.235, 0.88)
RIGHT_LABEL = (0.775, 0.12)
BAND_LABEL = (0.393, 0.155)

#: Both labels sit in the band, which the oracle curve crosses at every t, so
#: each is placed in the clear space on one side of it: the region names below
#: the curve, where the band's floor is still zero and the half-guard chord is
#: far above, and the oracle's own name above it, where the band is widest.
ORACLE_LABEL = (0.605, 0.755)


def bounds(t):
    """Lower and upper admissible guard factor, clipped to [0, 1]."""
    lo = np.clip((t - 1.0) / (t + 1.0), 0.0, 1.0)
    hi = np.clip(2.0 * t / (1.0 + t), 0.0, 1.0)
    return lo, hi


def build():
    fig, ax = style.new_figure(width_fraction=WIDTH_FRACTION, aspect=1.45,
                               grid=False)
    t = np.geomspace(T_MIN, T_MAX, 1500)
    lo, hi = bounds(t)

    ax.fill_between(t, hi, 1.0, color=LEFT_COLOR, linewidth=0, zorder=1)
    ax.fill_between(t, 0.0, lo, color=RIGHT_COLOR, linewidth=0, zorder=1)
    ax.fill_between(t, lo, hi, color=BAND_COLOR, linewidth=0, zorder=1)
    # The edges themselves, white over the fills, as in Figures 2.1 and 4.1.
    ax.plot(t, hi, color="white", linewidth=1.8, zorder=5)
    ax.plot(t, lo, color="white", linewidth=1.8, zorder=5)

    # Solid, black and in front, at the weight it carries in Figure 3.2: the
    # oracle is not one curve among the boundaries but the thing the guard is
    # trying to hit, and it should read the same way in both figures.
    ax.plot(t, t / (1.0 + t), color="black", zorder=6, linewidth=3.4)

    # The improvement threshold. Left of it the guard has no lower edge at all
    # -- the tuned member is behind the classical mean there, so no amount of
    # guarding can be too much -- and right of it the window closes from below.
    ax.axvline(1.0, color=style.GREY_LINE, linewidth=1.3, zorder=3,
               linestyle=(0, (1, 2.4)))

    # The half-guard chord and the window (1/3, 3) it buys.
    ax.axhline(G_HALF, color=style.GREY_LINE, linewidth=1.3, zorder=3,
               linestyle=(0, (1, 2.4)))
    ax.plot([1.0 / 3.0, 3.0], [G_HALF, G_HALF], marker="o", linestyle="none",
            markersize=8, color=style.ACC, markeredgecolor="white",
            markeredgewidth=1.3, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(mticker.FixedLocator(T_TICKS))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(T_LABELS))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=18, va="center")

    G = r"\mathrm{G}^{\mathrm{C}}(\alpha)"
    ax.text(LEFT_LABEL[0], LEFT_LABEL[1], r"$\mathrm{C}\succ %s$" % G,
            transform=ax.transAxes, color="white", fontsize=17,
            ha="center", va="center", zorder=7)
    ax.text(RIGHT_LABEL[0], RIGHT_LABEL[1],
            r"$\mathrm{P}_{\!\!+\!}\succ %s$" % G,
            transform=ax.transAxes, color="white", fontsize=17,
            ha="center", va="center", zorder=7)
    ax.text(BAND_LABEL[0], BAND_LABEL[1],
            "$%s\\succ$\n$\\mathrm{C},\\;\\mathrm{P}_{\\!\\!+\\!}$" % G,
            transform=ax.transAxes, color="white", fontsize=14,
            ha="center", va="center", linespacing=1.25, zorder=7)
    ax.text(ORACLE_LABEL[0], ORACLE_LABEL[1], r"$\alpha^{\star}$",
            transform=ax.transAxes,
            color="black", fontsize=20, ha="center", va="center",
            zorder=7)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    style.save(build(), NAME)
