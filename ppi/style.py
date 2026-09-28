"""
Shared style for every figure in the dissertation.

Everything the figures have in common lives here, so that changing it once
changes the whole paper:

* the colour scheme, taken verbatim from the LaTeX preamble of the paper
* Computer Modern fonts, matching the body text of the thesis
* a white background and no figure title -- titles are set in LaTeX
* deliberately large axis and tick labels, so that they stay legible after
  LaTeX scales a figure down into a 0.49\\textwidth subfigure
* the output path

Nothing here is specific to a single figure.
"""

import logging
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# The Computer Modern fonts bundled with matplotlib carry nonsensical creation
# dates in their head table, and fontTools says so on every embed. Harmless.
logging.getLogger("fontTools").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
# Verbatim from settings.tex, so that an estimator printed in colour in the
# text (\textcolor{ppp}{...}, Table A.1) carries that same colour in the plots.

CLA = "#8B1A1A"  # \cla   classical mean                        C
PPI = "#1F3FD0"  # \ppi   PPI at a fixed lambda_0               P_0
PPP = "#1E7B1E"  # \ppp   PPI++                                 P_+
DFX = "#F5A623"  # \dfx   Double-PPI at a fixed nu_0            D_0
DOL = "#9B59B6"  # \dol   Double-PPI at the tuned nu-hat        D_nu-hat
DSH = "#D6217F"  # \dsh   Double-PPI at the shrunken u*nu-hat    D_u nu-hat
DBL = "#E6007E"  # \dsh2  both orders shrunken                   S(D_u nu-hat)
ACC = "#B47100"  # \acc   accent / brown

# settings.tex defines a second set of names for four of the same values.
GENC, A3C, GAUC, SIMC = PPP, ACC, CLA, PPI

#: Legend labels, in the notation of notation.tex.
ESTIMATOR_LABEL = {
    "S0": r"$\mathrm{G}^{\mathrm{C}}_{0}$",
    # Section 5.1's two parametrisations carry the subscript of the notation,
    # "l" and "c" for liberal and conservative; the (h, kappa) behind each is
    # fixed once in the text. Bare, the guarded weight is P_+ by convention.
    "SEl": r"$\mathrm{G}^{\mathrm{C}}_{+,\mathrm{l}}$",
    "SEc": r"$\mathrm{G}^{\mathrm{C}}_{+,\mathrm{c}}$",
    "SEb": r"$\mathrm{G}^{\mathrm{C}}_{+}(4,\frac{3}{2})$",
    "SEa": r"$\mathrm{G}^{\mathrm{C}}_{+}(\frac{3}{2},\frac{1}{2})$",
    "SEg": r"$\mathrm{G}^{\mathrm{C}}_{+}(\frac{1}{2},\frac{1}{2})$",
    "C": r"$\mathrm{C}$",
    "P0": r"$\mathrm{P}_0$",
    "P+": r"$\mathrm{P}_{\!\!+\!}$",
    "D0": r"$\mathrm{D}_0$",
    "Dnu": r"$\mathrm{D}_{\!\!+\!}$",
    "S(P+)": r"$\mathrm{S}(\mathrm{P}_{\!\!+\!})$",
    "S(D0)": r"$\mathrm{S}(\mathrm{D}_0)$",
    "S(Dnu)": r"$\mathrm{G}^{\mathrm{C}}_{0}[\mathrm{D}_{\!\!+\!}]$",
}

#: Every weight of Table A.1, keyed by the symbol the thesis uses. The
#: shrunken ones are added below as the 55% tints, as the table colours them.
ESTIMATOR_COLOR = {
    "C": CLA,
    "P0": PPI,
    "P+": PPP,
    "D0": DFX,
    "Dnu": DOL,
    # Section 4 renames the tuned second-order weight D_+ and adds the
    # second-order shrunken one. The old key stays as the alias it always was.
    "D+": DOL,
    "Duv": DSH,
}

#: Cross-fit PPI++ is not in the table above; it is the brown accent.
CFP = ACC
SSP = PPP

#: Neutral greys for reference lines, annotations and axes.
#: Annotation red. Deliberately not CLA: a box drawn round a region is not a
#: member and must not be mistaken for the classical mean.
HIGHLIGHT = "#E8112D"

GREY_LINE = "#8A8A8A"
GREY_TEXT = "#4D4D4D"

#: xcolor mixes a shrunken weight at 55%, e.g. \textcolor{ppp!55}{S(P_+)}.
SHRUNKEN_MIX = 55


def tint(color, percent=SHRUNKEN_MIX):
    """Reproduce the xcolor expression ``color!percent`` (mixed with white).

    ``tint(PPP)`` is therefore exactly the ``ppp!55`` used for the shrunken
    weight S(P_+) in Table A.1.
    """
    r, g, b = mcolors.to_rgb(color)
    w = percent / 100.0
    return (r * w + (1 - w), g * w + (1 - w), b * w + (1 - w))


def sequential_colors(k, cmap="viridis", lo=0.05, hi=0.88):
    """`k` colours along a colormap, for an ordered family of curves.

    The ends are trimmed: the pale extreme of most colormaps disappears on
    white paper. Perceptually uniform and legible in greyscale, unlike the
    estimator palette above, which encodes identity rather than order.
    """
    return mpl.colormaps[cmap](np.linspace(lo, hi, k))


def discrete_colorbar(fig, ax, values, colors, label, ticks=None,
                      labelpad=18, fmt="%g"):
    """A colour key for an ordered family drawn with `sequential_colors`.

    Every band carries its own label, centred on it. The bar is built in index
    space rather than in the values themselves, so a family whose levels are
    not evenly spaced -- kappa runs 0.1, 0.2, 0.5, 1, 1.5, 2.25, 3 -- still
    gets bands of equal height with the right number on each. Reading a level
    off the boundary between two bands, which an evenly spaced bar invites, is
    then impossible by construction.

    `ticks` is kept for callers that want only some of the bands labelled.
    """
    values = np.asarray(values, dtype=float)
    index = np.arange(len(values), dtype=float)

    mappable = mpl.cm.ScalarMappable(
        norm=mpl.colors.BoundaryNorm(np.append(index - 0.5, index[-1] + 0.5),
                                     len(values)),
        cmap=mpl.colors.ListedColormap(colors),
    )
    bar = fig.colorbar(mappable, ax=ax, pad=0.03)
    if ticks is None:
        shown = index
        names = [fmt % v for v in values]
    else:
        keep = [int(np.argmin(np.abs(values - t))) for t in ticks]
        shown = index[keep]
        names = [fmt % values[i] for i in keep]
    bar.set_ticks(shown)
    bar.set_ticklabels(names)
    bar.set_label(label, rotation=0, labelpad=labelpad, va="center")
    bar.outline.set_linewidth(0)
    bar.ax.tick_params(length=0, pad=7)
    return bar


for _base in ("P+", "D0", "Dnu", "D+", "Duv"):
    ESTIMATOR_COLOR["S(%s)" % _base] = tint(ESTIMATOR_COLOR[_base])


# Section 5 names its shrunken P_+ rules directly. They are all the same
# tint, and are told apart by line style where a figure shows more than one:
# SEb is the selected member, SEa the earlier mean-optimal one it replaced.
for _name in ("S0", "SEb", "SEa", "SEg"):
    ESTIMATOR_COLOR[_name] = tint(ESTIMATOR_COLOR["P+"])

# Section 5's real-data figures shrink three different weights, so each
# shrunken series is the tint of its own base. The labels are written out
# rather than assembled, so that the maths stays inside one pair of $.
for _base in ("P+", "D0", "Dnu"):
    ESTIMATOR_COLOR["SE_" + _base] = tint(ESTIMATOR_COLOR[_base])

# Section 4, the ten members of Table A.1.
ESTIMATOR_LABEL["D+"] = r"$\mathrm{D}_{\!\!+\!}$"
ESTIMATOR_LABEL["Duv"] = r"$\mathrm{G}^{\mathrm{P}_{\!\!+\!}}_{0}$"
ESTIMATOR_LABEL["S(D+)"] = r"$\mathrm{S}(\mathrm{D}_{\!\!+\!})$"
ESTIMATOR_LABEL["S(Duv)"] = r"$\mathrm{G}^{\mathrm{C},\mathrm{P}_{\!\!+\!}}_{0}$"

# Chapter 3's guard notation: the superscript says what the guard protects
# against, the subscript 0 that the factor is fixed at one half.
ESTIMATOR_LABEL["S(P+)"] = r"$\mathrm{G}^{\mathrm{C}}_{0}[\mathrm{P}_{\!\!+\!}]$"
ESTIMATOR_LABEL["S(D0)"] = r"$\mathrm{G}^{\mathrm{C}}_{0}[\mathrm{D}_{0}]$"
ESTIMATOR_LABEL["S(D+)"] = r"$\mathrm{G}^{\mathrm{C}}_{0}[\mathrm{D}_{\!\!+\!}]$"

ESTIMATOR_LABEL["SEb_Both"] = r"$\mathrm{G}^{\mathrm{C},\mathrm{P}_{\!\!+\!}}_{+}(4,\frac{3}{2})$"
ESTIMATOR_LABEL["S0_Both"] = r"$\mathrm{G}^{\mathrm{C},\mathrm{P}_{\!\!+\!}}_{0}(\frac{1}{2},\frac{1}{2})$"
ESTIMATOR_LABEL["SEb_Gnu"] = r"$\mathrm{G}^{\mathrm{P}_{\!\!+\!}}_{+}(4,\frac{3}{2})$"
ESTIMATOR_LABEL["S0_Gnu"] = r"$\mathrm{G}^{\mathrm{P}_{\!\!+\!}}_{0}(\frac{1}{2})$"
ESTIMATOR_LABEL["SEb_D0"] = r"$\mathrm{G}^{\mathrm{C}}_{+}[\mathrm{D}_0]$"
ESTIMATOR_LABEL["SEb_Dnu"] = r"$\mathrm{G}^{\mathrm{C}}_{+}[\mathrm{D}_{\!\!+\!}]$"
ESTIMATOR_LABEL["S0_P+"] = r"$\mathrm{G}^{\mathrm{C}}_{0}(\frac{1}{2})$"
ESTIMATOR_LABEL["SEb_P+"] = r"$\mathrm{G}^{\mathrm{C}}_{+}(4,\frac{3}{2})$"
ESTIMATOR_LABEL["SEg_P+"] = r"$\mathrm{S}_{\mathrm{E}}(\mathrm{P}_{\!\!+\!})$, Gauss"
ESTIMATOR_LABEL["SEa_P+"] = r"$\mathrm{S}_{\mathrm{E}}(\frac{3}{2},\frac{1}{2})(\mathrm{P}_{\!\!+\!})$"
ESTIMATOR_LABEL["SE_P+"] = r"$\mathrm{S}_{\mathrm{E}}(\mathrm{P}_{\!\!+\!})$"
ESTIMATOR_LABEL["SE_D0"] = r"$\mathrm{S}_{\mathrm{E}}(\mathrm{D}_0)$"
ESTIMATOR_LABEL["SE_Dnu"] = r"$\mathrm{S}_{\mathrm{E}}(\mathrm{D}_{\widehat{\nu}})$"


# ---------------------------------------------------------------------------
# Page geometry
# ---------------------------------------------------------------------------
# a4paper with margin=3cm  ->  \textwidth = 21cm - 6cm = 15cm.
TEXTWIDTH_IN = 15.0 / 2.54
SUBFIG_FRACTION = 0.49  # the width of a side-by-side subfigure in the thesis

#: Every figure is drawn oversized and then shrunk by LaTeX. Building all of
#: them at the *same* reduction is what keeps type equally large across the
#: paper, whether a figure sits alone or in a pair of subfigures. The canvas
#: size therefore follows from the width the figure will occupy in the text.
RENDER_SCALE = 0.53
ASPECT = 4.0 / 3.0

#: Canvas point sizes. Multiply by RENDER_SCALE for the size on the page:
#: axis labels land at about 14pt against the 12pt body text.
_FONT_SIZE = 21
_LABEL_SIZE = 27
_TICK_SIZE = 23
_LEGEND_SIZE = 21


def figsize_for(width_fraction=SUBFIG_FRACTION, aspect=ASPECT):
    """Canvas size for a figure included at ``width_fraction`` of \textwidth.

    Pass the same number to ``\includegraphics[width=...\textwidth]`` and the
    labels come out the size intended, at any width.
    """
    width_in = width_fraction * TEXTWIDTH_IN / RENDER_SCALE
    return (width_in, width_in / aspect)


#: The default slot: one of two subfigures side by side.
FIGSIZE = figsize_for()


def use_thesis_style():
    """Apply the global rcParams. Called by :func:`new_figure`."""
    mpl.rcParams.update({
        # Computer Modern, to match the thesis body font. cmr10 has no proper
        # minus sign, hence unicode_minus=False.
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,

        "font.size": _FONT_SIZE,
        "axes.labelsize": _LABEL_SIZE,
        "xtick.labelsize": _TICK_SIZE,
        "ytick.labelsize": _TICK_SIZE,
        "legend.fontsize": _LEGEND_SIZE,

        # White background everywhere, also in the saved file.
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,

        # No frame on the top and right; heavier lines to survive shrinking.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.4,
        "lines.linewidth": 2.8,
        "lines.solid_capstyle": "round",
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.9,
        "grid.linestyle": (0, (1, 3)),

        "legend.frameon": False,
        "legend.handlelength": 1.9,
        "legend.borderaxespad": 0.2,

        # Vector output with embedded TrueType, no rasterisation.
        "pdf.fonttype": 42,
        "pdf.compression": 6,
        "figure.dpi": 110,
    })


def new_figure(width_fraction=SUBFIG_FRACTION, aspect=ASPECT, figsize=None,
               grid=True):
    """Return a ``(fig, ax)`` pair in the thesis style.

    `width_fraction` is the fraction of \textwidth the figure will be given in
    LaTeX; it only sets the canvas size, so that the type is the same size on
    the page for every figure. Never sets a title: captions and subcaptions
    are written in LaTeX.
    """
    use_thesis_style()
    if figsize is None:
        figsize = figsize_for(width_fraction, aspect)
    fig, ax = plt.subplots(figsize=figsize)
    if grid:
        ax.grid(True, which="major", zorder=0)
        # Minor gridlines carry the logarithmic zone of a broken axis, where
        # major ticks are far apart by construction.
        ax.grid(True, which="minor", zorder=0, linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)
    return fig, ax


# ---------------------------------------------------------------------------
# The piecewise sample-size axis
# ---------------------------------------------------------------------------
# The interesting behaviour sits at small n, but the large-n limit still has to
# be visible. The axis is therefore linear up to `break_at` and logarithmic
# above it, joined continuously, so that one decade above the break occupies
# `decade_width` axis units against the 100 units spent below it.

N_BREAK = 100.0
N_DECADE_WIDTH = 60.0

#: No curve in the thesis is drawn below this sample size. Six is the smallest
#: n at which the cross-fit half fold is not degenerate, so anything under it
#: would be an artefact of the formulae rather than a statement about them.
#: This is a floor on the *data*, not on the axis.
N_MIN = 6

#: Axes always begin at zero, whatever the data does. The empty stretch from 0
#: to N_MIN is deliberate: it shows that the small-n regime was left out.
AXIS_MIN = 0


def piecewise_transform(break_at, decade_width):
    """Forward and inverse of the linear-then-logarithmic scale."""

    def forward(v):
        v = np.asarray(v, dtype=float)
        safe = np.maximum(v, 1e-12)
        return np.where(v <= break_at,
                        v,
                        break_at + decade_width * np.log10(safe / break_at))

    def inverse(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= break_at,
                        y,
                        break_at * 10.0 ** ((y - break_at) / decade_width))

    return forward, inverse


#: One decade above a break takes this fraction of the span below it. The same
#: proportion is used on every broken axis in the thesis.
DECADE_FRACTION = 0.6


def set_piecewise_axis(ax, axis="x", axis_min=AXIS_MIN, break_at=100.0,
                       v_max=1000.0, decade_width=None, ticks=(),
                       minor_ticks=(), mark_break=True):
    """A linear zone up to `break_at`, one logarithmic decade above it.

    Each zone is regular in its own terms, and the faint rule at the join
    tells the reader that the slope change there belongs to the scale.
    """
    if decade_width is None:
        decade_width = DECADE_FRACTION * break_at
    forward, inverse = piecewise_transform(break_at, decade_width)

    if axis == "x":
        ax.set_xscale("function", functions=(forward, inverse))
        set_major, set_labels = ax.set_xticks, ax.set_xticklabels
        set_lim, rule = ax.set_xlim, ax.axvline
    else:
        ax.set_yscale("function", functions=(forward, inverse))
        set_major, set_labels = ax.set_yticks, ax.set_yticklabels
        set_lim, rule = ax.set_ylim, ax.axhline

    shown = [t for t in ticks if axis_min <= t <= v_max]
    set_major(shown)
    set_labels(["%g" % t for t in shown])
    if minor_ticks:
        set_major([t for t in minor_ticks if axis_min <= t <= v_max],
                  minor=True)
    set_lim(axis_min, v_max)

    if mark_break and axis_min < break_at < v_max:
        rule(break_at, color=GREY_LINE, linewidth=1.0,
             linestyle=(0, (4, 3)), alpha=0.55, zorder=1.5)
    return ax


def windowed_transform(lo, hi, compression=0.1):
    """Full resolution inside ``[lo, hi]``, evenly compressed outside it.

    Piecewise linear rather than logarithmic, so that equal steps outside the
    window keep equal spacing on the axis: with `compression` = 0.1 a step of
    ten outside is drawn the same length as a step of one inside, and the tick
    labels of both zones then sit on one common rhythm.
    """

    def forward(v):
        v = np.asarray(v, dtype=float)
        return (np.clip(v, lo, hi)
                + compression * (np.maximum(v - hi, 0.0)
                                 - np.maximum(lo - v, 0.0)))

    def inverse(y):
        y = np.asarray(y, dtype=float)
        return (np.clip(y, lo, hi)
                + (np.maximum(y - hi, 0.0)
                   - np.maximum(lo - y, 0.0)) / compression)

    return forward, inverse


def set_windowed_axis(ax, lo, hi, v_min, v_max, axis="y", compression=0.1,
                      ticks=(), minor_ticks=(), mark_window=True):
    """Give one band of an axis its full resolution and compress the rest.

    Use when the differences worth reading live in a narrow band but the same
    curves also take values far outside it, so that a plain linear axis would
    flatten the band and a clipped one would hide the excursions.
    """
    forward, inverse = windowed_transform(lo, hi, compression)

    if axis == "x":
        ax.set_xscale("function", functions=(forward, inverse))
        set_major, set_labels = ax.set_xticks, ax.set_xticklabels
        set_lim, rule = ax.set_xlim, ax.axvline
    else:
        ax.set_yscale("function", functions=(forward, inverse))
        set_major, set_labels = ax.set_yticks, ax.set_yticklabels
        set_lim, rule = ax.set_ylim, ax.axhline

    shown = [t for t in ticks if v_min <= t <= v_max]
    set_major(shown)
    set_labels(["%g" % t for t in shown])
    if minor_ticks:
        set_major([t for t in minor_ticks if v_min <= t <= v_max], minor=True)
    set_lim(v_min, v_max)

    if mark_window:
        for edge in (lo, hi):
            if v_min < edge < v_max:
                rule(edge, color=GREY_LINE, linewidth=1.0,
                     linestyle=(0, (4, 3)), alpha=0.55, zorder=1.5)
    return ax


def set_piecewise_n_axis(ax, n_max=1000, axis_min=AXIS_MIN, break_at=N_BREAK,
                         decade_width=N_DECADE_WIDTH, ticks=None,
                         minor_ticks=None, mark_break=True):
    """Put a linear-then-compressed sample-size scale on the x axis.

    Linear and evenly spaced from 0 to `break_at`, a single logarithmic decade
    above it. With the defaults, 0 to 100 takes about 63% of the axis and 100
    to 1000 the remaining 37%. `mark_break` draws a faint rule at the join, so that
    the slope change every curve shows there reads as the scale and not as
    the function.

    The axis starts at `axis_min` (zero), while the curves themselves start at
    :data:`N_MIN`; the two are deliberately different.
    """

    # Below the break, linear with a label every 20 and a tick every 10;
    # above it, one decade with the standard subdivision 200, ..., 900.
    if ticks is None:
        ticks = [0, 20, 40, 60, 80, 100, 1000]
    if minor_ticks is None:
        minor_ticks = ([10, 30, 50, 70, 90]
                       + [200, 300, 400, 500, 600, 700, 800, 900])
    return set_piecewise_axis(ax, axis="x", axis_min=axis_min,
                              break_at=break_at, v_max=n_max,
                              decade_width=decade_width, ticks=ticks,
                              minor_ticks=minor_ticks, mark_break=mark_break)


#: Section 5 shows n to 200: full resolution to 50, then evenly compressed at
#: 1/5, so a step of 50 above the break is as long as a step of 10 below it.
#: 1/3, not 1/5: the labels above the break are three digits wide and need
#: more room than the two-digit ones below it, so an equal tick rhythm would
#: run 100 into 150.
CH6_N_BREAK, CH6_N_COMPRESSION = 50.0, 1.0 / 3.0
CH6_N_TICKS = [0, 25, 50, 100, 150, 200]


def set_chapter6_n_axis(ax, n_max=200):
    """The sample-size axis used throughout Section 5."""
    return set_windowed_axis(ax, 0.0, CH6_N_BREAK, 0.0, n_max, axis="x",
                             compression=CH6_N_COMPRESSION, ticks=CH6_N_TICKS)


def set_linear_n_axis(ax, n_max=50, axis_min=AXIS_MIN, step=10, minor_step=5):
    """A plain, evenly spaced sample-size axis, for ranges short enough not to
    need the break of :func:`set_piecewise_n_axis`."""
    ax.set_xlim(axis_min, n_max)
    ax.set_xticks(np.arange(axis_min, n_max + 1, step))
    ax.set_xticks(np.arange(axis_min, n_max + 1, minor_step), minor=True)
    return ax


def set_rho_axis(ax, axis="x", step=0.2, minor_step=0.1):
    """The full correlation range, 0 to 1, evenly divided."""
    ticks = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    minor = np.round(np.arange(0.0, 1.0 + 1e-9, minor_step), 10)
    if axis == "x":
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(ticks)
        ax.set_xticks(minor, minor=True)
    else:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(ticks)
        ax.set_yticks(minor, minor=True)
    return ax


def n_grid(n_min=None, n_max=1000, points=4000):
    """A grid of sample sizes, dense where the piecewise axis is dense.

    Never goes below :data:`N_MIN`, so no figure can accidentally show the
    degenerate small-n regime.
    """
    n_min = N_MIN if n_min is None else max(float(n_min), float(N_MIN))
    return np.unique(np.concatenate([
        np.linspace(n_min, min(N_BREAK, n_max), points // 2),
        np.geomspace(n_min, n_max, points // 2),
    ]))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
PROJECT_DIR = _HERE.parent

#: Where the vector figures land. Set PPI_FIGURE_DIR to point this at the
#: graphics folder of a LaTeX project instead.
GRAPHICS_DIR = Path(os.environ.get("PPI_FIGURE_DIR",
                                   PROJECT_DIR / "output" / "pdf"))
#: Raster copies for quick inspection; the paper never uses these.
PREVIEW_DIR = Path(os.environ.get("PPI_PREVIEW_DIR",
                                  PROJECT_DIR / "output" / "png"))


def save(fig, name, preview=True):
    """Write ``name.pdf`` to :data:`GRAPHICS_DIR` and return its path.

    A PNG copy goes to :data:`PREVIEW_DIR` unless `preview` is false.
    """
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = GRAPHICS_DIR / (name + ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)

    if preview:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW_DIR / (name + ".png"), dpi=150,
                    bbox_inches="tight", pad_inches=0.02)

    print("wrote", pdf_path)
    return pdf_path

# The two rules of Section 5.1 on each weight they can guard. The word says
# how much of the tuned weight the rule lets through, the colour which member
# is being guarded, and the dash pattern repeats the word for a reader who is
# looking at a curve rather than at the key. "(2)" marks the two-level guard,
# which applies the same parametrisation at both orders.
# In Section 5 the superscript is always C -- every one of these guards the
# weight against the classical mean -- and the bracket names the weight that
# is being guarded. The two-level rule appears here as the guard of D_+,
# which is what it is at the first order.
_PP = r"\mathrm{P}_{\!\!+\!}"
_DP = r"\mathrm{D}_{\!\!+\!}"
for _r, _sub in (("SEl", "l"), ("SEc", "c")):
    def _g(inner, _sub=_sub):
        return r"$\mathrm{G}^{\mathrm{C}}_{+,\mathrm{%s}}[%s]$" % (_sub, inner)
    ESTIMATOR_LABEL["%s_P+" % _r] = _g(_PP)
    ESTIMATOR_LABEL["%s_D0" % _r] = _g(r"\mathrm{D}_0")
    ESTIMATOR_LABEL["%s_Dnu" % _r] = _g(_DP)
    ESTIMATOR_LABEL["%s_Both" % _r] = _g(_DP)
    ESTIMATOR_LABEL["%s_Gnu" % _r] = (
        r"$\mathrm{G}^{%s}_{+,\mathrm{%s}}$" % (_PP, _sub))
