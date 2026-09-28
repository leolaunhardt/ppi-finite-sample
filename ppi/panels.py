"""
Panel builders shared by figures that differ only in their parameters.

Keeping the drawing in one place is what makes a pair of subfigures, or two
figures about two families, actually look like one system. Each figure script
still owns the parameters it varies, which is what one wants to edit.
"""

import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import style, theory

#: The rules are functions of t-hat_lambda; this is the range shown.
T_MAX = 10.0

#: The half-value point of the direct plug-in rule, shared by both families
#: as the natural fixed h.
DP_H, DP_KAPPA = 2.0, 1.0




#: The t axis of the guard window, so the two figures of Section 3.2 are read
#: on one scale. Logarithmic because the family is a shift in log t: the
#: half-value point enters alpha_E only through log h, so equal ratios in h
#: are equal distances here and the curves come out as one shape translated.
EXPFAM_T = (0.05, 20.0)
EXPFAM_TICKS = [0.05, 1.0 / 3.0, 1.0, 3.0, 10.0, 20.0]
EXPFAM_LABELS = [r"$\frac{1}{20}$", r"$\frac{1}{3}$", "1", "3", "10", "20"]


def exponential_family_panel(rules, width_fraction=0.6, oracle=True,
                             marks=False, legend_at=(0.03, 0.995)):
    """The selected members of the exponential family against the oracle.

    `rules` is a sequence of ``(h, kappa, label, key)``. The calibration
    family itself is not drawn: once the two parametrisations are fixed, the
    curves that were needed to choose them only compete with them for the
    reader's attention, and what matters is how each stands to the oracle it
    approximates.
    """
    fig, ax = style.new_figure(width_fraction=width_fraction)
    t = np.geomspace(EXPFAM_T[0], EXPFAM_T[1], 3000)

    ax.axhline(theory.HALF_VALUE, color=style.GREY_LINE, linewidth=1.2,
               linestyle=(0, (4, 3)), zorder=2)

    for h, kappa, label, key in rules:
        color, dash = SHRINK_STYLE[key]
        ax.plot(t, theory.s_exponential_family(t, h=h, kappa=kappa),
                color=color, linestyle=dash, linewidth=3.2, label=label,
                zorder=4)
        if marks:
            # Every member of the family passes through one half at t = h,
            # whatever the curvature; the marker is what makes h readable.
            ax.plot([h], [theory.HALF_VALUE], marker="o", linestyle="none",
                    markersize=9, color=color, markeredgecolor="white",
                    markeredgewidth=1.4, zorder=5)

    if oracle:
        ax.plot(t, theory.oracle_shrinkage(t), color="black", linewidth=3.4,
                zorder=6, label=r"$\alpha^\star$")

    ax.set_xscale("log")
    ax.set_xlim(*EXPFAM_T)
    ax.xaxis.set_major_locator(mticker.FixedLocator(EXPFAM_TICKS))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(EXPFAM_LABELS))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    style.set_rho_axis(ax, axis="y")
    ax.set_xlabel(r"$\widehat{t}$")
    ax.set_ylabel(r"$\widehat{\alpha}$", rotation=0, labelpad=20,
                  va="center")

    fig.tight_layout()
    ax.legend(loc="upper left", bbox_to_anchor=legend_at,
              frameon=False, handlelength=2.0, borderaxespad=0.0)
    return fig


def shrinkage_family_panel(rule, param, values, fixed, label,
                           width_fraction=style.SUBFIG_FRACTION,
                           t_max=T_MAX, oracle=True, highlight=None,
                           ylabel=True, legend_cols=3):
    """One panel of the exponential-family figure.

    `rule` is called as ``rule(t, h=..., kappa=...)``. `param` names the
    argument that varies over `values`, `fixed` gives the other one, and
    `label` formats the legend entry, e.g. ``"$h=%g$"``.

    `oracle` draws the oracle rule alpha* = t/(1+t) of Proposition 3.2 as the
    benchmark every family member is trying to approximate. Its own half-value
    point is h = 1. It is drawn black, heavy and in front: it is not one curve
    among several but the thing they are all approximating, and the family
    should read as a set of approximations to it rather than as a bundle in
    which it happens to lie.
    """
    fig, ax = style.new_figure(width_fraction=width_fraction)
    colors = style.sequential_colors(len(values))
    t = np.linspace(0.0, t_max, 2001)

    # Every rule of Section 3.2 passes through one half at t = h.
    ax.axhline(theory.HALF_VALUE, color=style.GREY_LINE, linewidth=1.2,
               linestyle=(0, (4, 3)), zorder=2)

    for color, value in zip(colors, values):
        kwargs = dict(fixed)
        kwargs[param] = value
        ax.plot(t, rule(t, **kwargs), color=color, label=label % value,
                linewidth=1.7, zorder=3)

    for kwargs, hl_label, key in (highlight or ()):
        # The two parametrisations of Section 5.1, in the colour and dash they
        # carry there, and heavier than the family they are drawn from.
        color, dash = SHRINK_STYLE[key]
        ax.plot(t, rule(t, **kwargs), color=color, linestyle=dash,
                linewidth=3.6, label=hl_label, zorder=4)

    if oracle:
        ax.plot(t, theory.oracle_shrinkage(t), color="black",
                linewidth=3.4, zorder=6, label=r"$\alpha^\star$")

    ax.set_xlim(0.0, t_max)
    ax.set_xticks(np.arange(0, t_max + 1, 2))
    ax.set_xticks(np.arange(0, t_max + 1, 1), minor=True)
    style.set_rho_axis(ax, axis="y")   # 0 to 1 in fifths, the range of alpha

    ax.set_xlabel(r"$\widehat{t}$")
    if ylabel:
        ax.set_ylabel(r"$\widehat{\alpha}$", rotation=0, labelpad=20,
                      va="center")

    # Lay the axes out first: a legend outside them would otherwise make
    # tight_layout squeeze the plot into a strip.
    fig.tight_layout()

    # Above the panel rather than below it, so the two subfigures of the pair
    # carry their keys at the top where the eye meets them first.
    # savefig(bbox_inches="tight") takes it into the saved area.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=legend_cols, frameon=False,
              handlelength=1.7, columnspacing=1.3,
              fontsize=style._LEGEND_SIZE * 0.80)
    return fig


# ---------------------------------------------------------------------------
# Section 4: which member of the family is best, and what the others cost
# ---------------------------------------------------------------------------

from matplotlib.colors import ListedColormap       # noqa: E402
from matplotlib.patches import Patch, Rectangle    # noqa: E402

from . import unified                              # noqa: E402

#: The correlation is evaluated just off both ends. At rho = 0 the
#: parameterisation sigma_y = lambda*/rho is singular, though every relative
#: risk still has a limit; at rho = 1 the Double-PPI risks are exactly zero,
#: so excess over the best member would divide by zero.
RHO_EPS = 2e-3

#: Excess risk is shown linearly to here, then one logarithmic decade.
EXCESS_BREAK, EXCESS_MAX = 10.0, 100.0
EXCESS_TICKS = [0, 2, 4, 6, 8, 10, 100]
EXCESS_MINOR = [1, 3, 5, 7, 9, 20, 30, 40, 50, 60, 70, 80, 90]

#: The strip under zero that carries the colour of whichever member is best.
BAR_BOTTOM = -1.3


def _legend_below(ax, ncol, handles=None, labels=None, pad=0.05):
    """Put the key under the axis label, whatever the figure size.

    The drop is measured from the label itself rather than fixed, so a wide
    panel does not end up with a wide gap.
    """
    ax.figure.canvas.draw()
    inv = ax.transAxes.inverted()
    y = ax.xaxis.label.get_window_extent().transformed(inv).y0 - pad
    kwargs = {} if handles is None else dict(handles=handles, labels=labels)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              columnspacing=1.3, handlelength=1.5, **kwargs)


def switch_points(names, n, s=0.5, lam_star=0.05, points=20001, law=None):
    """The correlations at which the winner changes below the belief line.

    Under the belief line no risk in Table A.1 involves lambda*, so the map is
    a set of vertical rules there and reading them off the argmin is exact up
    to the grid. This is used instead of the closed-form cutoffs so that the
    rules a figure draws are the ones its own colours change at.
    """
    rho = np.linspace(RHO_EPS, 1.0 - RHO_EPS, points)
    lam = np.full_like(rho, lam_star)
    if law is None:
        risk = unified.risk_table(names, n, rho, lam, s=s)
    else:
        from . import laws
        risk = laws.risk_table(law, names, n, rho, lam, s=s)
    best = np.argmin(risk, axis=0)
    cut = np.flatnonzero(np.diff(best)) + 1
    return rho[cut]


def _legend_right_grid(fig, ax, names, ncol=2, pad=0.035, col_width=0.23,
                       swatch=(0.17, 0.048), gap=0.25, absent=()):
    """A key to the right of the panel, on a grid rather than in a legend.

    Matplotlib packs the columns of a legend independently, so rows drift
    apart as soon as the labels differ in height -- and here they do: a label
    whose superscript carries its own subscript, such as G^{P_+}_0, is forty
    per cent taller than a bare C. Placing the swatches on a fixed grid keeps
    the pairs level with one another, which is the whole point of ordering
    them in pairs.

    Swatches and labels are placed in figure coordinates rather than inside an
    invisible axis. An axis would contribute its whole reserved rectangle to
    the saved bounding box, so a short one-column key would carry a blank
    strip on its right and push the map off centre on the page; drawn this way
    the figure is trimmed to the labels themselves, whatever `ncol` is.

    `col_width`, `swatch` and `gap` are all fractions of one column, so the
    swatches come out the same size whether the key has one column or two.

    Members named in `absent` are bracketed. They are in the key because the
    reader is meant to see that they exist, but they win nowhere on the plane,
    so their colour never appears in the map and the reader would otherwise
    hunt for it.
    """
    box = ax.get_position()
    col = col_width * box.width
    rows = int(np.ceil(len(names) / ncol))

    for i, nm in enumerate(names):
        c, row = divmod(i, rows)
        x = box.x1 + pad + c * col
        y = box.y0 + box.height * (1.0 - (row + 0.5) / rows)
        fig.add_artist(Rectangle((x, y - swatch[1] * box.height / 2.0),
                                 swatch[0] * col, swatch[1] * box.height,
                                 facecolor=style.ESTIMATOR_COLOR[nm],
                                 edgecolor="none", transform=fig.transFigure))
        label = style.ESTIMATOR_LABEL[nm]
        if nm in absent:
            label = "$(" + label.strip("$") + ")$"
        fig.text(x + gap * col, y, label,
                 va="center", ha="left", fontsize=style._LEGEND_SIZE)


def regions_panel(names, n, s_shrink=0.5, width_fraction=0.8, lam_max=2.0,
                  vertical_thresholds=True, grid_points=(900, 700),
                  legend_cols=5, rules=None, legend="below",
                  legend_all=False, t_rule=False, outline=None,
                  t_axis=None, law=None):
    """Figure 4.1 or 4.2: the best member over (rho, lambda*).

    Each point is coloured by the estimator with the smallest relative mean
    squared error there. Only the members that actually win somewhere are
    named in the key.
    """
    fig, ax = style.new_figure(width_fraction=width_fraction, grid=False)
    K_n = unified.K(n)

    rho = np.linspace(RHO_EPS, 1.0 - RHO_EPS, grid_points[0])
    lam = np.linspace(1e-3, lam_max, grid_points[1])
    R, L = np.meshgrid(rho, lam)

    if law is None:
        risk = unified.risk_table(names, n, R, L, s=s_shrink, K_n=K_n)
    else:
        from . import laws
        risk = laws.risk_table(law, names, n, R, L, s=s_shrink)
    best = np.argmin(risk, axis=0)

    colors = [style.ESTIMATOR_COLOR[nm] for nm in names]
    ax.pcolormesh(rho, lam, best, cmap=ListedColormap(colors),
                  vmin=-0.5, vmax=len(names) - 0.5, shading="nearest",
                  rasterized=True)

    guide = dict(color="white", linewidth=1.6, linestyle=(0, (4, 3)),
                 zorder=3)
    ax.axhline(0.5, **guide)                       # P0 > C, at lambda* = 1/2
    if outline:
        # Where `outline` would win if it were the only correction on offer,
        # that is among the members that carry no guard at all. In the full
        # map this region has been taken over, so it is shown as a contour
        # rather than a colour.
        plain = [nm for nm in unified.UNSHRUNKEN_ALL if nm not in ("Duv",)]
        if law is None:
            plain_risk = unified.risk_table(plain, n, R, L, s=s_shrink, K_n=K_n)
        else:
            from . import laws
            plain_risk = laws.risk_table(law, plain, n, R, L, s=s_shrink)
        won = np.argmin(plain_risk, axis=0)
        ax.contour(rho, lam, (won == plain.index(outline)).astype(float),
                   levels=[0.5], colors=[style.ESTIMATOR_COLOR[outline]],
                   linewidths=2.4, linestyles="dotted", zorder=6)

    if t_rule:
        # t = 1 is the improvement condition of Corollary 2.1; under
        # normality it is rho = 1/sqrt(n-2), and it is drawn heavier than the
        # four cutoffs because it is the one threshold that predates this
        # chapter.
        ax.axvline(np.sqrt(1.0 / (n - 2.0)), color="white", linewidth=3.4,
                   zorder=4)
    if rules is not None:
        for r in rules:
            ax.axvline(float(r), **guide)
    elif vertical_thresholds:
        ax.axvline(np.sqrt(1.0 / (n - 2.0)), **guide)          # P+ > C
        ax.axvline(np.sqrt(K_n / (1.0 + K_n)), **guide)        # Dnu > P+

    if t_axis:
        _t_top_axis(ax, n, t_axis)

    style.set_rho_axis(ax, axis="x")
    ax.set_ylim(0.0, lam_max)
    ax.set_yticks(np.arange(0.0, lam_max + 1e-9, 0.5))

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\lambda^\star$", rotation=0, labelpad=28, va="center")

    fig.tight_layout()
    won = {names[i] for i in set(best.ravel().tolist())}
    if legend_all:
        # Every member named, whether or not it wins anywhere: the reader is
        # meant to see that two of them never do. Those two are bracketed,
        # since their colour is nowhere in the map to be matched against.
        present = [nm for nm in unified.LEGEND_ACROSS if nm in names]
    else:
        present = [nm for nm in unified.LEGEND_DOWN if nm in won]
        present += [nm for nm in names if nm in won and nm not in present]
    absent = [nm for nm in present if nm not in won]
    handles = [Patch(facecolor=style.ESTIMATOR_COLOR[nm]) for nm in present]
    labels = [style.ESTIMATOR_LABEL[nm] for nm in present]
    if legend == "below":
        _legend_below(ax, legend_cols, handles, labels)
    elif legend == "right":
        _legend_right_grid(fig, ax, present, ncol=legend_cols, absent=absent)
    return fig




# ---------------------------------------------------------------------------
# Section 4: the ten-member picture
# ---------------------------------------------------------------------------
# The excess panels are drawn on one logarithmic decade ladder rather than the
# broken axis of the earlier version. A break resolves the small differences
# but puts a kink in every curve that is an artefact of the axis; five decades
# of log do the same job and leave the curves honest. Everything below the
# floor is within a hundredth of a percent of the best member and is read as
# "as good as the winner".
EXCESS_LOG_LIM = (0.01, 1000.0)
EXCESS_LOG_TICKS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

#: The strip that carries the winner's colour, as a fraction of the decade
#: ladder measured from the floor.
EXCESS_STRIP = 0.055

#: The excess panels sit at 0.49\textwidth in the thesis but are drawn on a
#: wider canvas, so that their type comes out smaller than nominal and matches
#: the region map they sit under, which is given 0.8\textwidth.
EXCESS_CANVAS = 0.592


def _legend_right(ax, handles, labels, pad=0.02, ncol=1, **kwargs):
    """Put the key outside the axis, to the right."""
    return ax.legend(handles=handles, labels=labels, loc="upper left",
                     bbox_to_anchor=(1.0 + pad, 1.0), ncol=ncol, frameon=False,
                     handlelength=1.4, borderaxespad=0.0, **kwargs)


#: The members are drawn at one weight: they are the estimators one can
#: actually run, and the panel is about them.
MAIN_WIDTH = 3.0

#: The optimum is not a member but the bound they are measured against, so it
#: is black, as the oracle is in Figures 3.1 and 3.2, and the heaviest line in
#: the panel: everything else is read against it.
STAR_WIDTH = 4.6

#: Where the magnified corner sits inside the panel and how big it is, in axis
#: fractions. Bottom right: the cloud is bottom left and the lines leave
#: towards the top. Only the left edge, the width and the height are fixed;
#: the bottom is set just above the horizontal baseline, which sits at a
#: different height in the two orders, so that the box never cuts it.
ZOOM_BOX = (0.735, 0.225, 0.225)
ZOOM_GAP = 0.055

#: The marginal of the predictor, hung under the frame: its position and
#: height as fractions of the panel. Drawn downwards, so that the axis it
#: hangs from is the axis the points would have stood on.
MARGINAL_BOX = (0.0, -0.30, 1.0, 0.22)
#: Height of the marginal strip, in inches, so that it is the same
#: thickness on the page in every panel.
MARGINAL_HEIGHT = 0.95
MARGINAL_BINS = 80


def slope_intuition_panel(x, y, xlabel, ylabel, members, star, frame,
                          highlight=None, zoom=None, marginal=False,
                          marginal_label="trees", width_fraction=0.8,
                          legend_where="right", legend_cols=1,
                          marker_labels=True, aspect=1.0,
                          pad_height=None, pad_width=None,
                          legend_blanks=0):
    """A weight of the thesis as the slope of a line, on real data.

    Both orders have the same shape. At the first, the weight is the slope of
    Y on F; at the second, of F~Y~ on F~^2. In each case the believed weight is
    one, so the belief is the diagonal of a frame whose two axes carry the same
    scale, and the question is how far the truth is from it.

    `members` is a sequence of ``(slope, colour, label)`` and `star` the pair
    ``(slope, label)`` of the optimum, which is drawn black and heaviest: it is
    the bound the members are measured against, not one of them. Every line is
    drawn through the population centroid, so that the panel compares slopes
    and nothing else; the estimated ones are computed from `highlight`.

    Writing the estimator as the mean of Z = Y - w(F - mu_f), its variance is
    Var(Y - wF)/n up to the finite-population factors, and an intercept does
    not enter a variance -- so each line's contribution to the risk is the
    vertical scatter of the points about it, and the optimum is the slope that
    minimises that scatter.
    """
    x_bar, y_bar = float(x.mean()), float(y.mean())
    # The data frame is square, so the canvas has to leave room for the
    # marginal strip and the legend on top, or the square axes shrinks and
    # the saved figure carries dead white beside it.
    fig, ax = style.new_figure(width_fraction=width_fraction, aspect=aspect)
    if not marginal:
        ax.scatter(x, y, s=7, color=style.GREY_LINE, alpha=0.28, linewidths=0,
                   zorder=2)
    # The mean of the predictor, which is what the correction pivots on: the
    # estimator moves the labelled mean by w times the distance from it.
    ax.axvline(x_bar, color=style.GREY_LINE, linewidth=1.3, zorder=2,
               linestyle=(0, (1, 2.4)))

    grid = np.array([0.0, frame])
    lines = []
    for slope, colour, label in members:
        lines.append((slope, colour, MAIN_WIDTH, 5))
        ax.plot(grid, y_bar + slope * (grid - x_bar), color=colour,
                linewidth=MAIN_WIDTH, zorder=5, label=label)
    lines.append((star[0], "black", STAR_WIDTH, 6))
    ax.plot(grid, y_bar + star[0] * (grid - x_bar), color="black",
            linewidth=STAR_WIDTH, zorder=6, label=star[1])

    # Empty rows at the foot of the key, so that a panel with fewer members
    # can be given a key as tall as its neighbour's and the two frames start
    # at the same depth.
    for _ in range(legend_blanks):
        ax.plot([], [], color="none", label=" ")

    if highlight is not None:
        ax.scatter(x[highlight], y[highlight], s=80, color="black", zorder=7,
                   edgecolor="white", linewidths=1.3,
                   label=("%d felled trees" % len(highlight))
                   if marker_labels else "_nolegend_")
    ax.scatter([x_bar], [y_bar], s=140, marker="P", color=style.GREY_TEXT,
               zorder=8, edgecolor="white", linewidths=1.3,
               label="centroid" if marker_labels else "_nolegend_")

    axm = None
    if zoom:
        bottom = y_bar / frame + ZOOM_GAP
        axin = ax.inset_axes((ZOOM_BOX[0], bottom, ZOOM_BOX[1], ZOOM_BOX[2]))
        if not marginal:
            axin.scatter(x, y, s=7, color=style.GREY_LINE, alpha=0.28,
                         linewidths=0, zorder=2)
        axin.axvline(x_bar, color=style.GREY_LINE, linewidth=1.1, zorder=2,
                     linestyle=(0, (1, 2.4)))
        for slope, colour, lw, zo in lines:
            axin.plot(grid, y_bar + slope * (grid - x_bar), color=colour,
                      linewidth=lw, zorder=zo)
        if highlight is not None:
            axin.scatter(x[highlight], y[highlight], s=34, color="black",
                         zorder=7, edgecolor="white", linewidths=0.9)
        axin.scatter([x_bar], [y_bar], s=80, marker="P",
                     color=style.GREY_TEXT, zorder=8, edgecolor="white",
                     linewidths=1.0)
        win = tuple(zoom)
        axin.set_xlim(win[0], win[1])
        axin.set_ylim(win[2], win[3])
        axin.set_xticks(win[:2])
        axin.set_yticks(win[2:])
        axin.tick_params(labelsize=style._TICK_SIZE * 0.42, length=2, pad=1.5,
                         colors=style.GREY_TEXT)
        axin.set_facecolor("white")
        axin.grid(False)
        for spine in axin.spines.values():
            # The thesis style hides the top and right spines everywhere; an
            # inset needs all four, or it does not read as a window.
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_color(style.GREY_TEXT)
        ax.indicate_inset_zoom(axin, edgecolor=style.GREY_LINE, linewidth=0.9,
                               alpha=1.0)

    # The same scale on both axes, so the believed weight is the diagonal and
    # the slopes can be compared by eye.
    ax.set_xlim(0.0, frame)
    ax.set_ylim(0.0, frame)
    ax.set_aspect("equal", adjustable="box")
    # ... and the same ticks on both, so that a step is the same length in
    # either direction and the diagonal really looks like one.
    ticks = [t for t in mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
             .tick_values(0.0, frame) if -1e-9 <= t <= frame + 1e-9]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if axm is not None:
        axm.set_xticks(ticks)
    if not marginal:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if marginal:
        # The label sits between the frame and the strip, under the tick
        # numbers it belongs to, so the strip reads as an appendix to the
        # frame rather than as a second panel with its own axis. The band it
        # needs is measured in inches, not in axes fractions, so that panels
        # of different size put the same gap around the same type, and it is
        # reserved before the layout runs rather than taken back afterwards.
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv = fig.dpi_scale_trans.inverted()
        pad = 0.10
        ticks = ax.get_xticklabels()
        tick_h = (max(t.get_window_extent(rend).height for t in ticks)
                  / fig.dpi) if ticks else 0.0
        lab = ax.text(0.5, 0.0, xlabel, transform=ax.transAxes, ha="center",
                      va="top", fontsize=style._LABEL_SIZE)
        fig.canvas.draw()
        lab_h = lab.get_window_extent(rend).transformed(inv).height

        band = tick_h + 2 * pad + lab_h + MARGINAL_HEIGHT
        fig.tight_layout(rect=(0.0, band / fig.get_size_inches()[1], 1.0, 1.0))
        fig.canvas.draw()

        h_ax = ax.get_window_extent().transformed(inv).height
        lab.set_y(-(tick_h + pad) / h_ax)
        y_top = -(tick_h + 2 * pad + lab_h) / h_ax

        # The population as the empirical distribution of the predictor,
        # hanging under the frame rather than as a cloud inside it: what the
        # panel has to show about it is the skewness, not its shape in two
        # dimensions, and a cloud on a linear scale collapses into a blot.
        axm = ax.inset_axes((0.0, y_top - MARGINAL_HEIGHT / h_ax, 1.0,
                             MARGINAL_HEIGHT / h_ax))
        axm.hist(x, bins=np.linspace(0.0, frame, MARGINAL_BINS + 1),
                 color=style.GREY_LINE, linewidth=0, bottom=0.8)
        axm.set_yscale("log")
        axm.set_ylim(0.8, float(len(x)))
        axm.axvline(x_bar, color=style.GREY_LINE, linewidth=1.3, zorder=3,
                    linestyle=(0, (1, 2.4)))
        axm.set_xlim(0.0, frame)
        axm.invert_yaxis()
        decades = [d for d in (1, 10, 100, 1000) if d <= len(x)]
        axm.set_yticks(decades)
        axm.set_yticklabels([str(d) for d in decades])
        axm.yaxis.set_minor_formatter(mticker.NullFormatter())
        axm.set_ylabel(marginal_label, fontsize=style._TICK_SIZE * 0.5,
                       labelpad=2)
        axm.tick_params(axis="y", labelsize=style._TICK_SIZE * 0.5, length=2,
                        pad=1.5)
        # The numbers are already above, on the frame.
        axm.tick_params(axis="x", labelbottom=False, length=0)
        axm.grid(False)
        for side in ("top", "right"):
            axm.spines[side].set_visible(False)
    if legend_where == "top":
        # Just above the frame, but flush with the left edge of the canvas
        # rather than centred on it: an equal aspect leaves the square frame
        # narrower than the canvas, and a key centred on it would hang out on
        # the right, which the saved figure would carry as dead white.
        fig.canvas.draw()
        box = ax.get_position()
        ax.legend(loc="lower left", bbox_to_anchor=(-box.x0 / box.width, 1.02),
                  ncol=legend_cols, frameon=False,
                  fontsize=style._LEGEND_SIZE * 0.85, handlelength=1.8,
                  columnspacing=1.5)
        if pad_width:
            # Same idea sideways, so that two panels can be given identical
            # boxes in LaTeX and therefore come out at identical scale.
            fig.canvas.draw()
            w_fig = fig.get_size_inches()[0]
            tb = fig.get_tightbbox(fig.canvas.get_renderer())
            if pad_width > tb.width:
                sp = fig.text((tb.x1 + pad_width - tb.width) / w_fig, 0.5,
                              ".", alpha=0.0, fontsize=6, ha="left",
                              va="center")
                for _ in range(6):
                    fig.canvas.draw()
                    tb = fig.get_tightbbox(fig.canvas.get_renderer())
                    gap = pad_width - tb.width
                    if abs(gap) < 1e-3:
                        break
                    sp.set_x(sp.get_position()[0] + gap / w_fig)

        if pad_height:
            # Panels that stand side by side have to be the same height on the
            # page, and their keys need not have the same number of rows. An
            # invisible spacer stretches the saved box to the agreed height;
            # its own extent is not known in advance, so it is nudged until
            # the box measures what was asked for.
            fig.canvas.draw()
            h_fig = fig.get_size_inches()[1]
            tb = fig.get_tightbbox(fig.canvas.get_renderer())
            if pad_height > tb.height:
                # Above, not below: the panels are top-aligned in LaTeX, and
                # what has to line up is the frames, not the keys. A panel
                # whose key has one row less gets that row back as blank space
                # over it, so both frames start at the same depth.
                spacer = fig.text(0.5, (tb.y1 + pad_height - tb.height) / h_fig,
                                  ".", alpha=0.0, fontsize=6, va="bottom",
                                  ha="center")
                for _ in range(6):
                    fig.canvas.draw()
                    tb = fig.get_tightbbox(fig.canvas.get_renderer())
                    gap = pad_height - tb.height
                    if abs(gap) < 1e-3:
                        break
                    spacer.set_y(spacer.get_position()[1] + gap / h_fig)
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,
                  frameon=False, fontsize=style._LEGEND_SIZE * 0.85,
                  handlelength=1.8, labelspacing=0.8, borderaxespad=0.0)
    return fig


def _t_top_axis(ax, n, t_axis):
    """A second scale on top, in t_C, over an axis that carries rho.

    Under normality t_C = (n-1) rho^2/(1+rho^2), so the thresholds of
    Table A.3, which are statements about t_C, can be read off directly
    instead of being converted in the reader's head. The label carries the
    subscript: t on its own is the generic statistic of Figure 3.1, which
    stands for either order.
    """
    axt = ax.twiny()
    axt.set_xlim(*ax.get_xlim())
    pos = [np.sqrt(t / (n - 1.0 - t)) for t in t_axis]
    axt.xaxis.set_major_locator(mticker.FixedLocator(pos))
    axt.xaxis.set_major_formatter(
        mticker.FixedFormatter(["%g" % t for t in t_axis]))
    axt.set_xlabel(r"$t_{\mathrm{C}}$", labelpad=8)
    axt.tick_params(length=3)
    return axt


def excess_log_panel(names, n, lam_star, s=0.5, ylabel=True,
                     width_fraction=EXCESS_CANVAS, points=1600,
                     ylim=EXCESS_LOG_LIM, law=None, t_axis=None):
    """What the wrong member costs, on a single logarithmic scale.

    Excess relative mean squared error over the best member at each rho. The
    winner is exactly zero and therefore off a logarithmic axis altogether;
    the strip along the floor carries its colour instead, which is what makes
    the sequence of optima readable without tracing a curve.
    """
    fig, ax = style.new_figure(width_fraction=width_fraction)
    rho = np.linspace(RHO_EPS, 1.0 - RHO_EPS, points)
    if law is None:
        best, excess = unified.best_and_excess(names, n, rho, lam_star, s=s)
    else:
        from . import laws
        best, excess = laws.best_and_excess(law, names, n, rho, lam_star, s=s)

    lo, hi = ylim
    top = lo * (hi / lo) ** EXCESS_STRIP
    for i, nm in enumerate(names):
        ax.fill_between(rho, lo, top, where=(best == i),
                        color=style.ESTIMATOR_COLOR[nm], linewidth=0,
                        zorder=4, step="mid")
    for i, nm in enumerate(names):
        ax.plot(rho, np.maximum(excess[i], lo * 0.5),
                color=style.ESTIMATOR_COLOR[nm], zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_yticks(EXCESS_LOG_TICKS)
    ax.set_yticklabels([("%g" % t) for t in EXCESS_LOG_TICKS])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    style.set_rho_axis(ax, axis="x")
    ax.set_xlabel(r"$\rho$")
    if ylabel:
        # Only the leftmost panel of a row carries the title; the ticks stay
        # on every panel, since the ladder is what makes them comparable.
        ax.set_ylabel("excess MSE, %")
    if t_axis:
        _t_top_axis(ax, n, t_axis)
    fig.tight_layout()
    return fig


def members_legend(names=None, cols=5, width_fraction=1.0, height=1.05):
    """A standalone key for the ten members, to sit above a figure.

    Two rows: every shrunken weight sits directly under the weight it
    shrinks, and the columns run from no second order at all on the left to
    both orders shrunken on the right.
    """
    # A matplotlib legend fills column by column, so the pairs are passed
    # in reading-down order and come out as five columns of two.
    names = unified.LEGEND_DOWN if names is None else names
    fig, ax = style.new_figure(width_fraction=width_fraction, grid=False,
                               figsize=(style.figsize_for(width_fraction)[0], height))
    ax.axis("off")
    ax.legend(handles=[Patch(facecolor=style.ESTIMATOR_COLOR[nm]) for nm in names],
              labels=[style.ESTIMATOR_LABEL[nm] for nm in names],
              loc="center", ncol=cols, frameon=False, handlelength=1.5,
              columnspacing=1.4)
    fig.tight_layout()
    return fig
# ---------------------------------------------------------------------------
# Delta against each parameter in turn
# ---------------------------------------------------------------------------

#: The point the three panels pivot around. Whichever parameter is on the x
#: axis varies; the other two are held here.
PIVOT = dict(n=30, rho=0.9, lam_star=0.75)

LAMBDA_MAX = 2.0
DELTA_N_MAX = 1000
# ---------------------------------------------------------------------------
# Gain over PPI++, in percentage points of Delta
# ---------------------------------------------------------------------------

#: Line styles for the second factor of the design, in the order given.
GAIN_STYLES = ["-", (0, (5, 2.5)), (0, (1, 2)), (0, (7, 2, 1, 2))]
# ---------------------------------------------------------------------------
# Section 5: the shrinkage rules
# ---------------------------------------------------------------------------

#: P_+ keeps its own colour; its shrunken versions are the 55% tint of it, as
#: Table A.1 colours them, and are told apart by line style.
SHRINK_STYLE = {
    "P+": (style.PPP, "-"),
    "S0": (style.tint(style.PPP), "-"),
    # The two parametrisations of Section 5.1, in the dash patterns they
    # carry there (GUARD_DASH, defined further down with the atlas styles):
    # the longer the stroke, the more of the tuned weight the rule lets
    # through.
    "SEl": (style.tint(style.PPP), (0, (6, 2))),
    "SEc": (style.tint(style.PPP), (0, (1, 2))),
    "SEb": (style.tint(style.PPP), (0, (5, 2.5))),
    "SEa": (style.tint(style.PPP), (0, (1, 2))),
    "SEg": (style.tint(style.PPP), (0, (1, 2))),
}
SHRINK_ORDER = ["P+", "S0", "SEl", "SEc"]


def shrink_curves(ax, x, delta, names=None, zorder=3):
    """Draw the Section 5 rules, plus the classical mean as the zero line."""
    ax.axhline(0.0, color=style.CLA, zorder=2, label="C")
    for name in (names or SHRINK_ORDER):
        color, dash = SHRINK_STYLE[name]
        ax.plot(x, delta[name], color=color, linestyle=dash,
                label=style.ESTIMATOR_LABEL[name], zorder=zorder)
    return ax


# ---------------------------------------------------------------------------
# Section 5: the binary belief map
# ---------------------------------------------------------------------------

#: The belief region of Section 5.1: a predictor the analyst suspects is
#: mediocre but cannot rule out as useless, lambda* between 0.05 and 0.47.
BELIEF_PPV = (0.45, 0.63)
BELIEF_NPV = (0.60, 0.84)
BELIEF_PREVALENCE = 0.2
BELIEF_N = 20
BELIEF_GRID = 220
# ---------------------------------------------------------------------------
# Section 5: mean gain against worst case
# ---------------------------------------------------------------------------

#: The members named on the plot. Anything else would crowd it. The label
#: offsets differ between the two panels because their horizontal scales do.
TRADEOFF_MARKS = [
    ("s0.50", r"fixed $s=\frac{1}{2}$"),
    ("h1.50_k0.50", r"$\mathrm{S}_{\mathrm{E}}(\frac{3}{2},\frac{1}{2})$"),
    ("h6.00_k0.75", r"$\mathrm{S}_{\mathrm{E}}(6,\frac{3}{4})$"),
]
# ---------------------------------------------------------------------------
# Section 5: the benchmark datasets
# ---------------------------------------------------------------------------

#: Solid for the unshrunken weights, dashed for their shrunken versions.
REAL_ORDER = ["P0", "P+", "D0", "Dnu", "SE_P+", "SE_D0", "SE_Dnu"]
REAL_DASH = (0, (5, 2.5))


#: The vertical range is taken from these alone. The weights that estimate nu
#: overshoot by hundreds of percent at the smallest n, and letting that set the
#: scale flattens everything worth reading; they simply leave the frame.
REAL_REFERENCE = ["C", "P0", "P+", "SE_P+"]
REAL_PAD = 0.10


#: Against PPI++ the classical mean is no longer a competitor but the thing
#: both sides beat, by hundreds of percent, so it is dropped; the comparison
#: is among the prediction-powered weights. Shrinkage is left out too --- it
#: is the subject of the modified datasets that follow, and here it would
#: double the number of curves without changing which weight wins. D_nu-hat
#: sets no scale and leaves the frame at the smallest n.
REAL_ORDER_PPLUS = ["P0", "D0", "Dnu"]
REAL_REFERENCE_PPLUS = ["P0", "D0"]

#: Sample sizes of the re-based panel. Both axes are logarithmic there: the
#: range now runs to 400, and under Delta_{P+} every curve decays towards
#: zero, so equal ratios rather than equal differences are what the eye
#: should be able to compare at either end.
REAL_N_MAX_PPLUS = 1000
REAL_X_TICKS = [6, 10, 30, 100, 300, 1000]

#: Below this magnitude the axis is linear, above it logarithmic in each
#: direction. Under Delta_{P+} every curve decays monotonically towards zero
#: over roughly two orders of magnitude, so a linear axis would resolve the
#: small sample sizes and flatten the large ones into a single line.
REAL_LINTHRESH = 1.0
REAL_SYMLOG_TICKS = [-300, -100, -30, -10, -3, -1, 0, 1, 3, 10, 30, 100]
# ---------------------------------------------------------------------------
# Shared legends
# ---------------------------------------------------------------------------

def legend_block(entries, ncol=1, width_fraction=0.32, handlelength=1.9,
                 fontsize=None):
    """A key shaped like one cell of a subfigure grid.

    The three-by-three grids of Section 5 carry eight populations, which
    leaves the lower right cell free. A key drawn on the same canvas as a
    panel fills it exactly and keeps the grid square, where a strip below the
    panels would add a row of its own.
    """
    fig, ax = style.new_figure(width_fraction=width_fraction, grid=False)
    ax.set_axis_off()
    handles = []
    for entry in entries:
        kind = entry[0]
        if kind == "blank":
            # An empty cell, so that a row-wise grouping can be laid out in a
            # key that matplotlib fills column by column.
            handles.append(Line2D([], [], color="none", label=" "))
            continue
        kind, color, dash, label = entry[:4]
        lw = entry[4] if len(entry) > 4 else None
        if kind == "patch":
            handles.append(Patch(facecolor=color, label=label))
        else:
            handles.append(Line2D([], [], color=color, linestyle=dash,
                                  label=label, linewidth=lw))
    ax.legend(handles=handles, labels=[h.get_label() for h in handles],
              loc="center", ncol=ncol, frameon=False,
              handlelength=handlelength, labelspacing=0.9,
              fontsize=fontsize or style._LEGEND_SIZE)
    fig.tight_layout()
    return fig


def legend_strip(entries, ncol, width_fraction=0.95, handlelength=1.6):
    """A figure containing nothing but a key.

    A grid of subfigures that all share a key looks lopsided if one of them
    carries it. This puts the key in its own strip, to be included as a row
    between the panels and the caption, so every panel is the same size.

    `entries` are ``(kind, colour, linestyle, label)``, with kind "line" or
    "patch", and optionally a fifth field giving the line width, for a key
    whose panels do not draw every series at the same weight.
    """
    style.use_thesis_style()
    width = width_fraction * style.TEXTWIDTH_IN / style.RENDER_SCALE
    fig = plt.figure(figsize=(width, 0.8))

    handles = []
    for entry in entries:
        kind = entry[0]
        if kind == "blank":
            # An empty cell, so that a row-wise grouping can be laid out in a
            # key that matplotlib fills column by column.
            handles.append(Line2D([], [], color="none", label=" "))
            continue
        kind, color, dash, label = entry[:4]
        width = entry[4] if len(entry) > 4 else None
        if kind == "patch":
            handles.append(Patch(facecolor=color, label=label))
        else:
            handles.append(Line2D([], [], color=color, linestyle=dash,
                                  label=label, linewidth=width))
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="center", ncol=ncol, frameon=False,
               handlelength=handlelength, columnspacing=1.4)
    return fig


# ---------------------------------------------------------------------------
# The relative gain of estimated Double-PPI, averaged over n
# ---------------------------------------------------------------------------

#: The sample sizes averaged over in both panels.
GAIN_N_GRID = np.arange(10, 301, 10)
GAIN_RHO_MAX = 0.999


def _gain_pct(n, rho):
    """MSE(D_nu)/MSE(P_+) - 1 in percent, in closed form."""
    from . import unified

    K_n = unified.K_of_n(n)
    r_d = unified.relative_mse("Dnu", n=n, rho=rho, lam_star=0.75, K_n=K_n)
    r_p = unified.relative_mse("P+", n=n, rho=rho, lam_star=0.75, K_n=K_n)
    return 100.0 * (r_d / r_p - 1.0)
# ---------------------------------------------------------------------------
# Section 5: bootstrap against Gaussian plug-in
# ---------------------------------------------------------------------------

VARHAT_GAUSS_DASH = "-"
VARHAT_BOOT_DASH = (0, (5, 2.5))
#: Figure 5.4 is a single wide panel, so its canvas is flattened until it is
#: no taller than one of the half-width panels it shares the chapter with:
#: 0.8 * W / VARHAT_ASPECT <= 0.49 * W / ASPECT.
VARHAT_ASPECT = 2.25
# ---------------------------------------------------------------------------
# Section 5: selecting (h, kappa) for the exponential family
# ---------------------------------------------------------------------------

#: Curvatures drawn as curves. The full grid is finer; these are enough to
#: show the shape, and kappa = 0.1 is kept because its curve degenerating to a
#: point is the visual statement that the family collapses to the fixed rule.
HYPER_KAPPAS = [0.1, 0.2, 0.5, 1.0, 1.5, 2.25, 3.0]

#: Axis windows. The candidates that are more conservative than the fixed rule
#: simply leave the frame.
HYPER_WINDOW = {"belief": (-4.0, -1.30, 11.0), "all": (-30.4, -22.0, 12.6)}


def hyper_tradeoff_panel(name, width_fraction=style.SUBFIG_FRACTION,
                         mark=None, offsets=None, ylabel=True,
                         colorbar=True, source=None, window=None):
    """Mean against worst case over one regime, for the exponential family.

    Horizontal: the mean Delta_C over the regime's cells, so further left is a
    better average. Vertical: the worst case over the same cells, so lower is
    safer. Each curvature traces a curve as h runs over its grid; the heavy
    line is the non-dominated front, the candidates an analyst would actually
    choose along.
    """
    from . import hyper as _default
    hyper = source or _default

    mean, worst, pp_m, pp_w, hf_m, hf_w = hyper.criteria(name)
    fig, ax = style.new_figure(width_fraction=width_fraction)

    colors = style.sequential_colors(len(HYPER_KAPPAS))
    for color, k in zip(colors, HYPER_KAPPAS):
        j = int(np.argmin(np.abs(hyper.K_GRID - k)))
        ax.plot(mean[:, j], worst[:, j], color=color, linewidth=2.0, zorder=3)

    fm, fw, _, _ = hyper.pareto_front(name)
    ax.plot(fm, fw, color=style.GREY_TEXT, linewidth=3.2, zorder=4)

    ax.scatter([pp_m], [pp_w], s=110, color=style.PPP, zorder=6)
    ax.scatter([hf_m], [hf_w], s=110, color=style.ACC, zorder=6)

    marks = list(mark or [])
    offsets = offsets or [(10, 10)] * len(marks)
    for (h, k, label), offset in zip(marks, offsets):
        i = int(np.argmin(np.abs(hyper.H_GRID - h)))
        j = int(np.argmin(np.abs(hyper.K_GRID - k)))
        ax.scatter([mean[i, j]], [worst[i, j]], s=60, facecolor="white",
                   edgecolor=style.GREY_TEXT, linewidth=1.6, zorder=7)
        ax.annotate(label, (mean[i, j], worst[i, j]),
                    textcoords="offset points", xytext=offset, fontsize=15,
                    color=style.GREY_TEXT, zorder=7)

    lo, hi, top = window or HYPER_WINDOW[name]
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.5, top)
    ax.set_xlabel(r"mean $\Delta_{\mathrm{C}}$ in %")
    if ylabel:
        ax.set_ylabel("worst case\n" + r"$\Delta_{\mathrm{C}}$ in %")

    fig.tight_layout()
    if colorbar:
        style.discrete_colorbar(fig, ax, HYPER_KAPPAS, colors, r"$\kappa$")
    return fig


def hyper_breakdown_panel(name, what, width_fraction=style.SUBFIG_FRACTION,
                          y_top=None, mark=None, offsets=None,
                          colorbar=True, source=None):
    """Mean or worst case against h, one curve per curvature.

    This is the trade-off panel opened out along its own parameter: it shows
    where the family stops paying, which the (mean, worst) plane hides because
    h is only implicit there.
    """
    from . import hyper as _default
    hyper = source or _default

    mean, worst, pp_m, pp_w, hf_m, hf_w = hyper.criteria(name)
    surface = mean if what == "mean" else worst
    reference = pp_m if what == "mean" else pp_w

    fig, ax = style.new_figure(width_fraction=width_fraction)
    ax.axhline(reference, color=style.PPP, linewidth=2.0, zorder=2,
               linestyle=(0, (5, 2.5)))

    colors = style.sequential_colors(len(HYPER_KAPPAS))
    for color, k in zip(colors, HYPER_KAPPAS):
        j = int(np.argmin(np.abs(hyper.K_GRID - k)))
        ax.plot(hyper.H_GRID, surface[:, j], color=color, linewidth=2.0,
                zorder=3)

    marks = list(mark or [])
    offsets = offsets or [(10, 10)] * len(marks)
    for (h, k, hl_label), offset in zip(marks, offsets):
        i = int(np.argmin(np.abs(hyper.H_GRID - h)))
        j = int(np.argmin(np.abs(hyper.K_GRID - k)))
        ax.scatter([hyper.H_GRID[i]], [surface[i, j]], s=60,
                   facecolor="white", edgecolor=style.GREY_TEXT,
                   linewidth=1.6, zorder=7)
        ax.annotate(hl_label, (hyper.H_GRID[i], surface[i, j]),
                    textcoords="offset points", xytext=offset, fontsize=15,
                    color=style.GREY_TEXT, zorder=7)

    ax.set_xscale("log")
    ax.set_xlim(hyper.H_GRID[0], hyper.H_GRID[-1])
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
    if y_top is not None:
        ax.set_ylim(top=y_top)
    if what == "mean":
        # Room under the lowest curve for the annotation of the member that
        # sits there; without it the label falls off the frame.
        ax.set_ylim(bottom=float(surface.min()) - 1.9)

    ax.set_xlabel("$h$")
    ax.set_ylabel(("mean " if what == "mean" else "worst case\n")
                  + r"$\Delta_{\mathrm{C}}$ in %")

    fig.tight_layout()
    if colorbar:
        style.discrete_colorbar(fig, ax, HYPER_KAPPAS, colors, r"$\kappa$")
    return fig


# ---------------------------------------------------------------------------
# Section 5.3: the twenty real-data populations
# ---------------------------------------------------------------------------

#: Against PPI++ (parts A and B) only the unshrunken weights are compared;
#: against the classical mean (part C) the two shrinkage rules join them.
ATLAS_ORDER_PPLUS = ["P0", "D0", "Dnu"]

#: Section 5.3, the strong predictors. The first-order guards belong to the
#: weak-predictor corner, where the belief is what is at stake; here the only
#: guarded member worth carrying is the two-level one, and it is carried on
#: every panel so that the same five lines can be read across the grid.
ATLAS_ORDER_PPLUS_G = ["P0", "D0", "Dnu"] + ["SEl_Both", "SEc_Both"]

#: Section 5.3 as printed: PPI at the fixed weight is the baseline, and the
#: classical mean is drawn but kept out of `ATLAS_REFERENCE_P0`, so that it
#: appears wherever it happens to fall in frame without setting the scale.
#: The guard of PPI++ rides with it and stays out of the reference too: on
#: the six well calibrated populations it falls inside the frame anyway, and
#: on the two badly calibrated ones it would undo the zoom on the lower group.
ATLAS_ORDER_P0 = (["P+"] + ["SEl_P+", "SEc_P+"] + ["D0", "Dnu"] + ["SEl_Both", "SEc_Both"] + ["C"])
ATLAS_REFERENCE_P0 = ["P+", "D0", "Dnu"] + ["SEl_Both", "SEc_Both"]

#: Section 5.3, the binary grid: PPI++ as the baseline, with its guard.
ATLAS_ORDER_BINARY = ATLAS_ORDER_PPLUS_G + ["SEl_P+", "SEc_P+"]

#: The shrinkage panel carries P_+ alone: the Double-PPI weights belong to the
#: Double-PPI groups, and in this corner their shrunken versions are governed
#: by the bootstrap's difficulty with the D_0 weight rather than by the rule.
#: Both rules of Section 5.1, against the fixed rule at one half.
#: Two families side by side: P_+ with the first-order rules in green, and
#: the tuned Double-PPI weight with its two-level counterpart in purple.
ATLAS_ORDER_C = (["P+"] + ["SEl_P+", "SEc_P+"] + ["S0_P+", "Dnu"] + ["SEl_Both", "SEc_Both"])

#: Section 5.3d: the same key with PPI at a fixed weight added. On those two
#: populations lambda* is the point of the section, so the estimator that
#: believes lambda_0 = 1 has to be in the picture rather than left out of it.
ATLAS_ORDER_C_PPI = ["P0"] + ATLAS_ORDER_C

#: Section 5.3: guarding the second-order weight. The two guarded versions
#: inherit D_nu-hat's colour, dashed for the estimated rule and dotted for the
#: fixed one, exactly as the first-order rules do on P_+.
#: Shrinkage on both levels at once: the second-order weight towards zero
#: (back to PPI++) and the resulting weight towards zero (back to the
#: classical mean). Estimated and fixed form side by side.
ATLAS_ORDER_GUARD = ["D0", "Dnu"] + ["SEl_Both", "SEc_Both"] + ["S0_Both"]
ATLAS_ORDER_GUARD_C = ["P+", "Dnu"] + ["SEl_Both", "SEc_Both"] + ["S0_Both"]

# Section 5.1's three parametrisations, distinguished by dash pattern rather
# than by colour: the colour is spoken for by the member being guarded, so the
# rhythm of the line has to carry the strength. The pattern is ordered the way
# the rules are -- the longer the stroke, the more of the tuned weight the rule
# lets through -- so a reader can rank three curves without the key:
#
#   liberal       dashes           almost all of the weight survives
#   conservative  dots             the weight is held back hardest
#
# The fixed half-guard is not a member of that family, so it gets a rhythm of
# its own rather than a place on the scale.
GUARD_DASH = {"SEl": (0, (6, 2)),
              "SEc": (0, (1, 2))}
FIXED_DASH = (0, (3, 1, 1, 1, 1, 1))

#: The oracle floor Delta^star. Grey and thin, because it is not a member but
#: the level none of them can pass: the weight lambda* known rather than
#: estimated. Free of n, so it is a horizontal rule.
ORACLE_STYLE = dict(color=style.GREY_TEXT, linewidth=1.5,
                    linestyle=(0, (4, 3)), zorder=2.5)
ORACLE_LABEL = r"$\Delta^\star$"

#: Which member's colour each guarded weight inherits.
_GUARDED_COLOR = {"P+": style.PPP, "D0": style.ESTIMATOR_COLOR["D0"],
                  "Dnu": style.ESTIMATOR_COLOR["Dnu"],
                  "Gnu": style.ESTIMATOR_COLOR["Dnu"],
                  "Both": style.ESTIMATOR_COLOR["Dnu"]}

ATLAS_STYLE = {}
for _r, _dash in GUARD_DASH.items():
    for _w, _col in _GUARDED_COLOR.items():
        ATLAS_STYLE["%s_%s" % (_r, _w)] = (style.tint(_col), _dash)
for _w, _col in _GUARDED_COLOR.items():
    ATLAS_STYLE["S0_%s" % _w] = (style.tint(_col), FIXED_DASH)

ATLAS_X_TICKS = [10, 20, 50, 100, 300, 1000]

#: These figures are grids of four to six panels, so the house sizes leave the
#: curves very little room. Axis titles and ticks are set smaller here, and the
#: vertical title is carried by the leftmost panel of each row alone.
ATLAS_LABEL_SIZE = 21
ATLAS_TICK_SIZE = 18
#: Three panels to a row leave no room for five labels.
ATLAS_X_TICKS_NARROW = [10, 50, 100, 300, 1000]

#: House weight is 2.8. The strong-predictor grid puts five curves on a third
#: of the text width, where they read as a band rather than as five lines, and
#: asks for this instead. Passed explicitly, so no other panel is affected.
ATLAS_LINEWIDTH = 2.3
ATLAS_LINTHRESH = 1.0
#: Decades only. A finer ladder collides at this panel size once the frame
#: spans several orders of magnitude.
ATLAS_TICKS = [-100, -10, -1, 0, 1, 10, 100, 1000]

#: Delta can never fall below -100, and above this the curve has left every
#: comparison worth making, so it leaves the frame instead of setting the
#: scale.
ATLAS_FLOOR, ATLAS_CAP = -100.0, 2000.0

#: The vertical title, one per admissible baseline.
ATLAS_DELTA_LABEL = {
    "C": r"$\Delta_{\mathrm{C}}$ in %",
    "P0": r"$\Delta_{\mathrm{P}_0}$ in %",
    "P+": r"$\Delta_{\mathrm{P}_{\!+}}$ in %",
}


def _log_ladder(lo, hi):
    """The 1-2-5 ladder between two magnitudes, as a list."""
    out, step = [], 1.0
    while step <= hi * 10.0:
        for m in (1.0, 2.0, 5.0):
            v = m * step
            if lo <= v <= hi:
                out.append(v)
        step *= 10.0
    while len(out) > 5:
        # These panels are a third of the text width; six labels on a
        # vertical axis collide. Thinning keeps the ends and the ratio.
        out = out[::2]
    return out


def atlas_panel(name, baseline="P+", width_fraction=style.SUBFIG_FRACTION,
                reference=None, ylim=None, yscale="symlog", guard=False,
                ylabel=True, compact=True, order=None, xscale="log",
                x_min=None, linewidth=None, oracle=True):
    """Delta over n on one population of Section 5.3.

    Both axes are logarithmic: n spans a factor of thirty and the risks span
    orders of magnitude, so equal ratios rather than equal differences are
    what should be comparable across the frame.
    """
    from . import populations

    if baseline == "C":
        order = order or (ATLAS_ORDER_GUARD_C if guard else ATLAS_ORDER_C)
    else:
        order = order or (ATLAS_ORDER_GUARD if guard else ATLAS_ORDER_PPLUS)
    # The plain sweep runs to n = 1000 but carries no guarded weight, so it
    # can only be used when none is asked for.
    plain = (baseline == "P+" and not guard
             and not any(k in ATLAS_STYLE for k in order))
    d = populations.dataset(name, plain=plain)
    fig, ax = style.new_figure(width_fraction=width_fraction)

    if baseline == "C":
        # The sweep stores Delta against the classical mean, so that one is
        # already in the right units; any other baseline is a change of
        # reference on the risk ratio.
        series = {k: d[k] for k in order}
    else:
        base = 1.0 + d[baseline] / 100.0
        series = {k: 100.0 * ((1.0 + d[k] / 100.0) / base - 1.0)
                  for k in order}
    zero_color = style.ESTIMATOR_COLOR.get(baseline, style.CLA)
    zero_label = style.ESTIMATOR_LABEL.get(baseline, "C")

    ax.axhline(0.0, color=zero_color, zorder=2, label=zero_label,
               linewidth=linewidth)
    for key in order:
        color, dash = ATLAS_STYLE.get(
            key, (style.ESTIMATOR_COLOR.get(key, style.GREY_TEXT), "-"))
        ax.plot(d["n"], series[key], color=color, linestyle=dash,
                label=style.ESTIMATOR_LABEL.get(key, key), zorder=3,
                linewidth=linewidth)

    # The oracle weight lambda* is exact and free of n, so the floor is a
    # horizontal rule. It is drawn under the members and kept in the frame's
    # reference set, since a floor that falls outside the window is worse than
    # none at all: the reader would take the lowest curve for the limit.
    floor = None
    if oracle and baseline in ("C", "P0"):
        floor = populations.oracle_floor(name, baseline)
        ax.axhline(floor, label=ORACLE_LABEL, **ORACLE_STYLE)

    if xscale == "linear":
        # Evenly spaced in n itself, and beginning at zero whatever the grid
        # does, as every sample-size axis in the thesis does. The grid is
        # geometric, so the points crowd the left of such a frame.
        left = float(style.AXIS_MIN if x_min is None else x_min)
        right = float(d["n"].max())
        ax.set_xlim(left, right)
        # Round numbers, plus the left edge itself: a frame that starts at the
        # first grid point rather than at zero has to say where it starts.
        auto = mticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(
            left, right)
        keep = [t for t in auto
                if left + 0.10 * (right - left) < t <= right]
        ax.xaxis.set_major_locator(mticker.FixedLocator([left] + keep))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    else:
        ax.set_xscale("log")
        ax.set_xlim(float(d["n"].min()), float(d["n"].max()))
        ticks = (ATLAS_X_TICKS_NARROW if width_fraction < 0.4
                 else ATLAS_X_TICKS)
        shown_x = [t for t in ticks if d["n"].min() <= t <= d["n"].max()]
        if width_fraction < 0.4 and len(shown_x) > 3:
            # Three panels to a row: decades only, and the grid's own end
            # where it stops short of the next decade, so the right edge of
            # a truncated grid is labelled too.
            shown_x = [10, 100, 1000]
            top = int(d["n"].max())
            shown_x = [t for t in shown_x if t <= top]
            if top >= 2 * shown_x[-1]:
                shown_x.append(top)
        if len(shown_x) < 2:
            # A population small enough that the standard ladder gives one
            # tick or none: label the ends of its own range instead.
            lo, hi = int(d["n"].min()), int(d["n"].max())
            shown_x = sorted({lo, int(round((lo * hi) ** 0.5)), hi})
        ax.set_xticks(shown_x)
        ax.set_xticklabels(["%g" % t for t in shown_x])
    # Over a single decade matplotlib labels the minor ticks as well, which
    # collides with the labels above.
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    if ylim is None:
        parts = [series[k] for k in (reference or order)]
        if floor is not None:
            parts.append(np.array([floor]))
        ref = np.concatenate(parts)
        lo, hi = float(ref.min()), float(ref.max())
        if yscale == "linear":
            # The baseline sits at zero by construction and needs no room of
            # its own; the frame follows the curves that carry the comparison.
            pad = 0.08 * max(hi - lo, 1e-6)
            ylim = (lo - pad, hi + pad)
        elif lo > 0 or hi < 0:
            # Every curve on one side of the baseline. Forcing zero into the
            # frame would then spend most of it on a stretch no curve visits
            # and compress the comparison into a sliver, so the frame stays on
            # the side the curves are on and the scale is a plain logarithm of
            # the distance from the baseline.
            ylim = ((lo * 1.12, hi / 1.12) if hi < 0
                    else (lo / 1.12, hi * 1.12))
        else:
            lo = max(lo * 1.3 if lo < 0 else -0.4, ATLAS_FLOOR)
            hi = min(hi * 1.3 if hi > 0 else 0.4, ATLAS_CAP)
            ylim = (lo, hi)

    if yscale == "linear":
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    else:
        ax.set_yscale("symlog", linthresh=ATLAS_LINTHRESH, linscale=0.45)
        if ylim[0] > 0 or ylim[1] < 0:
            # A one-sided frame carries no decade of ATLAS_TICKS in general,
            # so it gets its own ladder over the distances it actually spans.
            sign = 1.0 if ylim[0] > 0 else -1.0
            mags = _log_ladder(min(abs(ylim[0]), abs(ylim[1])),
                               max(abs(ylim[0]), abs(ylim[1])))
            ticks_y = [sign * m for m in mags]
        else:
            ticks_y = ATLAS_TICKS
            if max(abs(ylim[0]), abs(ylim[1])) >= 50.0:
                # The linear band around zero is then a sliver of the frame
                # and its own labels collide with the zero line.
                ticks_y = [t for t in ticks_y if abs(t) != 1]
        shown = [t for t in ticks_y if ylim[0] <= t <= ylim[1]]
        ax.set_yticks(shown)
        ax.set_yticklabels(["%g" % t for t in shown])
    ax.set_ylim(*ylim)

    ax.set_xlabel("$n$")
    if ylabel:
        ax.set_ylabel(ATLAS_DELTA_LABEL.get(
            baseline, r"$\Delta_{\mathrm{C}}$ in %"))
    if compact:
        ax.xaxis.label.set_size(ATLAS_LABEL_SIZE)
        ax.yaxis.label.set_size(ATLAS_LABEL_SIZE)
        ax.tick_params(labelsize=ATLAS_TICK_SIZE)
    fig.tight_layout()
    return fig


#: The correlations Figure 4.6 traces, light to dark.
GAIN_RHOS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]


#: The correlations for which the shrunken Double-PPI weight is drawn as well.
GAIN_SHRUNK_RHOS = [0.0, 0.25, 0.5]

#: Dash-dot, so the classical mean is told apart from the tuned weights by the
#: rhythm of its line rather than by a dotting that reads as "fainter". It is
#: not a weaker version of them but a different member, and at two of these
#: correlations the better one over part of the range.
GAIN_CLASSICAL_DASH = (0, (5, 2, 1, 2))

#: The classical mean is drawn only where it has something to say. At rho = 0
#: it is PPI++'s own limit and its curve carries no information the zero line
#: does not already carry.
GAIN_CLASSICAL_RHOS = [0.25, 0.5]

#: The strips on which PPI++ beats both D_+ and C at the same correlation:
#: (rho, n from, n to, y from, y to), the interval between the two crossings,
#: padded in n so that the narrow one at rho = 1/2 is visible at all.
#: Computed from K_n, not read off.
#:
#: Each box carries its own height, hugging the D_+ curve of its own
#: correlation rather than reaching up to the C curve. A box tall enough to
#: hold both would swallow the rho = 0 curve as well, which runs from +1.3 to
#: +0.4 across the first strip and has nothing to do with it. Complete
#: separation is not available -- the two curves cross in value inside the
#: strip -- but at this height rho = 0 only clips the lower right corner.
#: The floor sits just under zero rather than under the curves: what the box
#: marks is the stretch of the zero line on which PPI++ is ahead of both, so
#: it has no reason to reach down to where the curves eventually go.
GAIN_STRIPS = [(0.25, 18.0, 33.0, -0.30, 0.65),
               (0.5, 5.90, 7.00, -0.30, 2.0)]


#: A single wide panel with the key outside it, so it is drawn flatter than
#: the house ratio: the curves span four decades in n and the eye follows them
#: across rather than up.
GAIN_ASPECT = 1.95


def gain_delta_panel(rhos=GAIN_RHOS, n_min=6, n_max=1000,
                     width_fraction=0.8, points=1400, inset=True,
                     shrunk_rhos=(), classical_rhos=GAIN_CLASSICAL_RHOS, s=0.5,
                     aspect=GAIN_ASPECT, legend=True, strips=()):
    """Double-PPI against PPI++ over the sample size, as Delta of eq:gain-rel.

    The quantity is MSE(D_+)/MSE(P_+) - 1 in percent, so negative means
    Double-PPI is the better of the two and -50 says it halves the risk of
    PPI++. It is evaluated from the closed form

        2 (K_n - r) / (n + 2 r),   r = rho^2/(1 - rho^2),

    rather than as a ratio of the two risks, which is what lets rho = 0 be
    drawn: there the ratio is 0/0 in the parametrisation of `bias2_var` but
    the comparison itself is the perfectly finite 2 K_n / n.

    The vertical axis is an inverse hyperbolic sine rather than the symmetric
    logarithm used elsewhere. Both are linear near zero and logarithmic in the
    tails, but symlog joins the two pieces with a corner at the threshold,
    which puts a visible kink in every curve that crosses it; asinh is smooth
    everywhere and leaves the curves their own shape. The inset resolves the
    corner from n = 200 on, at a hundredth of the vertical range of the main
    panel, and the box it magnifies is drawn there with connectors.
    """
    fig, ax = style.new_figure(width_fraction=width_fraction, aspect=aspect)
    n = style.n_grid(n_min=n_min, n_max=n_max, points=points)
    K_n = unified.K_of_n(n)
    colors = list(style.sequential_colors(len(rhos)))[::-1]

    def delta(rho):
        r = rho ** 2 / (1.0 - rho ** 2)
        return 100.0 * 2.0 * (K_n - r) / (n + 2.0 * r)

    def delta_shrunk(rho):
        """The same comparison for S(D_+), the shrunken second-order weight.

        Written out rather than taken from `relative_mse`, whose
        parametrisation divides by rho and cannot be evaluated at rho = 0.
        Both risks are relative to the classical mean, so lambda* cancels:

            rel(S(D_+)) = (1-rho^2) + (1-s)^2 rho^2
                          + s^2 (1-rho^2)(1+2K_n)/(n-1),
            rel(P_+)    = (1-rho^2) + (1+rho^2)/(n-1).
        """
        q = rho ** 2
        num = ((1.0 - q) + (1.0 - s) ** 2 * q
               + s ** 2 * (1.0 - q) * (1.0 + 2.0 * K_n) / (n - 1.0))
        den = (1.0 - q) + (1.0 + q) / (n - 1.0)
        return 100.0 * (num / den - 1.0)

    def delta_classical(rho):
        """The classical mean against PPI++, in the same units.

        Its relative risk is one by definition, so the comparison is just the
        reciprocal of PPI++'s own: positive means PPI++ is the better of the
        two, negative that the predictor is not worth using at that n.
        """
        q = rho ** 2
        return 100.0 * (1.0 / ((1.0 - q) + (1.0 + q) / (n - 1.0)) - 1.0)

    ax.axhline(0.0, color=style.ESTIMATOR_COLOR["P+"], zorder=2)
    for color, rho in zip(colors, rhos):
        ax.plot(n, delta(rho), color=color, zorder=3)
    for rho in (shrunk_rhos or []):
        color = colors[int(np.argmin(np.abs(np.asarray(rhos) - rho)))]
        ax.plot(n, delta_shrunk(rho), color=color, zorder=4,
                linestyle=(0, (5, 2.2)))
    for rho in (classical_rhos or []):
        color = colors[int(np.argmin(np.abs(np.asarray(rhos) - rho)))]
        ax.plot(n, delta_classical(rho), color=color, zorder=4,
                linestyle=GAIN_CLASSICAL_DASH)

    ax.set_xscale("log")
    ax.set_xlim(n_min, n_max)
    # 500 and 1000 collide at this width; the decade is readable without it.
    ax.set_xticks([6, 10, 20, 50, 100, 200, 1000])
    ax.set_xticklabels(["6", "10", "20", "50", "100", "200", "1000"])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yscale("asinh", linear_width=0.5)
    ax.set_ylim(-100.0, 20.0)
    ax.set_yticks([-100, -30, -10, -3, -1, 0, 1, 3, 10])
    ax.set_yticklabels(["-100", "-30", "-10", "-3", "-1", "0", "1", "3", "10"])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\Delta_{\mathrm{P}_{\!+}}$ in %")

    # The strips on which the zero line beats both curves of one correlation.
    for _rho, n_lo, n_hi, y_lo, y_hi in (strips or []):
        ax.add_patch(Rectangle((n_lo, y_lo), n_hi - n_lo, y_hi - y_lo,
                               facecolor="none", edgecolor=style.HIGHLIGHT,
                               linewidth=1.8, zorder=8))

    fig.tight_layout()
    if legend:
        handles = [Line2D([], [], color=c) for c in colors]
        labels = [r"$\rho = %g$" % r for r in rhos]
        if classical_rhos:
            handles.append(Line2D([], [], color=style.GREY_TEXT,
                                  linestyle=GAIN_CLASSICAL_DASH))
            labels.append(r"$\mathrm{C}$")
        # Under the panel, not beside it: a flat frame has no room for a
        # column of seven entries.
        ax.legend(handles=handles, labels=labels, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False,
                  handlelength=1.6, columnspacing=1.6)

    if inset:
        # Nine curves leave no empty rectangle inside the frame, and the key
        # now sits underneath, so the magnified corner goes into the free
        # right margin where it can hide nothing.
        box = ax.get_position()
        axin = fig.add_axes([box.x1 + 0.035, box.y0 + 0.26 * box.height,
                             0.26 * box.width, 0.48 * box.height])
        axin.set_zorder(10)
        axin.patch.set_facecolor("white")
        for color, rho in zip(colors, rhos):
            axin.plot(n, delta(rho), color=color, zorder=3)
        for rho in (shrunk_rhos or []):
            color = colors[int(np.argmin(np.abs(np.asarray(rhos) - rho)))]
            axin.plot(n, delta_shrunk(rho), color=color, zorder=4,
                      linestyle=(0, (5, 2.2)))
        for rho in (classical_rhos or []):
            color = colors[int(np.argmin(np.abs(np.asarray(rhos) - rho)))]
            axin.plot(n, delta_classical(rho), color=color, zorder=4,
                      linestyle=GAIN_CLASSICAL_DASH)
        axin.axhline(0.0, color=style.ESTIMATOR_COLOR["P+"], zorder=2)
        axin.set_xscale("log")
        axin.set_xlim(200, n_max)
        # Nothing happens between zero and one, so the window stops there.
        axin.set_ylim(-1.0, 0.08)
        axin.xaxis.set_major_locator(mticker.FixedLocator([200, 1000]))
        axin.xaxis.set_major_formatter(mticker.FixedFormatter(["200", "1000"]))
        axin.xaxis.set_minor_locator(mticker.NullLocator())
        axin.yaxis.set_major_locator(mticker.FixedLocator([-1.0, -0.5, 0.0]))
        axin.yaxis.set_major_formatter(
            mticker.FixedFormatter(["-1", "-0.5", "0"]))
        axin.tick_params(labelsize=style._TICK_SIZE * 0.60)
        axin.grid(True, which="major", zorder=0)
        axin.set_axisbelow(True)
        for spine in axin.spines.values():
            spine.set_linewidth(0.9)
        ax.indicate_inset_zoom(axin, edgecolor=style.GREY_TEXT,
                               linewidth=1.1, alpha=1.0)

    return fig
