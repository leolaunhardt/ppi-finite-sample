"""
Appendix: every member on every population of Section 5, one panel each.

Drawn from the same cache as the panels of Section 5 (``ppi/populations.py``:
50 000 replicates per cell, B = 500, the grid of Section 5.1), so that the
appendix and the chapter never disagree. Unlike the figures of Section 5,
which compare against one baseline at a time, these put all members on one
scale, Delta against the classical mean, so that the thirteen populations can
be read against one another. The two estimated guards are the liberal and the
conservative rule of eq:two-rules.

Writes: output/pdf/app_legend.pdf and output/pdf/app_<group>_<population>.pdf

Suggested caption
-----------------
\\caption[Every member on every population]{\\textbf{Every member on every
population.} $\\Delta_{\\mathrm C}$ over $n$, the classical mean at zero and
the oracle floor $\\Delta^\\star$ dashed in grey. Same simulation as
Section~\\ref{sec:empirical}.}
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppi import populations, panels, style

#: Members, in the order the key lists them: the first order, its guards and
#: the fixed half-guard; then the second order and its two-level guards.
ORDER = ["P0", "P+", "SEl_P+", "SEc_P+", "S0_P+",
         "D0", "Dnu", "SEl_Both", "SEc_Both"]

GROUPS = [
    ("strong", ["freesolv", "nwp_temp", "nwp_precip", "amazon_base",
                "amazon_tuned", "amazon_top", "agb_chave05_dry",
                "cu_scanner3"]),
    ("binary", ["pas_galaxy_spiral", "tree_cover_binary"]),
    ("weak", ["compas_violent", "lsi_lowbr", "moral_old"]),
]

#: Decade ticks for a frame that straddles the baseline.
DECADES = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000]
LABEL_SIZE, TICK_SIZE = 19, 15
LINEWIDTH = 2.1


def _style(key):
    return panels.ATLAS_STYLE.get(
        key, (style.ESTIMATOR_COLOR.get(key, style.GREY_TEXT), "-"))


def _ladder(lo, hi):
    """The 1-2-5 magnitudes inside [lo, hi], thinned to at most five."""
    out, step = [], 1.0
    while step <= hi * 10:
        for mult in (1.0, 2.0, 5.0):
            if lo <= mult * step <= hi:
                out.append(mult * step)
        step *= 10.0
    while len(out) > 5:
        out = out[::2]
    return out


def panel(name, ylabel):
    """One population: every member against the classical mean."""
    d = populations.dataset(name)
    series = {k: d[k] for k in ORDER}
    floor = populations.oracle_floor(name, "C")
    stacked = np.concatenate([series[k] for k in ORDER] + [[floor]])
    lo, hi = float(stacked.min()), float(stacked.max())

    fig, ax = style.new_figure(width_fraction=0.32)
    # The baseline is drawn only where the frame reaches it: the vertical
    # scale follows the curves, not the zero line.
    if lo < 0.0 < hi:
        ax.axhline(0.0, color=style.CLA, linewidth=LINEWIDTH, zorder=2)
    ax.axhline(floor, **panels.ORACLE_STYLE)
    for k in ORDER:
        color, dash = _style(k)
        ax.plot(d["n"], series[k], color=color, linestyle=dash,
                linewidth=LINEWIDTH, zorder=3)

    one_sided = hi < 0 or lo > 0
    span = (max(abs(lo), abs(hi)) / max(min(abs(lo), abs(hi)), 1e-9)
            if one_sided else np.inf)
    if one_sided and span < 10.0:
        # Less than a decade on one side: a linear frame resolves it.
        pad = 0.07 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    else:
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.45)
        if one_sided:
            sign = 1.0 if lo > 0 else -1.0
            ylim = (lo * 1.12, hi / 1.12) if hi < 0 else (lo / 1.12, hi * 1.12)
            ticks = [sign * mag for mag in
                     _ladder(min(abs(ylim[0]), abs(ylim[1])),
                             max(abs(ylim[0]), abs(ylim[1])))]
        else:
            ylim = (lo * 1.25 if lo < 0 else -0.5,
                    hi * 1.25 if hi > 0 else 0.5)
            ticks = [t for t in DECADES if ylim[0] <= t <= ylim[1]]
            if max(abs(ylim[0]), abs(ylim[1])) >= 50.0:
                ticks = [t for t in ticks if abs(t) != 1]
        ax.set_ylim(*ylim)
        ax.set_yticks(ticks)
        ax.set_yticklabels(["%g" % t for t in ticks])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xscale("log")
    ax.set_xlim(float(d["n"].min()), float(d["n"].max()))
    # Decades only, plus the grid's own end where it stops short of the next
    # decade, as the narrow panels of Section 5 do.
    top = int(d["n"].max())
    xt = [t for t in (10, 100, 1000) if t <= top]
    if top >= 2 * xt[-1]:
        xt.append(top)
    if len(xt) < 2:
        xt = sorted({int(d["n"].min()), int(round((d["n"].min() * top) ** 0.5)), top})
    ax.set_xticks(xt)
    ax.set_xticklabels(["%g" % t for t in xt])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel("$n$")
    if ylabel:
        ax.set_ylabel(r"$\Delta_{\mathrm{C}}$ in %")
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    fig.tight_layout()
    return fig


def legend():
    """Three rows, filled column by column: first order, second order, rest."""
    rows = [["P0", "P+", "SEl_P+", "SEc_P+", "S0_P+"],
            ["D0", "Dnu", "SEl_Both", "SEc_Both", None],
            ["C", "__oracle__", None, None, None]]
    entries = []
    for col in range(5):
        for row in rows:
            key = row[col]
            if key is None:
                entries.append(("blank", None, None, None))
            elif key == "__oracle__":
                o = panels.ORACLE_STYLE
                entries.append(("line", o["color"], o["linestyle"],
                                panels.ORACLE_LABEL, o["linewidth"]))
            elif key == "C":
                entries.append(("line", style.CLA, "-", r"$\mathrm{C}$",
                                LINEWIDTH))
            else:
                color, dash = _style(key)
                entries.append(("line", color, dash,
                                style.ESTIMATOR_LABEL[key], LINEWIDTH))
    return panels.legend_strip(entries, ncol=5, width_fraction=0.95)


def build():
    style.save(legend(), "app_legend")
    for tag, names in GROUPS:
        for i, name in enumerate(names):
            style.save(panel(name, ylabel=(i % 3 == 0)),
                       "app_%s_%s" % (tag, name))


if __name__ == "__main__":
    build()
