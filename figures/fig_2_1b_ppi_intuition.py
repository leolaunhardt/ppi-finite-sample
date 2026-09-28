"""
The weights as slopes, on the tree example of the introduction.
Section 2.1 and Section 3.1.

Three panels, all on the dry-forest allometry of Section 5.3:

    ppi_intuition_trees    first order, the members of Section 2.1
    ppi_intuition_all      first order, all five members
    ppi_intuition_second   second order, the members of Section 3.1

The weight is the slope of a line, and every member is one line. Both axes of
a panel carry the same scale, so the believed weight one is the diagonal and
the slopes can be compared by eye. All lines are drawn through the population
centroid, so the panel compares slopes and nothing else.

The slopes drawn are the *effective* weights, what multiplies Fbar - mu_f. For
the estimated members these are computed from the ten felled trees exactly as
the estimators of Definition 2.1 and Section 3.1 compute them. In particular
the weight of PPI++ is sigma^_fy / sigma_f^2, the labelled covariance over the
variance of *all* the predictions, which is 0.51 here -- not the least-squares
slope of the twenty points, which is 2.08 and is the separate lambda^_LS.

The optimum is not a member and is drawn black and heaviest: it is the bound
the members are measured against. Writing the estimator as the mean of
Z = Y - w(F - mu_f), its variance is Var(Y - wF)/n up to the finite-population
factors, and an intercept does not enter a variance, so each line's
contribution to the risk is the vertical scatter of the points about it and
lambda* is the slope that minimises it. On these trees that scatter falls from
Var(Y) at the horizontal to 45% of it at lambda_0 = 1 and 16.7% at lambda*,
which is 1 - rho^2 exactly; one order up it falls to 12.3% at nu*, or 1 - r^2.

The population is not drawn as a cloud but as its marginal, hung under the
frame on a logarithmic count with a decade ladder of its own. That is what the
panels need to say about it: the median tree carries 106 kg against a mean of
624, and the few giants at the far right are what fix the slope. The dotted
vertical is the mean the correction pivots on.

Frames hold 97% of the 4004 trees at either order. The small inset carries no
argument of its own: it magnifies the cluster at the origin far enough to show
that it is eighteen of the twenty felled trees and not one.

Suggested caption (the pair: ppi_intuition_all and ppi_intuition_second)

    \caption[Every weight is the slope of a line]{\textbf{Every weight is
    the slope of a line.} Twenty felled trees out of $M = 4004$ (black), with
    each estimator's effective weight drawn as a line through the population
    centroid and the optimum in black. Both axes of a panel carry the same
    scale, so the believed weight $1$ is the diagonal, and the vertical
    scatter of the population about a line is that line's contribution to the
    risk. At the first order (left) \textcolor{cla}{$\mathrm{C}$} is flat and
    \textcolor{ppi}{$\mathrm{P}_0$} stops at the diagonal, while
    \textcolor{ppp}{$\mathrm{P}_{+}$} chases $\lambda^\star = 2.40$ and, on
    ten trees, falls short; \textcolor{dfx}{$\mathrm{D}_0$} and
    \textcolor{dol}{$\mathrm{D}_{+}$} correct it. At the second order (right)
    the same picture runs in $(\widetilde{F}^{\,2},
    \widetilde{F}\widetilde{Y})$, where $\nu^\star = 3.08$ is far from the
    believed $1$. The marginal of the predictions hangs under each frame on a
    logarithmic count: the few giants at the right fix the slope. Inset: the
    cluster at the origin holds most of the twenty trees.}

Writes: output/pdf/ppi_intuition_{trees,all,second}.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ppi import panels, populations, style

DATASET = "agb_chave05_dry"
WIDTH_FRACTION = 0.8
#: The pair of first- and second-order panels sits two to a line. The
#: canvas is taller than wide so the square frame can use its full width.
PAIR_WIDTH_FRACTION = 0.52
PAIR_ASPECT = 0.8

#: The felled trees, a cell of the Section 5.3 grid.
N_LABELLED = 20
SEED = 66362

#: Everything is carried in 10^3 kg, so that both orders read on the same
#: prefix: the second order is then in (10^3 kg)^2 with no further factor.
UNIT = 1e3
FIRST_UNIT = r"$10^3$ kg"
SECOND_UNIT = r"$(10^3\,$kg$)^2$"

#: 97% of the population on both axes, at either order.
FRAME_FIRST = 9.0
FRAME_SECOND = 30.0

#: The magnified corner, as a window: just the cluster of small felled trees,
#: so that it reads as several points rather than one.
ZOOM_FIRST = (0.0, 1.40, 0.0, 1.40)
ZOOM_SECOND = (0.0, 0.80, 0.0, 0.80)


def _weights():
    """The effective weight of every member, from the ten felled trees.

    The effective weight is what multiplies Fbar - mu_f. The complement makes
    Fbar - Fbar' = (M/(M-n))(Fbar - mu_f), and the estimated weights carry the
    reciprocal factor in their denominator, so it cancels and what is left is
    the covariance over the variance of all the predictions.
    """
    Y, F = populations.load(DATASET)
    F, Y = F / UNIT, Y / UNIT
    idx = np.random.default_rng(SEED).choice(len(F), N_LABELLED, replace=False)
    n = N_LABELLED
    Fs, Ys = F[idx], Y[idx]
    dF, dY = Fs - Fs.mean(), Ys - Ys.mean()
    s_fy = float((dF * dY).sum() / (n - 1))
    s_ff = float((dF ** 2).sum() / (n - 1))
    sig_f2 = float(np.var(F, ddof=1))          # the pooled block is the population
    u, v = dF ** 2, dF * dY
    du = u - u.mean()
    nu_hat = float((du * (v - v.mean())).sum() / (du ** 2).sum())
    gap = s_ff - sig_f2
    return F, Y, idx, dict(
        P_plus=s_fy / sig_f2,
        D_0=(s_fy - gap) / sig_f2,
        D_plus=(s_fy - nu_hat * gap) / sig_f2,
        nu_hat=nu_hat)


def _star(x, y):
    return float(((x - x.mean()) * (y - y.mean())).mean()
                 / ((x - x.mean()) ** 2).mean())


def build_trees():
    """Section 2.1: the classical mean, PPI and PPI++."""
    F, Y, idx, w = _weights()
    members = [(0.0, style.CLA, r"$\mathrm{C}$: $\lambda_0 = 0$"),
               (1.0, style.PPI, r"$\mathrm{P}_0$: $\lambda_0 = 1$"),
               (w["P_plus"], style.PPP,
                r"$\mathrm{P}_{\!\!+\!}$: $\widehat{\lambda} = %.2f$" % w["P_plus"])]
    return panels.slope_intuition_panel(
        F, Y, r"predicted biomass $F$ [" + FIRST_UNIT + "]",
        r"true biomass $Y$ [" + FIRST_UNIT + "]",
        members, (_star(F, Y), r"$\lambda^\star = %.2f$" % _star(F, Y)),
        FRAME_FIRST, highlight=idx, zoom=ZOOM_FIRST, marginal=True,
        width_fraction=WIDTH_FRACTION, legend_where="right")


def build_all(pad_height=None, pad_width=None):
    """All five members at the first order, for the pair with the second."""
    F, Y, idx, w = _weights()
    members = [(0.0, style.CLA, r"$\mathrm{C}$: $\lambda_0 = 0$"),
               (1.0, style.PPI, r"$\mathrm{P}_0$: $\lambda_0 = 1$"),
               (w["P_plus"], style.PPP,
                r"$\mathrm{P}_{\!\!+\!}$: $\widehat{\lambda} = %.2f$" % w["P_plus"]),
               (w["D_0"], style.DFX,
                r"$\mathrm{D}_0$: $\widehat{\lambda} = %.2f$" % w["D_0"]),
               (w["D_plus"], style.DOL,
                r"$\mathrm{D}_{\!\!+\!}$: $\widehat{\lambda} = %.2f$" % w["D_plus"])]
    return panels.slope_intuition_panel(
        F, Y, r"predicted biomass $F$ [" + FIRST_UNIT + "]",
        r"true biomass $Y$ [" + FIRST_UNIT + "]",
        members, (_star(F, Y), r"$\lambda^\star = %.2f$" % _star(F, Y)),
        FRAME_FIRST, highlight=idx, zoom=ZOOM_FIRST, marginal=True,
        width_fraction=PAIR_WIDTH_FRACTION, legend_where="top", legend_cols=2,
        marker_labels=False, aspect=PAIR_ASPECT, pad_height=pad_height,
        pad_width=pad_width)


def build_second(pad_height=None, pad_width=None):
    """Section 3.1: the second-order weight, on the same trees."""
    F, Y, idx, w = _weights()
    f, y = F - F.mean(), Y - Y.mean()
    a, b = f * f, f * y
    members = [(0.0, style.PPP, r"$\mathrm{P}_{\!\!+\!}$: $\nu_0 = 0$"),
               (1.0, style.DFX, r"$\mathrm{D}_0$: $\nu_0 = 1$"),
               (w["nu_hat"], style.DOL,
                r"$\mathrm{D}_{\!\!+\!}$: $\widehat{\nu} = %.2f$" % w["nu_hat"])]
    return panels.slope_intuition_panel(
        a, b, r"$\widetilde{F}^{\,2}$ [" + SECOND_UNIT + "]",
        r"$\widetilde{F}\,\widetilde{Y}$ [" + SECOND_UNIT + "]",
        members, (_star(a, b), r"$\nu^\star = %.2f$" % _star(a, b)),
        FRAME_SECOND, highlight=idx, zoom=ZOOM_SECOND, marginal=True,
        width_fraction=PAIR_WIDTH_FRACTION, legend_where="top", legend_cols=2,
        marker_labels=False, aspect=PAIR_ASPECT, pad_height=pad_height,
        pad_width=pad_width, legend_blanks=2)


if __name__ == "__main__":
    style.save(build_trees(), "ppi_intuition_trees")
    # The pair shares a height, so measure both before writing either.
    import matplotlib.pyplot as plt
    boxes = []
    for fn in (build_all, build_second):
        f = fn()
        f.canvas.draw()
        boxes.append(f.get_tightbbox(f.canvas.get_renderer()))
        plt.close(f)
    tall = max(b.height for b in boxes)
    wide = max(b.width for b in boxes)
    style.save(build_all(pad_height=tall, pad_width=wide), "ppi_intuition_all")
    style.save(build_second(pad_height=tall, pad_width=wide),
               "ppi_intuition_second")
