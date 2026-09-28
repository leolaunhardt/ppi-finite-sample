"""
Named Monte-Carlo experiments, so that figures drawn from the same run agree.

Each function returns a dict of arrays and caches it under ``simdata/``. Two
figures that show the same cell in different units therefore never disagree,
and neither pays for the simulation twice.

The two runs below no longer feed any figure: Figures 3.2 and 3.3 are drawn
from Proposition 3.1 in closed form. They are kept because they are the
evidence that the closed form is right -- at N = 1000 they reproduce it to
within Monte-Carlo error for every correlation up to 0.9 -- and because they
are what shows where the N -> infinity limit stops being adequate: at rho = 1
and n = 50 the simulated risk is twice the limit. Section 5, whose shrinkage
rules are truncated nonlinear functions of an estimated statistic and have no
closed form, will use this machinery for real.
"""

import numpy as np

from . import simulate as sim

# ---------------------------------------------------------------------------
# Figure 3.2 -- cross-fit against single-sample PPI++, Gaussian case
# ---------------------------------------------------------------------------

#: A weak predictor, chosen so that the range shown sits mostly below the
#: threshold of Corollary 3.2: SSP only pays from n = 27, CFP from n = 54.
CF_VS_SS_RHO = 0.2

#: Finite, and far larger than n, as the PPI regime assumes.
CF_VS_SS_N_UNLABELLED = 1000

#: Even sample sizes only, so that the cross-fit folds n/2 are exact.
CF_VS_SS_N_VALUES = np.arange(6, 51, 2)

#: The upper end of the range Section 5 states.
CF_VS_SS_REPS = 200_000


def cf_vs_ss(force=False):
    """MSE of C, SSP and CFP over n, at a fixed correlation.

    Returns ``{"n": ..., "C": ..., "SSP": ..., "CFP": ...}`` with the mean
    squared errors, plus the parameters of the run.
    """

    def compute():
        n_values, mse = sim.sweep_n(
            CF_VS_SS_N_VALUES,
            N=CF_VS_SS_N_UNLABELLED,
            rho=CF_VS_SS_RHO,
            reps=CF_VS_SS_REPS,
            seed=1000,
        )
        return dict(n=n_values,
                    rho=np.array(CF_VS_SS_RHO),
                    N=np.array(CF_VS_SS_N_UNLABELLED),
                    reps=np.array(CF_VS_SS_REPS),
                    **mse)

    return sim.run_or_load("fig_3_2_cf_vs_ss", compute, force=force)


# ---------------------------------------------------------------------------
# Figure 3.3 -- SSP against classical inference over a range of correlations
# ---------------------------------------------------------------------------

#: The whole correlation range, in tenths.
CORR_SWEEP_RHOS = np.round(np.arange(0.0, 1.01, 0.1), 2)

CORR_SWEEP_N_UNLABELLED = CF_VS_SS_N_UNLABELLED
CORR_SWEEP_N_VALUES = CF_VS_SS_N_VALUES
CORR_SWEEP_REPS = 100_000


def corr_sweep(force=False):
    """MSE of C and SSP over n, at every correlation in `CORR_SWEEP_RHOS`.

    Returns ``{"n": ..., "rhos": ..., "C": (rho, n), "SSP": (rho, n), ...}``.
    """

    def compute():
        out = {name: [] for name in sim.ESTIMATORS}
        for j, rho in enumerate(CORR_SWEEP_RHOS):
            _, mse = sim.sweep_n(
                CORR_SWEEP_N_VALUES,
                N=CORR_SWEEP_N_UNLABELLED,
                rho=float(rho),
                reps=CORR_SWEEP_REPS,
                seed=2000 + 100 * j,
            )
            for name in sim.ESTIMATORS:
                out[name].append(mse[name])
            print("  rho = %.1f done" % rho)

        return dict(n=CORR_SWEEP_N_VALUES,
                    rhos=CORR_SWEEP_RHOS,
                    N=np.array(CORR_SWEEP_N_UNLABELLED),
                    reps=np.array(CORR_SWEEP_REPS),
                    **{k: np.array(v) for k, v in out.items()})

    return sim.run_or_load("fig_3_3_corr_sweep", compute, force=force)


# ---------------------------------------------------------------------------
# Section 5, Shrinkage: protection, price, and exposure over all correlations
# ---------------------------------------------------------------------------

from . import gaussian as shrink  # noqa: E402

#: Section 5.1 fixes the setting: jointly normal with unit variances, so
#: lambda* = rho, and an unlabelled sample of 10^4.
SHRINK_N_UNLABELLED = 10_000
SHRINK_REPS = 200_000

# The two exponential members are found by searching both parameters, not
# picked by hand; see `best_member`. Two things came out of that search.
#
# Over the whole range there is a proper interior optimum at (h, kappa) =
# (1/2, 1/2), worth -28.37% in mean, against -23.30% for the fixed rule.
#
# Inside the belief region t_lambda <= 3 the optimum is degenerate. It runs to
# kappa -> 0, and there s_E(h, kappa) = 2^-(h/t)^kappa -> 2^-1 = 1/2 for every
# t > 0: the exponential family's limit *is* the fixed rule. The unconstrained
# best, (1/4, 1/20), scores -6.613% against the fixed rule's -6.556%, a gain
# of six hundredths of a point for a rule that is no longer adaptive at all.
# Below the threshold, in other words, there is nothing to adapt to, and
# shrinking by a constant half is already the right answer.
#
# The member drawn is the one the formal selection of Section 5.1 returns; see
# `hyper.py`. Over the unrestricted grid it is the best mean subject to a worst
# case of at most half a point, (h, kappa) = (4, 3/2), which keeps 99.3% of the
# mean gain of unshrunken PPI++ while cutting its worst case from +11.5 to
# +0.5. It replaces the earlier (3/2, 1/2), a mean-optimal rather than a robust
# choice.
H_BELIEF, KAPPA_BELIEF = 4.0, 1.5

#: The earlier, mean-optimal member, kept alongside for comparison.
H_ALT, KAPPA_ALT = 1.5, 0.5
H_GENERAL, KAPPA_GENERAL = 0.5, 0.5

SHRINK_RULES = {
    "C": None,
    "P+": None,
    "S0": shrink.s_fixed(0.5),
    "SEb": shrink.s_exponential(H_BELIEF, KAPPA_BELIEF),
    "SEg": shrink.s_exponential(H_GENERAL, KAPPA_GENERAL),
}

SHRINK_PROTECTION_RHO = 0.1
SHRINK_PRICE_N = 50
SHRINK_N_VALUES = np.array([6, 8, 10, 12, 15, 18, 22, 26, 30, 36, 42, 50, 60,
                            72, 86, 100, 120, 145, 170, 200])
SHRINK_RHOS = np.round(np.arange(0.0, 0.981, 0.02), 3)
SHRINK_BAND_RHOS = np.round(np.arange(0.0, 0.951, 0.05), 3)


def shrink_protection(force=False):
    """Delta against n at a correlation where power tuning is a mistake."""

    def compute():
        mse = shrink.sweep(SHRINK_RULES, SHRINK_N_VALUES, SHRINK_N_UNLABELLED,
                           [SHRINK_PROTECTION_RHO], SHRINK_REPS, seed=6100)
        return dict(n=SHRINK_N_VALUES,
                    **{k: v[0] for k, v in mse.items()})

    return sim.run_or_load("fig_6_1a_protection", compute, force=force)


def shrink_price(force=False):
    """Delta against the correlation at a sample size worth using."""

    def compute():
        mse = shrink.sweep(SHRINK_RULES, [SHRINK_PRICE_N],
                           SHRINK_N_UNLABELLED, SHRINK_RHOS, SHRINK_REPS,
                           seed=6200)
        return dict(rho=SHRINK_RHOS,
                    **{k: v[:, 0] for k, v in mse.items()})

    return sim.run_or_load("fig_6_1b_price", compute, force=force)


def shrink_band(force=False):
    """Delta over n for the whole correlation range, every rule."""

    def compute():
        mse = shrink.sweep(SHRINK_RULES, SHRINK_N_VALUES, SHRINK_N_UNLABELLED,
                           SHRINK_BAND_RHOS, SHRINK_REPS // 2, seed=6300)
        return dict(n=SHRINK_N_VALUES, rhos=SHRINK_BAND_RHOS, **mse)

    return sim.run_or_load("fig_6_1c_band", compute, force=force)


# ---------------------------------------------------------------------------
# Choosing the two exponential members by optimisation
# ---------------------------------------------------------------------------

#: Candidate members of the exponential family. The family has two
#: parameters, so the search is over both: h locates the rule and kappa
#: steepens it, and the two trade against each other.
SHRINK_H_GRID = np.round(np.arange(0.5, 8.01, 0.5), 3)
SHRINK_KAPPA_GRID = np.round(np.arange(0.25, 3.01, 0.25), 3)

#: The grid the average is taken over. t_lambda = (n-1) rho^2/(1+rho^2) is the
#: sufficient summary of Corollary 2.1, so each (n, rho) cell is classified by
#: the t it produces.
SHRINK_OPT_N = np.array([8, 12, 20, 30, 50, 80, 130, 200])
SHRINK_OPT_RHOS = np.round(np.arange(0.02, 0.981, 0.04), 3)
SHRINK_OPT_REPS = 60_000

#: The belief region of the trade-off section.
BELIEF_T_MAX = 3.0


def _key(h, kappa):
    return "h%.2f_k%.2f" % (h, kappa)


def shrink_hk_search(force=False):
    """Mean Delta of the exponential rule over a grid of (h, kappa).

    Every candidate is evaluated on the same draws, so the comparison between
    them carries no Monte-Carlo error of its own.
    """

    def compute():
        rules = {"C": None, "P+": None}
        for h in SHRINK_H_GRID:
            for k in SHRINK_KAPPA_GRID:
                rules[_key(h, k)] = shrink.s_exponential(float(h), float(k))
        mse = shrink.sweep(rules, SHRINK_OPT_N, SHRINK_N_UNLABELLED,
                           SHRINK_OPT_RHOS, SHRINK_OPT_REPS, seed=6400)
        return dict(n=SHRINK_OPT_N, rhos=SHRINK_OPT_RHOS,
                    h=SHRINK_H_GRID, kappa=SHRINK_KAPPA_GRID, **mse)

    return sim.run_or_load("fig_6_1_hk_search", compute, force=force)


def best_member(t_max=None):
    """The (h, kappa) with the largest mean reduction on the chosen region.

    Returns ``(h, kappa, mean_delta, table)`` with the table of mean Delta
    over the whole grid, shape ``(h, kappa)``.
    """
    d = shrink_hk_search()
    n = d["n"].astype(float)[None, :]
    rho = d["rhos"].astype(float)[:, None]
    t_lambda = (n - 1.0) * rho ** 2 / (1.0 + rho ** 2)
    mask = (np.ones_like(t_lambda, dtype=bool) if t_max is None
            else t_lambda <= t_max)

    table = np.empty((len(d["h"]), len(d["kappa"])))
    for i, h in enumerate(d["h"]):
        for j, k in enumerate(d["kappa"]):
            delta = 100.0 * (d[_key(h, k)] / d["C"] - 1.0)
            table[i, j] = delta[mask].mean()

    i, j = np.unravel_index(int(np.argmin(table)), table.shape)
    return float(d["h"][i]), float(d["kappa"][j]), float(table[i, j]), table


# ---------------------------------------------------------------------------
# Section 5, the trade-off: mean gain against worst case
# ---------------------------------------------------------------------------

#: The two panels now average over two different grids, because they answer
#: two different questions. The belief panel is laid out in t_lambda, the
#: quantity the analyst has a belief about; the other in the correlation,
#: which is what varies when nothing is known.
TRADEOFF_N = np.arange(10, 301, 10)
TRADEOFF_T = np.round(np.arange(0.1, 3.001, 0.1), 3)
TRADEOFF_RHO = np.round(np.arange(0.0, 1.001, 0.05), 3)
TRADEOFF_RHO_MAX = 0.999
TRADEOFF_REPS = 40_000

#: The two families compared. The plug-in family is left out.
TRADEOFF_S_GRID = np.round(np.arange(0.05, 1.001, 0.05), 3)
TRADEOFF_H_GRID = np.round(np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]), 3)
TRADEOFF_KAPPA_GRID = np.round(np.array([0.25, 0.5, 0.75, 1.0, 1.5]), 3)


def tradeoff_cells(kind):
    """The (n, rho, t) cells one panel averages over.

    "belief" walks a grid in t and recovers the correlation at each sample
    size from rho^2 = t/(n-1-t); "all" walks a grid in the correlation and
    reports the t it implies. Cells needing a correlation above
    TRADEOFF_RHO_MAX are dropped.
    """
    rows = []
    for n in TRADEOFF_N:
        if kind == "belief":
            for t in TRADEOFF_T:
                denom = n - 1.0 - t
                if denom <= 0:
                    continue
                rho2 = t / denom
                if 0.0 < rho2 <= TRADEOFF_RHO_MAX ** 2:
                    rows.append((int(n), float(np.sqrt(rho2)), float(t)))
        else:
            for rho in TRADEOFF_RHO:
                r = min(float(rho), TRADEOFF_RHO_MAX)
                t = (n - 1.0) * r ** 2 / (1.0 + r ** 2)
                rows.append((int(n), r, float(t)))
    return rows


def tradeoff(kind, force=False):
    """Delta of every candidate rule on every cell of one panel's grid."""

    def compute():
        cells = tradeoff_cells(kind)
        rules = {"C": None, "P+": None}
        for s_ in TRADEOFF_S_GRID:
            rules["s%.2f" % s_] = shrink.s_fixed(float(s_))
        for h in TRADEOFF_H_GRID:
            for k in TRADEOFF_KAPPA_GRID:
                rules[_key(h, k)] = shrink.s_exponential(float(h), float(k))

        out = {name: np.empty(len(cells)) for name in rules}
        for i, (n, rho, _) in enumerate(cells):
            rng = np.random.default_rng(7100 + i)
            total = {name: 0.0 for name in rules}
            done = 0
            while done < TRADEOFF_REPS:
                m = min(20000, TRADEOFF_REPS - done)
                got = shrink.cell(rng, m, n, SHRINK_N_UNLABELLED, rho, rules)
                for name in rules:
                    total[name] += got[name]
                done += m
            for name in rules:
                out[name][i] = total[name] / TRADEOFF_REPS

        arr = np.array(cells, dtype=float)
        delta = {k: 100.0 * (v / out["C"] - 1.0) for k, v in out.items()}
        return dict(n=arr[:, 0], rho=arr[:, 1], t=arr[:, 2], **delta)

    return sim.run_or_load("fig_6_2_tradeoff_" + kind, compute, force=force)


def tradeoff_summary(kind):
    """Mean and worst-case Delta of every rule over one panel's cells."""
    d = tradeoff(kind)
    skip = {"n", "rho", "t"}
    return {k: (float(v.mean()), float(v.max()))
            for k, v in d.items() if k not in skip}


# ---------------------------------------------------------------------------
# Section 5, benchmark datasets
# ---------------------------------------------------------------------------

from . import realdata  # noqa: E402

REAL_N_VALUES = np.array([6, 8, 10, 12, 15, 18, 22, 26, 30, 36, 42, 50, 60,
                          72, 86, 100, 120, 145, 170, 200, 240, 290, 340, 400,
                          500, 650, 800, 1000])
REAL_REPS = 100_000

#: How much Gaussian noise is added to the predictions in the modified
#: version, in units of their own standard deviation. Section 6.2 needs the
#: datasets moved out of the regime where every predictor is already good.
REAL_NOISE = 4.0


def real_dataset(name, noise=0.0, force=False):
    """Delta over n for one benchmark dataset, optionally degraded."""
    tag = "fig_6_4_%s%s" % (name, "" if noise == 0 else "_noise%g" % noise)

    def compute():
        delta = realdata.sweep(name, REAL_N_VALUES, REAL_REPS,
                               seed=6600, noise=noise)
        return dict(n=realdata.n_values_for(name, REAL_N_VALUES).astype(float),
                    **delta)

    return sim.run_or_load(tag, compute, force=force)


# ---------------------------------------------------------------------------
# Figure 5.3, rebuilt: the belief region on the t axis
# ---------------------------------------------------------------------------

#: Both panels sit at the same sample size, so that (a) is exactly the
#: belief-region zoom of (b) in t coordinates.
SHRINK_PANEL_N = 20

#: Panel (a) runs over the belief region itself. At a fixed n the correlation
#: follows from t by rho^2 = t/(n-1-t), so t = 3 is rho = 0.433 here.
SHRINK_T_GRID = np.round(np.arange(0.05, 3.001, 0.05), 3)

#: Panel (b) runs over the whole correlation range at the same n.
SHRINK_RHO_GRID = np.round(np.arange(0.02, 0.981, 0.02), 3)


def _rho_from_t(t, n):
    return float(np.sqrt(t / (n - 1.0 - t)))


def shrink_t_search(force=False):
    """Mean Delta over the belief region, for a grid of (h, kappa)."""

    def compute():
        rules = {"C": None, "P+": None, "S0": shrink.s_fixed(0.5)}
        for h in SHRINK_H_GRID:
            for k in SHRINK_KAPPA_GRID:
                rules[_key(h, k)] = shrink.s_exponential(float(h), float(k))
        rhos = [_rho_from_t(t, SHRINK_PANEL_N) for t in SHRINK_T_GRID]
        mse = shrink.sweep(rules, [SHRINK_PANEL_N], SHRINK_N_UNLABELLED,
                           rhos, SHRINK_REPS, seed=6700)
        return dict(t=SHRINK_T_GRID, rho=np.array(rhos),
                    **{k: v[:, 0] for k, v in mse.items()})

    return sim.run_or_load("fig_6_1_t_search", compute, force=force)


def best_member_on_t(kappa_min=None):
    """The (h, kappa) with the smallest mean Delta over the belief region."""
    d = shrink_t_search()
    best, table = None, {}
    for h in SHRINK_H_GRID:
        for k in SHRINK_KAPPA_GRID:
            if kappa_min is not None and k < kappa_min:
                continue
            m = float(np.mean(100.0 * (d[_key(h, k)] / d["C"] - 1.0)))
            table[(float(h), float(k))] = m
            if best is None or m < best[2]:
                best = (float(h), float(k), m)
    return best, table


#: The rules the risk-scale panels of the hypertuning appendix draw: the
#: classical baseline, unguarded PPI++, the fixed half-guard, and the two
#: members of eq:two-rules, liberal (4, 1) and conservative (8, 3/2). The
#: threshold statistic is the paired bootstrap with the B of the selection
#: run, so that the panels show the rules on exactly the input they were
#: chosen on.
PANEL_RULES = {
    "C": None,
    "P+": None,
    "S0": shrink.s_fixed(0.5),
    "SEl": shrink.s_exponential(4.0, 1.0),
    "SEc": shrink.s_exponential(8.0, 1.5),
}
PANEL_N_BOOT = 200


def shrink_panel_price(force=False):
    """Delta over the whole correlation range at the panel sample size."""

    def compute():
        mse = shrink.sweep(PANEL_RULES, [SHRINK_PANEL_N], SHRINK_N_UNLABELLED,
                           SHRINK_RHO_GRID, SHRINK_REPS, seed=6800,
                           n_boot=PANEL_N_BOOT)
        return dict(rho=SHRINK_RHO_GRID, **{k: v[:, 0] for k, v in mse.items()})

    return sim.run_or_load("appB_price_n20_boot", compute, force=force)


def shrink_panel_protection():
    """Delta over the belief region, read out of the parameter search."""
    d = shrink_t_search()
    keys = {"C": "C", "P+": "P+", "S0": "S0", "SEb": _key(H_BELIEF, KAPPA_BELIEF)}
    return dict(t=d["t"], rho=d["rho"], **{k: d[v] for k, v in keys.items()})


# ---------------------------------------------------------------------------
# Figure 5.3, final form
# ---------------------------------------------------------------------------

#: Panel (a) keeps n on the axis and picks the correlation so that n covers
#: the belief region: at rho = 0.2, t_lambda = (n-1)/26, so n = 6 to 79 is
#: t = 0.19 to 3.00. The threshold t = 1 falls at n = 27, inside the frame.
SHRINK_PROT_RHO = 0.2
SHRINK_PROT_N = np.arange(6, 80, 2)

#: Panel (b) shows the whole correlation range and carries an inset on the
#: transition zone, where the rules part company from unshrunken PPI++.
SHRINK_ZOOM_RANGE = (0.30, 0.60)
SHRINK_ZOOM_RHOS = np.round(np.arange(0.30, 0.6001, 0.005), 4)


def shrink_protection_n(force=False):
    """Delta over n at the correlation that spans the belief region."""

    def compute():
        mse = shrink.sweep(PANEL_RULES, SHRINK_PROT_N, SHRINK_N_UNLABELLED,
                           [SHRINK_PROT_RHO], SHRINK_REPS, seed=6900,
                           n_boot=PANEL_N_BOOT)
        return dict(n=SHRINK_PROT_N,
                    t=(SHRINK_PROT_N - 1.0) * SHRINK_PROT_RHO ** 2
                      / (1.0 + SHRINK_PROT_RHO ** 2),
                    **{k: v[0] for k, v in mse.items()})

    return sim.run_or_load("appB_protection_n_boot", compute, force=force)


def shrink_price_zoom(force=False):
    """Delta over the transition zone in the correlation, at the panel n."""

    def compute():
        mse = shrink.sweep(PANEL_RULES, [SHRINK_PANEL_N], SHRINK_N_UNLABELLED,
                           SHRINK_ZOOM_RHOS, SHRINK_REPS, seed=7000,
                           n_boot=PANEL_N_BOOT)
        return dict(rho=SHRINK_ZOOM_RHOS,
                    **{k: v[:, 0] for k, v in mse.items()})

    return sim.run_or_load("appB_price_zoom_boot", compute, force=force)


# ---------------------------------------------------------------------------
# Section 5: bootstrap against Gaussian plug-in, undertrained predictor
# ---------------------------------------------------------------------------
