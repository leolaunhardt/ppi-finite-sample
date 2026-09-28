"""
The twenty real-data populations of Section 5.3.

Each population is a finite set of (F, Y) pairs; a replicate draws n of them
without replacement as the labelled sample and treats the complement as the
unlabelled block, exactly as in `realdata.py`. The populations, their
provenance and the reference Monte-Carlo results are in ``data/``; see
``data/PPI_HANDOVER.md`` for the rebuild recipes and the source citations.

Two departures from `realdata.py`, both chosen deliberately:

* the shrinkage statistic is the paired bootstrap of ``eq:boot`` rather than
  the Gaussian plug-in, because several of these populations are strongly
  skewed and Section 4 showed that this is exactly where the two inputs
  part company;
* the two members of the exponential family selected in Section 5.1 are
  carried side by side -- liberal and conservative -- so that the price of
  guarding can be read as a range rather than at one setting.

nu-hat is re-estimated inside every bootstrap resample. Holding it fixed
understates the resampling variability of the Double-PPI covariance badly on
heavy-tailed predictors.
"""

import numpy as np
import pandas as pd

from pathlib import Path

from . import gaussian as shrink
from . import simulate as sim
from .realdata import sample_without_replacement

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

#: The two members selected in Section 5.1 on the bootstrap surfaces, named
#: for how much of the tuned weight they let through. Liberal takes almost all
#: of the mean gain that guarding can reach and still cuts the worst case by
#: four fifths; conservative gives up mean gain to hold the worst case at zero.
#:
#:   liberal       alpha_E(4, 1)       mean -29.16%   worst +1.94%
#:   conservative  alpha_E(8, 3/2)     mean -26.99%   worst +0.46%
SHRINK_RULES = {"SEl": (4.0, 1.0), "SEc": (8.0, 1.5)}

#: Order the figures list them in, liberal first.
RULE_ORDER = ["SEl", "SEc"]

#: The rule the Gaussian-plug-in comparison is carried on, so that the two
#: inputs are compared at one fixed parametrisation rather than two.
GAUSS_RULE = "SEl"

#: Weights that carry a random covariance and can therefore be shrunk.
SHRINKABLE = ["P+", "D0", "Dnu"]

#: The fixed rule of eq:s-fixed, carried on P_+ as the simple alternative.
FIXED_S = 0.5

#: The believed first-order weight of P_0, as Section 5.3 sets it.
LAMBDA_0 = 1.0

#: The moderate rule fed by the Gaussian plug-in of eq:gauss-var instead of
#: the bootstrap, so that the two inputs of Section 3.3 can be compared on the
#: populations where the rule actually matters.
GAUSS_KEY = "SEg_P+"

ESTIMATORS = (["C", "P0", "P+", "D0", "Dnu", "S0_P+", GAUSS_KEY]
              + ["%s_%s" % (r, w) for r in RULE_ORDER for w in SHRINKABLE]
              + ["%s_Gnu" % r for r in RULE_ORDER]
              + ["%s_Both" % r for r in RULE_ORDER]
              + ["S0_Gnu", "S0_Both"])

#: Without shrinkage no bootstrap is needed, and the sample size can then run
#: far higher for the same cost.
ESTIMATORS_PLAIN = ["C", "P0", "P+", "D0", "Dnu"]

#: Bootstrap resamples per replicate, the B = 500 of Section 5.1.
N_BOOT = 500


#: Four bundles: the twenty of Section 5.3, the six drawn from the PPI
#: follow-up literature, the three Amazon populations rebuilt by
#: ``data/make_amazon.py`` and the three PurpleAir sites carrying the EPA
#: correction, rebuilt by ``data/make_purpleair_corrected.py``. All are
#: mean-estimation problems and share a format.
BUNDLES = [("populations_subset.npz", "population_table_subset.csv"),
           ("populations_lit6.npz", "population_table_lit6.csv"),
           ("populations_amazon.npz", "population_table_amazon.csv"),
           ("populations_pa_corr.npz", "population_table_pa_corr.csv"),
           ("populations_pa_chen.npz", "population_table_pa_chen.csv"),
           ("populations_pa_places.npz", "population_table_pa_places.csv"),
           ("populations_chave.npz", "population_table_chave.csv"),
           ("populations_xrf.npz", "population_table_xrf.csv")]


def _table():
    frames = [pd.read_csv(DATA_DIR / t) for _, t in BUNDLES
              if (DATA_DIR / t).exists()]
    return pd.concat(frames, ignore_index=True, sort=False)


def load(name):
    """Return ``(Y, F)`` of one population, from whichever source holds it.

    The bundles are not part of the repository; ``data/README.md`` says how
    each one is built. Bundles that are absent are skipped, so the thirteen
    populations of the thesis can be rebuilt one bundle at a time.
    """
    for npz, _ in BUNDLES:
        if not (DATA_DIR / npz).exists():
            continue
        P = np.load(DATA_DIR / npz)
        if name + "__Y" in P.files:
            return P[name + "__Y"], P[name + "__F"]
    from . import realdata
    if name in realdata.DATASETS:          # the four ppi_py mean benchmarks
        return realdata.load(name)
    raise KeyError(name)


def summary(name, t_at=20):
    """Population characteristics, all computed from the population itself.

    nu* = Cov(F~^2, F~Y~)/Var(F~^2), r^2 its squared correlation with the
    covariance estimator, and t(m) the exact finite-m statistic of
    Definition 4.2. Computing them here rather than reading a table keeps the
    three bundles on one footing.
    """
    Y, F = load(name)
    f, y = F - F.mean(), Y - Y.mean()
    s2f, s2y = float((f * f).mean()), float((y * y).mean())
    sfy = float((f * y).mean())
    a, b = f * f, f * y
    C = float(np.cov(a, b, ddof=1)[0, 1])
    Vff, Vfy = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    s_f2y2 = float((a * y * y).mean())
    m = t_at
    V = s_f2y2 / m + (s2f * s2y - (m - 2) * sfy ** 2) / (m * (m - 1))
    return dict(M=len(Y), rho=sfy / np.sqrt(s2f * s2y), lam_star=sfy / s2f,
                nu_star=C / Vff, r2=C * C / (Vff * Vfy),
                r2_normal=2 * (sfy ** 2 / (s2f * s2y)) / (1 + sfy ** 2 / (s2f * s2y)),
                t=sfy ** 2 / V, skew_y=float(pd.Series(Y).skew()),
                kurt_y=float(pd.Series(Y).kurtosis()),
                # Excess kurtosis of the product: the fourth moment that
                # drives Var(sigma^_fy) and therefore t, which is why t is not
                # a function of rho.
                kurt_fy=float(pd.Series(f * y).kurtosis()),
                binary_F=bool(len(np.unique(F)) <= 2))


def oracle_floor(name, baseline="C"):
    """Delta of the oracle weight lambda*, in percent on `baseline`'s scale.

    The oracle is the member of the family that knows lambda*, so no estimator
    built on the same shell can go below it: it is the floor every panel of
    Section 5.3 is approaching from above.

    It is exact rather than simulated, and free of n. Writing the estimator as
    the sample mean of Z_i = Y_i - lambda (F_i - mu_f), a draw without
    replacement gives Var(Zbar) = (sigma_Z^2/n)(1 - n/M) M/(M-1), so the same
    finite-population factors multiply every fixed weight and cancel in the
    ratio. What survives is the population variance ratio

        sigma_Z^2 / sigma_y^2 ,   sigma_Z^2 = sigma_y^2 - 2 lam sigma_fy
                                              + lam^2 sigma_f^2 ,

    which at lam = lambda* is 1 - rho^2 and at lam = lambda_0 = 1 is the
    denominator below. On the P_0 scale this is the Delta^star of Section 5.3.
    """
    Y, F = load(name)
    f, y = F - F.mean(), Y - Y.mean()
    s2f, s2y = float((f * f).mean()), float((y * y).mean())
    sfy = float((f * y).mean())
    rho2 = sfy ** 2 / (s2f * s2y)

    if baseline == "C":
        return 100.0 * (-rho2)
    if baseline == "P0":
        # sigma_Z^2 of the fixed weight lambda_0 = 1, in units of sigma_y^2.
        base = (s2y - 2.0 * LAMBDA_0 * sfy + LAMBDA_0 ** 2 * s2f) / s2y
        return 100.0 * ((1.0 - rho2) / base - 1.0)
    if baseline == "P+":
        raise ValueError("P_+ is an estimated weight, so its scale moves with "
                         "n; the floor is not a horizontal line there")
    raise KeyError(baseline)


def _boot_draws(rng, Fs, Ys, u_var, n, n_boot):
    """Paired bootstrap of the labelled sample.

    Returns the resampled covariance, prediction variance and OLS weight
    ``(cov_b, var_b, nu_b)``, each of shape ``(replicates, n_boot)``. nu-hat is
    re-estimated inside every resample: holding it fixed understates the
    resampling variability of the Double-PPI covariance badly on heavy-tailed
    predictors.
    """
    m = Fs.shape[0]
    idx = rng.integers(0, n, size=(m, n_boot, n))
    Fb = np.take_along_axis(np.broadcast_to(Fs[:, None, :], (m, n_boot, n)),
                            idx, axis=2)
    Yb = np.take_along_axis(np.broadcast_to(Ys[:, None, :], (m, n_boot, n)),
                            idx, axis=2)
    dF = Fb - Fb.mean(axis=2, keepdims=True)
    dY = Yb - Yb.mean(axis=2, keepdims=True)
    cov_b = (dF * dY).sum(axis=2) / (n - 1.0)
    var_b = (dF ** 2).sum(axis=2) / (n - 1.0)

    u, v = dF ** 2, dF * dY
    du = u - u.mean(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu_b = ((du * (v - v.mean(axis=2, keepdims=True))).sum(axis=2)
                / (du ** 2).sum(axis=2))
    return cov_b, var_b, np.nan_to_num(nu_b)


def _snr(centre, boot):
    """t = centre^2 / Var_boot(centre), the variance centred at the estimate."""
    v = ((boot - centre[:, None]) ** 2).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, centre ** 2 / v, 0.0)


def cell(rng, reps, n, Y, F, theta, f_var, n_boot=N_BOOT, chunk=None,
         shrink_too=True):
    """Squared error of every estimator over `reps` samples of size n."""
    keys = ESTIMATORS if shrink_too else ESTIMATORS_PLAIN
    total = {k: 0.0 for k in keys}
    N = len(F)
    n_un = N - n
    sum_f, sumsq_f = float(F.sum()), float((F ** 2).sum())
    rules = {r: shrink.s_exponential(h, k) for r, (h, k) in SHRINK_RULES.items()}

    if chunk is None:                       # keep the bootstrap block bounded
        chunk = max(32, min(20000, 20_000_000 // max(n_boot * n, 1)))

    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        idx = sample_without_replacement(rng, m, n, N)
        Fs, Ys = F[idx], Y[idx]

        u_mean = (sum_f - Fs.sum(axis=1)) / n_un
        u_var = ((sumsq_f - (Fs ** 2).sum(axis=1))
                 - n_un * u_mean ** 2) / (n_un - 1.0)

        f_bar, y_bar = Fs.mean(axis=1), Ys.mean(axis=1)
        dF, dY = Fs - f_bar[:, None], Ys - y_bar[:, None]
        cov = (dF * dY).sum(axis=1) / (n - 1.0)
        var_f = (dF ** 2).sum(axis=1) / (n - 1.0)
        var_y = (dY ** 2).sum(axis=1) / (n - 1.0)

        u, v = dF ** 2, dF * dY
        du = u - u.mean(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            nu_hat = ((du * (v - v.mean(axis=1, keepdims=True))).sum(axis=1)
                      / (du ** 2).sum(axis=1))
        nu_hat = np.nan_to_num(nu_hat)

        gap = var_f - u_var
        d0, dnu = cov - gap, cov - nu_hat * gap
        denom = (1.0 + n / n_un) * f_var
        correction = f_bar - u_mean

        # P_0 carries the same finite-N factor as the tuned weights. Without
        # it the complement construction leaves it with an effective weight of
        # 1 + n/N_unl rather than lambda_0: writing Fbar' as the residual makes
        # Fbar - Fbar' = (1 + n/N_unl)(Fbar - mu_f), and only the estimated
        # weights cancel that through their denominator. With the factor the
        # correction term is lambda_0 (Fbar - mu_f), which is what PPI means.
        weights = {"P0": np.full(m, 1.0 / (1.0 + n / n_un)),
                   "P+": cov / denom, "D0": d0 / denom, "Dnu": dnu / denom}

        err = {"C": y_bar}
        for name, W in weights.items():
            err[name] = y_bar - W * correction

        if shrink_too:
            cov_b, var_b, nu_b = _boot_draws(rng, Fs, Ys, u_var, n, n_boot)
            gap_b = var_b - u_var[:, None]

            t_hat = {"P+": _snr(cov, cov_b),
                     "D0": _snr(d0, cov_b - gap_b),
                     "Dnu": _snr(dnu, cov_b - nu_b * gap_b)}
            # Var(sigma^PPI(nu)) = V* + V_ff (nu - nu*)^2 is exactly quadratic
            # in nu, so shrinking nu-hat by s minimises (1-s)^2 nu*^2 +
            # s^2 Var(nu-hat): Proposition 3.2 one level down, same oracle and
            # same threshold at t_nu = 1.
            t_nu = _snr(nu_hat, nu_b)

            err["S0_P+"] = y_bar - FIXED_S * weights["P+"] * correction

            # The Gaussian plug-in: E[F~^2 Y~^2] is replaced by its normal
            # value, which turns t into a function of the sample correlation
            # alone. Fed to the same rule, it isolates what the choice of
            # variance estimator costs.
            with np.errstate(divide="ignore", invalid="ignore"):
                rho2 = np.where(var_f * var_y > 0,
                                cov ** 2 / (var_f * var_y), 0.0)
            t_gauss = (n - 1.0) * rho2 / (1.0 + rho2)
            err[GAUSS_KEY] = (y_bar - rules[GAUSS_RULE](t_gauss, n)
                              * weights["P+"] * correction)
            for rname, rule in rules.items():
                for w in SHRINKABLE:
                    err["%s_%s" % (rname, w)] = (
                        y_bar - rule(t_hat[w], n) * weights[w] * correction)

            # Second order alone, and both orders together. For the combined
            # rules the outer statistic is the signal-to-noise of the guarded
            # covariance itself; the inner factor is held at its point value
            # inside the bootstrap, which avoids a nested resampling.
            second = [(r, rules[r](t_nu, n)) for r in RULE_ORDER]
            second.append(("S0", np.full(m, FIXED_S)))
            for tag, s2 in second:
                cov_g = cov - s2 * nu_hat * gap
                w_g = cov_g / denom
                err["%s_Gnu" % tag] = y_bar - w_g * correction
                # The two-level guard applies its own rule again on the outside,
                # so each parametrisation is carried through both levels rather
                # than mixed with another one.
                if tag == "S0":
                    s1 = FIXED_S
                else:
                    s1 = rules[tag](
                        _snr(cov_g, cov_b - s2[:, None] * nu_b * gap_b), n)
                err["%s_Both" % tag] = y_bar - s1 * w_g * correction

        for k in keys:
            total[k] += float(((err[k] - theta) ** 2).sum())
        done += m

    return {k: v / reps for k, v in total.items()}


#: Replicates per cell, the 50 000 of Section 5.1 at every n. Every estimator
#: in a cell shares its draws, so the differences the figures show are far
#: better determined than this number suggests for the levels.
REPS = 50_000


def reps_for(n):
    return REPS


#: The grid of Section 5.1: nine points from 10 to 1000, four per decade.
N_GRID = np.array([10, 18, 32, 56, 100, 180, 320, 560, 1000])

#: The plain sweep once ran further than the bootstrap one; both grids are
#: now the one the text states.
N_GRID_PLAIN = N_GRID

#: Every population's grid is truncated at a fifth of it so the complement
#: stays large: with the unlabelled block the residual of the labelled draw,
#: Fbar - Fbar' is inflated by M/(M-n), and a fifth holds that factor at or
#: below 1.25 everywhere.
MAX_SHARE = 0.20


def n_values(name, plain=False):
    """The grid points a population is large enough to carry.

    The cap floor(M/5) itself is appended wherever it lies below the top of
    the grid, so that n_max = floor(M/5) is the largest sample size of every
    truncated population: freesolv (M = 642) ends at 128, agb_chave05_dry
    (M = 4004) at 800, cu_scanner3 (M = 99) at 19.
    """
    M = len(load(name)[0])
    g = N_GRID_PLAIN if plain else N_GRID
    cap = int(MAX_SHARE * M)
    kept = g[g <= cap]
    if cap < g[-1] and cap not in kept:
        kept = np.append(kept, cap)
    return kept


#: The plain sweep carries the same replication as the bootstrap one.
PLAIN_REPS = REPS


def sweep(name, reps_fn=reps_for, seed=0, n_boot=N_BOOT, plain=False):
    """Delta against the classical mean over n, for one population."""
    Y, F = load(name)
    theta = float(Y.mean())
    f_var = float(np.var(F, ddof=1))
    ns = n_values(name, plain=plain)
    keys = ESTIMATORS_PLAIN if plain else ESTIMATORS
    mse = {k: np.empty(len(ns)) for k in keys}
    for i, n in enumerate(ns):
        r = PLAIN_REPS if plain else int(reps_fn(n))
        got = cell(np.random.default_rng(seed + 977 * i), r,
                   int(n), Y, F, theta, f_var, n_boot=n_boot,
                   shrink_too=not plain)
        for k in keys:
            mse[k][i] = got[k]
    out = {k: 100.0 * (v / mse["C"] - 1.0) for k, v in mse.items()}
    return dict(n=ns.astype(float), **out)


def dataset(name, force=False, plain=False):
    """Cached sweep. `plain` drops the shrinkage rules and extends the grid."""
    tag = ("atlasx_" if plain else "atlas_") + name
    return sim.run_or_load(tag, lambda: sweep(name, plain=plain), force=force)
