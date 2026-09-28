"""
Monte-Carlo machinery for the empirical figures.

The estimators are coded from their definitions in Section 3.2 and 3.3, in the
single-sample, finite-N form the thesis evaluates:

* classical mean          Definition 3.1  (``def:classical``)
* SSP, single-sample PPI++ Definition 3.2 (``def:ssp``)
* CFP, cross-fit PPI++     Definition 3.3 (``def:cfp``)

Everything is measured by mean squared error against the true theta, and
reported relative to the classical mean through Delta of Section 5,
equation ``eq:delta``.

Within one cell all estimators see the same replicates, as Section 5 requires,
so the Monte-Carlo error of a *difference* between two methods is far smaller
than that of either quantity on its own.
"""

from pathlib import Path

import numpy as np

SIMDATA_DIR = Path(__file__).resolve().parent.parent / "simdata"


# ---------------------------------------------------------------------------
# The estimators
# ---------------------------------------------------------------------------

def _sample_cov(a, b):
    """Row-wise sample covariance with the n-1 denominator."""
    n = a.shape[1]
    da = a - a.mean(axis=1, keepdims=True)
    db = b - b.mean(axis=1, keepdims=True)
    return (da * db).sum(axis=1) / (n - 1)


def _sample_var(a):
    return a.var(axis=1, ddof=1)


def theta_classical(Y):
    """Definition 3.1: the sample mean of the labelled outcomes."""
    return Y.mean(axis=1)


def theta_ssp(F, Y, Fu):
    """Definition 3.2, single-sample PPI++.

    theta = Ybar - lambda_hat (Fbar - Fbar'), with
    lambda_hat = Cov(F, Y) / ((1 + n/N) Var(F)).

    Var(F) is taken over the labelled *and* the unlabelled predictions. The
    definition writes only ``Varhat(F)``, but that is the reading which agrees
    with Proposition 3.1 at c = 1 and with the explicit
    ``Varhat(F^(k) (+) F'^(k))`` of Definition 3.3; taking the variance on the
    labelled block alone turns the weight into the OLS slope, whose variance
    is a different quantity altogether. See the note in the module header of
    figures/fig_3_2a_cf_vs_ss_nominal.py.
    """
    n, N = F.shape[1], Fu.shape[1]
    pooled = np.concatenate([F, Fu], axis=1)
    lam = _sample_cov(F, Y) / ((1.0 + n / N) * _sample_var(pooled))
    return Y.mean(axis=1) - lam * (F.mean(axis=1) - Fu.mean(axis=1))


def theta_cfp(F, Y, Fu):
    """Definition 3.3, cross-fit PPI++.

    The labelled and the unlabelled samples are each split in half. Fold k
    supplies the weight, the complementary fold k' the means, and the two fold
    estimators are averaged. The denominator of the weight uses the labelled
    and unlabelled predictions of fold k together, as ``eq:cfp-lambda``
    prescribes.
    """
    n, N = F.shape[1], Fu.shape[1]
    h, hu = n // 2, N // 2
    folds = [(slice(0, h), slice(0, hu)), (slice(h, 2 * h), slice(hu, 2 * hu))]

    out = 0.0
    for k in (0, 1):
        lab, unlab = folds[k]           # fold k: the weight
        lab_c, unlab_c = folds[1 - k]   # fold k': the means

        pooled = np.concatenate([F[:, lab], Fu[:, unlab]], axis=1)
        lam = (_sample_cov(F[:, lab], Y[:, lab])
               / ((1.0 + n / N) * _sample_var(pooled)))

        out = out + (Y[:, lab_c].mean(axis=1)
                     - lam * (F[:, lab_c].mean(axis=1)
                              - Fu[:, unlab_c].mean(axis=1)))
    return out / 2.0


#: Name -> estimator, in the order the figures draw them.
ESTIMATORS = {
    "C": lambda F, Y, Fu: theta_classical(Y),
    "SSP": theta_ssp,
    "CFP": theta_cfp,
}


# ---------------------------------------------------------------------------
# The Gaussian data-generating process
# ---------------------------------------------------------------------------

def draw_gaussian(rng, reps, n, N, rho, sigma_f=1.0, sigma_y=1.0,
                  mu_f=0.0, mu_y=0.0):
    """Jointly normal (F, Y) pairs plus an independent unlabelled block.

    Returns ``(F, Y, Fu)`` of shapes ``(reps, n)``, ``(reps, n)``,
    ``(reps, N)``. The estimand is theta = mu_y.
    """
    Z1 = rng.standard_normal((reps, n))
    Z2 = rng.standard_normal((reps, n))
    F = mu_f + sigma_f * Z1
    Y = mu_y + sigma_y * (rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2)
    Fu = mu_f + sigma_f * rng.standard_normal((reps, N))
    return F, Y, Fu


def mse_at(n, N, rho, reps, seed=0, chunk=4000, **dgp):
    """Mean squared error of every estimator at one (n, N, rho) cell.

    Replicates are processed in chunks to keep the unlabelled block out of
    memory; every estimator sees the same replicates.
    """
    rng = np.random.default_rng(seed)
    theta_true = dgp.get("mu_y", 0.0)

    total = {name: 0.0 for name in ESTIMATORS}
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        F, Y, Fu = draw_gaussian(rng, m, n, N, rho, **dgp)
        for name, est in ESTIMATORS.items():
            total[name] += float(((est(F, Y, Fu) - theta_true) ** 2).sum())
        done += m

    return {name: total[name] / reps for name in ESTIMATORS}


def sweep_n(n_values, N, rho, reps, seed=0, **dgp):
    """Run :func:`mse_at` over a grid of sample sizes.

    Returns ``(n_values, {name: mse array})``. Each cell gets its own seed, so
    a rerun reproduces the figure exactly and adding a cell does not disturb
    the others.
    """
    n_values = np.asarray(n_values, dtype=int)
    mse = {name: np.empty(len(n_values)) for name in ESTIMATORS}
    for i, n in enumerate(n_values):
        cell = mse_at(int(n), N, rho, reps, seed=seed + i, **dgp)
        for name in ESTIMATORS:
            mse[name][i] = cell[name]
    return n_values, mse


def delta_percent(mse, baseline="C"):
    """Delta of equation ``eq:delta``: MSE(theta) / MSE(Ybar) - 1, in percent."""
    ref = mse[baseline]
    return {name: 100.0 * (mse[name] / ref - 1.0) for name in mse}


# ---------------------------------------------------------------------------
# Caching, so that two figures can share one simulation
# ---------------------------------------------------------------------------

def run_or_load(name, compute, force=False):
    """Return ``compute()``, cached in ``simdata/name.npz``.

    Delete the file, or pass ``force=True``, to run the simulation again.
    """
    SIMDATA_DIR.mkdir(parents=True, exist_ok=True)
    path = SIMDATA_DIR / (name + ".npz")

    if path.exists() and not force:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    result = compute()
    np.savez_compressed(path, **result)
    print("simulated and cached", path)
    return result
