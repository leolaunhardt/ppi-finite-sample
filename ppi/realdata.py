"""
The six benchmark datasets of Section 6.2.

Each dataset is a set of pairs (Y, F) with F the model's prediction. The
dataset is treated as the population: theta is the mean of Y over all of it,
the unlabelled block is the predictions of all of it, and a replicate draws n
labelled pairs from it with replacement. Mean squared error is then measured
against that population mean, so it is exactly defined and needs no model.

The estimators are the ones of Chapters 3 to 5, all in their single-sample
form:

    C        the sample mean
    P0       PPI at the fixed weight lambda_0 = 1
    P+       PPI++,               nu = 0 in the Double-PPI covariance
    D0       Double-PPI at the guessed nu_0 = 1
    Dnu      Double-PPI at the estimated nu-hat of eq:nu-ols
    S_E(.)   each of the three, shrunk by the exponential rule

The Double-PPI covariance estimator is eq:def-ppi-cov,

    sigma^D(nu) = Cov(F, Y) - nu (Var(F) - Var(F')),

with Var(F') taken over the whole population, and nu-hat the least squares
slope of the cross-products on the squared prediction residuals.
"""

from pathlib import Path

import numpy as np

from . import gaussian as shrink

#: The four ppi_py mean-estimation benchmarks, as the ``.npz`` files the ppi_py
#: examples download (keys ``Y`` and ``Yhat``). They are not redistributed
#: here; ``data/README.md`` says where to get them and where to put them.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ppi_py"

#: Order as Table 6.2 lists them.
#: Only the mean-estimation benchmarks. ppi_py's census examples are a linear
#: (`ppi_ols_ci`) and a logistic (`ppi_logistic_ci`) regression, so their
#: published estimand is a coefficient rather than a mean and they are out of
#: scope here. `alphafold` stays: its notebook reports an odds ratio, which is
#: a ratio of two means and is therefore covered by the theory of Chapter 3.
DATASETS = ["galaxies", "forest", "alphafold", "ballots"]

DATASET_LABEL = {
    "galaxies": r"\texttt{galaxies}",
    "forest": r"\texttt{forest}",
    "alphafold": r"\texttt{alphafold}",
    "ballots": r"\texttt{ballots}",
    "census_income": r"\texttt{census\_income}",
    "census_healthcare": r"\texttt{census\_healthcare}",
}


def load(name):
    """Return ``(Y, F)`` as float arrays."""
    with np.load(DATA_DIR / (name + ".npz"), allow_pickle=True) as z:
        return z["Y"].astype(float).ravel(), z["Yhat"].astype(float).ravel()


def summary(name):
    """Population correlation, calibration and size."""
    Y, F = load(name)
    rho = float(np.corrcoef(F, Y)[0, 1])
    lam = float(np.cov(F, Y, ddof=1)[0, 1] / np.var(F, ddof=1))
    return dict(n=len(Y), rho=rho, lam_star=lam,
                sigma_y=float(np.std(Y, ddof=1)),
                sigma_f=float(np.std(F, ddof=1)))


def degrade(F, noise, seed=12345):
    """Add noise to the predictions, holding lambda* fixed.

    Adding e ~ N(0, (noise sigma_f)^2) multiplies Var(F) by 1 + noise^2 and
    leaves Cov(F, Y) alone, so lambda* = Cov/Var falls by that factor and rho
    by its square root. Rescaling by 1/(1 + noise^2) puts lambda* back and
    leaves the correlation where the noise put it.
    """
    rng = np.random.default_rng(seed)
    sigma_f = float(np.std(F, ddof=1))
    noisy = F + rng.normal(0.0, noise * sigma_f, size=len(F))
    return noisy / (1.0 + noise ** 2)


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------

ESTIMATORS = ["C", "P0", "P+", "D0", "Dnu", "SE_P+", "SE_D0", "SE_Dnu"]

#: The exponential rule used throughout, at the parameters of Section 5.1.
SHRINK_RULE = shrink.s_exponential(1.5, 0.5)

#: Double-PPI's guessed weight.
NU_0 = 1.0


#: Below this population size a partial sort over the whole population is the
#: cheaper way to draw a sample without replacement; above it the sample is a
#: small enough share that redrawing collisions converges in a few rounds.
SMALL_POPULATION = 5000


def sample_without_replacement(rng, m, n, N, max_rounds=64):
    """`m` rows of `n` distinct indices into a population of size `N`.

    Both branches are uniform over the subsets of size n. The second redraws
    only the positions that collide; the procedure is equivariant under
    relabelling the population, so what it returns is a simple random sample
    and not merely an approximation to one.
    """
    if N <= SMALL_POPULATION:
        keys = rng.random((m, N))
        return np.argpartition(keys, n - 1, axis=1)[:, :n]

    idx = rng.integers(0, N, size=(m, n))
    for _ in range(max_rounds):
        order = np.argsort(idx, axis=1)
        srt = np.take_along_axis(idx, order, axis=1)
        repeat = np.zeros(srt.shape, dtype=bool)
        repeat[:, 1:] = srt[:, 1:] == srt[:, :-1]
        if not repeat.any():
            return idx
        collide = np.zeros_like(repeat)
        np.put_along_axis(collide, order, repeat, axis=1)
        idx[collide] = rng.integers(0, N, size=int(collide.sum()))
    raise RuntimeError("sampling did not converge at n=%d, N=%d" % (n, N))


def cell(rng, reps, n, Y, F, theta, f_mean, f_var, chunk=20000):
    """Squared error of every estimator over `reps` samples of size n.

    The labelled sample is drawn without replacement, and the unlabelled block
    is its complement, so that no observation is used as both. The complement
    has size N - n and its mean and variance are therefore random, which is
    what the finite-N forms of Section 3.2 assume. One simplification follows:
    the labelled and unlabelled predictions together are the whole population,
    so the pooled variance in the denominator of the weight is exactly the
    population variance rather than an estimate of it.
    """
    total = {name: 0.0 for name in ESTIMATORS}
    done = 0
    N = len(F)
    n_un = N - n
    if n_un < 2:
        raise ValueError("n = %d leaves no unlabelled block in N = %d" % (n, N))
    sum_f = float(F.sum())
    sumsq_f = float((F ** 2).sum())

    while done < reps:
        m = min(chunk, reps - done)
        idx = sample_without_replacement(rng, m, n, N)
        Fs, Ys = F[idx], Y[idx]

        # The unlabelled block is everything the labelled sample left behind.
        u_sum = sum_f - Fs.sum(axis=1)
        u_sumsq = sumsq_f - (Fs ** 2).sum(axis=1)
        u_mean = u_sum / n_un
        u_var = (u_sumsq - n_un * u_mean ** 2) / (n_un - 1.0)

        f_bar, y_bar = Fs.mean(axis=1), Ys.mean(axis=1)
        dF, dY = Fs - f_bar[:, None], Ys - y_bar[:, None]
        cov = (dF * dY).sum(axis=1) / (n - 1.0)
        var_f = (dF ** 2).sum(axis=1) / (n - 1.0)
        var_y = (dY ** 2).sum(axis=1) / (n - 1.0)

        # nu-hat of eq:nu-ols: cross-products regressed on squared residuals.
        u, v = dF ** 2, dF * dY
        du = u - u.mean(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            nu_hat = ((du * (v - v.mean(axis=1, keepdims=True))).sum(axis=1)
                      / (du ** 2).sum(axis=1))
        nu_hat = np.nan_to_num(nu_hat)

        # eq:def-ppi-cov, with Var(F') estimated on the unlabelled block.
        def weight(nu):
            d_cov = cov - nu * (var_f - u_var)
            return d_cov / ((1.0 + n / n_un) * f_var)

        correction = f_bar - u_mean

        # Gaussian plug-in statistic for the shrinkage rules.
        with np.errstate(divide="ignore", invalid="ignore"):
            rho2 = np.where(var_f * var_y > 0,
                            cov ** 2 / (var_f * var_y), 0.0)
        t_hat = (n - 1.0) * rho2 / (1.0 + rho2)
        s_hat = SHRINK_RULE(t_hat, n)

        w = {"P+": weight(0.0), "D0": weight(NU_0), "Dnu": weight(nu_hat)}
        # P_0 carries the same finite-N factor as the tuned weights; see the
        # note in populations.py. Without it the complement construction leaves its
        # effective weight at 1 + n/N_unl instead of lambda_0.
        est = {"C": y_bar,
               "P0": y_bar - correction / (1.0 + n / n_un)}
        for name, wt in w.items():
            est[name] = y_bar - wt * correction
            est["SE_" + name] = y_bar - s_hat * wt * correction

        for name in ESTIMATORS:
            total[name] += float(((est[name] - theta) ** 2).sum())
        done += m

    return {k: v / reps for k, v in total.items()}


#: With the unlabelled block defined as the complement, Fbar - Fbar' is
#: inflated by M/(M-n). A tenth of the population holds that factor below
#: 1.12, so that what the panels show is the estimator rather than the size of
#: the pool it was drawn from. Same convention as `populations.py`.
MAX_SHARE = 0.20


def n_values_for(name, n_values):
    """The grid of `n_values` that this population is large enough to carry."""
    M = len(load(name)[0])
    return np.asarray([n for n in n_values if n <= MAX_SHARE * M], dtype=int)


def sweep(name, n_values, reps, seed=0, noise=0.0):
    """Delta against the classical mean over n, for one dataset.

    `noise` degrades the predictor: independent Gaussian noise of that many
    prediction standard deviations is added, and the result is rescaled by
    1/(1 + noise^2) so that lambda* is left where it was. Only the
    correlation moves, by the factor 1/sqrt(1 + noise^2), which is what puts
    a dataset into the regime where shrinkage has something to do. Degrading
    the calibration as well would merely make PPI at a fixed weight look bad
    and would say nothing about shrinkage.
    """
    Y, F = load(name)
    n_values = n_values_for(name, n_values)
    if noise > 0.0:
        F = degrade(F, noise)

    theta = float(Y.mean())
    f_mean, f_var = float(F.mean()), float(np.var(F, ddof=1))

    mse = {k: np.empty(len(n_values)) for k in ESTIMATORS}
    for i, n in enumerate(n_values):
        got = cell(np.random.default_rng(seed + i), reps, int(n),
                   Y, F, theta, f_mean, f_var)
        for k in ESTIMATORS:
            mse[k][i] = got[k]
    return {k: 100.0 * (v / mse["C"] - 1.0) for k, v in mse.items()}
