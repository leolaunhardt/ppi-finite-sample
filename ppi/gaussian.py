"""
Monte-Carlo machinery for the shrinkage rules of Section 5.

The rules of Section 3.2 are truncated, nonlinear functions of an estimated
statistic, so unlike everything in Chapters 3 to 5 they have no closed form and
have to be simulated. What is simulated is only the labelled sample: the
unlabelled block of size N enters the estimators solely through its mean and
its sum of squares, and both of those are drawn exactly from their own
distributions instead of from N points. That is what makes N = 10^4 affordable.

    mean of N iid N(mu_f, sigma_f^2)      ~  N(mu_f, sigma_f^2/N)
    their centred sum of squares          ~  sigma_f^2 * chi^2_{N-1}

Both are exact and independent, so the fast path is not an approximation. It is
checked against a direct simulation in `validate_fast_path`.
"""

import numpy as np

from . import simulate as sim

#: Curvature of the exponential family throughout Section 5, as the text's
#: footnote fixes it: s_E(h) abbreviates s_E(h, 1).
KAPPA = 1.0


# ---------------------------------------------------------------------------
# Drawing one cell
# ---------------------------------------------------------------------------

def draw_labelled(rng, reps, n, rho, sigma_f=1.0, sigma_y=1.0):
    """Labelled pairs, jointly normal with unit variances by default."""
    Z1 = rng.standard_normal((reps, n))
    Z2 = rng.standard_normal((reps, n))
    F = sigma_f * Z1
    Y = sigma_y * (rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2)
    return F, Y


def draw_unlabelled_stats(rng, reps, N, sigma_f=1.0):
    """The unlabelled block's mean and centred sum of squares, drawn exactly."""
    mean = rng.normal(0.0, sigma_f / np.sqrt(N), size=reps)
    ss = sigma_f ** 2 * rng.chisquare(N - 1, size=reps)
    return mean, ss


def _pooled_var(F, F_bar, ss_lab, n, u_mean, u_ss, N):
    """Sample variance of the labelled and unlabelled predictions together."""
    total = n + N
    grand = (n * F_bar + N * u_mean) / total
    ss = (ss_lab + u_ss
          + n * (F_bar - grand) ** 2
          + N * (u_mean - grand) ** 2)
    return ss / (total - 1.0)


def cell(rng, reps, n, N, rho, rules, sigma_f=1.0, sigma_y=1.0, n_boot=None):
    """Mean squared error of every rule at one (n, N, rho) cell.

    `rules` maps a name to a callable ``s(t_hat, n)`` giving the shrinkage
    factor; the name ``"C"`` is reserved for the classical mean and ``None``
    as the callable means no shrinkage at all. With `n_boot` the threshold
    statistic is the paired bootstrap of eq:boot with that many resamples,
    the input of Section 3.3; without it the Gaussian plug-in of eq:t-gauss.
    """
    F, Y = draw_labelled(rng, reps, n, rho, sigma_f, sigma_y)
    u_mean, u_ss = draw_unlabelled_stats(rng, reps, N, sigma_f)

    F_bar, Y_bar = F.mean(axis=1), Y.mean(axis=1)
    dF, dY = F - F_bar[:, None], Y - Y_bar[:, None]
    ss_lab = (dF ** 2).sum(axis=1)
    cov = (dF * dY).sum(axis=1) / (n - 1.0)
    var_f_lab = ss_lab / (n - 1.0)
    var_y_lab = (dY ** 2).sum(axis=1) / (n - 1.0)

    # Definition 3.2: the denominator uses labelled and unlabelled together.
    var_pooled = _pooled_var(F, F_bar, ss_lab, n, u_mean, u_ss, N)
    lam = cov / ((1.0 + n / N) * var_pooled)
    correction = F_bar - u_mean

    if n_boot:
        from .hyper import boot_t          # here, to avoid a circular import
        t_hat = boot_t(rng, F, Y, cov, n, n_boot)
    else:
        # Gaussian plug-in of eq:t-gauss: t = (n-1) rho_hat^2 / (1 + rho_hat^2).
        with np.errstate(divide="ignore", invalid="ignore"):
            rho2 = np.where(var_f_lab * var_y_lab > 0,
                            cov ** 2 / (var_f_lab * var_y_lab), 0.0)
        t_hat = (n - 1.0) * rho2 / (1.0 + rho2)

    out = {}
    for name, rule in rules.items():
        if name == "C":
            err = Y_bar
        else:
            s = 1.0 if rule is None else rule(t_hat, n)
            err = Y_bar - s * lam * correction
        out[name] = float((err ** 2).sum())
    return out


def sweep(rules, n_values, N, rho_values, reps, seed=0, chunk=20000,
          n_boot=None, **dgp):
    """Mean squared error over a grid of (n, rho). Shape ``(rho, n)`` each.

    With `n_boot` the bootstrap block ``(chunk, B, n)`` is what bounds the
    memory, so the chunk shrinks with n and B as in ``hyper.py``.
    """
    n_values = np.asarray(n_values, dtype=int)
    rho_values = np.atleast_1d(np.asarray(rho_values, dtype=float))
    mse = {name: np.empty((len(rho_values), len(n_values))) for name in rules}

    for j, rho in enumerate(rho_values):
        for i, n in enumerate(n_values):
            rng = np.random.default_rng(seed + 1000 * j + i)
            block = chunk
            if n_boot:
                block = max(64, min(chunk, 20_000_000 // max(n_boot * int(n), 1)))
            total = {name: 0.0 for name in rules}
            done = 0
            while done < reps:
                m = min(block, reps - done)
                got = cell(rng, m, int(n), N, float(rho), rules,
                           n_boot=n_boot, **dgp)
                for name in rules:
                    total[name] += got[name]
                done += m
            for name in rules:
                mse[name][j, i] = total[name] / reps
    return mse


def delta(mse, baseline="C"):
    """Delta of ``eq:delta`` in percent, against the classical mean."""
    ref = mse[baseline]
    return {k: 100.0 * (v / ref - 1.0) for k, v in mse.items()}


# ---------------------------------------------------------------------------
# The rules compared in Section 5
# ---------------------------------------------------------------------------

def s_fixed(value):
    """A constant shrinkage factor, equation ``eq:s-fixed``."""
    return lambda t, n: value


def s_exponential(h, kappa=KAPPA):
    """The tuned exponential family, equation ``eq:expfamily``, on t-hat."""
    def rule(t, n):
        with np.errstate(divide="ignore", over="ignore"):
            ratio = np.where(t > 0, h / np.where(t > 0, t, 1.0), np.inf)
        return np.exp2(-(ratio ** kappa))
    return rule


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_fast_path(n=20, N=400, rho=0.3, reps=200_000, seed=7):
    """Compare the sufficient-statistic path against drawing all N points."""
    rules = {"C": None, "P+": None, "S0": s_fixed(0.5), "SE4": s_exponential(4.0)}

    rng = np.random.default_rng(seed)
    fast = {k: 0.0 for k in rules}
    done = 0
    while done < reps:
        m = min(20000, reps - done)
        got = cell(rng, m, n, N, rho, rules)
        for k in rules:
            fast[k] += got[k]
        done += m
    fast = {k: v / reps for k, v in fast.items()}

    rng = np.random.default_rng(seed)
    slow = {k: 0.0 for k in rules}
    done = 0
    while done < reps:
        m = min(2000, reps - done)
        F, Y = draw_labelled(rng, m, n, rho)
        Fu = rng.standard_normal((m, N))
        F_bar, Y_bar = F.mean(1), Y.mean(1)
        dF, dY = F - F_bar[:, None], Y - Y_bar[:, None]
        cov = (dF * dY).sum(1) / (n - 1.0)
        var_f = (dF ** 2).sum(1) / (n - 1.0)
        var_y = (dY ** 2).sum(1) / (n - 1.0)
        pooled = np.concatenate([F, Fu], axis=1).var(axis=1, ddof=1)
        lam = cov / ((1.0 + n / N) * pooled)
        corr = F_bar - Fu.mean(1)
        rho2 = cov ** 2 / (var_f * var_y)
        t = (n - 1.0) * rho2 / (1.0 + rho2)
        for k, rule in rules.items():
            err = Y_bar if k == "C" else Y_bar - (1.0 if rule is None
                                                  else rule(t, n)) * lam * corr
            slow[k] += float((err ** 2).sum())
        done += m
    slow = {k: v / reps for k, v in slow.items()}

    return {k: (fast[k], slow[k], 100.0 * (fast[k] / slow[k] - 1.0))
            for k in rules}
