"""
Closed-form results transcribed from the thesis.

Every function names the numbered result of the paper it comes from.
The text is the source of truth: if a formula changes there, it changes here,
and nowhere else.
"""

import numpy as np

#: Fold-size constant c of Section 3.3. c = 1 is the single sample used by
#: single-sample PPI++ (SSP), c = 1/2 the half fold of cross-fit PPI++ (CFP).
C_SINGLE = 1.0
C_CROSSFIT = 0.5


def ppp_variance_gaussian(n, sigma_y2, sigma_fy, sigma_f2, c=C_SINGLE):
    """Finite-sample variance of PPI++ under joint normality.

    Proposition 3.1 (``prop:gauss-var``), the case N -> infinity::

        Var = sigma_y^2/n * (1 + 1/(cn - 1))
              - (cn - 2)/(n(cn - 1)) * sigma_fy^2/sigma_f^2
    """
    n = np.asarray(n, dtype=float)
    return (sigma_y2 / n * (1.0 + 1.0 / (c * n - 1.0))
            - (c * n - 2.0) / (n * (c * n - 1.0)) * sigma_fy ** 2 / sigma_f2)


def ppp_relative_mse_gaussian(n, rho, c=C_SINGLE):
    """Delta of PPI++ against the classical mean, Gaussian case, N -> infinity.

    Proposition 3.1 divided by the classical risk sigma_y^2/n collapses to

        MSE / MSE(Ybar) = 1 - rho^2 + (1 + rho^2)/(cn - 1),

    in which the scale sigma_y^2/sigma_f^2 has cancelled, so only (n, rho, c)
    remain. Returned as Delta of ``eq:delta`` in percent:

        Delta = 100 [ (1 + rho^2)/(cn - 1) - rho^2 ].

    The two terms are the whole finite-sample story: a gain of rho^2 for using
    the predictions at all, against a price of (1 + rho^2)/(cn - 1) for having
    to estimate the weight. They balance at rho^2 = 1/(cn - 2), which is
    Corollary 3.2.
    """
    n = np.asarray(n, dtype=float)
    rho = np.asarray(rho, dtype=float)
    return 100.0 * ((1.0 + rho ** 2) / (c * n - 1.0) - rho ** 2)


def gaussian_improvement_threshold(n, c=C_SINGLE):
    """The correlation PPI++ needs to beat the classical mean, under normality.

    Corollary 3.2 (``cor:thres-gauss``) states the condition as
    ``1/(cn - 2) < rho^2``; solved for the correlation this is the curve

        rho(n) = 1/sqrt(cn - 2),

    above which PPI++ has the smaller variance. Returns NaN where the
    condition cannot be met by any correlation, that is where cn - 2 <= 1.
    """
    n = np.asarray(n, dtype=float)
    denom = c * n - 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(denom > 0.0, 1.0 / np.sqrt(np.abs(denom)), np.nan)
    return np.where(rho <= 1.0, rho, np.nan)


def gaussian_threshold_onset(c=C_SINGLE):
    """Smallest n at which the threshold of Corollary 3.2 is attainable.

    The curve reaches rho = 1 where cn - 2 = 1, so n = 3/c: n = 3 for the
    single sample and n = 6 for the cross-fit half fold.
    """
    return 3.0 / c


# ---------------------------------------------------------------------------
# Shrinkage, Section 4.2
# ---------------------------------------------------------------------------

def var_sigma_hat_gaussian(n, sigma_f2=1.0, sigma_y2=1.0, rho=0.0, c=C_SINGLE):
    """Var(sigma_hat_fy) under joint normality, Section 3.4.

        Var = (sigma_fy^2 + sigma_f^2 sigma_y^2) / (cn - 1)
    """
    n = np.asarray(n, dtype=float)
    sigma_fy2 = rho ** 2 * sigma_f2 * sigma_y2
    return (sigma_fy2 + sigma_f2 * sigma_y2) / (c * n - 1.0)


def t_lambda_gaussian(n, rho, c=C_SINGLE):
    """The signal-to-noise statistic of the weight, equation ``eq:tlam-gauss``.

        t_lambda = lambda*^2 / Var(lambda_hat) = (cn - 1) rho^2 / (1 + rho^2)

    The scale factor sigma_y^2/sigma_f^2 cancels between signal and noise, so
    only (n, rho) survive. The improvement threshold sits at t_lambda = 1.
    """
    n = np.asarray(n, dtype=float)
    rho = np.asarray(rho, dtype=float)
    return (c * n - 1.0) * rho ** 2 / (1.0 + rho ** 2)


def oracle_shrinkage(t_lambda):
    """The oracle shrinkage factor, Proposition 3.2 (``eq:oracle``).

        s* = t_lambda / (1 + t_lambda)

    It is 1/2 exactly at the threshold t_lambda = 1, the half-value point.
    """
    t = np.asarray(t_lambda, dtype=float)
    return t / (1.0 + t)


def delta_shrunken_gaussian(n, rho, s, c=C_SINGLE):
    """Relative MSE of the shrunken PPI++ weight against the classical mean.

    From the proof of Proposition 3.2, with the scale factors cancelled,

        MSE(theta_S(s)) / MSE(Ybar)
            = 1 - 2 s rho^2 + s^2 (rho^2 + Var(sigma_hat)/sigma_f^4),

    returned as Delta of ``eq:delta`` in percent. ``s = 1`` gives the
    unshrunken P_+, and ``s = oracle_shrinkage(t_lambda_gaussian(...))``
    reduces to the 1 - s* rho^2 of the proposition.

    The estimator is unbiased here, so this is equally its relative variance.
    """
    n = np.asarray(n, dtype=float)
    rho = np.asarray(rho, dtype=float)
    var_sig = var_sigma_hat_gaussian(n, rho=rho, c=c)
    ratio = 1.0 - 2.0 * s * rho ** 2 + s ** 2 * (rho ** 2 + var_sig)
    return 100.0 * (ratio - 1.0)


# ---------------------------------------------------------------------------
# The shrinkage rules, Section 3.2 and Table 4.1
# ---------------------------------------------------------------------------
# All four are functions of the single scalar t_lambda_hat, and all but the
# fixed rule are ordered by their half-value point h, the t at which they
# reach one half.

def _ratio(t, h):
    """(h/t)^1 with t = 0 sent to infinity rather than to a warning."""
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0.0, h / np.where(t > 0.0, t, 1.0), np.inf)


def s_fixed(t0):
    """The fixed rule, equation ``eq:s-fixed``: the oracle at a believed t0."""
    return oracle_shrinkage(t0)


def s_direct_plugin(t):
    """The direct plug-in rule, equation ``eq:s-dp``.

        s_DP = (1 - 1/t)_+

    Half-value point h = 2, and it switches off entirely at t <= 1. It is the
    member (h=2, kappa=1) of the plug-in family below.
    """
    return np.clip(1.0 - _ratio(t, 1.0), 0.0, None)


def s_plugin_family(t, h=2.0, kappa=1.0):
    """The tuned plug-in family, equation ``eq:family``.

        s_P(h, kappa) = (1 - (1/2)(h/t)^kappa)_+

    Normalised so that it equals exactly 1/2 at t = h, whatever kappa. It
    truncates to zero at t <= h 2^(-1/kappa), which is the kink the
    exponential family removes.
    """
    return np.clip(1.0 - 0.5 * _ratio(t, h) ** kappa, 0.0, None)


def s_exponential_family(t, h=2.0, kappa=1.0):
    """The tuned exponential family, equation ``eq:expfamily``.

        s_E(h, kappa) = 2^(-(h/t)^kappa)

    Also exactly 1/2 at t = h, so the two families are comparable through h.
    Smooth throughout: it decays to zero as t -> 0 instead of truncating, and
    approaches 1 like 1 - h ln2 / t, the same order as the direct plug-in.
    """
    return np.exp2(-(_ratio(t, h) ** kappa))


#: The half-value point of Definition 4.2: every rule above passes through it.
HALF_VALUE = 0.5
