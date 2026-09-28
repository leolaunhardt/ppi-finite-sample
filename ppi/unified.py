"""
The unified framework of Section 5.1 in the Gaussian case, Table A.1.

Every estimator is a weight W applied to the same estimator shell
theta_W = Ybar - W(Fbar - Fbar'), and is described by a squared bias and a
variance of that weight. They combine, by equation ``eq:bias-var``, into

    MSE(theta_W) = sigma_y^2 (1 - rho^2)/n + (sigma_f^2/n)[Bias(W)^2 + Var(W)]

which relative to the classical mean sigma_y^2/n is

    MSE(theta_W)/MSE(Ybar) = (1 - rho^2) + (rho/lambda*)^2 [Bias(W)^2 + Var(W)]

since lambda* = rho sigma_y/sigma_f. The pair (rho, lambda*) therefore fixes
everything: rho is the correlation, lambda* the calibration, and their ratio
is the scale sigma_y/sigma_f. Throughout sigma_f = 1 and lambda_0 = nu_0 = 1.
"""

import numpy as np

from . import simulate as sim

#: The believed weights of Table A.3: PPI at a fixed 1, Double-PPI likewise.
LAMBDA_0 = 1.0
NU_0 = 1.0

#: The second-order shrinkage factor of Section 4: u = 1/2 is the belief
#: t_nu = 1, exactly as s = 1/2 is the belief t_lambda = 1 one level up.
U_SHRINK = 0.5

#: Unshrunken weights, in the order the legends list them. "D+" is the tuned
#: second-order weight (the old key "Dnu" is kept as its alias) and "Duv" the
#: second-order shrunken one, u*nu-hat.
UNSHRUNKEN = ["C", "P0", "P+", "D0", "Dnu"]
UNSHRUNKEN_ALL = ["C", "P0", "P+", "D0", "D+", "Duv"]

#: Their shrunken counterparts. The classical mean and PPI at a fixed weight
#: have no random weight to shrink.
SHRUNKEN = ["S(P+)", "S(D0)", "S(Dnu)"]
SHRUNKEN_ALL = ["S(P+)", "S(D0)", "S(D+)", "S(Duv)"]

ALL_ESTIMATORS = UNSHRUNKEN + SHRUNKEN

#: The ten members of Table A.1, in the order the table lists them.
ALL_TEN = UNSHRUNKEN_ALL + SHRUNKEN_ALL

#: The order a key lists them in: each shrunken weight directly under the
#: weight it shrinks, and the columns running from no second order at all to
#: both orders shrunken. Read down for a vertical key, across for a
#: two-row horizontal one.
LEGEND_PAIRS = [("C", "P0"), ("P+", "S(P+)"), ("D0", "S(D0)"),
                ("D+", "S(D+)"), ("Duv", "S(Duv)")]
LEGEND_DOWN = [nm for pair in LEGEND_PAIRS for nm in pair]
#: Kept for a key that is filled row by row rather than column by column.
LEGEND_ACROSS = ([p[0] for p in LEGEND_PAIRS] + [p[1] for p in LEGEND_PAIRS])


# ---------------------------------------------------------------------------
# The heteroskedasticity constant
# ---------------------------------------------------------------------------

def _kn_compute(n, reps=4_000_000, seed=0, chunk=100_000):
    """K_n of Lemma 3.3, by Monte Carlo from the definition in its proof.

    With X standard normal, Xt its centred sample and Z = Xt^2,

        w_i = (Z_i - Zbar + 2/(n-1)) Xt_i,
        D   = sum_i (Z_i - Zbar)^2 + (2/(n-1)) sum_i Z_i,
        K_n = E[ (sum_i w_i^2 - (sum_i w_i)^2/n) / D^2 ].

    Every term matters at small n. Dropping the 2/(n-1) corrections and the
    centring of Z gives the asymptotically equivalent expression, which is
    what an earlier version of this function computed; it is 27% too large at
    n = 6 and 6% at n = 20, and agrees only from n = 500. The values below
    reproduce Table B.3 to its stated Monte-Carlo error.
    """
    rng = np.random.default_rng(seed)
    total, done = 0.0, 0
    while done < reps:
        m = min(chunk, reps - done)
        X = rng.standard_normal((m, n))
        Xt = X - X.mean(axis=1, keepdims=True)
        Z = Xt ** 2
        Zc = Z - Z.mean(axis=1, keepdims=True)
        w = (Zc + 2.0 / (n - 1.0)) * Xt
        D = (Zc ** 2).sum(axis=1) + 2.0 * Z.sum(axis=1) / (n - 1.0)
        num = (w ** 2).sum(axis=1) - w.sum(axis=1) ** 2 / n
        total += float((num / D ** 2).sum())
        done += m
    return total / reps


def K(n, reps=4_000_000, force=False):
    """Cached K_n. Table B.3 lists 0.10481 at n = 20."""
    key = "K_n_%d_%d_v2" % (n, reps)
    return float(sim.run_or_load(
        key, lambda: {"K": np.array(_kn_compute(n, reps=reps))},
        force=force)["K"])


#: Sample sizes at which K_n is simulated for the interpolated curve. n K_n
#: is smooth and tends to 5/2, so interpolating it in log n is accurate well
#: inside the Monte-Carlo error.
K_GRID = [6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80, 100, 130, 170, 220,
          300, 400, 550, 750, 1000]


def _k_curve():
    """The cached (n, K_n) grid behind :func:`K_of_n`."""

    def compute():
        ks = [_kn_compute(n, reps=1_000_000 if n <= 100 else 200_000, seed=n)
              for n in K_GRID]
        return {"n": np.array(K_GRID, dtype=float), "K": np.array(ks)}

    d = sim.run_or_load("K_n_curve_v2", compute)
    return d["n"], d["K"]


def K_of_n(n):
    """K_n at arbitrary, possibly non-integer or array-valued n."""
    grid_n, grid_k = _k_curve()
    n = np.asarray(n, dtype=float)
    return np.interp(np.log(n), np.log(grid_n), grid_n * grid_k) / n


def K_for(n):
    """K_n for a scalar n exactly, for an array by interpolation."""
    arr = np.asarray(n)
    return K(int(arr)) if arr.ndim == 0 else K_of_n(arr)


# ---------------------------------------------------------------------------
# Bias and variance of every weight, Table A.1
# ---------------------------------------------------------------------------

def bias2_var(name, n, rho, lam_star, s=0.5, K_n=None):
    """Return ``(Bias(W)^2, Var(W))`` for one estimator, with sigma_f = 1.

    `s` is used only by the shrunken weights.
    """
    rho = np.asarray(rho, dtype=float)
    lam_star = np.asarray(lam_star, dtype=float)
    if K_n is None:
        K_n = K_for(n)

    sigma_y2 = (lam_star / rho) ** 2                  # sigma_f = 1
    v_star = sigma_y2 * (1.0 - rho ** 2) / (n - 1.0)  # V*/sigma_f^4

    # The weights with no estimation error still have to carry the shape of
    # whichever argument is an array, so that a panel varying n gets a curve
    # for them rather than a single point.
    zero = np.zeros(np.broadcast(rho, lam_star, np.asarray(n, float)).shape)

    var = {
        "C": zero,
        "P0": zero,
        "P+": sigma_y2 * (1.0 + rho ** 2) / (n - 1.0),
        "D0": v_star + 2.0 * (NU_0 - lam_star) ** 2 / (n - 1.0),
        "Dnu": v_star * (1.0 + 2.0 * K_n),
        "D+": v_star * (1.0 + 2.0 * K_n),
        # Table A.1, second-order shrunken row: shrinking nu-hat by u
        # multiplies its own error by u and buys back a bias (1-u) nu*, with
        # nu* = lambda* under normality and sigma_f = 1.
        "Duv": (v_star * (1.0 + 2.0 * U_SHRINK ** 2 * K_n)
                + 2.0 * (1.0 - U_SHRINK) ** 2 * lam_star ** 2 / (n - 1.0)),
    }

    if name in var:
        bias2 = {"C": lam_star ** 2,
                 "P0": (LAMBDA_0 - lam_star) ** 2}.get(name, 0.0 * rho)
        return bias2, var[name]

    if name.startswith("S(") and name.endswith(")"):
        # eq:shrink-biasvar: shrinking multiplies the deviation from zero by s
        base = name[2:-1]
        return (1.0 - s) ** 2 * lam_star ** 2, s ** 2 * var[base]

    raise KeyError(name)


def relative_mse(name, n, rho, lam_star, s=0.5, K_n=None):
    """MSE of one estimator divided by the MSE of the classical mean."""
    rho = np.asarray(rho, dtype=float)
    bias2, var = bias2_var(name, n, rho, lam_star, s=s, K_n=K_n)
    return (1.0 - rho ** 2) + (rho / lam_star) ** 2 * (bias2 + var)


def risk_table(names, n, rho, lam_star, s=0.5, K_n=None):
    """Stack `relative_mse` over `names`, shape ``(len(names),) + rho.shape``."""
    if K_n is None:
        K_n = K_for(n)
    return np.stack([relative_mse(nm, n, rho, lam_star, s=s, K_n=K_n)
                     for nm in names])


def best_and_excess(names, n, rho, lam_star, s=0.5, K_n=None):
    """Return ``(index of the best estimator, excess risk in percent)``.

    Excess is measured against the best member at each point, so the winner
    is exactly zero there.
    """
    risk = risk_table(names, n, rho, lam_star, s=s, K_n=K_n)
    best = np.nanargmin(risk, axis=0)
    excess = 100.0 * (risk / np.nanmin(risk, axis=0) - 1.0)
    return best, excess
