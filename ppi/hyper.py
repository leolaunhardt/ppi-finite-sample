"""
Formal hyperparameter selection for the tuned exponential family.

Section 5.1, supporting the choice of (h, kappa) in

    s_E(h, kappa)(t) = 2^(-(h/t)^kappa).

The rule is a nonlinear function of the *estimated* t-hat, so it has no closed
form and must be simulated. t-hat is the paired bootstrap of eq:boot, the
default of Section 3.3: every replicate resamples its own labelled pairs B
times and reads Var(sigma-hat) off the resamples, centred at the point
estimate. The trick that makes a fine (h, kappa) grid affordable is that the
shrinkage factor enters the risk only through two moments. Writing the error
of the shrunken estimator as

    e = Ybar - s * lambda_hat * (Fbar - Fbar'),      s = s(t_hat),

and putting  B := lambda_hat (Fbar - Fbar') Ybar  and  C := [lambda_hat
(Fbar - Fbar')]^2, the risk of *any* rule on the same replicates is

    MSE(s) = E[Ybar^2] - 2 E[s B] + E[s^2 C].

Since s depends on the replicate only through t_hat, binning the replicates by
t_hat and storing the per-bin sums of B and C reduces one cell to a small
summary from which every candidate rule is read off in O(bins). One simulation
therefore serves the whole grid, and all rules share the same random numbers,
so the *differences* between them are far better determined than their levels.

Bins are uniform in u = t/(1+t), which resolves the small-t region where the
rules differ and compresses the large-t tail where they all agree. The
bootstrap t-hat is unbounded above, so the compression matters more here than
it did for the plug-in, whose t is capped at n - 1.
"""

import time

import numpy as np

from . import gaussian as shrink
from . import simulate as sim

#: Unlabelled block, as everywhere in Section 5.
N_UNLABELLED = 10_000

#: Bins in u = t/(1+t). Validated in `validate_binning` to well under 0.01 pp.
N_BINS = 2000

#: Bootstrap resamples per replicate, the B of eq:boot, as everywhere else.
N_BOOT = 200


def _chunk_for(n, n_boot, cap=20_000):
    """Replicates per block, so that the (m, B, n) bootstrap stays bounded."""
    return max(64, min(cap, 20_000_000 // max(n_boot * n, 1)))


def boot_t(rng, F, Y, cov, n, n_boot=N_BOOT):
    """t-hat of the covariance from the paired bootstrap, equation eq:boot.

    The pairs are resampled together -- an unpaired resample would destroy the
    very dependence whose estimation error is being quantified -- and the
    variance is centred at the point estimate rather than at the bootstrap
    mean, exactly as eq:boot writes it. Identical to the ``_snr`` path of
    ``populations.py``, so the simulation and the real populations feed their
    rules the same statistic.
    """
    m = F.shape[0]
    idx = rng.integers(0, n, size=(m, n_boot, n))
    Fb = np.take_along_axis(np.broadcast_to(F[:, None, :], (m, n_boot, n)),
                            idx, axis=2)
    Yb = np.take_along_axis(np.broadcast_to(Y[:, None, :], (m, n_boot, n)),
                            idx, axis=2)
    s_f, s_y = Fb.sum(axis=2), Yb.sum(axis=2)
    cov_b = ((Fb * Yb).sum(axis=2) - s_f * s_y / n) / (n - 1.0)
    v = ((cov_b - cov[:, None]) ** 2).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0.0, cov ** 2 / np.where(v > 0.0, v, 1.0), 0.0)


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------

def cell_summary(n, rho, reps, seed, N=N_UNLABELLED, nbins=N_BINS,
                 n_boot=N_BOOT, chunk=None):
    """Simulate one (n, rho) cell and return its sufficient summary.

    Returns ``(sum_A, count, sum_t, sum_B, sum_C, reps)`` with the last four
    binned by t_hat; ``sum_A`` is the classical mean's summed squared error.
    """
    rng = np.random.default_rng(seed)
    if chunk is None:
        chunk = _chunk_for(n, n_boot)
    sum_A = 0.0
    count = np.zeros(nbins)
    sum_t = np.zeros(nbins)
    sum_B = np.zeros(nbins)
    sum_C = np.zeros(nbins)

    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        F, Y = shrink.draw_labelled(rng, m, n, rho)
        u_mean, u_ss = shrink.draw_unlabelled_stats(rng, m, N)

        F_bar, Y_bar = F.mean(axis=1), Y.mean(axis=1)
        dF, dY = F - F_bar[:, None], Y - Y_bar[:, None]
        ss_lab = (dF ** 2).sum(axis=1)
        cov = (dF * dY).sum(axis=1) / (n - 1.0)

        var_pooled = shrink._pooled_var(F, F_bar, ss_lab, n, u_mean, u_ss, N)
        lam = cov / ((1.0 + n / N) * var_pooled)
        correction = F_bar - u_mean

        t_hat = boot_t(rng, F, Y, cov, n, n_boot)

        g = lam * correction
        B = g * Y_bar
        C = g * g

        sum_A += float((Y_bar ** 2).sum())
        u = t_hat / (1.0 + t_hat)
        idx = np.clip((u * nbins).astype(int), 0, nbins - 1)
        count += np.bincount(idx, minlength=nbins)
        sum_t += np.bincount(idx, weights=t_hat, minlength=nbins)
        sum_B += np.bincount(idx, weights=B, minlength=nbins)
        sum_C += np.bincount(idx, weights=C, minlength=nbins)
        done += m

    return sum_A, count, sum_t, sum_B, sum_C, reps


def bin_centres(count, sum_t, nbins=N_BINS):
    """Representative t of each bin: the mean t_hat of the replicates in it."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(count > 0, sum_t / np.where(count > 0, count, 1.0), 0.0)
    return t


def delta_from_summary(summary, s_of_t):
    """Delta_C in percent of a rule given as a function of the bin centres."""
    sum_A, count, sum_t, sum_B, sum_C, reps = summary
    t = bin_centres(count, sum_t)
    s = np.where(count > 0, s_of_t(t), 0.0)
    mse = (sum_A - 2.0 * float((s * sum_B).sum())
           + float((s * s * sum_C).sum())) / reps
    return 100.0 * (mse / (sum_A / reps) - 1.0)


# ---------------------------------------------------------------------------
# The candidate grid
# ---------------------------------------------------------------------------

def s_exp(t, h, kappa):
    """s_E(h, kappa) evaluated on an array of t, with t = 0 sent to s = 0."""
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        ratio = np.where(t > 0.0, h / np.where(t > 0.0, t, 1.0), np.inf)
        out = np.exp2(-(ratio ** kappa))
    return np.where(np.isfinite(out), out, 0.0)


def delta_grid(summary, h_grid, k_grid):
    """Delta_C of every (h, kappa) on one cell. Shape ``(len(h), len(k))``."""
    sum_A, count, sum_t, sum_B, sum_C, reps = summary
    t = bin_centres(count, sum_t)
    live = count > 0
    t, sB, sC = t[live], sum_B[live], sum_C[live]
    base = sum_A / reps

    out = np.empty((len(h_grid), len(k_grid)))
    lnt = np.log(np.where(t > 0, t, 1.0))
    pos = t > 0
    for j, kappa in enumerate(k_grid):
        for i, h in enumerate(h_grid):
            with np.errstate(over="ignore", invalid="ignore"):
                e = np.exp(kappa * (np.log(h) - lnt))
                s = np.where(pos, np.exp2(-e), 0.0)
            s = np.where(np.isfinite(s), s, 0.0)
            mse = (sum_A - 2.0 * float((s * sB).sum())
                   + float((s * s * sC).sum())) / reps
            out[i, j] = 100.0 * (mse / base - 1.0)
    return out


def delta_reference(summary):
    """Delta_C of P_+ (no shrinkage) and of the fixed rule at s = 1/2."""
    return (delta_from_summary(summary, lambda t: np.ones_like(t)),
            delta_from_summary(summary, lambda t: np.full_like(t, 0.5)))


# ---------------------------------------------------------------------------
# Validation of the binning
# ---------------------------------------------------------------------------

def validate_binning(n=20, rho=0.3, reps=100_000, seed=99, n_boot=N_BOOT,
                     probes=((1.5, 0.5), (4.0, 1.0), (0.5, 2.0))):
    """Binned Delta against an exact per-replicate evaluation, same draws.

    The replicates are binned and evaluated exactly in the same pass, so the
    comparison isolates the binning error rather than the difference between
    two random streams.
    """
    rng = np.random.default_rng(seed)
    chunk = _chunk_for(n, n_boot)

    sum_A = 0.0
    count = np.zeros(N_BINS)
    sum_t = np.zeros(N_BINS)
    sum_B = np.zeros(N_BINS)
    sum_C = np.zeros(N_BINS)
    exact_sse = {pr: 0.0 for pr in probes}

    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        F, Y = shrink.draw_labelled(rng, m, n, rho)
        u_mean, u_ss = shrink.draw_unlabelled_stats(rng, m, N_UNLABELLED)
        F_bar, Y_bar = F.mean(axis=1), Y.mean(axis=1)
        dF, dY = F - F_bar[:, None], Y - Y_bar[:, None]
        ss_lab = (dF ** 2).sum(axis=1)
        cov = (dF * dY).sum(axis=1) / (n - 1.0)
        var_pooled = shrink._pooled_var(F, F_bar, ss_lab, n, u_mean, u_ss,
                                        N_UNLABELLED)
        lam = cov / ((1.0 + n / N_UNLABELLED) * var_pooled)
        corr = F_bar - u_mean
        t_hat = boot_t(rng, F, Y, cov, n, n_boot)

        g = lam * corr
        sum_A += float((Y_bar ** 2).sum())
        u = t_hat / (1.0 + t_hat)
        idx = np.clip((u * N_BINS).astype(int), 0, N_BINS - 1)
        count += np.bincount(idx, minlength=N_BINS)
        sum_t += np.bincount(idx, weights=t_hat, minlength=N_BINS)
        sum_B += np.bincount(idx, weights=g * Y_bar, minlength=N_BINS)
        sum_C += np.bincount(idx, weights=g * g, minlength=N_BINS)

        for h, k in probes:
            err = Y_bar - s_exp(t_hat, h, k) * g
            exact_sse[(h, k)] += float((err ** 2).sum())
        done += m

    summary = (sum_A, count, sum_t, sum_B, sum_C, reps)
    mse_c = sum_A / reps

    out = {}
    for h, k in probes:
        exact = 100.0 * (exact_sse[(h, k)] / reps / mse_c - 1.0)
        binned = delta_from_summary(summary, lambda t: s_exp(t, h, k))
        out[(h, k)] = (exact, binned, binned - exact)
    return out


# ---------------------------------------------------------------------------
# The grids the selection runs over
# ---------------------------------------------------------------------------

#: Sample sizes. Spans the range of Section 5 and adds n = 1000 so that the
#: asymptotic behaviour of the optimum can be read off.
N_GRID = [10, 15, 20, 30, 50, 100, 200, 300, 1000]

#: Replicates per cell, reduced for the large n where each draw costs more and
#: the risks themselves are smaller. All rules share the draws of a cell, so
#: differences between rules are far better determined than these numbers
#: suggest for their levels.
def reps_for(n):
    if n <= 100:
        return 200_000
    if n <= 300:
        return 100_000
    return 50_000


#: t of the belief grid, and the tighter band that counts as "at the
#: threshold" -- a factor of sqrt(2) either side of t = 1, symmetric in log t.
T_GRID = np.round(np.arange(0.1, 3.001, 0.1), 3)
T_NEAR = np.round(np.arange(0.7, 1.401, 0.1), 3)

#: Correlations of the unrestricted grid.
RHO_GRID = np.round(np.arange(0.0, 1.001, 0.05), 3)
RHO_MAX = 0.999

#: Candidates. h is log-spaced because the family is a shift in log t; kappa
#: is linear. The range is deliberately wider than any plausible answer, so
#: that the point where the family degenerates is inside the picture.
H_GRID = np.round(np.exp(np.linspace(np.log(0.2), np.log(20.0), 41)), 4)
K_GRID = np.round(np.concatenate([np.arange(0.1, 2.001, 0.1),
                                  np.array([2.25, 2.5, 3.0, 4.0])]), 3)


def t_cells():
    """(n, rho, t) of the t-indexed grid: rho recovered from t and n."""
    rows = []
    for n in N_GRID:
        for t in T_GRID:
            denom = n - 1.0 - t
            if denom <= 0:
                continue
            rho2 = t / denom
            if 0.0 < rho2 <= RHO_MAX ** 2:
                rows.append((int(n), float(np.sqrt(rho2)), float(t)))
    return rows


def rho_cells():
    """(n, rho, t) of the correlation-indexed grid, t implied."""
    rows = []
    for n in N_GRID:
        for rho in RHO_GRID:
            r = min(float(rho), RHO_MAX)
            rows.append((int(n), r, (n - 1.0) * r ** 2 / (1.0 + r ** 2)))
    return rows


def _sweep(cells, tag, seed0):
    """Delta_C of the whole (h, kappa) grid on every cell, plus references."""
    nh, nk = len(H_GRID), len(K_GRID)
    grid = np.empty((len(cells), nh, nk))
    d_pp = np.empty(len(cells))
    d_half = np.empty(len(cells))
    t0 = time.time()
    for i, (n, rho, _t) in enumerate(cells):
        summary = cell_summary(n, rho, reps_for(n), seed0 + 17 * i)
        grid[i] = delta_grid(summary, H_GRID, K_GRID)
        d_pp[i], d_half[i] = delta_reference(summary)
        print("  %s cell %3d/%3d  n=%4d rho=%.3f  %6.1f min elapsed"
              % (tag, i + 1, len(cells), n, rho, (time.time() - t0) / 60.0),
              flush=True)
    arr = np.array(cells, dtype=float)
    return dict(n=arr[:, 0], rho=arr[:, 1], t=arr[:, 2],
                grid=grid, pplus=d_pp, half=d_half,
                h=np.asarray(H_GRID), kappa=np.asarray(K_GRID))


def sweep_t(force=False):
    return sim.run_or_load("hyper_boot_t",
                           lambda: _sweep(t_cells(), "t", 8100), force=force)


def sweep_rho(force=False):
    return sim.run_or_load("hyper_boot_rho",
                           lambda: _sweep(rho_cells(), "rho", 8200),
                           force=force)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def regime(name):
    """The (data, boolean mask over its cells) of one of the three regimes."""
    if name == "all":
        d = sweep_rho()
        return d, np.ones(len(d["n"]), dtype=bool)
    d = sweep_t()
    if name == "threshold":
        return d, (d["t"] >= T_NEAR[0] - 1e-9) & (d["t"] <= T_NEAR[-1] + 1e-9)
    if name == "belief":
        return d, np.ones(len(d["n"]), dtype=bool)
    raise KeyError(name)


def criteria(name, n_only=None):
    """Mean and worst-case Delta_C of every candidate over one regime.

    `n_only` restricts to a single sample size. Returns ``(mean, worst)``,
    each of shape ``(len(H_GRID), len(K_GRID))``, plus the two references.
    """
    d, mask = regime(name)
    if n_only is not None:
        mask = mask & (d["n"] == n_only)
    g = d["grid"][mask]
    return (g.mean(axis=0), g.max(axis=0),
            float(d["pplus"][mask].mean()), float(d["pplus"][mask].max()),
            float(d["half"][mask].mean()), float(d["half"][mask].max()))


def _at(arr, i, j):
    return float(arr[i, j])


def select(name, n_only=None, tol=0.10):
    """The three optima of one regime, with the plateau around each.

    Returns a dict with, for each of "worst", "mean" and "mixed":
    the (h, kappa) attaining it, its mean and worst-case Delta, and the
    rectangle of candidates within `tol` percentage points of the optimum.
    """
    mean, worst, pp_mean, pp_worst, hf_mean, hf_worst = criteria(name, n_only)
    out = {"reference": {"P+": (pp_mean, pp_worst),
                         "s=1/2": (hf_mean, hf_worst)}}

    iw, jw = np.unravel_index(np.argmin(worst), worst.shape)
    out["worst"] = dict(h=float(H_GRID[iw]), kappa=float(K_GRID[jw]),
                        mean=_at(mean, iw, jw), worst=_at(worst, iw, jw))

    im, jm = np.unravel_index(np.argmin(mean), mean.shape)
    out["mean"] = dict(h=float(H_GRID[im]), kappa=float(K_GRID[jm]),
                       mean=_at(mean, im, jm), worst=_at(worst, im, jm))

    feasible = worst <= 0.0
    if feasible.any():
        masked = np.where(feasible, mean, np.inf)
        ix, jx = np.unravel_index(np.argmin(masked), masked.shape)
        out["mixed"] = dict(h=float(H_GRID[ix]), kappa=float(K_GRID[jx]),
                            mean=_at(mean, ix, jx), worst=_at(worst, ix, jx),
                            feasible=True)
    else:
        out["mixed"] = dict(feasible=False,
                            best_worst=float(worst.min()))

    for key, arr in (("mean", mean), ("worst", worst)):
        best = float(arr.min())
        near = arr <= best + tol
        hs, ks = np.where(near)
        out["plateau_" + key] = dict(
            h=(float(H_GRID[hs.min()]), float(H_GRID[hs.max()])),
            kappa=(float(K_GRID[ks.min()]), float(K_GRID[ks.max()])),
            count=int(near.sum()), of=int(arr.size))
    out["mean_surface"] = mean
    out["worst_surface"] = worst
    return out


def frontier(name, bounds, n_only=None):
    """Best mean Delta subject to a ceiling on the worst case.

    The unconstrained minimax is degenerate -- the worst case falls to zero
    only as h grows without bound, where the rule becomes the classical mean
    and gives up every gain -- so the informative object is this frontier
    rather than a single minimax point.
    """
    mean, worst, _, _, _, _ = criteria(name, n_only)
    rows = []
    for b in bounds:
        ok = worst <= b
        if not ok.any():
            rows.append((b, None, None, None, None))
            continue
        masked = np.where(ok, mean, np.inf)
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        rows.append((b, float(H_GRID[i]), float(K_GRID[j]),
                     float(mean[i, j]), float(worst[i, j])))
    return rows


def pareto_front(name, n_only=None):
    """The non-dominated candidates of one regime.

    A candidate is dominated if another attains both a lower mean and a lower
    worst case. What survives is the curve an analyst actually chooses along;
    everything else is beaten on both counts at once. Returned sorted by mean,
    as ``(mean, worst, h, kappa)`` columns.
    """
    mean, worst, _, _, _, _ = criteria(name, n_only)
    hh, kk = np.meshgrid(H_GRID, K_GRID, indexing="ij")
    m, w = mean.ravel(), worst.ravel()
    h, k = hh.ravel(), kk.ravel()

    keep = np.ones(len(m), dtype=bool)
    for i in range(len(m)):
        if ((m <= m[i]) & (w <= w[i]) & ((m < m[i]) | (w < w[i]))).any():
            keep[i] = False
    order = np.argsort(m[keep])
    return (m[keep][order], w[keep][order], h[keep][order], k[keep][order])
