# A Bigger and Cheaper Lunch

Code for the dissertation *A Bigger and Cheaper Lunch: Proposing DoublePPI and
Guarded Tuning in Finite-Sample Prediction-Powered Inference* (Leo M.
Launhardt, MSc Statistical Science, University of Oxford). It reproduces
every figure and every number the paper reports.

Prediction-powered inference corrects an estimate computed from model
predictions by the average error those predictions make on a small labelled
sample. PPI++ weights that correction by a tuned parameter, which has to be
estimated from the same small sample. This repository studies what that
estimation costs at finite `n`, and two ways of reducing it:

* **DoublePPI** applies prediction powering to the tuning parameter itself.
* **Guarded tuning** shrinks the weight toward zero when it cannot be
  estimated reliably, so the estimator falls back on the classical mean
  rather than behind it.

---

## The estimators

Ten members, written throughout as a weight `W` applied to one correction,
`theta_hat = Ybar - lambda_W (Fbar - Fbar')`.

| Symbol | Name | Weight |
| --- | --- | --- |
| `C` | classical mean | `0` |
| `P0` | PPI at a fixed weight | `lambda_0`, canonically `1` |
| `P+` | PPI++ | `sigma_hat_fy / sigma_f^2` |
| `D0` | DoublePPI at a fixed `nu_0` | second order fixed at `1` |
| `D+` | DoublePPI at the estimated `nu_hat` | second order tuned |
| `G^C(alpha)[W]` | first-order guard of `W` against `C` | `alpha * lambda_W` |
| `G^P+(beta)` | second-order guard of `D+` against `P+` | second order `beta * nu_hat` |
| `G^C,P+(alpha,beta)` | both guards at once | |

Index `0` marks a fixed factor, index `+` an estimated one. The guard factor
is a function of one statistic, the signal-to-noise ratio `t` of the weight it
guards; the oracle is `gamma* = t / (1 + t)`.

The definitions live in exactly three places, and every figure comes from one
of them:

| File | What it is |
| --- | --- |
| `ppi/unified.py` | closed-form squared bias and variance of all ten members, general and under normality, and the constant `K_n` by Monte Carlo |
| `ppi/populations.py` | the members evaluated on a real, finite population, plus the paired bootstrap that estimates `t` |
| `ppi/gaussian.py` | the members evaluated on simulated Gaussian pairs |

`ppi/theory.py` holds the individual closed forms of Sections 2 and 3
(thresholds, the oracle guard, the exponential rule).

## The data

Thirteen populations, each treated as finite and complete: every label exists
and only its scarcity is simulated. One replicate draws `n` of the `M` pairs
without replacement as the labelled sample and takes the remaining `M - n` as
the unlabelled block, so no observation is used twice.

| Group | Populations |
| --- | --- |
| Strong predictors | hydration energy, temperature, precipitation, three Amazon review sets, tree biomass, soil copper |
| Binary predictors | spiral galaxies, tree cover |
| Weak predictors | COMPAS, LSI-R, Moral Machine |

None of the data is redistributed here. The code reads each population from
`data/populations_*.npz`, one `__Y` and one `__F` array per population, and
`data/README.md` gives, for every population, the original source, the
citation, the columns that become `Y` and `F`, and the preprocessing, so
that the bundles can be rebuilt from the sources. `data/make_*.py` are the
build scripts that exist as code; the rest of the recipes are written out in
that file. The four mean-estimation benchmarks of the PPI literature used in
Appendix D go into `data/ppi_py/` the same way.

Loading, the sample-size grid and its cap at `M / 5` are in
`ppi/populations.py`.

---

## Running it

```bash
pip install -r requirements.txt

python build_all.py              # every figure
python build_all.py 4            # everything in Section 4
python build_all.py fig_5_4a     # one figure
```

Figures land in `output/pdf` as PDF and `output/png` as PNG; set
`PPI_FIGURE_DIR` to write the PDFs somewhere else, for instance into the
graphics folder of a LaTeX project.

Simulation results are cached in `simdata/`, which is not tracked. The first
run of a figure that needs a new simulation is slow; every later run is
immediate. Everything in Sections 2 to 4 is analytic apart from the constant
`K_n` (Monte Carlo, minutes) and builds at once. The two expensive parts are
the finite-population sweeps of Section 5, 50 000 replicates per cell with a
paired bootstrap of `B = 500` (about two and a half hours for the thirteen
populations on four cores), and the `(h, kappa)` search of Appendix B (about
nine hours). Both need the data bundles of `data/README.md`; without them
`build_all.py 5` and the Appendix B figures cannot run.

## Layout

```
ppi/            the package: estimators, simulations, figure panels
figures/        one script per figure, named after the paper
data/           README with every data source, plus the build scripts
build_all.py    driver
LICENSE         MIT
```

Every script in `figures/` is named after the figure it draws and can be run
on its own. `figures/make_legends.py` writes the shared keys,
`figures/make_appendix_panels.py` the three appendix figures.

Python 3.9 or later; numpy, pandas, scipy and matplotlib. Developed and run
under Python 3.9.6.
