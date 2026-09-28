"""
Finite-sample prediction-powered inference: DoublePPI and guarded tuning.

The code behind the figures and tables of the dissertation. Three layers.

Estimators
    ``unified``     closed-form bias and variance of all ten members, general
                    and under normality, and the constant K_n
    ``theory``      the individual closed forms of Sections 2 and 3
    ``populations`` the members evaluated on a real, finite population by
                    complement sampling; the bootstrap lives here too
    ``gaussian``    the members evaluated on simulated Gaussian pairs

Data
    ``populations`` loading and the sample-size grid; ``data/make_*.py``
                    build the population bundles from the original sources
    ``realdata``    the four mean-estimation benchmarks of the PPI literature

Selection and drawing
    ``hyper``       the (h, kappa) search for the exponential guard
    ``experiments`` the simulations of Section 5.1
    ``panels``      one function per kind of figure
    ``style``       colours, type sizes and the output paths
    ``simulate``    a cache so that a figure is drawn, not recomputed
"""
