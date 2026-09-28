"""PurpleAir across places rather than across time.

The three site populations of Section 5.3 are hourly series at one location,
so their rows are serially dependent: the i.i.d. sampling that PPI assumes is
an approximation there. This file builds the cross-sectional counterpart. A
unit is a location, its label is the winter mean of the regulatory monitor it
is collocated with, and its prediction is the winter mean of the low-cost
sensor. Averaging over the winter removes the time dimension entirely, so the
autocorrelation that the hourly populations carry is gone.

Two granularities, because California does not offer more:

``pa_places``  one row per PurpleAir device that sits within 5 km of a
               regulatory monitor, M = 301. Several devices may share a
               monitor, so Y repeats across those rows.
``pa_sites``   one row per regulatory monitor, its prediction the mean of the
               devices collocated with it, M = 56. Every row is a distinct
               location and a distinct label, but 56 is a very small
               population.

Each comes in the three calibrations of Section 5.3: the raw factory reading,
the EPA correction of Barkjohn et al. (2021) and the linear calibration of
Chen (2021). Both corrections are linear in the sensor reading, the
temperature and the humidity, so applying them hour by hour and then averaging
is the same as applying them to the winter means, which is what is done here.

Input: dat.pkl, the pickled form of data/tidy/CA/dat.RData of
github.com/xiangchen-stat/AirCalibrate, which is Chen's merged hourly table
of every California collocation within 5 km.

Run:  python data/make_purpleair_places.py <directory holding dat.pkl>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

RADIUS = 5000.0
#: Barkjohn, Gantt & Clements (2021).
EPA_COEF = (5.75, 0.524, 0.0, -0.0862)
#: Chen (2021), his linear model, refitted and verified against his own object.
CHEN_COEF = (4.53336948, 0.32221390, 0.02933756, -0.03025920)


def populations(src):
    dat = pd.read_pickle(Path(src) / "dat.pkl")
    dat = dat[dat["dist"] < RADIUS].copy()
    dat["cf1"] = (dat["pm2.5_cf1_a"] + dat["pm2.5_cf1_b"]) / 2.0
    dat["pa"] = list(zip(dat["lat_pa"].round(6), dat["lon_pa"].round(6)))
    dat["epa"] = list(zip(dat["lat_epa"].round(6), dat["lon_epa"].round(6)))
    out = {}
    # device level: average the winter within a device, then keep the row
    dev = dat.groupby(["pa", "epa"]).agg(Y=("pm2.5_epa", "mean"), cf1=("cf1", "mean"),
                                         temp=("temp", "mean"), hum=("hum", "mean"))
    out["pa_places"] = dev.reset_index()
    # site level: average the devices of a monitor
    out["pa_sites"] = dev.groupby("epa").mean().reset_index()
    return out


def calibrate(df, coef):
    b0, b1, b2, b3 = coef
    return b0 + b1 * df["cf1"].values + b2 * df["temp"].values + b3 * df["hum"].values


def summary(F, Y, m=10.0):
    f, y = F - F.mean(), Y - Y.mean()
    s2f, s2y, sfy = (f * f).mean(), (y * y).mean(), (f * y).mean()
    a, b = f * f, f * y
    C = float(np.cov(a, b, ddof=1)[0, 1])
    Vff, Vfy = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    V = (a * y * y).mean() / m + (s2f * s2y - (m - 2) * sfy ** 2) / (m * (m - 1))
    return dict(M=len(Y), rho=round(float(sfy / np.sqrt(s2f * s2y)), 4),
                lam=round(float(sfy / s2f), 4), nu=round(float(C / Vff), 4),
                r2=round(float(C * C / (Vff * Vfy)), 4),
                t10=round(float(sfy ** 2 / V), 3),
                mean_Y=float(Y.mean()), sd_Y=float(Y.std(ddof=1)),
                mean_F=float(F.mean()), sd_F=float(F.std(ddof=1)),
                skew_Y=round(float(pd.Series(Y).skew()), 3),
                exkurt_Y=round(float(pd.Series(Y).kurtosis()), 3),
                exkurt_FY=round(float(pd.Series(f * y).kurtosis()), 2))


def main(src):
    pops = populations(src)
    arrays, rows = {}, []
    for base, df in pops.items():
        Y = np.ascontiguousarray(df["Y"].values, float)
        for tag, F in (("", np.ascontiguousarray(df["cf1"].values, float)),
                       ("_corr", np.ascontiguousarray(calibrate(df, EPA_COEF))),
                       ("_chen", np.ascontiguousarray(calibrate(df, CHEN_COEF)))):
            key = base + tag
            arrays[key + "__Y"], arrays[key + "__F"] = Y, F
            s = summary(F, Y)
            rows.append(dict(dataset=key, name=key, binary_Y=False, binary_F=False,
                             paper="Chen (2021); Barkjohn et al. (2021)",
                             estimand="mean winter PM2.5 across monitored locations",
                             Y_ist="winter mean of the regulatory monitor",
                             F_ist={"": "raw sensor reading",
                                    "_corr": "EPA correction applied",
                                    "_chen": "Chen's linear calibration applied"}[tag],
                             repo="github.com/xiangchen-stat/AirCalibrate",
                             datei="built by data/make_purpleair_places.py",
                             split_im_paper="collocation radius %d m" % RADIUS, **s))
            print("%-16s M=%4d rho=%.3f lam*=%6.3f nu*=%7.3f r2=%.3f t(10)=%5.2f skewY=%5.2f"
                  % (key, s["M"], s["rho"], s["lam"], s["nu"], s["r2"], s["t10"], s["skew_Y"]))
    np.savez_compressed(HERE / "populations_pa_places.npz", **arrays)
    pd.DataFrame(rows).to_csv(HERE / "population_table_pa_places.csv", index=False)
    print("wrote", HERE / "populations_pa_places.npz")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
