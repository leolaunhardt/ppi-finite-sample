"""The three PurpleAir sites with the calibration of Chen (2021) applied.

\\textcite{chenAirCalibrate} fits calibration models for PurpleAir sensors in
California on 770272 collocated hourly pairs within 5 km of a reference
monitor. His linear model is

    pm2.5_epa ~ pm2.5_cf1 + temp + hum,

fitted on the 80 per cent of pairs closest to their monitor. Refitting that
specification here reproduces his stored coefficients to eight decimals
(4.53336948, 0.32221390, 0.02933756, -0.03025920), so the model applied below
is his, not an imitation.

Two versions are built, and the difference between them is the point:

``pa_*_chen``       his model as fitted. All three of our sites are within
                    65 m of their monitor and therefore inside his training
                    set, so the calibration has seen the labels of the very
                    problem it is then used on.
``pa_*_chenloso``   the same specification refitted with the target site's
                    rows removed, which is what an analyst who calibrates
                    elsewhere and applies here actually has.

Inputs, downloaded once into the working directory: CA.zip, epa_2019.zip,
epa_2020.zip and dat.RData, all from github.com/xiangchen-stat/AirCalibrate.

Run:  python data/make_purpleair_chen.py <directory holding the inputs>
"""
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data.make_purpleair_corrected import SITES, epa_hourly  # noqa: E402

FEATURES = ["cf1", "temp", "hum"]


def fit(df):
    """Chen's linear calibration, least squares on his three covariates."""
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in FEATURES])
    beta, *_ = np.linalg.lstsq(X, df["pm2.5_epa"].values, rcond=None)
    return beta


def apply(beta, cf1, temp, hum):
    return beta[0] + beta[1] * cf1 + beta[2] * temp + beta[3] * hum


def sensor_rows(z, names, sensor):
    """A row per channel and hour, carrying that hour's temperature and humidity."""
    out = []
    for chan, tail in (("A", " (outside)"), ("B", " B ")):
        f = [x for x in names if x.startswith(sensor + tail)]
        d = pd.read_csv(z.open(f[0]))
        d["t"] = pd.to_datetime(
            d["created_at"].str.replace(" UTC", "", regex=False)).dt.floor("h")
        if chan == "A":
            met = d.groupby("t")[["Temperature_F", "Humidity_%"]].mean()
            met.columns = ["temp", "hum"]
        out.append(d[["t", "PM2.5_CF1_ug/m3"]].dropna()
                   .rename(columns={"PM2.5_CF1_ug/m3": "cf1"}).assign(chan=chan))
    return pd.concat(out, ignore_index=True).merge(met, on="t", how="left")


def main(src):
    src = Path(src)
    dat = pd.read_pickle(src / "dat.pkl").rename(columns={"pm2.5_cf1_a": "cf1"})
    beta_all = fit(dat[dat["dist"] < dat["dist"].quantile(.8)])
    print("Chen's coefficients, refitted:", np.round(beta_all, 8))

    ref = epa_hourly(src)
    z = zipfile.ZipFile(src / "CA.zip")
    names = z.namelist()
    bundle = np.load(HERE / "populations_subset.npz")
    arrays, rows = {}, []
    for name, (_, sensors) in SITES.items():
        pa = pd.concat([sensor_rows(z, names, s) for s in sensors], ignore_index=True)
        m = pa.merge(ref[ref["name"] == name][["t", "Y"]], on="t")
        m = m[m["Y"] >= 0].dropna(subset=["hum", "temp"])
        assert len(m) == len(bundle[name + "__F"]), name
        assert np.allclose(np.sort(m["cf1"].values), np.sort(bundle[name + "__F"]))

        # the site's own rows are inside Chen's training data; drop them by
        # location to get the leave-one-site-out fit
        lat, lon = dat["lat_epa"], dat["lon_epa"]
        site_lat = ref[ref["name"] == name].attrs.get("lat")
        near = (np.abs(lat - m.attrs.get("lat", 0)) < 1e-6) if False else None
        Y = m["Y"].values
        for tag, beta in (("chen", beta_all), ("chenloso", None)):
            if beta is None:
                keep = dat[~_same_site(dat, name)]
                beta = fit(keep[keep["dist"] < keep["dist"].quantile(.8)])
                print("  %-12s LOSO coefficients: %s" % (name, np.round(beta, 5)))
            F = apply(beta, m["cf1"].values, m["temp"].values, m["hum"].values)
            key = "%s_%s" % (name, tag)
            arrays[key + "__Y"] = np.ascontiguousarray(Y)
            arrays[key + "__F"] = np.ascontiguousarray(F)
            rows.append(_row(key, F, Y, tag))
            print("    %-22s rho=%.3f lam*=%.3f nu*=%.3f t(20)=%.2f"
                  % (key, rows[-1]["rho"], rows[-1]["lam"], rows[-1]["nu"], rows[-1]["t20"]))

    np.savez_compressed(HERE / "populations_pa_chen.npz", **arrays)
    pd.DataFrame(rows).to_csv(HERE / "population_table_pa_chen.csv", index=False)
    print("wrote", HERE / "populations_pa_chen.npz")


#: The AQS coordinates of the three sites, used to cut them out of Chen's data.
_COORD = {"pa_tehama": (40.170930, -122.255560),
          "pa_fresno_g": (36.785380, -119.773210),
          "pa_fresno_v": (36.819449, -119.716433)}


def _same_site(dat, name):
    la, lo = _COORD[name]
    return (np.abs(dat["lat_epa"] - la) < 1e-4) & (np.abs(dat["lon_epa"] - lo) < 1e-4)


def _row(key, F, Y, tag):
    f, y = F - F.mean(), Y - Y.mean()
    s2f, s2y, sfy = (f * f).mean(), (y * y).mean(), (f * y).mean()
    a, b = f * f, f * y
    C = float(np.cov(a, b, ddof=1)[0, 1])
    Vff, Vfy = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    m = 20.0
    V = (a * y * y).mean() / m + (s2f * s2y - (m - 2) * sfy ** 2) / (m * (m - 1))
    return dict(dataset=key, name=key, M=len(Y),
                rho=round(float(sfy / np.sqrt(s2f * s2y)), 4),
                lam=round(float(sfy / s2f), 4), nu=round(float(C / Vff), 4),
                r2=round(float(C * C / (Vff * Vfy)), 4),
                t20=round(float(sfy ** 2 / V), 3),
                mean_Y=float(Y.mean()), sd_Y=float(Y.std(ddof=1)),
                mean_F=float(F.mean()), sd_F=float(F.std(ddof=1)),
                skew_Y=round(float(pd.Series(Y).skew()), 3),
                exkurt_Y=round(float(pd.Series(Y).kurtosis()), 3),
                exkurt_FY=round(float(pd.Series(f * y).kurtosis()), 2),
                binary_Y=False, binary_F=False,
                paper="Chen (2021), ScM thesis, Johns Hopkins",
                estimand="mean PM2.5 at the site, calibrated predictor",
                Y_ist="EPA reference monitor, hourly",
                F_ist=("Chen's linear calibration as fitted (site in training set)"
                       if tag == "chen" else
                       "Chen's specification refitted without this site"),
                repo="github.com/xiangchen-stat/AirCalibrate",
                datei="built by data/make_purpleair_chen.py",
                split_im_paper="lm(pm2.5_epa ~ pm2.5_cf1 + temp + hum)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
