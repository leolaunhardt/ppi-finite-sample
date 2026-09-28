"""The three PurpleAir sites with the published EPA correction applied.

The populations in ``populations_subset.npz`` carry the sensor's raw factory
reading as F. The correction of Barkjohn, Gantt & Clements (2021, AMT 14,
4617), which the EPA applies before publishing PurpleAir data on its own maps,
is

    PM2.5_corrected = 0.524 * PA_cf1 - 0.0862 * RH + 5.75,

with RH the relative humidity reported by the same sensor. The humidity is not
in the bundle, so the three populations are rebuilt here from the sources named
in ``PPI_HANDOVER.md`` and the correction is applied row by row.

The rebuild reproduces the bundle exactly: same number of rows, and the sorted
F and Y agree to machine precision on all three sites. The rule recovered from
that check is

* one row per sensor channel (A and B) and hour, merged with the reference on
  the UTC hour, the sensor within 0.5 km of the AQS site;
* where a site reports several instruments (POC), the reference is their hourly
  mean --- this is what makes \\texttt{pa\\_fresno\\_g} come out at $M=14052$;
* rows with a negative reference value are dropped;
* the humidity of an hour comes from the A channel, which is the only one that
  reports it, and is used for both channels of that sensor.

Inputs, downloaded once into the working directory:

    CA.zip        raw/CA/CA.zip           of github.com/xiangchen-stat/AirCalibrate
    epa_2019.zip  raw/EPA/hourly_88101_2019.zip
    epa_2020.zip  raw/EPA/hourly_88101_2020.zip

Run:  python data/make_purpleair_corrected.py <directory holding the three zips>
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

#: Barkjohn, Gantt & Clements (2021), the US-wide correction.
SLOPE, RH_COEF, INTERCEPT = 0.524, -0.0862, 5.75

#: AQS site, its coordinates, and the collocated PurpleAir sensors.
SITES = {
    "pa_tehama":   (("06", "103", "0007"), ["CARB_SMOKE_TCAPCD_REDBLUFF"]),
    "pa_fresno_g": (("06", "019", "0011"), ["IMPROVE_FRES2"]),
    "pa_fresno_v": (("06", "019", "5001"), ["908 Villa 1", "908 Villa 2", "908 Villa 3"]),
}


def epa_hourly(src):
    """Hourly reference series of the three sites, averaged over instruments."""
    want = {s: n for n, (s, _) in SITES.items()}
    frames = []
    for year in (2019, 2020):
        z = zipfile.ZipFile(src / ("epa_%d.zip" % year))
        reader = pd.read_csv(
            z.open(z.namelist()[0]), chunksize=500_000, low_memory=False,
            usecols=["State Code", "County Code", "Site Num", "Date GMT",
                     "Time GMT", "Sample Measurement"],
            dtype={"State Code": str, "County Code": str, "Site Num": str})
        for chunk in reader:
            key = list(zip(chunk["State Code"], chunk["County Code"], chunk["Site Num"]))
            chunk = chunk.assign(site=key)
            hit = chunk[chunk["site"].isin(want)]
            if len(hit):
                frames.append(hit)
    epa = pd.concat(frames, ignore_index=True)
    epa["name"] = epa["site"].map(want)
    epa["t"] = pd.to_datetime(epa["Date GMT"] + " " + epa["Time GMT"])
    # several instruments at one site are averaged within the hour
    return (epa.groupby(["name", "t"])["Sample Measurement"].mean()
            .rename("Y").reset_index())


def purpleair(z, names, sensor):
    """One sensor: a row per channel and hour, with the A-channel humidity."""
    out = []
    for chan, tail in (("A", " (outside)"), ("B", " B ")):
        f = [x for x in names if x.startswith(sensor + tail)]
        d = pd.read_csv(z.open(f[0]))
        d["t"] = pd.to_datetime(
            d["created_at"].str.replace(" UTC", "", regex=False)).dt.floor("h")
        if chan == "A":
            rh = d.groupby("t")["Humidity_%"].mean().rename("rh")
        out.append(d[["t", "PM2.5_CF1_ug/m3"]].dropna().assign(chan=chan))
    return pd.concat(out, ignore_index=True).merge(rh, on="t", how="left")


def main(src):
    src = Path(src)
    ref = epa_hourly(src)
    z = zipfile.ZipFile(src / "CA.zip")
    names = z.namelist()
    bundle = np.load(HERE / "populations_subset.npz")
    arrays, rows = {}, []
    for name, (_, sensors) in SITES.items():
        pa = pd.concat([purpleair(z, names, s) for s in sensors], ignore_index=True)
        m = pa.merge(ref[ref["name"] == name][["t", "Y"]], on="t")
        m = m[m["Y"] >= 0].dropna(subset=["rh"])
        F, Y, RH = (m["PM2.5_CF1_ug/m3"].values, m["Y"].values, m["rh"].values)

        ok = (len(F) == len(bundle[name + "__F"])
              and np.allclose(np.sort(F), np.sort(bundle[name + "__F"]))
              and np.allclose(np.sort(Y), np.sort(bundle[name + "__Y"])))
        print("%-12s M=%6d  rebuild matches bundle: %s" % (name, len(F), ok))
        if not ok:
            raise SystemExit("rebuild does not reproduce %s" % name)

        Fc = SLOPE * F + RH_COEF * RH + INTERCEPT
        key = name + "_corr"
        arrays[key + "__Y"] = np.ascontiguousarray(Y)
        arrays[key + "__F"] = np.ascontiguousarray(Fc)
        f, y = Fc - Fc.mean(), Y - Y.mean()
        s2f, s2y, sfy = (f * f).mean(), (y * y).mean(), (f * y).mean()
        a, b = f * f, f * y
        C = float(np.cov(a, b, ddof=1)[0, 1])
        Vff, Vfy = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
        mm = 20.0
        V = (a * y * y).mean() / mm + (s2f * s2y - (mm - 2) * sfy ** 2) / (mm * (mm - 1))
        rows.append(dict(dataset=key, name=key, M=len(Y),
                         rho=round(float(sfy / np.sqrt(s2f * s2y)), 4),
                         lam=round(float(sfy / s2f), 4),
                         nu=round(float(C / Vff), 4),
                         r2=round(float(C * C / (Vff * Vfy)), 4),
                         t20=round(float(sfy ** 2 / V), 3),
                         mean_Y=float(Y.mean()), sd_Y=float(Y.std(ddof=1)),
                         mean_F=float(Fc.mean()), sd_F=float(Fc.std(ddof=1)),
                         skew_Y=round(float(pd.Series(Y).skew()), 3),
                         exkurt_Y=round(float(pd.Series(Y).kurtosis()), 3),
                         exkurt_FY=round(float(pd.Series(f * y).kurtosis()), 2),
                         binary_Y=False, binary_F=False,
                         paper="Barkjohn, Gantt & Clements (2021), AMT 14, 4617",
                         estimand="mean PM2.5 at the site, corrected predictor",
                         Y_ist="EPA reference monitor, hourly",
                         F_ist="PurpleAir CF_1 with the EPA correction applied",
                         repo="github.com/xiangchen-stat/AirCalibrate",
                         datei="built by data/make_purpleair_corrected.py",
                         split_im_paper="0.524*PA - 0.0862*RH + 5.75"))
        print("             corrected: rho=%.3f lam*=%.3f nu*=%.3f t(20)=%.2f"
              % (rows[-1]["rho"], rows[-1]["lam"], rows[-1]["nu"], rows[-1]["t20"]))

    np.savez_compressed(HERE / "populations_pa_corr.npz", **arrays)
    pd.DataFrame(rows).to_csv(HERE / "population_table_pa_corr.csv", index=False)
    print("wrote", HERE / "populations_pa_corr.npz")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
