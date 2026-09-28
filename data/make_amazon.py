"""Three Amazon populations, built from the PAS bundle of Li & Ignatiadis.

The bundle stores 200 products side by side, with between 251 and 913 reviews
each. Three populations are formed from it:

``amazon_tuned``  all 74913 reviews of the 200 products pooled, predicted by
                  the authors' fine-tuned BERT;
``amazon_base``   the same population, predicted by the off-the-shelf
                  nlptown BERT;
``amazon_top``    the mean rating of the single product with the most reviews,
                  B007JFMH8M, all 913 of them, fine-tuned BERT.

The first two answer a question about the 200 products at once, the third the
per-product question of the source paper.

Source: Li & Ignatiadis (2025), Prediction-Powered Adaptive Shrinkage, ICML,
arXiv 2502.14166; reviews from the SNAP Amazon Fine Food Reviews corpus.

Run from the project root:  python data/make_amazon.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 1
TOP = "B007JFMH8M"

src = np.load(HERE / "populations_lit6.npz")
names = json.load(open(HERE / "group_names.json"))["pas_amazon_bert_tuned"]["group_names"]
G = src["pas_amazon_bert_tuned__G"]
groups, counts = np.unique(G, return_counts=True)
K = int(counts.min())
top_g = int(groups[names.index(TOP)])
assert int(counts[list(groups).index(top_g)]) == counts.max()


def pooled(model):
    """Every review of every product."""
    key = "pas_amazon_bert_" + model
    return src[key + "__Y"], src[key + "__F"], src[key + "__G"]


out, rows = {}, []
YT, FT, GT = pooled("tuned")
YB, FB, _ = pooled("base")
mask = src["pas_amazon_bert_tuned__G"] == top_g
built = [("amazon_tuned", YT, FT, "fine-tuned BERT of the authors",
          "mean true rating over the pooled 200 products"),
         ("amazon_base", YB, FB, "off-the-shelf nlptown BERT",
          "mean true rating over the pooled 200 products"),
         ("amazon_top", src["pas_amazon_bert_tuned__Y"][mask],
          src["pas_amazon_bert_tuned__F"][mask], "fine-tuned BERT of the authors",
          "mean rating of product " + TOP)]

for name, Y, F, model, estimand in built:
    Y = np.ascontiguousarray(np.asarray(Y, float))
    F = np.ascontiguousarray(np.asarray(F, float))
    out[name + "__Y"], out[name + "__F"] = Y, F
    f, y = F - F.mean(), Y - Y.mean()
    s2f, s2y, sfy = (f * f).mean(), (y * y).mean(), (f * y).mean()
    a, b = f * f, f * y
    C = float(np.cov(a, b, ddof=1)[0, 1])
    Vff, Vfy = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    m = 20.0
    V = (a * y * y).mean() / m + (s2f * s2y - (m - 2) * sfy ** 2) / (m * (m - 1))
    rows.append(dict(dataset=name, name=name, M=len(Y),
                     n_groups=(200 if name != "amazon_top" else 1),
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
                     paper="Li & Ignatiadis (2025), ICML, arXiv 2502.14166",
                     estimand=estimand, Y_ist="true star rating 1-5",
                     F_ist=model,
                     repo="github.com/listar2000/prediction-powered-adaptive-shrinkage",
                     datei="built by data/make_amazon.py from populations_lit6.npz",
                     split_im_paper="all reviews pooled; smallest product has "
                                    "%d reviews" % K))
    print("%-13s M=%6d  rho=%.3f  lam*=%.3f  nu*=%.3f  t(20)=%.2f"
          % (name, len(Y), rows[-1]["rho"], rows[-1]["lam"], rows[-1]["nu"],
             rows[-1]["t20"]))

np.savez_compressed(HERE / "populations_amazon.npz", **out)
pd.DataFrame(rows).to_csv(HERE / "population_table_amazon.csv", index=False)
print("wrote", HERE / "populations_amazon.npz")
