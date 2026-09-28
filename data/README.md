# Data: sources and rebuild recipes

No data is redistributed with this repository. This file says, for every
population the thesis uses, where the original data comes from, what to
cite, which columns become the label `Y` and the prediction `F`, and how the
population was formed. Five of the recipes below were re-run from the
public sources and reproduce the bundles exactly (marked *verified*); two
are written from the population's documented description (marked *check*).

## How the code expects the data

`ppi/populations.py` reads each population from a bundle
`data/populations_<bundle>.npz` holding two float arrays per population,
`<name>__Y` and `<name>__F`, of equal length `M`. A companion
`data/population_table_<bundle>.csv` is optional; only its `name` column is
used. Bundles that are absent are skipped, so the populations can be
rebuilt one bundle at a time. To write a bundle:

```python
import numpy as np
np.savez_compressed("data/populations_subset.npz",
                    freesolv__Y=Y, freesolv__F=F, ...)
```

The four benchmark datasets of Appendix D (`galaxies`, `forest`,
`alphafold`, `ballots`) are read by `ppi/realdata.py` from
`data/ppi_py/<name>.npz` with keys `Y` and `Yhat`, the files the
[ppi_py](https://github.com/aangelopoulos/ppi_py) examples download.

Every population is a finite set of `(Y, F)` pairs. One replicate of the
simulation draws `n` pairs without replacement as the labelled sample and
treats the rest as the unlabelled block; the mean of `Y` over all `M` pairs
is the estimand. The checks below (`M`, means, `rho`, `lambda*`) are what a
correct rebuild reproduces; `rho` and `lambda*` are also Tables 5.1 to 5.3
of the thesis.

| name | bundle | M | mean Y | mean F | rho | lambda* |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `freesolv` | subset | 642 | -3.8030 | -3.4861 | .933 | .857 |
| `nwp_temp` | subset | 36 826 | 276.4870 | 275.8187 | .842 | .927 |
| `nwp_precip` | subset | 4 043 | 2.9034 | 3.4696 | .743 | .813 |
| `amazon_base` | amazon | 74 913 | 4.1719 | 3.9999 | .800 | .790 |
| `amazon_tuned` | amazon | 74 913 | 4.1719 | 4.2178 | .888 | .871 |
| `amazon_top` | amazon | 913 | 4.5827 | 4.7514 | .748 | .799 |
| `agb_chave05_dry` | chave | 4 004 | 1134.1393 | 624.3058 | .913 | 2.398 |
| `cu_scanner3` | xrf | 99 | 106.8310 | 216.8384 | .909 | .349 |
| `pas_galaxy_spiral` | lit6 | 106 551 | 0.5240 | 0.4496 | .747 | .750 |
| `tree_cover_binary` | lit6 | 67 968 | 0.4939 | 0.5529 | .826 | .830 |
| `compas_violent` | subset | 6 172 | 0.1121 | 0.3642 | .182 | .231 |
| `lsi_lowbr` | subset | 1 953 | 0.0876 | 0.3518 | .094 | .176 |
| `moral_old` | subset | 60 000 | 0.2324 | 0.4072 | .164 | .141 |

The bundle names are those in `populations.BUNDLES`; the grouping is
historical and carries no meaning. Everything can go into one bundle.

## Strong predictors

### `freesolv`, hydration energy (*verified*)

* Source: the FreeSolv database, https://github.com/MobleyLab/FreeSolv,
  file `database.txt` (semicolon-separated, comment lines start with `#`).
* Cite: Mobley (2013), *Experimental and Calculated Small Molecule
  Hydration Free Energies*, eScholarship; Duarte Ramos Matos et al. (2017),
  *J. Chem. Eng. Data* 62(5), 1559–1569.
* `Y` = column 4 (index 3), the experimental hydration free energy in
  kcal/mol. `F` = column 6 (index 5), the value calculated by the Mobley
  group (GAFF force field, AM1-BCC charges, explicit water). All 642 rows.

```bash
git clone --depth 1 https://github.com/MobleyLab/FreeSolv.git
```
```python
rows = [l.split(";") for l in open("FreeSolv/database.txt") if not l.startswith("#")]
Y = np.array([float(r[3]) for r in rows]); F = np.array([float(r[5]) for r in rows])
```

### `nwp_temp`, temperature (*verified*)

* Source: dataset `srft` of the R package `ensembleBMA` (the University of
  Washington mesoscale ensemble, 48-h surface temperature forecasts and
  station observations, Pacific Northwest). The CRAN mirror on GitHub holds
  it as `data/srft.rda`.
* Cite: Raftery, Gneiting, Balabdaoui and Polakowski (2005), *Monthly
  Weather Review* 133(5), 1155–1174.
* `Y` = `observation` (kelvin). `F` = the mean of the eight members `CMCG`,
  `ETA`, `GASP`, `GFS`, `JMA`, `NGPS`, `TCWB`, `UKMO`. Rows with a missing
  member or observation are dropped; 36 826 remain.

```bash
git clone --depth 1 https://github.com/cran/ensembleBMA.git
pip install rdata          # pure-Python reader for .rda files
```
```python
import rdata
srft = rdata.read_rda("ensembleBMA/data/srft.rda")["srft"]
members = ["CMCG", "ETA", "GASP", "GFS", "JMA", "NGPS", "TCWB", "UKMO"]
d = srft.dropna(subset=members + ["observation"])
Y = d["observation"].to_numpy(float); F = d[members].astype(float).mean(axis=1).to_numpy()
```

### `nwp_precip`, precipitation (*verified*)

* Source: dataset `prcpDJdata` of the same package, 48-h precipitation
  forecasts of nine members and station observations, December–January.
* Cite: as above (Raftery et al. 2005; the data are those of Sloughter,
  Raftery, Gneiting and Fraley 2007, *Monthly Weather Review* 135, 3209–3220).
* Both sides on the square-root scale. `Y` = `sqrt(observations)`.
  `F` = the mean over the nine members `avn/gfs`, `cent`, `cmcg`, `eta`,
  `gasp`, `jma`, `ngps`, `tcwb`, `ukmo` of the square root of each member's
  forecast (mean of square roots, not square root of the mean). Rows with a
  missing value dropped; 4 043 remain.

```python
p = rdata.read_rda("ensembleBMA/data/prcpDJdata.rda")["prcpDJdata"]
mem = ["avn/gfs", "cent", "cmcg", "eta", "gasp", "jma", "ngps", "tcwb", "ukmo"]
d = p.dropna(subset=mem + ["observations"])
Y = np.sqrt(d["observations"].to_numpy(float)); F = np.sqrt(d[mem].astype(float)).mean(axis=1).to_numpy()
```

### `amazon_base`, `amazon_tuned`, `amazon_top`, product reviews

* Source: the data bundle of Li and Ignatiadis (2025), *Prediction-Powered
  Adaptive Shrinkage Estimation*, ICML 2025, code and data at
  https://github.com/listar2000/prediction-powered-adaptive-shrinkage:
  `data/amazon/amazon_review.h5` (fine-tuned BERT) and
  `data/amazon/amazon_review_raw.h5` (the off-the-shelf
  `nlptown/bert-base-multilingual-uncased-sentiment`). The reviews are the
  SNAP *Amazon Fine Food Reviews* corpus; the bundle holds the 200 products
  with the most reviews.
* Cite: Li and Ignatiadis (2025); Stanford Network Analysis Project (2014),
  *Amazon Fine Food Reviews*, Kaggle; McAuley and Leskovec (2013), WWW '13,
  897–908, which SNAP asks to be cited for the corpus; Devlin et al. (2019)
  for BERT.
* `Y` = the star rating 1–5, `F` = the BERT prediction of it.
  `amazon_tuned` and `amazon_base` pool all 74 913 reviews of the 200
  products (fine-tuned and off-the-shelf model respectively); `amazon_top`
  is the single product with the most reviews, `B007JFMH8M`, 913 reviews,
  fine-tuned model.
* Two steps. First read each `.h5` file into three arrays of length
  74 913, the true rating, the predicted rating and the product index
  0–199, and store them as `pas_amazon_bert_tuned__Y`, `__F`, `__G` and
  `pas_amazon_bert_base__Y`, `__F`, `__G` in `data/populations_lit6.npz`,
  with the product ids of the 200 groups in `data/group_names.json` as
  `{"pas_amazon_bert_tuned": {"group_names": [...]}}` (*check* the column
  names inside the `.h5` files against the authors' loading code). Then
  `python data/make_amazon.py` forms the three thesis populations from
  that bundle.

### `agb_chave05_dry`, tree biomass (*check*)

* Source: the pantropical direct-harvest data of Chave et al. (2014),
  *Global Change Biology* 20(10), 3177–3190, 4 004 felled trees from 58
  sites, distributed with the paper (Chave's site
  http://chave.ups-tlse.fr/pantropical_allometry.htm, file
  `Chave_GCB_Direct_harvest_data.csv`; columns: site, trunk diameter `D` in
  cm, height `H` in m, wood specific gravity `WSG`, measured dry aboveground
  biomass in kg).
* Cite: Chave et al. (2014) for the data, Chave et al. (2005), *Oecologia*
  145(1), 87–99, for the equation.
* `Y` = the measured dry aboveground biomass of the tree in kg. `F` = the
  Chave et al. (2005) dry-forest model with height applied to every tree,
  `AGB = 0.112 (WSG * D^2 * H)^0.916`, so that the prediction uses only
  diameter, height and wood density. All 4 004 trees; nothing is fitted.
* *Check*: this is the population's documented construction (Table 5.1a of
  the thesis); the original build script is not in this repository. A
  correct rebuild has `M = 4004`, mean `Y = 1134.1` kg, mean `F = 624.3` kg
  and `lambda* = 2.398`; the largest tree has `Y = 76 063.5` kg and
  `F = 18 725.6` kg.

### `cu_scanner3`, soil copper (*check*)

* Source: the supplementary data of Declercq et al. (2019), *Remote
  Sensing* 11(21), 2490, https://doi.org/10.3390/rs11212490, homogenised
  soil samples from ten countries measured in the laboratory and with three
  portable X-ray fluorescence scanners.
* Cite: Declercq et al. (2019).
* `Y` = the laboratory copper concentration of a sample, `F` = the copper
  reading of scanner 3 for the same sample; the 99 samples with both
  values.
* *Check*: as for the trees, the recipe is the documented construction. A
  correct rebuild has `M = 99`, mean `Y = 106.8`, mean `F = 216.8`,
  `lambda* = 0.349`, `Y` in `[5.3, 511.0]` and `F` in `[11.7, 1395.3]`.

## Binary predictors

### `pas_galaxy_spiral`, spiral galaxies

* Source: the same Li and Ignatiadis bundle,
  `data/galaxy/galaxy_test_prediction.csv`. Labels are Galaxy Zoo 2
  classifications (Willett et al. 2013, *MNRAS* 435(4), 2835–2860) of SDSS
  images (York et al. 2000); predictions come from the authors' ResNet50
  (He et al. 2016) trained on about 50 000 held-out images.
* `Y` = the volunteer classification binarised, classes 3, 4 and 5 counting
  as spiral. `F` = the ResNet50 prediction binarised the same way. All 102
  spatial bins of the source paper are pooled into one population of
  106 551 galaxies (the bin is not used).

### `tree_cover_binary`, tree cover

* Source: the Predict-then-Debias bundle of Kluger, Lu, Zrnic, Wang and
  Bates (2025), https://github.com/DanKluger/Predict-Then-Debias_Bootstrap,
  file `UpdatedExperimentsJune2026/Datasets/TreecoverBinarizedFormatted.RData`,
  two row-aligned data frames `GoodDat` (measured) and `ProxyDat`
  (predicted). Labels derive from the USFS CONUS 2021 tree canopy layer
  (USDA Forest Service 2023); predictions are the MOSAIKS satellite
  featuriser of Rolf et al. (2021), *Nature Communications* 12, 4392.
* `Y` = `GoodDat$Forest`, the indicator of tree cover above ten percent.
  `F` = `ProxyDat$Forest`, the same indicator from the prediction. All
  67 968 tiles.
* Read the `.RData` with `rdata.read_rda(path)`; it returns both frames.
  In other files of that repository the proxied column is a covariate
  rather than the outcome, so check which column differs between the two
  frames before using any of them.

## Weak predictors

### `compas_violent`, COMPAS recidivism (*verified*)

* Source: ProPublica's COMPAS analysis,
  https://github.com/propublica/compas-analysis, file
  `compas-scores-two-years.csv`.
* Cite: Angwin, Larson, Mattu and Kirchner (2016), *Machine Bias*, ProPublica.
* ProPublica's standard filter is applied: `days_b_screening_arrest` in
  [-30, 30], `is_recid != -1`, `c_charge_degree != "O"`,
  `score_text != "N/A"`, leaving 6 172 defendants. `Y` = `is_violent_recid`,
  re-arrest for a violent offence within two years. `F` =
  `v_decile_score / 10`, the violent-recidivism decile score at intake.

```python
c = pd.read_csv("compas-analysis/compas-scores-two-years.csv")
c = c[(c.days_b_screening_arrest.between(-30, 30)) & (c.is_recid != -1)
      & (c.c_charge_degree != "O") & (c.score_text != "N/A")]
Y = c["is_violent_recid"].to_numpy(float); F = c["v_decile_score"].to_numpy(float) / 10
```

### `lsi_lowbr`, LSI-R recidivism (*verified*)

* Source: Lin, Jung, Goel and Skeem (2020), *Science Advances* 6(7),
  eaaz0652, data at https://github.com/stanford-policylab/recidivism-predictions:
  `data/public/surveys/lsi_low_br_response.csv` (one row per crowdsourced
  forecast) and
  `data/public/individuals/lsi_vignettes_site_low_br_outcome_only.csv` (one
  row per person, the low-base-rate site).
* `F` = the mean of `predicted_decision / 100` over all forecasts of a
  person (`individual_id` in the survey file, `id` in the outcome file;
  cast both to numeric before merging). `Y` = `outcome`. One of the 1 954
  persons has no forecast and is dropped; 1 953 remain.

```python
r = pd.read_csv(".../lsi_low_br_response.csv"); o = pd.read_csv(".../lsi_vignettes_site_low_br_outcome_only.csv")
m = (r.groupby(pd.to_numeric(r.individual_id))["predicted_decision"].mean() / 100).rename("F")
j = o.merge(m, left_on=pd.to_numeric(o.id), right_index=True)
Y = j["outcome"].to_numpy(float); F = j["F"].to_numpy(float)
```

### `moral_old`, sparing the elderly

* Source: the Mixed Subjects Design replication package of Broska, Howes
  and van Loon (2025), *Sociological Methods & Research* 54(3), 1074–1109,
  https://github.com/davidbroska/MixedSubjects, file
  `Data/3_gpt4turbo_wp_20241118.csv.gz`; the dilemmas are those of the
  Moral Machine experiment (Awad et al. 2018, *Nature* 563, 59–64).
* Restrict to `AttributeLevel == "Old"` (without a restriction the human
  mean is exactly one half by construction), then thin to 60 000 rows at
  random. `Y` = `Saved`, one participant's decision to spare the character.
  `F` = `gpt4turbo_wp_Saved`, GPT-4-turbo's answer to the same dilemma.

## The four PPI benchmarks (Appendix D, Table D.1)

`galaxies`, `forest`, `alphafold` and `ballots` are the mean-estimation
examples of Angelopoulos et al. (2023), downloaded by the notebooks in
`examples/` of https://github.com/aangelopoulos/ppi_py as `<name>.npz`
files with keys `Y` and `Yhat` (`forest` is `examples/baselines/forest_badmodel.npz`
in that repository, a deliberately weakened linear model). Place the four
files in `data/ppi_py/`; `ppi/realdata.py` reads them from there.

## Scripts in this folder

* `make_amazon.py`: the three Amazon populations from the Li and
  Ignatiadis bundle.
* `make_purpleair_*.py`: PurpleAir air-quality populations that were built
  for an earlier version of the study and are not used in the thesis;
  kept for reference.
