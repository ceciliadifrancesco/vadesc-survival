# VaDeSC: reproduction and robustness analysis

Course project for CS-E4885 Machine Learning in Biomedicine.

We re-implement VaDeSC ([Manduchi et al., ICLR 2022](https://arxiv.org/abs/2106.05763)) in PyTorch,
compare it with standard survival models, and look at when its clusters are stable and meaningful.
The original TensorFlow code is at [i6092467/vadesc](https://github.com/i6092467/vadesc).

## Setup

```bash
pip install -r requirements.txt
```

## Data

- SUPPORT is downloaded automatically into `data/` from the VaDeSC repository.
- FLChain, GBSG2 and WHAS500 come with scikit-survival.

## Structure

```
src/
  data.py       loading and preprocessing
  vadesc.py     model and training
  baselines.py  Cox PH, random survival forest, Weibull AFT, DSM
  ssc.py        semi-supervised clustering (Bair & Tibshirani)
  dsm/          Deep Survival Machines, official code (auton-survival, MIT)
  metrics.py    C-index, integrated Brier score, log-rank test
  simulate.py   synthetic data (port of the official simulation code)
  missing.py    missingness simulation and missingness-aware VaDeSC (MissVaDeSC)
  paper_setup.py  SUPPORT with the official VaDeSC preprocessing and split (and variants)
scripts/
  collect_poster_results.py  copies result tables from the notebooks into results/poster/
imputation/
  data_imputation.py  VAE imputation of SUPPORT
notebooks/
  00_datasets.ipynb                           the datasets: what they are, variables, censoring, missing values
  01_vadesc_support.ipynb                     VaDeSC and baselines on SUPPORT
  02_simulated_benchmark.ipynb                simulated data, one seed
  03_simulated_benchmark_multi.ipynb          simulated data, 5 seeds
  04_vae_support_imputation.ipynb             VAE imputation of SUPPORT
  05_vae_vadesc_on_VAE_imputed_support.ipynb  VaDeSC on VAE-imputed SUPPORT
  06_missingness.ipynb                        missingness on simulated data and SUPPORT
  07_controlled_missingness.ipynb             values removed from GBSG2/WHAS500, FLChain
  08_ssc_baseline.ipynb                       semi-supervised clustering baseline
  09_dsm_baseline.ipynb                       Deep Survival Machines baseline
  10_imputation_comparison.ipynb              median vs VAE imputation for VaDeSC (3 splits, 5 seeds)
  11_cluster_interpretation.ipynb             clinical description of the VaDeSC clusters on SUPPORT
  12_poster_figures.ipynb                     poster figures (from results/poster/, saved to figures/)
  13_paper_setup.ipynb                        the paper's SUPPORT setup: why its C-index is higher than ours
  14_robustness.ipynb                         number of clusters, survival distribution, stability across splits
  15_missingness_support_splits.ipynb         all ways of handling missing values on SUPPORT (3 splits, 5 seeds)
  16_support_10_splits.ipynb                  SUPPORT on 10 splits: leakage, imputation ablation, missingness variants, tuning
  17_real_10_splits.ipynb                     FLChain, GBSG2, WHAS500 on 10 splits, with a tuned VaDeSC

Run notebooks from the `notebooks/` folder.

## Reproducing the results

Each notebook runs on its own (`QUICK = True` at the top gives a short check where available).
Times are for a CPU without a GPU.

| Result | Notebook | Time |
|---|---|---|
| Dataset description, Table 1 | 00 | 1 min |
| SUPPORT benchmark, leakage, seed stability | 01 | 45 min |
| Simulated benchmark (60k patients, 1 seed / 10k patients, 5 seeds) | 02, 03 | hours (GPU recommended) |
| VAE imputation | 04, 05, 10 | 30-60 min each |
| Missingness experiments (mask, MissVaDeSC) | 06, 07 | 1 h each |
| Semi-supervised clustering baseline | 08 | 30 min |
| Deep Survival Machines baseline | 09 | 30 min |
| Clinical interpretation of the clusters, comparison with Cox risk groups | 11 | 10 min |
| Why the paper's SUPPORT C-index is higher (official preprocessing) | 13 | 20 min |
| Number of clusters, survival distribution, stability across splits | 14 | 45 min |
| Imputation and missingness-aware VaDeSC on SUPPORT (3 splits, 5 seeds) | 15 | 15 min (10 parallel jobs) |
| SUPPORT on 10 splits: leakage, imputation ablation, missingness variants, tuning | 16 | 1.5 h (8 parallel jobs) |
| FLChain, GBSG2, WHAS500 on 10 splits, tuned VaDeSC | 17 | 2 h (6 parallel jobs) |
| Poster figures | `scripts/collect_poster_results.py`, then 12 | 1 min |

Results that later notebooks reuse are saved in `results/` (e.g. `results/sim_by_seed.csv` from
notebook 03 is used by 08 and 09).

## Licence

The code written for this project is under the MIT licence (see `LICENSE`). Two parts come from other projects and keep their own terms: `src/dsm/` is copied from [auton-survival](https://github.com/autonlab/auton-survival) (MIT), and `src/simulate.py` is ported from the [official VaDeSC code](https://github.com/i6092467/vadesc) by Manduchi et al., which has no licence, so the rights to that code remain with its authors.
