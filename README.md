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
  baselines.py  Cox PH, random survival forest, Weibull AFT
  metrics.py    C-index, integrated Brier score, log-rank test
notebooks/
  01_vadesc_support.ipynb
```

Run notebooks from the `notebooks/` folder.
