# GA-QSVM: Flexible Genetic Algorithm for Quantum Support Vector Machines

Fork of [vutuanhai237/GA-QSVM](https://github.com/vutuanhai237/GA-QSVM), the code
for the paper [Flexible Genetic Algorithm for Quantum Support Vector Machines](https://arxiv.org/abs/2511.19160).
The goal of this fork is a simplified, step-by-step reproduction of the paper,
one figure at a time, as a base for further modifications.

```
@misc{duc2025flexiblegeneticalgorithmquantum,
      title={Flexible Genetic Algorithm for Quantum Support Vector Machines},
      author={Nguyen Minh Duc and Vu Tuan Hai and Le Bin Ho and Tran Nguyen Lan},
      year={2025},
      eprint={2511.19160},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2511.19160},
}
```

## Setup

```bash
uv sync --dev
uv run pytest -q
```

## Reproduction roadmap

| Step | Paper | Status | Command |
|------|-------|--------|---------|
| 1. Data preparation and PCA | Section 5.1, Figure 3 | done | `uv run python experiments/fig3_pca.py` |
| 2. GA core and hyperparameter study | Section 4, Figure 4 | pending | |
| 3. Optimal 7-qubit circuits | Figure 5 | pending | |
| 4. Model comparison (fixed split, k-fold) | Figure 6 | pending | |
| 5. Transfer learning | Figure 7 | pending | |
| 6. Quantum noise | extension | pending | |

### Step 1: data preparation (Figure 3)

All dataset handling lives in `ga_qsvm/data.py`:

- `load_dataset(name)`: full Digits, Wine, Breast Cancer (scikit-learn) or
  Fashion-MNIST (downloaded once into `data/`, same files as `keras.datasets`).
- `paper_split(name, n_features)`: the fixed stratified split used by the GA
  (100 train / 100 test; Wine 100 / 78), then MinMaxScaler and PCA to
  `n_features` components, both fitted on the training split only. The output
  is bit-identical to the original authors' `data/split.py`.
- `explained_variance_curve(name)`: cumulative PCA explained variance of the
  full MinMax-scaled dataset.

`experiments/fig3_pca.py` writes `results/fig3/fig3_pca.{pdf,png}` and CSVs.
Components needed for 95% explained variance:

| Dataset | This repo | Paper |
|---------|-----------|-------|
| Digits | 30 | 30 |
| Fashion | 188 | 200 |
| Wine | 10 | 10 |
| Breast Cancer | 10 | 10 |

The Fashion value is 187-188 for every reasonable preprocessing (MinMax, /255,
raw pixels, train-only), so the paper's 200 appears to be a rounded value.
Note that Table 2 of the paper lists 5620 Digits and 592 Breast Cancer
instances; scikit-learn provides 1797 and 569, which is what the code uses.

## Project structure

- `ga_qsvm/data.py`: datasets, paper split, preprocessing.
- `ga_qsvm/runners/`, `ga_qsvm/cli/`: GA training/evaluation entry points (to be reworked in step 2).
- `qoop/`: Quantum Object Optimizer package with the GA operators (to be trimmed in step 2).
- `experiments/`: one script per paper figure.
- `results/`: figure outputs.
- `docs/PLOTTING_STYLE.md`: plotting conventions for paper figures.

## Acknowledgments

This project uses the QOOP (Quantum Object Optimizer) package developed by Vu Tuan Hai, Nguyen Tan Viet, and Le Bin Ho.
