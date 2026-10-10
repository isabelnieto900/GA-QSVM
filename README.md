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
| 2a. GA core | Section 4, Algorithm 2 | done | `uv run python experiments/run_ga.py --dataset wine --qubits 3` |
| 2b. Hyperparameter study | Figure 4 | script ready | `uv run python experiments/fig4_hyperparams.py` |
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

### Step 2a: genetic algorithm (Section 4)

- `ga_qsvm/circuits.py`: gate pool `{H, RX, RY, RZ, CX}`, random circuit
  generator, crossover, mutation and normalizers.
- `ga_qsvm/kernels.py`: QSVM with the Fidelity Quantum Kernel (FQK) or the
  Projected Quantum Kernel (PQK, squlearn), and `QSVMFitness` (Eq. 8: test
  accuracy of a QSVM trained with the circuit as feature map).
- `ga_qsvm/ga.py`: `GAConfig` (metadata M of Eq. 12) and `run_ga` (Algorithm 2):
  elitist selection of the best half, one-point crossover, bit-flip mutation,
  stop when the best fitness exceeds 0.99 or does not improve for 50 generations.
- `experiments/run_ga.py`: command-line runner. Each run is saved to
  `results/ga/<dataset>-<kernel>-n<qubits>-seed<seed>/` (`best_circuit.qpy`,
  `best_circuit.txt`, `summary.json`, `history.csv`).

```bash
uv run python experiments/run_ga.py --dataset digits --kernel fqk --qubits 3 4 5 6 7 --seed 0
```

Defaults are the paper's default configuration: `d = 5n`, `nCX = 2n`,
`ncircuit = 16`, `p = 0.1`, 100 generations.

**Fast FQK.** Qiskit's `FidelityQuantumKernel` builds and samples one circuit
per pair of samples (1-4 min per fitness evaluation on a laptop). The default
`--fqk-backend statevector` computes the same kernel from statevectors
(\|<psi_i|psi_j>\|^2 with the same PSD projection as Qiskit). On 18 test
circuits the accuracies are identical and the kernel matrices differ by at
most 1e-14, about 100x faster. Use `--fqk-backend qiskit` for the original path.

**Paper text vs. code.** The GA is a line-by-line port of the authors' `qoop`
code (checked with identical random seeds: same circuits, same fitness
history). By default it reproduces what the code did, which differs from the
paper text in some places. Each difference can be switched on with a flag:

| Paper text | Original code (default) | Flag for the paper version / fix |
|------------|-------------------------|----------------------------------|
| Normalizer truncates circuits deeper than `d` and adds CX if fewer than `nCX` | Forces exactly `n` rotations: cuts after the n-th rotation or appends RX | `--normalizer-mode depth_cnot` |
| Crossover cuts both parents at a depth (Fig. 2b) | Cuts after rotation number `n/2` | `--crossover-mode depth` |
| `d` is the circuit depth | `d * n` gate slots, but a `zip` bug keeps only one of four random counts for the H gates, so circuits are shorter (n=5, d=25: about 42 gates, real depth about 17) | `--fill-all-slots` |
| `(nRx, nRy, nRz)` are part of the metadata | Sampled per circuit, biased toward RX (n=5: about 2.4 RX vs 1.2 RZ) | `--rotations RX RY RZ` or `--rotation-mode uniform` |
| Mutation replaces a gate with a different one | Can redraw the same gate | `--mutation-distinct` |
| Elitism keeps the best half | The copied parents are mutated too | `--keep-elites` |
| Fewer QSVM iterations during the GA | Solver always runs to convergence | `--max-iter N` |

Other notes:

- Algorithm 2 writes the stop condition as `f < tau`; the code stops when the
  best fitness is `> 0.99`, plus a patience of 50 generations not in the paper.
- The fitness is the accuracy on the same test split that is reported, as the
  paper states.
- The normalizer and mutation interact: a mutation that adds a rotation makes
  the normalizer truncate the tail of the circuit, so circuits tend to lose CX
  gates over the generations.
- Upstream commits between Feb 2025 and the 2026 refactor removed CX from the
  generator and did not pass `prob_mutate` to the mutation (always 0.1). This
  port follows the version consistent with the paper (2n CX, `p` is used).
- Runs are reproducible with `--seed`; the original had no global seed.

### Step 2b: hyperparameter study (Figure 4)

`experiments/fig4_hyperparams.py` runs the GA on Digits with n = 5 and varies
one hyperparameter per panel: (a) `d` in 5-25, (b) `ncircuit` in 4-20,
(c) `nCX` in 5-25, (d) `p` in 0.001-0.5. It plots the best fitness of each
generation, mean and standard deviation over repeated runs. Runs are saved in
`results/fig4/<base>-<kernel>/runs/` and skipped when they already exist, so
the study can be resumed; `--plot-only` redraws the figure.

The authors' script for this figure (`benchmark.py`, Sep 2025) differs from the
caption ("n = 5, ncircuit = 16, p = 0.1, d = 5n, nCX = 2n"):

- base values `d = 35`, `nCX = 14` (the `5n`, `2n` values for n = 7), not 25 and 10;
- PQK kernel, 200 generations, 10 runs per configuration, no early stopping;
- `prob_mutate` was not passed to the mutation, so every curve of panel (d)
  used p = 0.1. Here `p` is applied.

`--base original` (default) uses the script's values, `--base caption` the
caption's; both can be given at once. All GA runs are independent and run in
parallel, one per core (`--jobs`, default: all cores). One PQK fitness
evaluation takes about 1 s of one core, so the full original study
(18 configurations x 10 runs x 200 generations) is about 150 core-hours per
base: about 10 hours on 16 cores, or 45 hours on a 6-core laptop.
`--repeats` and `--num-generation` reduce it.

```bash
uv run python experiments/fig4_hyperparams.py --base original caption --repeats 3 --num-generation 100
uv run python experiments/fig4_hyperparams.py --base original caption --repeats 3 --num-generation 100 --plot-only
```

## Running the experiments on another machine

The GA studies are long; they can run on a bigger machine and the results can
be copied back. The code must be pushed to the fork first.

1. Install uv and the environment (Linux/macOS):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   git clone https://github.com/isabelnieto900/GA-QSVM.git
   cd GA-QSVM
   uv sync --dev
   uv run pytest -q
   ```

   Digits, Wine and Breast Cancer come with scikit-learn; nothing else is
   downloaded for the GA experiments.

2. Run Figure 4 with both base configurations in one command. It runs one GA
   per core (all cores by default; `--jobs N` to leave some free) and prints a
   line each time a run finishes. `nohup` keeps it running after closing the
   terminal:

   ```bash
   nohup uv run python experiments/fig4_hyperparams.py --base original caption > fig4.log 2>&1 &
   tail -f fig4.log
   ```

   The defaults reproduce the authors' study (PQK, 10 runs x 200 generations,
   about 300 core-hours for both bases). Use `--repeats 3 --num-generation 100`
   for a shorter version; keep the same values in every later command, since
   they select the runs folder. Each process needs a few hundred MB of RAM.

3. If the study is interrupted, launch the same command again: finished GA
   runs are skipped. To redraw the figures from whatever has finished:

   ```bash
   uv run python experiments/fig4_hyperparams.py --base original caption --plot-only
   ```

   The GA runs for the next figures (all datasets, kernels and qubit counts)
   can be parallelized the same way:

   ```bash
   uv run python experiments/run_ga.py --dataset digits wine cancer --kernel fqk pqk --qubits 3 4 5 6 7 --seed 0 --jobs 30
   ```

4. Copy the results back: everything is in `results/fig4/` (one folder per
   base, with `fig4_hyperparams.{pdf,png}`, `fig4_curves.csv`, `study.json`
   and `runs/`).

## Project structure

- `ga_qsvm/data.py`: datasets, paper split, preprocessing.
- `ga_qsvm/circuits.py`, `ga_qsvm/kernels.py`, `ga_qsvm/ga.py`: genetic algorithm.
- `experiments/`: one script per paper figure, plus `run_ga.py`.
- `results/`: figure outputs.
- `docs/PLOTTING_STYLE.md`: plotting conventions for paper figures.

## Acknowledgments

The GA operators are ported from the QOOP (Quantum Object Optimizer) package developed by Vu Tuan Hai, Nguyen Tan Viet, and Le Bin Ho.
