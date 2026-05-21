# Project Agent Notes

## Project Purpose

This repository contains research code for estimating normalizing constants with
path sampling, coupled MCMC, unbiased estimators, and stratified allocation over
the path parameter lambda.

The notebooks are the main experimental entry points. The `src/` package holds
the reusable components used by those experiments.

## Repository Map

- `src/path_MH.py`: Metropolis-Hastings kernels and coupled transition kernels.
- `src/debiasedalgo.py`: Coupled-chain simulation and unbiased MCMC estimator.
- `src/normal.py`: Normal-model path-sampling experiments, tuning of `k` and
  `m`, moment estimation, and importance sampling over lambda.
- `src/startified_estimator.py`: Stratified estimators, adaptive splitting,
  budget allocation, common random numbers, and MSE comparison helpers.
- `src/gaussian_LOO.py`: Gaussian linear regression leave-one-out experiments,
  Zanella-style defensive mixture sampling, and LOO tempering paths.
- `*.ipynb`: Experiment notebooks and exploratory workflows.

## Working Conventions

- Preserve the research/prototype style unless explicitly asked to refactor.
- Avoid reverting notebook or source changes that already exist in the working
  tree; they may be user experiments.
- Prefer small, focused changes and stage only the files touched for the task.
- Use explicit module imports instead of expanding `from src import *` in new
  code.
- Keep stochastic experiments reproducible when practical by accepting an RNG or
  seed rather than relying only on global `np.random` state.
- Be careful with dimensions: parts of the current MCMC code assume a
  one-dimensional state, while the Gaussian LOO code works with vector-valued
  regression parameters.

## Validation

- For import checks, use `python3` in this environment.
- The plotting utilities depend on packages such as `seaborn`, `matplotlib`,
  `pandas`, `numpy`, `scipy`, and `tqdm`; install or verify dependencies before
  treating an import failure as a code regression.
- There is no dedicated test suite yet, so prefer small smoke tests around the
  edited function or notebook workflow.

## Git Notes

- The default branch is `main` and the remote is `origin`.
- If unrelated files are already modified, leave them unstaged unless the user
  explicitly asks to include them.
- Commit messages should be concise and describe the actual project change.
