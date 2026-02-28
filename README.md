# Learning Gaussian Processes with Randomized Quantum Local Kernels

This repository contains the **experimental code** used in the paper:

> **Learning Gaussian Processes with Randomized Quantum Local Kernels**  
> Abdallah Aaraba · Soumaya Cherkaoui · Ola Ahmad · Jean‑Frédéric Laprade · Shengrui Wang

## Overview

Gaussian Processes (GPs) are powerful non‑parametric models, but their performance is often bottlenecked by the **choice of covariance kernel**. This project explores a **NISQ‑compatible** approach where a quantum device acts as a *kernel generator*:

- **Shallow, geometry‑aware random quantum circuits** encode inputs into many‑qubit states.
- **Local Pauli measurements** (estimated efficiently via **classical shadows**) yield quantum feature vectors.
- A **classical wrapper** turns these features into scalar kernels.
- A **multiple‑kernel learning (MKL)** layer fits a **convex mixture** of many random local quantum kernels by maximizing the **GP marginal likelihood**.

The main reproduction entry point is the notebook:

- `notebooks/experiments/noisy_shadow_gp_learning_soft_approx.ipynb`

---

## Table of contents
- [Repository at a glance](#repository-at-a-glance)
- [Quick start](#quick-start)
- [Reproducing the paper experiments](#reproducing-the-paper-experiments)
- [Key parameters](#key-parameters)
- [Outputs](#outputs)
- [Testing](#testing)
- [Citation](#citation)

---

## Repository at a glance

**Top-level**
- `data/` — synthetic GP dataset generation and query-kernel configs
- `notebooks/` — experiment notebooks (main paper notebook lives here)
- `src/` — circuits, parameter sampling, runners, feature maps, MKL learners
- `results/` — evaluation metrics (e.g., kernel alignment)
- `viz/` — plotting utilities for figures
- `tests/` — unit / smoke tests
- `requirements.txt` — base dependencies (Qiskit + Aer + POVM toolbox)

**Code layout (high-level)**

```text
.
├── data/
│   ├── synthetic.py            # synthetic GP datasets on shared X
│   └── kernel_configs.py       # RBF / Matérn / non-stationary kernels used as targets
├── notebooks/
│   └── experiments/
│       └── noisy_shadow_gp_learning_soft_approx.ipynb
├── src/
│   ├── circuits/
│   │   ├── circuit_building.py # SQP circuit family (generator + mixing blocks)
│   │   ├── configs.py          # CircuitArchitecture, ring layers, etc.
│   │   └── param_sampling/     # uniform + Sobol sampling policies
│   ├── runners/
│   │   ├── circuit_running.py  # Aer statevector runner (ExactResults)
│   │   └── feature_maps/
│   │       ├── shadow_feature_map.py # classical-shadow “soft” simulator (MoM)
│   │       ├── exact_feature_map.py  # exact expectations (sanity checks)
│   │       └── utils.py             # k-local / edge-correlator Pauli families
│   ├── models/
│   │   ├── noisy/              # MKL learners with noise + (optional) approximations
│   │   └── non_noisy/          # noiseless variants
│   └── utils/
│       └── settings.py         # paths (storage/, notebooks/, ...)
├── results/
│   └── metrics/                # kernel alignment, Frobenius errors, etc.
├── viz/
│   ├── data/                   # GP plots
│   └── models/                 # MKL visualizations
└── tests/
```

---

## Quick start

### 0) Create an environment

We recommend Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
```

### 1) Install dependencies

Start from the provided requirements:

```bash
pip install -r requirements.txt
```

### 2) Make imports work from notebooks

From the repository root:

```bash
export PYTHONPATH="$(pwd)"
```

(Windows PowerShell)

```powershell
setx PYTHONPATH $PWD
```

### 3) Launch Jupyter

```bash
jupyter lab
```

---

## Reproducing the paper experiments

Open and run:

- `notebooks/experiments/noisy_shadow_gp_learning_soft_approx.ipynb`

The notebook is written as the **reference implementation** of the full pipeline:
1. **Generate synthetic GP data** for multiple target kernels on the same input grid  
   (`data/synthetic.py`, `data/kernel_configs.py`).
2. **Build the SQP circuit family** (shallow local circuit on a ring)  
   (`src/circuits/circuit_building.py`).
3. **Sample R random circuit parameterizations** (Sobol low‑discrepancy sequence)  
   (`src/circuits/param_sampling/sobol_sampling.py`).
4. **Simulate circuit execution** using Aer statevectors to obtain `ExactResults`  
   (`src/runners/circuit_running.py`).
5. **Estimate local features** with a classical‑shadow **Median‑of‑Means** estimator  
   (fast “soft” approximation) (`src/runners/feature_maps/shadow_feature_map.py`).
6. **Wrap features into hybrid kernels** (the notebook implements the same normalized
   cosine wrapper used in the paper) and assemble Gram matrices.
7. **Learn mixture weights** by maximizing the GP marginal likelihood (MKL)  
   (`src/models/noisy/noisy_soft_mklearner.py`).
8. **Evaluate** predictive performance + kernel alignment and reproduce figures/tables  
   (`results/metrics/`, `viz/`).

> Tip: the repo is designed so you can swap the shadow estimator with the exact one
> (`ExactFeatureMapsRetriever`) to sanity-check noise effects.

---

## Key parameters

The notebook exposes the main knobs controlling compute vs accuracy:

- **Data**
  - `M`: number of input points
  - `noise_variance`: observation noise used when sampling GP labels

- **Circuits**
  - `n`: number of qubits
  - `L`: number of layers
  - `R`: number of random parameter draws (dictionary size)

- **Measurements**
  - `observables`: typically 1-qubit Paulis + 2-qubit correlators along the ring
  - `shots`: effective number of classical-shadow snapshots
  - helper: `get_theoretical_shots(eps, delta, locality, M, m, R)` in
    `src/runners/feature_maps/shadow_feature_map.py`

- **GP / MKL optimization**
  - `train_test_split`
  - `jitter`
  - optional: approximate gradients / log-det with low-rank + Hutchinson probes
    (see `src/models/noisy/noisy_mk_learner.py`)

---

## Outputs

By default, notebooks typically write plots and intermediate artifacts under a local
`storage/` folder (you can create it at repo root):

```bash
mkdir -p storage/experiments
```

Typical outputs:
- figures (`.png`, `.pdf`)
- kernel similarity metrics (alignment, relative Frobenius error)
- serialized intermediate objects (pickles / numpy arrays) when enabled in the notebook

---

## Testing

Run unit tests from the repo root:

```bash
pytest -q
```
