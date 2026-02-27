from __future__ import annotations
from data.kernel_configs import KernelConfig, _kernel_matrix

from typing import Dict, Any, Sequence, List, Optional

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _sample_gp_for_kernel(
        cfg: "KernelConfig",
        X: np.ndarray,
        mean: np.ndarray,
        noise_variance: float,
        seed: int,
) -> np.ndarray:
    """Worker function to sample a GP for a single kernel config.

    This runs in a separate process when using ProcessPoolExecutor.
    """
    rng = np.random.default_rng(seed)
    M = X.shape[0]

    K = _kernel_matrix(X, cfg)
    K_noisy = K + noise_variance * np.eye(M, dtype=float)
    y = rng.multivariate_normal(mean=mean, cov=K_noisy)
    return y


def _sample_gp_for_kernel_from_args(args) -> np.ndarray:
    """Thin wrapper to make executor.map picklable (no lambdas/closures)."""
    return _sample_gp_for_kernel(*args)


def make_dataset(
        M: int,
        input_dim: int,
        sampling: str = "uniform",
        noise_variance: float = 0.0,
        kernel_cfgs: Sequence[KernelConfig] = (),
        seed: Optional[int] = None,
        n_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate synthetic GP data for multiple kernels on the *same* inputs.

    Parameters
    ----------
    M:
        Number of input points.
    input_dim:
        Input dimensionality ζ.
    sampling:
        How to sample the inputs X ∈ ℝ^{M×ζ}. Currently supports:
          - "uniform": i.i.d. Unif([-1,1]) over each coordinate (default).
          - "normal":  i.i.d. N(0,1) over each coordinate.
    noise_variance:
        How much noise to add to the observations.
    kernel_cfgs:
        Sequence of KernelConfig describing the covariance kernels.
        For each kernel k_i we generate a label vector y_i ∈ ℝ^M by
        sampling from a zero-mean Gaussian process with covariance matrix
        K_i(X, X) defined by that kernel.
    seed:
        Optional random seed for reproducibility.
    n_jobs:
        Number of worker processes to use for parallel generation across
        kernels. If None, uses min(os.cpu_count(), len(kernel_cfgs)).
        If 1 or if len(kernel_cfgs) <= 1, falls back to sequential mode.

    Returns
    -------
    data:
        A dict with keys:
          - "inputs": np.ndarray of shape (M, input_dim)
          - "labels": np.ndarray of shape (num_kernels, M), where
                      the i-th row corresponds to kernel_cfgs[i].
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}.")
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")

    rng = np.random.default_rng(seed)

    # Sample inputs X
    if sampling == "uniform":
        X = rng.uniform(-1.0, 1.0, size=(M, input_dim))
    elif sampling == "normal":
        X = rng.normal(loc=0.0, scale=1.0, size=(M, input_dim))
    else:
        raise ValueError(f"Unsupported sampling scheme: {sampling!r}")

    if not kernel_cfgs:
        # No kernels: just return inputs and empty labels list
        return {"inputs": X, "labels": []}

    mean = np.zeros(M, dtype=float)

    num_kernels = len(kernel_cfgs)

    # Decide on number of workers
    if n_jobs is None:
        cpu_count = os.cpu_count() or 1
        n_jobs = min(cpu_count, num_kernels)
    else:
        if n_jobs <= 0:
            raise ValueError(f"n_jobs must be positive, got {n_jobs}.")

        n_jobs = min(n_jobs, num_kernels)

    # If only one job or one kernel, run sequentially (no multiprocessing overhead)
    if n_jobs == 1 or num_kernels == 1:
        labels: List[np.ndarray] = []
        # Use the same RNG, but split seeds per-kernel for reproducibility
        seeds = rng.integers(0, 2 ** 63 - 1, size=num_kernels, dtype=np.int64)
        for cfg, s in zip(kernel_cfgs, seeds):
            y = _sample_gp_for_kernel(cfg, X, mean, noise_variance, int(s))
            labels.append(y)
        return {"inputs": X, "labels": np.asarray(labels)}

    # Parallel path: generate independent seeds for each kernel and fan out
    seeds = rng.integers(0, 2 ** 63 - 1, size=num_kernels, dtype=np.int64)

    # Prepare arguments for each worker
    worker_args = [
        (cfg, X, mean, noise_variance, int(s))
        for cfg, s in zip(kernel_cfgs, seeds)
    ]

    # Run in parallel across kernels
    labels: List[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # executor.map returns results in the same order as worker_args
        for y in executor.map(_sample_gp_for_kernel_from_args, worker_args):
            labels.append(y)

    return {"inputs": X, "labels": np.asarray(labels)}
