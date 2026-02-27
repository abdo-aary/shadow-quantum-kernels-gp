from __future__ import annotations

from data.kernel_configs import KernelConfig, _kernel_matrix
import numpy as np
from typing import Dict, Any, Sequence


def build_true_test_grams(
    data: Dict[str, Any],
    kernel_cfgs: Sequence[KernelConfig],
    test_idx: np.ndarray,
) -> np.ndarray:
    """
    True Gram matrices on the test inputs for each generating kernel.

    Parameters
    ----------
    data
        Dict returned by `make_dataset`, with key "inputs" of shape (M, input_dim).
    kernel_cfgs
        Same list of KernelConfig used to generate the GP samples.
    test_idx
        1D integer array of test indices (e.g. mkl.test_idx).

    Returns
    -------
    true_test_grams : (num_gps, M_test, M_test) array
        true_test_grams[g] = K_g(X_test, X_test) for kernel_cfgs[g].
    """
    # Full input matrix, (M, input_dim)
    X = np.asarray(data["inputs"], dtype=float)

    # Restrict to test points, keep the full dimension (M_test, input_dim)
    X_test = X[test_idx]

    K_true_list = []
    for cfg in kernel_cfgs:
        # _kernel_matrix already implements all your kernel types
        K_true = _kernel_matrix(X_test, cfg)  # (M_test, M_test)
        K_true_list.append(K_true)

    return np.stack(K_true_list, axis=0)      # (num_gps, M_test, M_test)


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center a Gram matrix: Kc = (I - 1/n 11^T) K (I - 1/n 11^T)."""
    n = K.shape[0]
    one_n = np.ones((n, n), dtype=K.dtype) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n


def frobenius_relative_error(
    K_hat: np.ndarray,
    K_true: np.ndarray,
    *,
    center: bool = False,
    eps: float = 1e-12,
) -> float:
    """
    Relative Frobenius error ||K_hat - K_true||_F / ||K_true||_F.

    Both matrices must have the same shape (n, n).
    """
    K_hat = np.asarray(K_hat, dtype=float)
    K_true = np.asarray(K_true, dtype=float)
    if K_hat.shape != K_true.shape:
        raise ValueError(f"shape mismatch: {K_hat.shape} vs {K_true.shape}")

    if center:
        K_hat = _center_kernel(K_hat)
        K_true = _center_kernel(K_true)

    num = np.linalg.norm(K_hat - K_true, ord="fro")
    den = np.linalg.norm(K_true, ord="fro") + eps
    return float(num / den)


def kernel_alignment(
    K_hat: np.ndarray,
    K_true: np.ndarray,
    *,
    center: bool = True,
    eps: float = 1e-12,
) -> float:
    """
    Kernel alignment (cosine similarity) between two Gram matrices.

    If center=True, uses double-centered kernels (classic kernel alignment).
    """
    K_hat = np.asarray(K_hat, dtype=float)
    K_true = np.asarray(K_true, dtype=float)
    if K_hat.shape != K_true.shape:
        raise ValueError(f"shape mismatch: {K_hat.shape} vs {K_true.shape}")

    if center:
        K_hat = _center_kernel(K_hat)
        K_true = _center_kernel(K_true)

    v_hat = K_hat.ravel()
    v_true = K_true.ravel()

    num = float(np.dot(v_hat, v_true))
    den = float(np.linalg.norm(v_hat) * np.linalg.norm(v_true) + eps)
    return num / den
