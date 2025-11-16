from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, List, Literal, Optional

import numpy as np


KernelType = Literal["rbf", "matern32", "nonstat_amp", "nonstat_ls"]


@dataclass
class KernelConfig:
    """Configuration for a covariance kernel used in synthetic GP data.

    Attributes
    ----------
    name:
        Arbitrary name, used only for identification / logging.
    kernel_type:
        One of {"rbf", "matern32", "nonstat_amp", "nonstat_ls"}.
        - "rbf":       stationary squared–exponential (RBF) kernel
        - "matern32":  stationary Matérn-3/2 kernel
        - "nonstat_amp":
              non-stationary kernel with input-dependent amplitude s(x),
              i.e. k(x,x') = s(x) s(x') k_base(x,x').
        - "nonstat_ls":
              non-stationary kernel with input-dependent lengthscale along
              the first input dimension, using the Paciorek–Schervish style
              construction in 1D, optionally multiplied by a stationary
              RBF over the remaining dimensions.
    params:
        Hyperparameters specific to the kernel_type. See _kernel_matrix()
        docstring for expected keys.
    """
    name: str
    kernel_type: KernelType
    params: Dict[str, Any]


def _pairwise_sqdist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances between rows of X and Y.

    Parameters
    ----------
    X : (N, d) array
    Y : (M, d) array

    Returns
    -------
    D : (N, M) array
        D[i,j] = ||X[i] - Y[j]||^2.
    """
    X2 = np.sum(X ** 2, axis=1, keepdims=True)
    Y2 = np.sum(Y ** 2, axis=1, keepdims=True).T
    cross = X @ Y.T
    return np.clip(X2 + Y2 - 2.0 * cross, a_min=0.0, a_max=None)


def _kernel_matrix(X: np.ndarray, cfg: KernelConfig) -> np.ndarray:
    """Compute the Gram matrix K for inputs X under a given kernel config.

    All kernels are normalised so that K[i,i] = 1 for all i, and values
    lie in [0, 1].

    Expected hyperparameters in cfg.params:

    For kernel_type == "rbf":
        - lengthscale: float > 0
          k(x,x') = exp(-0.5 * ||x - x'||^2 / lengthscale^2)

    For kernel_type == "matern32":
        - lengthscale: float > 0
          r = ||x - x'|| / lengthscale
          k(x,x') = (1 + sqrt(3) r) * exp(-sqrt(3) r)

    For kernel_type == "nonstat_amp":
        - base_lengthscale: float > 0
        - a: float in [0,1]
        - b: float in [0,1] with a + b <= 1
        - c: float > 0        (slope for the sigmoid)
          We define s(t) = a + b * sigmoid(c t) where t is the first
          input dimension, and
              k(x,x') = s(t) s(t') * exp(-0.5 * ||x - x'||^2 / base_lengthscale^2)

    For kernel_type == "nonstat_ls":
        - ell_min: float > 0
        - ell_max: float > 0 with ell_max >= ell_min
        - c: float > 0        (slope for the sigmoid in ℓ(t))
        - ell_rest: float > 0 (lengthscale for remaining dims, optional;
                               if omitted, defaults to ell_min)
          For t, t' the first coordinate, define
              ℓ(t) = ell_min + (ell_max - ell_min) * sigmoid(c t)
          and
              k_1d(t,t') = sqrt( 2 ℓ(t) ℓ(t') / (ℓ(t)^2 + ℓ(t')^2) )
                          * exp( -(t - t')^2 / (ℓ(t)^2 + ℓ(t')^2) ).
          For d > 1 we multiply by a stationary RBF kernel over the
          remaining coordinates with lengthscale ell_rest.
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    p = cfg.params
    ktype = cfg.kernel_type

    if ktype == "rbf":
        ell = float(p.get("lengthscale", 0.2))
        D2 = _pairwise_sqdist(X, X)
        K = np.exp(-0.5 * D2 / (ell ** 2))

    elif ktype == "matern32":
        ell = float(p.get("lengthscale", 0.2))
        D2 = _pairwise_sqdist(X, X)
        R = np.sqrt(D2 + 1e-12) / ell
        sqrt3 = np.sqrt(3.0)
        K = (1.0 + sqrt3 * R) * np.exp(-sqrt3 * R)

    elif ktype == "nonstat_amp":
        ell = float(p.get("base_lengthscale", 0.3))
        a = float(p.get("a", 0.3))
        b = float(p.get("b", 0.7))
        c = float(p.get("c", 2.0))

        if a < 0.0 or b < 0.0 or a + b > 1.0:
            raise ValueError(
                f"nonstat_amp parameters must satisfy a >= 0, b >= 0, a + b <= 1; got a={a}, b={b}."
            )

        D2 = _pairwise_sqdist(X, X)
        K_base = np.exp(-0.5 * D2 / (ell ** 2))

        t = X[:, 0:1]  # (N,1)
        s = a + b / (1.0 + np.exp(-c * t))        # (N,1), in (a, a+b] ⊂ (0,1]
        amp = s @ s.T                             # (N,N)

        K = amp * K_base

    elif ktype == "nonstat_ls":
        ell_min = float(p.get("ell_min", 0.1))
        ell_max = float(p.get("ell_max", 0.8))
        c = float(p.get("c", 2.0))
        ell_rest = float(p.get("ell_rest", ell_min))

        if ell_max < ell_min:
            raise ValueError(
                f"nonstat_ls requires ell_max >= ell_min; got ell_min={ell_min}, ell_max={ell_max}."
            )

        t = X[:, 0:1]  # (N,1)
        ell_t = ell_min + (ell_max - ell_min) / (1.0 + np.exp(-c * t))  # (N,1)

        t1 = t
        t2 = t.T
        ell_sq_1 = ell_t ** 2
        ell_sq_2 = (ell_t ** 2).T
        denom = ell_sq_1 + ell_sq_2  # (N,N)

        D2_t = (t1 - t2) ** 2  # (N,N)

        num = 2.0 * np.sqrt(ell_sq_1 * ell_sq_2)
        prefac = np.sqrt(num / (denom + 1e-12))
        K_1d = prefac * np.exp(-D2_t / (denom + 1e-12))

        if d > 1:
            X_rest = X[:, 1:]
            D2_rest = _pairwise_sqdist(X_rest, X_rest)
            K_rest = np.exp(-0.5 * D2_rest / (ell_rest ** 2))
            K = K_1d * K_rest
        else:
            K = K_1d

    else:
        raise ValueError(f"Unknown kernel_type: {ktype}")

    # Numerical safety: clip to [0,1], enforce symmetry & diagonal 1
    K = np.clip(K, 0.0, 1.0)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)
    return K


def make_dataset(
    M: int,
    input_dim: int,
    sampling: str = "uniform",
    kernel_cfgs: Sequence[KernelConfig] = (),
    seed: Optional[int] = None,
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
    kernel_cfgs:
        Sequence of KernelConfig describing the covariance kernels.
        For each kernel k_i we generate a label vector y_i ∈ ℝ^M by
        sampling from a zero-mean Gaussian process with covariance matrix
        K_i(X, X) defined by that kernel.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    data:
        A dict with keys:
          - "inputs": np.ndarray of shape (M, input_dim)
          - "labels": List[np.ndarray] where each element has shape (M,)
                      and corresponds to one kernel in kernel_cfgs.
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

    labels: List[np.ndarray] = []

    if not kernel_cfgs:
        return {"inputs": X, "labels": labels}

    jitter = 1e-6
    mean = np.zeros(M, dtype=float)

    for cfg in kernel_cfgs:
        K = _kernel_matrix(X, cfg)
        K_jitter = K + jitter * np.eye(M)
        y = rng.multivariate_normal(mean=mean, cov=K_jitter)
        labels.append(y)

    return {"inputs": X, "labels": labels}
