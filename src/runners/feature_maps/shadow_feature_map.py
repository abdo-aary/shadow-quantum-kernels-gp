from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Optional

import numpy as np

from src.circuits.configs import CircuitArchitecture
from src.runners.circuit_running import ExactResults
import pickle

from src.runners.feature_maps.base_fmp_retriever import BaseFeatureMapsRetriever
import math


def get_theoretical_shots(
        eps: float,
        delta: float,
        locality: int,
        num_data_pts: int,
        num_obs: int,
        num_draws: int,
) -> int:
    """
    Theoretical number of classical-shadow snapshots ensuring
    sup-norm error <= eps for all (draw, data point, observable)
    with probability at least 1 - delta.

    Parameters
    ----------
    eps : float
        Target uniform accuracy ε > 0.
    delta : float
        Failure probability δ in (0, 1).
    locality : int
        k in 'k-local' for the observables.
    num_data_pts : int
        M = number of data points.
    num_obs : int
        Number of observables per state.
    num_draws : int
        R = number of circuit draws / feature-map draws.

    Returns
    -------
    int
        Recommended total number of snapshots N_shots.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1).")

    # 34 / eps^2 factor from Theorem 1
    prefactor = 34.0 / (eps ** 2)

    # Upper bound on shadow norm^2 for k-local observables
    shadow_norm_sq = (3.0 / 2.0) ** locality

    # Total number of “features’’ we care about in the union bound:
    num_features = num_draws * num_data_pts * num_obs

    # N ≈ (34/eps^2) * (3/2)^k * log(num_features / delta)
    n_shots = prefactor * shadow_norm_sq * math.log(num_features / delta)

    return math.ceil(n_shots)


@dataclass
class CSFeatureMapsRetriever(BaseFeatureMapsRetriever):
    """
    Simulate feature maps obtained via (Pauli) classical shadows with
    a Median-of-Means (MoM) estimator, using ExactResults as oracle
    for the underlying pure states.

    Given:
      - an architecture `arch` with `num_qubits = n`,
      - a list of Hermitian observables (typically SparsePauliOp's)
        acting on n qubits,
      - ExactResults with states of shape (R, M, 2**n),

    this class produces an array `cs_feature_maps` of shape
    (R, M, num_observables), where each entry is a noisy estimate of
    the exact expectation value Tr[O_j ρ_{r,m}] obtained from a
    virtual classical-shadow experiment with `shots` effective
    measurements and a Median-of-Means aggregation.

    Notes
    -----
    * For speed, we do NOT explicitly simulate each projective
      measurement. Instead:

        1. We compute the true expectations μ_{r,m,j} exactly via
           vectorized linear algebra.

        2. We model the estimator as a MoM estimator built from
           ±1-valued measurements with mean μ_{r,m,j} and variance
           (1 - μ_{r,m,j}^2). Using a CLT approximation, each batch
           mean is sampled from N(μ, (1 - μ^2)/batch_size), and we
           take the median across batches.

      This keeps the correct 1 / √(shots) scaling while avoiding a
      huge inner “per-shot” loop.

    * Observables are assumed to have eigenvalues in [-1, 1]
      (Pauli strings, 2-local correlators, etc.). We clip numerical
      drift so that |μ| ≤ 1.
    """

    arch: CircuitArchitecture
    observables: Sequence[Any]
    cs_feature_maps: Optional[np.ndarray] = None  # last computed tensor, if any

    # ------------------------------------------------------------------
    # initialization helpers
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._n_qubits = self.arch.num_qubits
        self._dim = 1 << self._n_qubits

        if not self.observables:
            raise ValueError("observables must be a non-empty sequence.")

        obs_mats = []
        for idx, obs in enumerate(self.observables):
            # Support Qiskit SparsePauliOp/Operator, or plain numpy arrays
            if hasattr(obs, "to_matrix"):
                mat = obs.to_matrix()  # type: ignore[assignment]
            else:
                mat = np.asarray(obs)

            mat = np.asarray(mat, dtype=complex)

            if mat.shape != (self._dim, self._dim):
                raise ValueError(
                    f"Observable at index {idx} has shape {mat.shape}, "
                    f"expected {(self._dim, self._dim)} for "
                    f"{self._n_qubits} qubits."
                )

            # Safety: require Hermitian (up to numerical tolerance)
            if not np.allclose(mat, mat.conj().T):
                raise ValueError(
                    f"Observable at index {idx} is not Hermitian."
                )

            obs_mats.append(mat)

        # Stack into one tensor of shape (num_obs, dim, dim)
        self._obs_mats = np.stack(obs_mats, axis=0)
        self._num_observables = self._obs_mats.shape[0]

    # ------------------------------------------------------------------
    # core helpers
    # ------------------------------------------------------------------
    def _compute_true_expectations(self, results: ExactResults) -> np.ndarray:
        """
        Compute the exact expectations μ_{r,m,j} = Tr(O_j ρ_{r,m}) for all
        draws r, inputs m and observables j.

        Returns
        -------
        mu : np.ndarray
            Array of shape (R, M, num_observables).
        """
        if results.arch != self.arch:
            raise ValueError(
                "Architecture mismatch between CSFeatureMapsRetriever "
                "and ExactResults."
            )

        states = results.states
        if states.ndim != 3:
            raise ValueError(
                f"Expected states with shape (R, M, 2**n), "
                f"got {states.shape}."
            )

        R, M, dim = states.shape
        if dim != self._dim:
            raise ValueError(
                f"State dimension mismatch: got {dim}, "
                f"expected {self._dim}."
            )

        # Flatten (R, M, dim) -> (N_states, dim)
        psi = states.reshape(-1, dim)  # (N, dim)
        psi_conj = psi.conj()  # (N, dim)
        # obs_mats: (O, dim, dim)

        # μ_{i,o} = Σ_{a,b} ψ*_i,a O_{o,a,b} ψ_i,b
        mu_flat = np.einsum(
            "ia,oab,ib->io",
            psi_conj,
            self._obs_mats,
            psi,
            optimize=True,
        ).real  # (N, num_obs)

        mu = mu_flat.reshape(R, M, self._num_observables)
        # Guard against tiny numerical overshoots
        np.clip(mu, -1.0, 1.0, out=mu)
        return mu

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_feature_maps(
            self,
            results: ExactResults,
            *,
            shots: Optional[int] = None,
            seed: Optional[int] = None,
            n_groups: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate classical-shadow-based feature maps via a Median-of-Means
        estimator.

        Parameters
        ----------
        results : ExactResults
            Must contain states of shape (R, M, 2**n) and an architecture
            identical to self.arch.
        shots : int
            Total number of effective measurement shots N used for the
            estimator. Larger values => smaller noise.
        seed : int or None, optional
            Seed for the underlying RNG (for reproducibility).
        n_groups : int or None, optional
            Number of MoM groups K. If None, we use a heuristic
                K ≈ min(16, sqrt(shots)), but always 1 ≤ K ≤ shots.

        Returns
        -------
        cs_feature_maps : np.ndarray
            Array of shape (R, M, num_observables) containing noisy
            estimates of the exact expectations.
        """
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}.")

        mu = self._compute_true_expectations(results)  # (R, M, O)

        # Choose number of groups for MoM
        if n_groups is None:
            # heuristic: about sqrt(shots) groups, capped at 16
            n_groups = max(1, min(16, int(np.sqrt(shots))))
        n_groups = min(n_groups, shots)
        batch_size = max(shots // n_groups, 1)

        # For ±1-valued measurements with mean μ, each batch mean
        # has variance (1 - μ^2) / batch_size.
        var_group = (1.0 - mu ** 2) / batch_size
        var_group = np.maximum(var_group, 0.0)

        rng = np.random.default_rng(seed)

        # Sample K batch means per (r, m, j) using a Gaussian
        # approximation for the CLT:
        #   m_k ~ N(μ, var_group)
        # noise shape: (R, M, O, K)
        noise = rng.normal(
            loc=0.0,
            scale=np.sqrt(var_group)[..., None],
            size=mu.shape + (n_groups,),
        )
        group_means = mu[..., None] + noise  # (R, M, O, K)

        # Median-of-Means estimator
        cs_est = np.median(group_means, axis=-1)  # (R, M, O)

        self.cs_feature_maps = cs_est
        return cs_est.astype(float)

    # ------------------------------------------------------------------
    # persistence (same flavor as ExactGramsRetriever)
    # ------------------------------------------------------------------
    def save(self, file: str | Path) -> None:
        """
        Serialize CSFeatureMapsRetriever to disk.

        We persist:
          - the architecture (arch)
          - the observables
          - the last computed feature maps (cs_feature_maps), if any

        Derived internal structures (_obs_mats, etc.) are recomputed
        on load via __post_init__.
        """
        path = Path(file)
        payload = {
            "cls": "CSFeatureMapsRetriever",
            "arch": self.arch,
            "observables": self.observables,
            "cs_feature_maps": self.cs_feature_maps,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file: str | Path) -> "CSFeatureMapsRetriever":
        """
        Load CSFeatureMapsRetriever from a file created by `save`.
        """
        path = Path(file)
        with path.open("rb") as f:
            obj = pickle.load(f)

        # Backward compatibility: whole-object pickle
        if isinstance(obj, CSFeatureMapsRetriever):
            return obj

        # Preferred dict payload
        if isinstance(obj, dict) and obj.get("cls") == "CSFeatureMapsRetriever":
            arch = obj["arch"]
            observables = obj["observables"]
            cs_feature_maps = obj.get("cs_feature_maps", None)

            retr = CSFeatureMapsRetriever(
                arch=arch,
                observables=observables,
            )
            retr.cs_feature_maps = cs_feature_maps
            return retr

        raise TypeError(
            f"File {file} does not contain a valid CSFeatureMapsRetriever payload."
        )
