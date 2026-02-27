from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import os

import numpy as np

from src.circuits.configs import CircuitArchitecture
from src.runners.circuit_running import ExactResults
from concurrent.futures import ProcessPoolExecutor


def _compute_exact_grams_for_r(args) -> np.ndarray:
    """Worker: compute all block Gram matrices for a single draw r.

    Parameters
    ----------
    args : tuple
        (states_r, block_infos, tensor_shape) where
          - states_r: np.ndarray, shape (M, 2**n)
          - block_infos: List[dict] with keys
                "axes_perm", "dim_block", "dim_env"
          - tensor_shape: tuple, e.g. (2,)*n

    Returns
    -------
    grams_r : np.ndarray
        Array of shape (B, M, M) with Gram matrices for this r.
    """
    states_r, block_infos, tensor_shape = args
    M, dim = states_r.shape
    B = len(block_infos)

    grams_r = np.empty((B, M, M), dtype=float)

    for b_idx, blk_info in enumerate(block_infos):
        axes_perm = blk_info["axes_perm"]
        dim_block = blk_info["dim_block"]
        dim_env = blk_info["dim_env"]

        # Collect reduced density matrices for all M states:
        rho_flat = np.empty((M, dim_block * dim_block), dtype=complex)

        for m in range(M):
            psi = states_r[m]  # (2**n,)

            # 1. reshape to tensor (2,)*n
            psi_tensor = psi.reshape(tensor_shape)

            # 2. permute axes so that kept block qubits come first
            psi_perm = np.transpose(psi_tensor, axes_perm)

            # 3. view as matrix (dim_block, dim_env)
            psi_mat = psi_perm.reshape(dim_block, dim_env)

            # 4. reduced density: ρ_b = psi_mat @ psi_mat†
            rho_b = psi_mat @ psi_mat.conj().T

            # 5. flatten for bulk Gram computation later
            rho_flat[m, :] = rho_b.reshape(-1)

        # Hilbert–Schmidt inner products:
        #   K[i,j] = Tr(ρ_i ρ_j) = ⟨ρ_i, ρ_j⟩_{HS}
        gram_rb = rho_flat.conj() @ rho_flat.T  # (M, M) complex Hermitian
        grams_r[b_idx, :, :] = gram_rb.real

    return grams_r


@dataclass
class ExactFeatureMapsRetriever:
    """
    Compute per draw Gram matrices from ExactResults.

    Given:
      - an architecture arch with num_qubits = n and B blocks,
      - ExactResults with states of shape (R, M, 2**n),

    this class produces an array `exact_grams` of shape (R, M, M) where

        exact_grams[r, i, j] = ,

    and ρ_{r,b}(x_i) is the reduced density matrix on block b obtained
    from the n-qubit pure state |ψ_{r}(x_i)⟩.
    """

    arch: CircuitArchitecture
    exact_grams: np.ndarray

    def __post_init__(self) -> None:
        self._n_qubits = self.arch.num_qubits
        self._blocks = list(self.arch.blocks)
        self._B = len(self._blocks)

        # Precompute axis permutations and dimensions for each block
        # respecting Qiskit's little-endian convention:
        #   - basis index ordering is |q_{n-1} ... q_1 q_0>
        #   - when reshaping to (2,)*n, axis 0 ↔ qubit n-1, ..., axis n-1 ↔ qubit 0
        n = self._n_qubits

        def qubit_to_axis(q: int) -> int:
            # map Qiskit qubit index -> axis index of psi.reshape((2,)*n)
            return n - 1 - q

        all_qubits = set(range(n))
        block_infos: List[dict] = []

        for block in self._blocks:
            block_qubits = tuple(sorted(block.qubits))
            k = len(block_qubits)

            if k == 0:
                raise ValueError("Block with empty qubit set is not allowed.")

            env_qubits = tuple(sorted(all_qubits - set(block_qubits)))

            keep_axes = tuple(qubit_to_axis(q) for q in block_qubits)
            env_axes = tuple(qubit_to_axis(q) for q in env_qubits)
            axes_perm = keep_axes + env_axes

            dim_block = 1 << k
            dim_env = 1 << (n - k)

            block_infos.append(
                dict(
                    block_qubits=block_qubits,
                    axes_perm=axes_perm,
                    dim_block=dim_block,
                    dim_env=dim_env,
                )
            )

        self._block_infos = block_infos
        self._tensor_shape = (2,) * self._n_qubits

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_exact_grams(self, results: ExactResults, n_jobs: int | None = None) -> np.ndarray:
        """
        Compute all (R, B) exact Gram matrices from ExactResults.

        Parameters
        ----------
        results : ExactResults
            Must contain states of shape (R, M, 2**n) and an architecture
            identical to self.arch.
        n_jobs : int or None, optional
            Number of worker processes to use across draws r.
            - If None (default): use min(os.cpu_count(), R).
            - If 1: run sequentially (original behavior).

        Returns
        -------
        exact_grams : np.ndarray
            Array of shape (R, B, M, M) with
                exact_grams[r, b]  =  Gram matrix for draw r, block b.
        """
        # --- sanity checks ---
        if results.arch != self.arch:
            raise ValueError(
                "Architecture mismatch between ExactGramsRetriever and ExactResults."
            )

        states = results.states
        if states.ndim != 3:
            raise ValueError(
                f"Expected states with 3 dimensions (R, M, 2**n), got shape {states.shape}."
            )

        R, M, dim = states.shape
        expected_dim = 1 << self._n_qubits
        if dim != expected_dim:
            raise ValueError(
                f"Statevector dimension mismatch: got {dim}, expected {expected_dim} "
                f"for n={self._n_qubits} qubits."
            )

        B = self._B

        # (R, B, M, M) – kernel values are real; we drop small imaginary parts
        exact_grams = np.empty((R, B, M, M), dtype=float)

        # Decide on number of workers
        if n_jobs is None:
            n_jobs = os.cpu_count() or 1
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {n_jobs}.")
        n_jobs = min(n_jobs, R)

        # If only one job or a single draw, fall back to sequential behavior
        # If only one job or a single draw, fall back to sequential behavior
        if n_jobs == 1 or R == 1:
            for r in range(R):
                states_r = states[r]  # (M, 2**n)
                for b_idx, blk_info in enumerate(self._block_infos):
                    axes_perm: Tuple[int, ...] = blk_info["axes_perm"]
                    dim_block: int = blk_info["dim_block"]
                    dim_env: int = blk_info["dim_env"]

                    rho_flat = np.empty((M, dim_block * dim_block), dtype=complex)

                    for m in range(M):
                        psi = states_r[m]

                        psi_tensor = psi.reshape(self._tensor_shape)
                        psi_perm = np.transpose(psi_tensor, axes_perm)
                        psi_mat = psi_perm.reshape(dim_block, dim_env)
                        rho_b = psi_mat @ psi_mat.conj().T
                        rho_flat[m, :] = rho_b.reshape(-1)

                    gram_rb = rho_flat.conj() @ rho_flat.T
                    exact_grams[r, b_idx, :, :] = gram_rb.real

            # store on the instance
            self.exact_grams = exact_grams
            return exact_grams

        # --- parallel path across r ---
        worker_args = [
            (states[r], self._block_infos, self._tensor_shape)
            for r in range(R)
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for r, grams_r in enumerate(
                    executor.map(_compute_exact_grams_for_r, worker_args)
            ):
                # grams_r has shape (B, M, M)
                exact_grams[r, :, :, :] = grams_r

        self.exact_grams = exact_grams
        return exact_grams

    def save(self, file: str | Path) -> None:
        """Serialize ExactGramsRetriever to disk.

        We persist:
          - the architecture (arch)
          - the last computed Gram tensor (exact_grams), if any

        Internal derived structures (_block_infos, _tensor_shape, etc.)
        are recomputed on load via __post_init__.
        """
        path = Path(file)
        payload = {
            "cls": "ExactGramsRetriever",
            "arch": self.arch,
            "exact_grams": self.exact_grams,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file: str | Path) -> "ExactGramsRetriever":
        """Load ExactGramsRetriever from a file created by `save`."""
        path = Path(file)
        with path.open("rb") as f:
            obj = pickle.load(f)

        # Backward compatibility: if someone pickled the whole object
        if isinstance(obj, ExactGramsRetriever):
            return obj

        # Preferred path: dict payload with arch and possibly exact_grams
        if isinstance(obj, dict) and obj.get("cls") == "ExactGramsRetriever":
            arch = obj["arch"]
            exact_grams = obj["exact_grams"]
            retr = ExactGramsRetriever(arch=arch, exact_grams=exact_grams)
            # may be None if no grams were computed yet at save time
            retr.exact_grams = obj.get("exact_grams", None)
            return retr

        raise TypeError(
            f"File {file} does not contain a valid ExactGramsRetriever payload."
        )
