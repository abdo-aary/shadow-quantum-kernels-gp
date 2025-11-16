from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.circuits.configs import CircuitArchitecture
from src.runners.circuit_running import ExactResults


@dataclass
class ExactGramsRetriever:
    """
    Compute per-(r, b) block-local Gram matrices from ExactResults.

    Given:
      - an architecture arch with num_qubits = n and B blocks,
      - ExactResults with states of shape (R, M, 2**n),

    this class produces an array `exact_grams` of shape (R, B, M, M) where

        exact_grams[r, b, i, j] = Tr( ρ_{r,b}(x_i) ρ_{r,b}(x_j) ),

    and ρ_{r,b}(x_i) is the reduced density matrix on block b obtained
    from the n-qubit pure state |ψ_{r}(x_i)⟩.
    """

    arch: CircuitArchitecture

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
    def get_exact_grams(self, results: ExactResults) -> np.ndarray:
        """
        Compute all (R, B) exact Gram matrices from ExactResults.

        Parameters
        ----------
        results : ExactResults
            Must contain states of shape (R, M, 2**n) and an architecture
            identical to self.arch.

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
        n = self._n_qubits

        # (R, B, M, M) – kernel values are real; we drop small imaginary parts
        exact_grams = np.empty((R, B, M, M), dtype=float)

        # --- main computation ---
        for r in range(R):
            # states_r: (M, 2**n)
            states_r = states[r]

            for b_idx, blk_info in enumerate(self._block_infos):
                axes_perm: Tuple[int, ...] = blk_info["axes_perm"]
                dim_block: int = blk_info["dim_block"]
                dim_env: int = blk_info["dim_env"]

                # Collect reduced density matrices for all M states:
                #   rhos[m] has shape (dim_block, dim_block)
                # We'll store them flattened to (M, dim_block**2).
                rho_flat = np.empty((M, dim_block * dim_block), dtype=complex)

                for m in range(M):
                    psi = states_r[m]  # (2**n,)

                    # 1. reshape to tensor (2,)*n
                    psi_tensor = psi.reshape(self._tensor_shape)

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
                exact_grams[r, b_idx, :, :] = gram_rb.real  # drop tiny imaginary noise

        return exact_grams
