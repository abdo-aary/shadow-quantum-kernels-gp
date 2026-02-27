from __future__ import annotations
from typing import Sequence

import numpy as np

from src.circuits.configs import CircuitArchitecture
from src.runners.circuit_running import ExactResults

from qiskit.quantum_info import Operator

from src.runners.feature_maps.base_fmp_retriever import BaseFeatureMapsRetriever


class ExactFeatureMapsRetriever(BaseFeatureMapsRetriever):
    """
    Compute per draw exact feature maps from ExactResults.

    Given:
      - an architecture arch with num_qubits = n and B blocks,
      - ExactResults with states of shape (R, M, 2**n),

    this class produces an array `exact_fmps` of shape (R, M, m) where each entry exact_fmps[r, x] = phi_r(x),
    which is an array of shape (m,) with m being the nuber of measured observables.
    """

    def __init__(self, arch: CircuitArchitecture, observables: Sequence[Operator]):
        self.arch = arch
        self.observables = observables

    def get_feature_maps(self, results: ExactResults) -> np.ndarray:
        states = results.states

        # 1) observables as matrices, shape (K, dim, dim)
        obs_mat = np.stack([op.to_matrix() for op in self.observables], axis=0)  # (K, 16, 16)

        # 2) Apply all observables to all states:
        #    Opsi[r, m, k, i] = sum_j obs_mat[k, i, j] * states[r, m, j]
        Opsi = np.einsum('kij,rmj->rmki', obs_mat, states)  # shape (num_draws, M, K, dim)

        # 3) Contract with bra <psi| to get <psi|O|psi>:
        #    fmps[r, m, k] = sum_i conj(states[r, m, i]) * Opsi[r, m, k, i]
        exact_fmps = np.einsum('rmi,rmki->rmk', states.conj(), Opsi).real  # shape (num_draws, M, K)

        self.fmps = exact_fmps

        return exact_fmps
