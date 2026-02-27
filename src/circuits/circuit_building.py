from math import pi, sqrt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src.circuits.configs import CircuitArchitecture


class CircuitFactory:
    """Factory for building SQP circuits as in the paper."""

    @staticmethod
    def createSQPcircuit(cfg: CircuitArchitecture, norm_jitter: float = 1e-6) -> QuantumCircuit:
        """
        Build an SQP circuit implementing

            U(x, ϑ) = W_L G_L(x) ... W_1 G_1(x),

        with layer structure.

        Parameters
        ----------
        cfg : CircuitArchitecture
            Architecture specification (num_qubits, layers, input_dim).
        norm_jitter: float
            To avoid division per zero errors

        Returns
        -------
        QuantumCircuit
            A parametrized Qiskit circuit with:
            - data parameters x[0:input_dim]
            - generator parameters xi_ℓ[j], deltas_{ℓ}[q]
            - evolution parameters alpha, beta and omega
        """
        # Input data parameters x ∈ R^{ζ}
        x = ParameterVector("x", cfg.input_dim)

        qc = QuantumCircuit(cfg.num_qubits, name="SQP")
        for q in qc.qubits:
            qc.h(q)

        # Loop over layers ℓ = 0, ..., L-1
        for ell, layer in enumerate(cfg.layers):
            # -------------------------------
            # 1) Generator G_ℓ(x)
            # -------------------------------
            xi_ell = ParameterVector(f"xi_{ell}", cfg.input_dim)
            delta_ell = ParameterVector(f"delta_{ell}", len(layer.qubits))

            norm_x_sq = sum(x[j] ** 2 for j in range(cfg.input_dim))
            # norm_x = norm_x_sq ** 0.5 + norm_jitter

            gamma_ell = sum(xi_ell[j] * x[j] for j in range(cfg.input_dim))     # / norm_x

            for i, q in enumerate(layer.qubits):
                injected_angle_q = delta_ell[i] * gamma_ell
                qc.rz(injected_angle_q, q)

            # -------------------------------
            # 1) Unitary W_ℓ(x)
            # -------------------------------
            alpha_ell = ParameterVector(f"alpha_{ell}", len(layer.qubits))
            # a) R_x rotations
            for i, q in enumerate(layer.qubits):
                injected_angle_q = alpha_ell[i]
                qc.rx(injected_angle_q, q)

            # b) ZZ couplings:
            # We use RZZ(α), which realizes exp(-i α/2 Z⊗Z);
            beta_ell = ParameterVector(
                f"beta_{ell}", length=len(layer.edges)
            )
            for e_idx, (q1, q2) in enumerate(layer.edges):
                qc.rzz(beta_ell[e_idx], q1, q2)

            # a) R_y rotations
            omega_ell = ParameterVector(f"omega_{ell}", len(layer.qubits))
            for i, q in enumerate(layer.qubits):
                injected_angle_q = omega_ell[i]
                qc.ry(injected_angle_q, q)

        return qc
