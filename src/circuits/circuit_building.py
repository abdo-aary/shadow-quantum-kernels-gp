from math import pi
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src.circuits.configs import CircuitArchitecture


class CircuitFactory:
    """Factory for building SQP circuits as in the paper."""

    @staticmethod
    def createSQPcircuit(cfg: CircuitArchitecture) -> QuantumCircuit:
        """
        Build an SQP circuit implementing

            U(x, ϑ) = W^{(L)} G_L(x) ... W^{(1)} G_1(x),

        with per-block structure W^{(b,ℓ)} and G_{b,ℓ}(x) as in eqs. (5) and (6).

        Parameters
        ----------
        cfg : CircuitArchitecture
            Architecture specification (num_qubits, blocks, input_dim, num_layers).

        Returns
        -------
        QuantumCircuit
            A parametrized Qiskit circuit with:
            - data parameters x[0:input_dim]
            - generator parameters xi_ℓ[j], deltas_{b,ℓ}[q]
            - evolution parameters T_{b,ℓ}[q], ZZ_{b,ℓ}[edge], Rx_{b,ℓ}[q]
        """
        # Input data parameters x ∈ R^{ζ}
        x = ParameterVector("x", cfg.input_dim)

        qc = QuantumCircuit(cfg.num_qubits, name="SQP")

        # Loop over layers ℓ = 0, ..., L-1
        for ell in range(cfg.num_layers):
            # ξ_ℓ ∈ R^{ζ}; same for all blocks at layer ℓ
            xi_ell = ParameterVector(f"xi_{ell}", cfg.input_dim)

            # Scalar projection x_ell^eff = ξ_ℓ · x (ParameterExpression)
            x_eff_ell = sum(xi_ell[j] * x[j] for j in range(cfg.input_dim))

            # Loop over blocks b
            for b, block in enumerate(cfg.blocks):
                qubits = list(block.qubits)
                edges = list(block.edges)

                # -------------------------------
                # 1) Generator G_{b,ℓ}(x)  (eq. (6) with x ↦ ξ_ℓ·x)
                # -------------------------------
                # δ^{(q)}_{b,ℓ} for q ∈ Q_b
                deltas_b_ell = ParameterVector(
                    f"deltas_{b}_{ell}", length=len(qubits)
                )

                for idx, q in enumerate(qubits):
                    # G^{(q)}_{b,ℓ}(x) ~ Rz( (ξ_ℓ·x) δ^{(q)}_{b,ℓ} )
                    angle = x_eff_ell * deltas_b_ell[idx]
                    qc.rz(angle, q)

                # -------------------------------
                # 2) Unitary W^{(b,ℓ)} (eq. (5))
                # -------------------------------

                # a) Local non-diagonal rotations Rx(γ^{(q)}_{b,ℓ})
                Rx_params_b_ell = ParameterVector(
                    f"Rx_{b}_{ell}", length=len(qubits)
                )
                for idx, q in enumerate(qubits):
                    qc.rx(Rx_params_b_ell[idx], q)

                # b) ZZ couplings: exp(i α_{b,ℓ}^{(q,q')} Z_q Z_{q'})
                # We use RZZ(2 * α), which realizes exp(-i α Z⊗Z);
                # this is equivalent up to α ↦ -α reparameterization.
                ZZ_params_b_ell = ParameterVector(
                    f"ZZ_{b}_{ell}", length=len(edges)
                )
                for e_idx, (q1, q2) in enumerate(edges):
                    qc.rzz(2 * ZZ_params_b_ell[e_idx], q1, q2)

                # c) Non-Clifford T-doping: T^{t^{(q)}_{b,ℓ}}_q
                # Implemented as a phase gate P(π/4 * t), where t ∈ {0, 1} in practice.
                T_params_b_ell = ParameterVector(
                    f"T_{b}_{ell}", length=len(qubits)
                )
                for idx, q in enumerate(qubits):
                    qc.p(pi / 4 * T_params_b_ell[idx], q)


            # At this point, for layer ℓ, we have applied:
            #   [⊗_b W^{(b,ℓ)}] [⊗_b G_{b,ℓ}(x)]
            # in time order G_{1}, W^{(1)}, G_{2}, W^{(2)}, ...
            # matching U = W^{(L)} G_L ... W^{(1)} G_1 with ℓ increasing.

        return qc
