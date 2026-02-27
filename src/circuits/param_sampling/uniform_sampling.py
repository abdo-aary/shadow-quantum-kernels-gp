from typing import Optional, List, Dict

from math import pi
import numpy as np

from qiskit import QuantumCircuit

from src.circuits.param_sampling.base_sampling import ParameterSamplingStrategy


class UniformParameterSamplingStrategy(ParameterSamplingStrategy):
    """
    Sample all *non-data* parameters independently:

      • ξ_ℓ[j]: for each layer ℓ, we have to have ‖ξ_ℓ‖₂ < π.

      • deltas must be less than 1, i.e. between [-1, 1].

      • alpha_*, beta_*, and omega_*: Uniform(-π, π).

      • x[*] parameters are *not* included; they’re handled by set_parameters_data.

    All mappings are Dict[str, float] keyed by parameter name.
    """

    def _sample_rng(self, seed: Optional[int] = None) -> np.random.Generator:
        if seed is None:
            return self._rng
        return np.random.default_rng(seed)

    def sample_random_assignment(
            self, qc: QuantumCircuit, seed: Optional[int] = None
    ) -> dict:
        """
        Return a dict {name: value} for all circuit parameters except the
        data parameters x[*].

        - For each layer ℓ, the ξ_ℓ parameters ("xi_ℓ[j]") are drawn as a random
          direction in ℝ^{input_dim}, rescaled so that ‖ξ_ℓ‖₂ = 1/√input_dim.

        - Phase-gate parameters P_* and all other non-data parameters are
          sampled independently from Uniform(-π, π).

        - Data parameters x[*] are skipped (the user binds them later with
          actual inputs x ∈ [-1, 1]^ζ).
        """
        rng = self._sample_rng(seed)
        assignment: dict[str, float] = {}

        zeta = self.circuit_cfg.input_dim

        # --------- 1) Handle ξ parameters layer-wise (direction + scaling) ---------
        # Group xi_* parameters by layer index ℓ, based on names "xi_<ell>[j]".
        xi_by_layer: dict[int, list[tuple[str, int]]] = {}

        for p in qc.parameters:
            name = p.name
            if name.startswith("xi_"):
                # name pattern: "xi_<ell>[j]"
                prefix, rest = name.split("[", 1)  # "xi_<ell>", "j]"
                ell_str = prefix.split("_", 1)[1]  # part after "xi_"
                ell = int(ell_str)
                j_str = rest.split("]", 1)[0]
                j = int(j_str)
                xi_by_layer.setdefault(ell, []).append((name, j))

        # For each layer ℓ, sample ξ_ℓ uniformly in the ℓ2-ball of radius π
        for ell, params_j in xi_by_layer.items():
            raw = rng.normal(size=zeta)
            norm = np.linalg.norm(raw)
            if norm < 1e-12:
                raw = np.ones(zeta)
                norm = np.linalg.norm(raw)

            # Direction on the unit sphere
            unit_dir = raw / norm

            # Radius with correct distribution for uniform-in-ball
            u = rng.random()  # in [0, 1)
            radius = pi * (u ** (1.0 / zeta))  # => ‖ξ_ℓ‖₂ < π almost surely

            xi_vec = radius * unit_dir

            for name, j in params_j:
                if j < 0 or j >= zeta:
                    raise ValueError(
                        f"xi parameter index j={j} out of bounds for input_dim={zeta}"
                    )
                assignment[name] = float(xi_vec[j])

        # --------- 2) Handle all other parameters (deltas_*, alpha_*, beta_*, omega_*.) ---------
        for p in qc.parameters:
            name = p.name

            # skip data parameters: user will bind x later
            if name.startswith("x["):
                continue

            # skip ξ parameters: already assigned above
            if name.startswith("xi_"):
                continue

            # deltas in [-1, 1]
            if name.startswith("delta_"):
                assignment[name] = float(rng.uniform(-1.0, 1.0))
            # all other non-data params (alpha_*, beta_*, omega_*, ...) in [-π, π]
            else:
                assignment[name] = float(rng.uniform(-pi, pi))

        return assignment

    def sample_R_random_assignments(
        self,
        qc: QuantumCircuit,
        num_draws: int,
        seed: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        if num_draws <= 0:
            raise ValueError(f"num_draws must be positive, got {num_draws}.")

        assignments: List[Dict[str, float]] = []

        if seed is None:
            for _ in range(num_draws):
                assignments.append(self.sample_random_assignment(qc, seed=None))
        else:
            base_rng = np.random.default_rng(seed)
            seeds = base_rng.integers(0, 2**32 - 1, size=num_draws)
            for s in seeds:
                assignments.append(self.sample_random_assignment(qc, seed=int(s)))

        return assignments
