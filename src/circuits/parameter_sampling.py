# parameter_sampling.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from math import pi, sqrt
import numpy as np

from qiskit import QuantumCircuit

from src.circuits.configs import CircuitArchitecture


class ParameterSamplingStrategy(ABC):
    """Base class: all mappings are Dict[str, float] keyed by parameter name."""

    @abstractmethod
    def sample_random_assignment(
        self, qc: QuantumCircuit, seed: Optional[int] = None
    ) -> Dict[str, float]:
        """Sample one set of *non-data* parameters for qc (no x[j] inside)."""
        ...

    @abstractmethod
    def sample_R_random_assignments(
        self,
        qc: QuantumCircuit,
        num_draws: int,
        seed: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Sample num_draws independent sets of non-data parameters."""
        ...

    @abstractmethod
    def set_parameters_data(
        self,
        qc: QuantumCircuit,
        non_data_parameters: Dict[str, float],
        X: np.ndarray,
    ) -> List[Dict[str, float]]:
        """
        Merge non-data parameters with per-input data parameters.

        non_data_parameters: Dict[name, value] for ξ, δ, P, Rx, ZZ, …
        X: shape (M, input_dim).

        Returns: list of length M, each element Dict[name, value] including
                 both non-data params and x[j] for that input.
        """
        ...


class UniformParameterSamplingStrategy(ParameterSamplingStrategy):
    """
    Sample all *non-data* parameters independently:

      • ξ_ℓ[j]: for each layer ℓ, sample η_ℓ ~ N(0, I), normalise it, then
        scale so ‖ξ_ℓ‖₂ = 1/√ζ (ζ = input_dim). So |ξ_ℓ·x| ≤ 1 for x∈[-1,1]^ζ.

      • P_* and all other non-data params (deltas_*, ZZ_*, Rx_*…):
        Uniform(-π, π).

      • x[*] parameters are *not* included; they’re handled by set_parameters_data.

    All mappings are Dict[str, float] keyed by parameter name.
    """

    def __init__(self, circuit_cfg: CircuitArchitecture, seed: Optional[int] = None):
        self.circuit_cfg = circuit_cfg
        self._rng = np.random.default_rng(seed)

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

        # For each layer ℓ, sample a direction and scale by 1/sqrt(zeta)
        for ell, params_j in xi_by_layer.items():
            raw = rng.normal(size=zeta)
            norm = np.linalg.norm(raw)
            if norm < 1e-12:
                # extremely unlikely; just fall back to a simple vector
                raw = np.ones(zeta)
                norm = np.linalg.norm(raw)
            dir_vec = raw / norm
            scaled = dir_vec / sqrt(zeta)  # ‖ξ_ℓ‖₂ = 1/√ζ

            for name, j in params_j:
                if j < 0 or j >= zeta:
                    raise ValueError(
                        f"xi parameter index j={j} out of bounds for input_dim={zeta}"
                    )
                assignment[name] = float(scaled[j])

        # --------- 2) Handle all other parameters (P_*, deltas_*, ZZ_*, Rx_*, etc.) ---------
        for p in qc.parameters:
            name = p.name

            # skip data parameters: user will bind x later
            if name.startswith("x["):
                continue

            # skip ξ parameters: already assigned above
            if name.startswith("xi_"):
                continue

            # Phase gates and all other non-data params: Uniform(-π, π)
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

    def set_parameters_data(
            self,
            qc: QuantumCircuit,
            non_data_parameters: dict,
            X: np.ndarray,
    ) -> List[dict]:
        """
        Merge non-data parameters and per-input data x into a list of
        name->value dictionaries.

        Parameters
        ----------
        qc:
            The parameterized circuit.
        non_data_parameters:
            Dict[name, value] for all non-data parameters (ξ, P, Rx, ZZ, δ, ...).
        X:
            Array of shape (M, input_dim) containing the input data.

        Returns
        -------
        List[dict]
            A list of length M. Entry m is a dict[name, value] containing
            both non-data parameters and data parameters x[j] = X[m, j].
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (M, input_dim), got shape {X.shape}.")

        M, d = X.shape
        if d != self.circuit_cfg.input_dim:
            raise ValueError(
                f"X.shape[1]={d} must match circuit_cfg.input_dim={self.circuit_cfg.input_dim}."
            )

        # Map each index j to the parameter name "x[j]"
        data_param_names: dict[int, str] = {}
        for p in qc.parameters:
            name = p.name
            if name.startswith("x["):
                inside = name.split("[", 1)[1].split("]", 1)[0]
                j = int(inside)
                if j in data_param_names:
                    raise ValueError(f"Duplicate data parameter for x[{j}] detected.")
                data_param_names[j] = name

        if len(data_param_names) != d:
            raise ValueError(
                f"Found {len(data_param_names)} data parameters x[*], "
                f"but input_dim={d}."
            )

        parameters_X: List[dict] = []
        for m in range(M):
            x_vec = X[m]
            params_m = dict(non_data_parameters)  # copy base non-data params
            for j, name in data_param_names.items():
                params_m[name] = float(x_vec[j])
            parameters_X.append(params_m)

        return parameters_X
