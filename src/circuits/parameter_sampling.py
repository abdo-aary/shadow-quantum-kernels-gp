# parameter_sampling.py
from abc import ABC, abstractmethod
from typing import Optional, List

from math import pi
import numpy as np

from qiskit import QuantumCircuit

from src.circuits.configs import CircuitArchitecture


class ParameterSamplingStrategy(ABC):
    @abstractmethod
    def get_random_assignment(self, qc: QuantumCircuit):
        ...

    @abstractmethod
    def get_random_circuit(self, qc: QuantumCircuit, seed: Optional[int] = None) -> QuantumCircuit:
        ...

    @abstractmethod
    def get_n_random_circuits(self, qc: QuantumCircuit,
                              num_circuits: int, seed: Optional[int] = None) -> List[QuantumCircuit]:
        ...


class UniformParameterSamplingStrategy(ParameterSamplingStrategy):
    """
    Sample all circuit parameters independently, with:
      - T-gate parameters  ~  Bernoulli({0,1})
      - all other *non-data* parameters ~ Uniform(-π, π)
      - data parameters x[*] are left unbound (not in the returned dict)

    The returned mapping is {Parameter: float}.
    """

    def __init__(self, circuit_cfg: CircuitArchitecture, seed: Optional[int] = None):
        self.circuit_cfg = circuit_cfg
        self._rng = np.random.default_rng(seed)

    def _get_rng(self, seed: Optional[int] = None) -> np.random.Generator:
        # Per-call seed overrides the instance RNG
        if seed is None:
            return self._rng
        return np.random.default_rng(seed)

    def get_random_assignment(self, qc: QuantumCircuit, seed: Optional[int] = None) -> dict:
        """
        Return a dict {Parameter: value} for all circuit parameters except the
        data parameters x[*]. T-gate parameters are sampled as Bernoulli({0,1}),
        all other parameters as Uniform(-π, π).
        """
        rng = self._get_rng(seed)
        assignment = {}

        for p in qc.parameters:
            name = p.name

            # Skip data parameters: user will bind x later with actual inputs
            if name.startswith("x["):
                continue

            # T-gate parameters: Bernoulli {0,1}
            if name.startswith("T_"):
                value = float(rng.integers(0, 2))  # 0 or 1
            else:
                # Everything else: uniform over (-π, π)
                value = float(rng.uniform(-pi, pi))

            assignment[p] = value

        return assignment

    def get_random_circuit(self, qc: QuantumCircuit, seed: Optional[int] = None) -> QuantumCircuit:
        """
        Return a new circuit where all non-data parameters are bound to random
        values (according to the strategy above), but x[*] remain symbolic.
        """
        param_assignment = self.get_random_assignment(qc=qc, seed=seed)
        qc_assigned = qc.assign_parameters(param_assignment, inplace=False)
        return qc_assigned

    def get_n_random_circuits(
        self,
        qc: QuantumCircuit,
        num_circuits: int,
        seed: Optional[int] = None,
    ) -> List[QuantumCircuit]:
        """
        Return a list of `num_circuits` circuits, each with independently
        sampled non-data parameters.

        - If `seed` is provided, the whole list is reproducible:
          same seed => same sequence of circuits.
        - Different entries in the list have different parameters (almost surely).
        """
        if num_circuits <= 0:
            raise ValueError(f"num_circuits must be positive, got {num_circuits}.")

        circuits: List[QuantumCircuit] = []

        if seed is None:
            # Use the instance RNG: successive calls will advance its state
            for _ in range(num_circuits):
                circuits.append(self.get_random_circuit(qc, seed=None))
        else:
            # Derive independent per-circuit seeds from a base seed
            base_rng = np.random.default_rng(seed)
            seeds = base_rng.integers(0, 2**32 - 1, size=num_circuits)

            for s in seeds:
                circuits.append(self.get_random_circuit(qc, seed=int(s)))

        return circuits
