from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import numpy as np

from qiskit import QuantumCircuit


class ParameterSamplingStrategy(ABC):
    """Base class: all mappings are Dict[str, float] keyed by parameter name."""

    def __init__(self, circuit_cfg, seed):
        self.circuit_cfg = circuit_cfg
        self.seed = seed
        self._rng = np.random.default_rng(seed)

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
