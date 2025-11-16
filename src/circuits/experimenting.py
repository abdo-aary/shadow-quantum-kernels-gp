from typing import Optional, List, Tuple
import numpy as np
from qiskit import QuantumCircuit

from src.circuits.configs import CircuitArchitecture
from src.circuits.circuit_building import CircuitFactory
from src.circuits.parameter_sampling import ParameterSamplingStrategy


class Experiment:
    """
    An experiment is defined for a configuration of (dataset, circuit_config, samping_strategy). The idea is that this
    will be translated operationally to a set of R pubs [pub_1,...,pub_R] to be given to be run by a CircuitRunner.
    Each pub_r is conceptually M circuits to be run where the non-data parameters (param_r) are fixed. I.e.,
    pub_r := [U(x_m, θ_r) : m in [M]. In qiskit this is set as the pub (qc, [params_{r,m} : m]).
    """
    def __init__(
        self,
        X: np.ndarray,
        circuit_cfg: CircuitArchitecture,
        samplingStrategy: ParameterSamplingStrategy,
    ):
        """
        Parameters
        ----------
        X:
            Input data of shape (M, input_dim).
        circuit_cfg:
            Architecture of the SQP circuit.
        samplingStrategy:
            Strategy used to sample non-data parameters and to merge
            them with data parameters.
        """
        self.qc = CircuitFactory.createSQPcircuit(cfg=circuit_cfg)
        self.X = X
        self.samplingStrategy = samplingStrategy

    def get_parametrization(
        self,
        qc: QuantumCircuit,
        non_data_parameters: dict,
    ) -> list[dict]:
        """Convenience wrapper around the sampling strategy."""
        return self.samplingStrategy.set_parameters_data(qc, non_data_parameters, self.X)

    def get_R_pubs(
        self, num_draws: int, seed: Optional[int] = None
    ):
        """
        Sample `num_draws` sets of non-data parameters and, for each,
        build a pub_r = (qc, param_matrix_r).

        Returns
        -------
        pubs:
            List of length `num_draws`. Each element is (qc, param_matrix_r),
            where param_matrix_r has shape (M, P) with P = len(qc.parameters)
            and rows corresponding to different inputs x_m.

        parameters_R:
            List of length `num_draws`. Entry r is the list of dicts
            returned by `set_parameters_data` for that draw:
            parameters_R[r][m][name] is the value of parameter `name`
            for input x_m and draw r.
        """
        random_non_data_params = self.samplingStrategy.sample_R_random_assignments(
            qc=self.qc, num_draws=num_draws, seed=seed
        )

        params = list(self.qc.parameters)  # fixed parameter order
        num_params = len(params)
        M = len(self.X)

        pubs = []
        parameters_R: list[list[dict]] = []

        for non_data in random_non_data_params:
            # list[dict[name, value]] for each x_m
            params_r_X = self.get_parametrization(qc=self.qc, non_data_parameters=non_data)
            parameters_R.append(params_r_X)

            # Build matrix (M, P) consistent with QC parameter order
            param_matrix = np.zeros((M, num_params), dtype=float)
            for m, param_dict in enumerate(params_r_X):
                for j, p in enumerate(params):
                    name = p.name
                    try:
                        param_matrix[m, j] = float(param_dict[name])
                    except KeyError as exc:
                        raise KeyError(
                            f"Missing parameter '{name}' in dict for input index m={m}"
                        ) from exc

            pubs.append((self.qc, param_matrix))

        return pubs, parameters_R
