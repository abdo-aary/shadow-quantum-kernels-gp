import numpy as np

from qiskit import QuantumCircuit

from src.circuits.configs import CircuitArchitecture, BlockSpec
from src.circuits.circuit_building import CircuitFactory
from src.circuits.parameter_sampling import UniformParameterSamplingStrategy
from src.circuits.experimenting import Experiment


def build_2q_architecture(input_dim: int = 1, num_layers: int = 1) -> CircuitArchitecture:
    return CircuitArchitecture(
        num_qubits=2,
        blocks=(BlockSpec(qubits=(0, 1), edges=((0, 1),)),),
        input_dim=input_dim,
        num_layers=num_layers,
    )


class ToyDeterministicSampling(UniformParameterSamplingStrategy):
    """Deterministic strategy: one Rx_*[0] carries theta, everything else is 0."""

    def __init__(self, circuit_cfg: CircuitArchitecture, theta: float):
        super().__init__(circuit_cfg=circuit_cfg, seed=0)
        self.theta = theta

    def sample_random_assignment(self, qc: QuantumCircuit, seed: int | None = None) -> dict[str, float]:
        assignment: dict[str, float] = {}
        chosen_rx: str | None = None

        for p in qc.parameters:
            name = p.name
            if name.startswith("x["):
                continue

            if name.startswith("Rx_") and name.endswith("[0]") and chosen_rx is None:
                assignment[name] = float(self.theta)
                chosen_rx = name
            else:
                assignment[name] = 0.0

        if chosen_rx is None:
            raise RuntimeError("ToyDeterministicSampling: no Rx_*[0] parameter found.")

        return assignment

    def sample_R_random_assignments(self, qc, num_draws: int, seed=None):
        if num_draws != 1:
            raise ValueError("ToyDeterministicSampling only supports num_draws=1.")
        return [self.sample_random_assignment(qc, seed=None)]


def test_get_R_pubs_param_matrix_matches_dict_assignments():
    # X: two points in R^1
    X = np.array([[0.3], [-0.8]], dtype=float)
    M, input_dim = X.shape
    assert input_dim == 1

    arch = build_2q_architecture(input_dim=input_dim, num_layers=1)
    theta = 0.7
    toy_sampler = ToyDeterministicSampling(circuit_cfg=arch, theta=theta)

    # Ground-truth name-dict assignments
    qc = CircuitFactory.createSQPcircuit(cfg=arch)
    non_data = toy_sampler.sample_random_assignment(qc)
    params_dicts = toy_sampler.set_parameters_data(qc, non_data, X)
    assert len(params_dicts) == M

    # Through Experiment.get_R_pubs
    exp = Experiment(X=X, circuit_cfg=arch, samplingStrategy=toy_sampler)
    pubs, params_R = exp.get_R_pubs(num_draws=1)

    assert len(pubs) == 1
    assert len(params_R) == 1

    qc2, param_matrix = pubs[0]
    assert qc2 is exp.qc
    assert param_matrix.shape == (M, len(qc2.parameters))

    # Recover name-dicts from param_matrix
    params_from_matrix = []
    param_list = list(qc2.parameters)
    for m in range(M):
        row_dict = {p.name: float(param_matrix[m, j]) for j, p in enumerate(param_list)}
        params_from_matrix.append(row_dict)

    # Compare reference vs reconstructed
    for d_ref, d_mat in zip(params_dicts, params_from_matrix):
        assert d_ref == d_mat

    # parameters_R returned by Experiment should equal params_dicts as well
    assert params_R[0] == params_dicts
