# tests/circuits/test_uniform_parameter_sampling.py

import math
import numpy as np

from qiskit.quantum_info import Statevector

from src.circuits.circuit_building import CircuitFactory
from src.circuits.configs import CircuitArchitecture, BlockSpec
from src.circuits.parameter_sampling import UniformParameterSamplingStrategy


def build_5q_architecture(input_dim: int = 2, num_layers: int = 2) -> CircuitArchitecture:
    return CircuitArchitecture(
        num_qubits=5,
        blocks=(
            BlockSpec(qubits=(0, 1), edges=((0, 1),)),
            BlockSpec(qubits=(2, 3, 4), edges=((2, 3), (3, 4))),
        ),
        input_dim=input_dim,
        num_layers=num_layers,
    )


def _bind_x_to_zero(qc, name_assignment):
    """Bind all non-data parameters from name_assignment and set x[*]=0."""
    mapping = {}
    for p in qc.parameters:
        if p.name.startswith("x["):
            mapping[p] = 0.0
        elif p.name in name_assignment:
            mapping[p] = float(name_assignment[p.name])
    if not mapping:
        return qc
    return qc.assign_parameters(mapping, inplace=False)


# 1) parameter ranges and ξ norms ---------------------------------------------

def test_uniform_sampling_parameter_ranges_and_xi_norms():
    arch = build_5q_architecture(input_dim=2, num_layers=2)
    qc = CircuitFactory.createSQPcircuit(arch)

    sampler = UniformParameterSamplingStrategy(arch, seed=123)
    assignment = sampler.sample_random_assignment(qc)

    x_params = [p for p in qc.parameters if p.name.startswith("x[")]
    xi_params = [p for p in qc.parameters if p.name.startswith("xi_")]
    other_params = [
        p for p in qc.parameters
        if not p.name.startswith("x[") and not p.name.startswith("xi_")
    ]

    # x[j] must not appear
    for p in x_params:
        assert p.name not in assignment

    # ξ-layers each have ‖ξℓ‖ = 1/sqrt(zeta)
    zeta = arch.input_dim
    xi_by_layer = {}
    for p in xi_params:
        prefix, rest = p.name.split("[", 1)
        ell_str = prefix.split("_", 1)[1]
        ell = int(ell_str)
        j_str = rest.split("]", 1)[0]
        j = int(j_str)
        if ell not in xi_by_layer:
            xi_by_layer[ell] = np.zeros(zeta)
        xi_by_layer[ell][j] = assignment[p.name]

    for ell, vec in xi_by_layer.items():
        norm = np.linalg.norm(vec)
        target = 1.0 / math.sqrt(zeta)
        assert math.isclose(norm, target, rel_tol=1e-6, abs_tol=1e-6)

    # all other non-x, non-xi in [-π, π]
    for p in other_params:
        v = assignment[p.name]
        assert -math.pi <= v <= math.pi


# 2) seed reproducibility (single draw) --------------------------------------

def test_uniform_sampling_seed_reproducibility():
    arch = build_5q_architecture(input_dim=2, num_layers=2)
    qc = CircuitFactory.createSQPcircuit(arch)

    sampler1 = UniformParameterSamplingStrategy(arch, seed=42)
    sampler2 = UniformParameterSamplingStrategy(arch, seed=42)

    assign1 = sampler1.sample_random_assignment(qc)
    assign2 = sampler2.sample_random_assignment(qc)

    assert assign1 == assign2


# 3) seed reproducibility (R draws) ------------------------------------------

def test_sample_R_random_assignments_seed_reproducibility():
    arch = build_5q_architecture(input_dim=2, num_layers=2)
    qc = CircuitFactory.createSQPcircuit(arch)

    sampler = UniformParameterSamplingStrategy(arch, seed=999)

    params_list_1 = sampler.sample_R_random_assignments(qc, num_draws=3, seed=123)
    params_list_2 = sampler.sample_R_random_assignments(qc, num_draws=3, seed=123)

    assert len(params_list_1) == len(params_list_2) == 3
    for a1, a2 in zip(params_list_1, params_list_2):
        assert a1 == a2


# 4) diversity of sampled assignments (different statevectors) ----------------

def test_sample_R_random_assignments_produce_diverse_samples():
    arch = build_5q_architecture(input_dim=2, num_layers=2)
    qc = CircuitFactory.createSQPcircuit(arch)

    sampler = UniformParameterSamplingStrategy(arch, seed=42)
    params_list = sampler.sample_R_random_assignments(qc, num_draws=4, seed=None)

    svs = []
    for assign in params_list:
        qc_full = _bind_x_to_zero(qc, assign)
        svs.append(Statevector.from_instruction(qc_full).data)

    all_equal = True
    for i in range(len(svs)):
        for j in range(i + 1, len(svs)):
            if not np.allclose(svs[i], svs[j]):
                all_equal = False
                break
        if not all_equal:
            break

    assert not all_equal


# 5) set_parameters_data correctness -----------------------------------------

def test_set_parameters_data_merges_non_data_and_data_correctly():
    arch = build_5q_architecture(input_dim=2, num_layers=2)
    qc = CircuitFactory.createSQPcircuit(arch)

    sampler = UniformParameterSamplingStrategy(arch, seed=7)
    non_data = sampler.sample_random_assignment(qc)

    X = np.array([[0.1, -0.2], [0.3, 0.5], [-0.4, 0.9]], dtype=float)
    params_X = sampler.set_parameters_data(qc, non_data, X)
    assert len(params_X) == X.shape[0]

    # map j -> "x[j]"
    x_names_by_index = {}
    for p in qc.parameters:
        if p.name.startswith("x["):
            inside = p.name.split("[", 1)[1].split("]", 1)[0]
            j = int(inside)
            x_names_by_index[j] = p.name

    assert len(x_names_by_index) == arch.input_dim

    for m, params_m in enumerate(params_X):
        # non-data params are present and equal
        for name, v in non_data.items():
            assert name in params_m
            assert params_m[name] == v

        # x[j] matches X[m, j]
        for j, name in x_names_by_index.items():
            assert math.isclose(params_m[name], X[m, j])


def test_set_parameters_data_raises_on_wrong_input_dim():
    import pytest

    arch = build_5q_architecture(input_dim=2, num_layers=2)
    qc = CircuitFactory.createSQPcircuit(arch)

    sampler = UniformParameterSamplingStrategy(arch, seed=0)
    non_data = sampler.sample_random_assignment(qc)

    X_bad = np.random.randn(5, 3)
    with pytest.raises(ValueError):
        sampler.set_parameters_data(qc, non_data, X_bad)
