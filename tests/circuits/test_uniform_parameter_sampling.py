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

def _bind_x_to_zero(qc):
    """
    Helper: bind all data parameters x[*] to 0.0, leaving the
    random non-data parameters fixed. This gives a fully numeric
    circuit so we can compare statevectors.
    """
    if not qc.parameters:
        return qc

    assignment = {
        p: 0.0
        for p in qc.parameters
        if p.name.startswith("x[")
    }
    if not assignment:
        return qc

    return qc.assign_parameters(assignment, inplace=False)


def test_uniform_sampling_T_and_other_params_ranges():
    arch = build_5q_architecture(input_dim=2, num_layers=2)
    qc = CircuitFactory.createSQPcircuit(arch)

    sampler = UniformParameterSamplingStrategy(arch, seed=123)
    assignment = sampler.get_random_assignment(qc)

    x_params = [p for p in qc.parameters if p.name.startswith("x[")]
    T_params = [p for p in qc.parameters if p.name.startswith("T_")]
    other_params = [
        p for p in qc.parameters
        if not p.name.startswith("x[") and not p.name.startswith("T_")
    ]

    # 1) x[*] params must NOT be sampled (user will bind them separately)
    for p in x_params:
        assert p not in assignment

    # 2) T params Bernoulli in {0,1}
    for p in T_params:
        v = assignment[p]
        assert v in (0.0, 1.0)

    # 3) All other (non-x, non-T) params in [-π, π]
    for p in other_params:
        v = assignment[p]
        assert -math.pi <= v <= math.pi

def test_uniform_sampling_seed_reproducibility():
    arch = build_5q_architecture(input_dim=2, num_layers=2)
    qc = CircuitFactory.createSQPcircuit(arch)

    sampler1 = UniformParameterSamplingStrategy(arch, seed=42)
    sampler2 = UniformParameterSamplingStrategy(arch, seed=42)

    assign1 = sampler1.get_random_assignment(qc)
    assign2 = sampler2.get_random_assignment(qc)

    # Compare via parameter names to avoid differing Parameter object identities
    a1 = {p.name: v for p, v in assign1.items()}
    a2 = {p.name: v for p, v in assign2.items()}

    assert a1 == a2

def test_uniform_sampling_get_random_circuit_binds_all_non_x():
    arch = build_5q_architecture(input_dim=2, num_layers=2)
    qc = CircuitFactory.createSQPcircuit(arch)

    sampler = UniformParameterSamplingStrategy(arch, seed=7)
    qc_assigned = sampler.get_random_circuit(qc)

    remaining_param_names = {p.name for p in qc_assigned.parameters}

    # All remaining parameters must be x[*] only
    for name in remaining_param_names:
        assert name.startswith("x[")

    # There should be exactly input_dim data parameters left
    assert len(remaining_param_names) == arch.input_dim


def test_get_n_random_circuits_seed_reproducibility():
    arch = build_5q_architecture(input_dim=2, num_layers=2)
    base_qc = CircuitFactory.createSQPcircuit(arch)

    sampler = UniformParameterSamplingStrategy(arch, seed=999)

    circuits1 = sampler.get_n_random_circuits(base_qc, num_circuits=3, seed=123)
    circuits2 = sampler.get_n_random_circuits(base_qc, num_circuits=3, seed=123)

    assert len(circuits1) == len(circuits2) == 3

    for c1, c2 in zip(circuits1, circuits2):
        b1 = _bind_x_to_zero(c1)
        b2 = _bind_x_to_zero(c2)

        sv1 = Statevector.from_instruction(b1).data
        sv2 = Statevector.from_instruction(b2).data

        assert np.allclose(sv1, sv2)

def test_get_n_random_circuits_produce_diverse_samples():
    arch = build_5q_architecture(input_dim=2, num_layers=2)
    base_qc = CircuitFactory.createSQPcircuit(arch)

    sampler = UniformParameterSamplingStrategy(arch, seed=42)

    circuits = sampler.get_n_random_circuits(base_qc, num_circuits=4)  # seed=None by default

    # Compute statevectors with x bound to 0 so only non-data params matter
    svs = []
    for c in circuits:
        b = _bind_x_to_zero(c)
        svs.append(Statevector.from_instruction(b).data)

    # Check that not all statevectors are equal
    all_equal = True
    for i in range(len(svs)):
        for j in range(i + 1, len(svs)):
            if not np.allclose(svs[i], svs[j]):
                all_equal = False
                break
        if not all_equal:
            break

    assert not all_equal
