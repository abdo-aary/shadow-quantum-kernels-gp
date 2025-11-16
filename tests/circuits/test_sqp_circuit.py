import math

from qiskit.quantum_info import Statevector, SparsePauliOp

from src.circuits.configs import CircuitArchitecture, BlockSpec
from src.circuits.circuit_building import CircuitFactory


def test_sqp_circuit_single_rx_on_q0():
    """
    Build the 5-qubit SQP circuit and assign parameters so that the
    only non-trivial gate is Rx(theta) on qubit 0. Then check that
    <Z0> equals cos(theta).
    """
    # --- 1) Architecture as in your example_usage.py ---
    arch = CircuitArchitecture(
        num_qubits=5,
        blocks=(
            BlockSpec(qubits=(0, 1), edges=((0, 1),)),
            BlockSpec(qubits=(2, 3, 4), edges=((2, 3), (3, 4))),
        ),
        input_dim=2,
        num_layers=2,
    )

    qc = CircuitFactory.createSQPcircuit(arch)

    theta = 0.7  # angle for Rx on qubit 0, layer 0, block 0

    # --- 2) Build an assignment dict: Parameter -> value ---
    assignment = {p: 0.0 for p in qc.parameters}  # everything zero

    # Set ONLY Rx on q0, layer 0, block 0 non-zero.
    # Check the parameter names by printing qc if needed.
    for p in qc.parameters:
        if p.name == "Rx_0_0[0]":
            assignment[p] = theta

    # Optional: make sure we actually set something
    assert any(val != 0.0 for val in assignment.values())

    # --- 3) Assign parameters ---
    qc_bound = qc.assign_parameters(assignment, inplace=False)

    # --- 4) Statevector and <Z0> expectation ---
    state = Statevector.from_instruction(qc_bound)

    op = SparsePauliOp.from_list([("IIIIZ", 1.0)])  # Z on qubit 0
    expect_qc = state.expectation_value(op).real

    # Theoretical expectation: starting from |0>, Rx(theta) gives <Z> = cos(theta)
    expect_theory = math.cos(theta)

    assert math.isclose(expect_qc, expect_theory, rel_tol=1e-10, abs_tol=1e-10)


def test_sqp_circuit_two_layer_single_qubit_with_generator():
    """
    Use two layers on qubit 0 with a non-trivial generator in layer 1:
        U(x) = Rx_1 * Rz_1(x) * Rx_0
    and check that <Z_0> matches the analytic expression
        cos(g1)*cos(g2) - sin(g1)*sin(g2)*cos(x).
    """
    # 5-qubit, 2-block architecture as before, but 1D input
    arch = CircuitArchitecture(
        num_qubits=5,
        blocks=(
            BlockSpec(qubits=(0, 1), edges=((0, 1),)),
            BlockSpec(qubits=(2, 3, 4), edges=((2, 3), (3, 4))),
        ),
        input_dim=1,
        num_layers=2,
    )

    qc = CircuitFactory.createSQPcircuit(arch)

    # Parameters
    x_val = 0.3
    g1 = 0.7    # Rx on q0, layer 0
    g2 = -0.4   # Rx on q0, layer 1

    # Start with everything = 0
    assignment = {p: 0.0 for p in qc.parameters}

    for p in qc.parameters:
        name = p.name

        # data + generator to realize angle = x
        if name == "x[0]":
            assignment[p] = x_val
        elif name == "xi_1[0]":
            assignment[p] = 1.0
        elif name == "deltas_0_1[0]":
            assignment[p] = 1.0

        # layer-0 Rx on q0: γ1
        elif name == "Rx_0_0[0]":
            assignment[p] = g1

        # layer-1 Rx on q0: γ2
        elif name == "Rx_0_1[0]":
            assignment[p] = g2

        # all other parameters stay at 0.0 (they’re already set)

    # Sanity: at least something is non-zero
    assert any(val != 0.0 for val in assignment.values())

    # Bind and simulate
    qc_bound = qc.assign_parameters(assignment, inplace=False)

    state = Statevector.from_instruction(qc_bound)

    # Remember Qiskit endianness: right-most Pauli acts on qubit 0
    op = SparsePauliOp.from_list([("IIIIZ", 1.0)])  # Z on qubit 0
    expect_qc = state.expectation_value(op).real

    # Analytic expectation
    expect_theory = (
        math.cos(g1) * math.cos(g2)
        - math.sin(g1) * math.sin(g2) * math.cos(x_val)
    )

    assert math.isclose(expect_qc, expect_theory, rel_tol=1e-10, abs_tol=1e-10)

def test_sqp_circuit_block0_zz_coupling_affects_x0():
    """
    Turn on Rx on q0 and q1 plus a ZZ coupling between them, and check that
    <X_0> matches the analytic expression:
        <X0> = sin(alpha) * cos(beta) * sin(2 * lambda)
    where lambda is the ZZ parameter (qc.rzz(2 * lambda, 0, 1)).
    """
    # 5-qubit, 2-block architecture, 1 layer (only ℓ = 0) and 1D input
    arch = CircuitArchitecture(
        num_qubits=5,
        blocks=(
            BlockSpec(qubits=(0, 1), edges=((0, 1),)),
            BlockSpec(qubits=(2, 3, 4), edges=((2, 3), (3, 4))),
        ),
        input_dim=1,
        num_layers=1,
    )

    qc = CircuitFactory.createSQPcircuit(arch)

    # Choose some non-trivial angles
    alpha = 0.4   # Rx on q0
    beta = 0.9    # Rx on q1
    lam = 0.5     # ZZ parameter; rzz is called with angle 2*lam

    # Start with everything 0
    assignment = {p: 0.0 for p in qc.parameters}

    # Turn on only the gates we want:
    for p in qc.parameters:
        name = p.name

        # Local rotations on block 0
        if name == "Rx_0_0[0]":        # q0
            assignment[p] = alpha
        elif name == "Rx_0_0[1]":      # q1
            assignment[p] = beta

        # ZZ coupling on edge (0,1) in block 0
        elif name == "ZZ_0_0[0]":
            assignment[p] = lam

        # all others remain 0.0 (Rz generators, T gates, other blocks, etc.)

    # Sanity check: some params must be non-zero
    assert any(val != 0.0 for val in assignment.values())

    # Assign parameters and simulate
    qc_bound = qc.assign_parameters(assignment, inplace=False)
    state = Statevector.from_instruction(qc_bound)

    # Measure X on qubit 0 (right-most Pauli acts on q0: IIIIX)
    op = SparsePauliOp.from_list([("IIIIX", 1.0)])
    expect_qc = state.expectation_value(op).real

    # Analytic expectation: <X0> = sin(alpha) * cos(beta) * sin(2 * lambda)
    expect_theory = math.sin(alpha) * math.cos(beta) * math.sin(2 * lam)

    assert math.isclose(expect_qc, expect_theory, rel_tol=1e-10, abs_tol=1e-10)

def test_sqp_circuit_single_qubit_with_P_phase():
    """
    Turn on Rx and T (phase) on qubit 0, check that <X_0> matches
    the analytic expression:

        <X0> = sin(alpha) * sin(phi),

    where phi = (pi/4) * t and t is the T_0_0[0] parameter.
    """
    # --- 1) Architecture: 5 qubits, 2 blocks, 1 layer ---
    arch = CircuitArchitecture(
        num_qubits=5,
        blocks=(
            BlockSpec(qubits=(0, 1), edges=((0, 1),)),
            BlockSpec(qubits=(2, 3, 4), edges=((2, 3), (3, 4))),
        ),
        input_dim=1,
        num_layers=1,
    )

    qc = CircuitFactory.createSQPcircuit(arch)

    # --- 2) Choose angles ---
    alpha = 0.6   # Rx on q0
    t = 1.2       # raw T parameter; phase gate angle is phi = pi/4 * t
    phi = math.pi / 4.0 * t

    # --- 3) Set all parameters to 0, then turn on Rx_0_0[0] and T_0_0[0] ---
    assignment = {p: 0.0 for p in qc.parameters}

    for p in qc.parameters:
        name = p.name

        # Local Rx on block 0, qubit 0
        if name == "Rx_0_0[0]":
            assignment[p] = alpha

        # P gate on block 0, qubit 0 (phase pi/4 * t)
        elif p.name == "P_0_0[0]":
            assignment[p] = phi  # directly the phase angle

        # Everything else stays 0: all generators, ZZ, T on other qubits, etc.

    assert any(val != 0.0 for val in assignment.values())

    # --- 4) Bind parameters and simulate ---
    qc_bound = qc.assign_parameters(assignment, inplace=False)
    state = Statevector.from_instruction(qc_bound)

    # --- 5) Observable: X on qubit 0 ---
    # Qiskit endianness: right-most Pauli acts on qubit 0 -> IIIIX
    op = SparsePauliOp.from_list([("IIIIX", 1.0)])
    expect_qc = state.expectation_value(op).real

    # --- 6) Theoretical expectation: <X0> = sin(alpha) * sin(phi) ---
    expect_theory = math.sin(alpha) * math.sin(phi)

    assert math.isclose(expect_qc, expect_theory, rel_tol=1e-10, abs_tol=1e-10)
