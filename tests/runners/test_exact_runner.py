import numpy as np
import pytest

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

from src.runners.circuit_running import ExactCircuitsRunner, ExactResults


def _simulate_statevector_with_aer(circuit: QuantumCircuit) -> np.ndarray:
    """
    Helper: run a single circuit with AerSimulator(statevector) and
    return the resulting statevector as a numpy array.
    """
    backend = AerSimulator(method="statevector", device="CPU")
    qc = circuit.copy()
    qc.save_statevector()
    tqc = transpile(qc, backend)
    result = backend.run(tqc).result()
    return np.array(result.data(0)["statevector"], dtype=complex)


@pytest.mark.parametrize("max_threads", [0, 1, 2])
def test_run_pubs_correct_states_and_shape(max_threads):
    """
    Check that:
      - returned array has shape (R, M, 2**n),
      - each [r, m, :] matches a direct Aer statevector simulation,
      - results are consistent for different max_threads settings.
    """
    theta = Parameter("θ")
    phi = Parameter("φ")

    # Two simple 1-qubit parametrized circuits
    qc1 = QuantumCircuit(1)
    qc1.ry(theta, 0)

    qc2 = QuantumCircuit(1)
    qc2.rx(phi, 0)

    # M = 3 parameterizations per circuit
    thetas = np.array([[0.0], [np.pi / 4], [np.pi / 2]], dtype=float)
    phis   = np.array([[0.0], [np.pi / 3], [np.pi    ]], dtype=float)

    pubs = [(qc1, thetas), (qc2, phis)]  # R = 2
    R = len(pubs)
    M = thetas.shape[0]
    n_qubits = qc1.num_qubits
    dim = 1 << n_qubits

    runner = ExactCircuitsRunner()

    backend = AerSimulator(method="statevector", device="CPU")
    results = runner.run_pubs(
        pubs=pubs,
        backend=backend,
        max_threads=max_threads,
    )

    assert isinstance(results, ExactResults)
    states = results.states

    # Shape should be (R, M, 2**n)
    assert states.shape == (R, M, dim)

    # Check numerics vs direct Aer simulation
    circuits = [qc1, qc2]
    params_list = [thetas, phis]
    param_symbols = [theta, phi]

    for r in range(R):
        qc = circuits[r]
        param_matrix = params_list[r]
        sym = param_symbols[r]

        for m in range(M):
            val = float(param_matrix[m, 0])
            bound_qc = qc.assign_parameters({sym: val})
            expected_sv = _simulate_statevector_with_aer(bound_qc)

            assert expected_sv.shape == (dim,)
            assert np.allclose(states[r, m, :], expected_sv, atol=1e-8)


def test_run_pubs_raises_for_inconsistent_pubs():
    """
    Check that run_pubs complains if:
      - circuits have different numbers of qubits, or
      - pubs don't all share the same M.
    """
    runner = ExactCircuitsRunner()

    # Inconsistent qubit numbers
    theta = Parameter("θ")
    qc_1q = QuantumCircuit(1)
    qc_1q.ry(theta, 0)

    qc_2q = QuantumCircuit(2)
    qc_2q.ry(theta, 0)  # only acts on qubit 0, but circuit has 2 qubits

    params_1 = np.array([[0.0], [np.pi / 2]], dtype=float)  # M = 2
    params_2 = np.array([[0.0], [np.pi / 2]], dtype=float)  # M = 2

    pubs_bad_qubits = [(qc_1q, params_1), (qc_2q, params_2)]

    with pytest.raises(ValueError, match="All circuits must have same number of qubits"):
        runner.run_pubs(pubs=pubs_bad_qubits)

    # Inconsistent M (different number of rows)
    qc_same = QuantumCircuit(1)
    qc_same.ry(theta, 0)

    params_M2 = np.array([[0.0], [np.pi / 2]], dtype=float)       # M = 2
    params_M3 = np.array([[0.0], [np.pi / 2], [np.pi]], dtype=float)  # M = 3

    pubs_bad_M = [(qc_same, params_M2), (qc_same, params_M3)]

    with pytest.raises(ValueError, match="All pubs must have the same M"):
        runner.run_pubs(pubs=pubs_bad_M)
