import numpy as np
import pytest

from src.circuits.configs import CircuitArchitecture, BlockSpec
from src.runners.circuit_running import ExactResults
from src.kernels.exact import ExactGramsRetriever


@pytest.fixture
def two_block_arch() -> CircuitArchitecture:
    """
    4-qubit architecture with two disjoint blocks:
    - block 0: qubits (0, 1)
    - block 1: qubits (2, 3)
    Edges don’t matter for the Gram computation, but we add a simple line edge.
    """
    blocks = (
        BlockSpec(qubits=(0, 1), edges=((0, 1),)),
        BlockSpec(qubits=(2, 3), edges=((2, 3),)),
    )
    return CircuitArchitecture(
        num_qubits=4,
        blocks=blocks,
        input_dim=1,
        num_layers=1,
    )


def test_exact_grams_matches_theory_known_states(two_block_arch):
    """
    Construct R=2, M=2 product states with known block components and check that
    the retrieved Gram matrices match the analytically computed ones.

    For each r, m we build:
        |psi_{r,m}> = |chi_{r,m}>_(q3,q2) ⊗ |phi_{r,m}>_(q1,q0)

    Block 0 is (q0, q1), block 1 is (q2, q3).
    """

    R, M, n = 2, 2, 4
    dim_block = 4
    dim = 2**n

    # 2-qubit basis vectors |00>, |01>, |10>, |11>
    def basis_2qb(idx: int) -> np.ndarray:
        v = np.zeros(dim_block, dtype=complex)
        v[idx] = 1.0
        return v

    e00 = basis_2qb(0)
    e01 = basis_2qb(1)
    e10 = basis_2qb(2)
    e11 = basis_2qb(3)

    plus2 = np.ones(dim_block, dtype=complex) / 2.0  # |++> on two qubits
    cat = np.array([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)], dtype=complex)

    # phi[r][m] → block 0 state (q1,q0), chi[r][m] → block 1 state (q3,q2)
    phi = [[None for _ in range(M)] for _ in range(R)]
    chi = [[None for _ in range(M)] for _ in range(R)]

    # r = 0
    phi[0][0] = e00          # |00>
    phi[0][1] = e11          # |11>
    chi[0][0] = e00          # |00>
    chi[0][1] = plus2        # |++>

    # r = 1
    phi[1][0] = e01          # |01>
    phi[1][1] = e10          # |10>
    chi[1][0] = e11          # |11>
    chi[1][1] = cat          # (|00> + |11>)/sqrt(2)

    # Build global 4-qubit states using Qiskit ordering (q3 q2 q1 q0):
    # |psi> = |chi>_(q3,q2) ⊗ |phi>_(q1,q0)
    states = np.zeros((R, M, dim), dtype=complex)
    for r in range(R):
        for m in range(M):
            states[r, m, :] = np.kron(chi[r][m], phi[r][m])

    results = ExactResults(states=states, arch=two_block_arch)
    retriever = ExactGramsRetriever(arch=two_block_arch)
    grams = retriever.get_exact_grams(results)

    assert grams.shape == (R, 2, M, M)

    # Analytic Gram matrices:
    # For pure block states, k_b(i,j) = Tr(ρ_i ρ_j) = |<φ_i | φ_j>|^2 (or the
    # corresponding block-1 chi states).
    expected = np.zeros_like(grams)

    # r = 0, block 0: |00>, |11>  → orthogonal
    expected[0, 0] = np.array([[1.0, 0.0],
                               [0.0, 1.0]])

    # r = 0, block 1: |00>, |++>
    # <00|++> = 1/2  ⇒  |<·|·>|^2 = 1/4
    expected[0, 1] = np.array([[1.0, 0.25],
                               [0.25, 1.0]])

    # r = 1, block 0: |01>, |10>  → orthogonal
    expected[1, 0] = np.array([[1.0, 0.0],
                               [0.0, 1.0]])

    # r = 1, block 1: |11>, (|00>+|11>)/√2
    # <11|cat> = 1/√2 ⇒ |<·|·>|^2 = 1/2
    expected[1, 1] = np.array([[1.0, 0.5],
                               [0.5, 1.0]])

    np.testing.assert_allclose(grams, expected, atol=1e-12)


def _random_normalized_states(R: int, M: int, n: int, seed: int = 123) -> np.ndarray:
    """Utility for the random test: Haar-like random normalized statevectors."""
    rng = np.random.default_rng(seed)
    dim = 2**n
    raw = rng.normal(size=(R, M, dim)) + 1j * rng.normal(size=(R, M, dim))
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    return raw / norms


def test_exact_grams_are_psd_and_symmetric_for_random_states(two_block_arch):
    """
    For random quantum states, each (r, b) Gram matrix must be:
      - symmetric
      - positive semi-definite (eigenvalues >= 0 up to numerical tolerance)
    """

    R, M, n = 3, 4, 4

    states = _random_normalized_states(R, M, n, seed=42)
    results = ExactResults(states=states, arch=two_block_arch)
    retriever = ExactGramsRetriever(arch=two_block_arch)

    grams = retriever.get_exact_grams(results)
    assert grams.shape == (R, 2, M, M)

    tol = 1e-10
    for r in range(R):
        for b in range(2):
            G = grams[r, b]

            # Symmetry
            assert np.allclose(G, G.T, atol=tol)

            # Positive semi-definite: eigenvalues >= -tol
            eigvals = np.linalg.eigvalsh(G)
            assert eigvals.min() >= -tol, f"Gram matrix not PSD for r={r}, b={b}"
