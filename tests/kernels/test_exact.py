import numpy as np
import pytest

from src.circuits.configs import CircuitArchitecture, BlockSpec

import pickle
from dataclasses import dataclass
from typing import List, Tuple

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
    retriever = ExactGramsRetriever(arch=two_block_arch, exact_grams=None)
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
    retriever = ExactGramsRetriever(arch=two_block_arch, exact_grams=None)

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


# ---------------------------------------------------------------------------
# Small dummy architecture for tests
# ---------------------------------------------------------------------------

@dataclass(eq=True)
class DummyBlock:
    qubits: Tuple[int, ...]


@dataclass(eq=True)
class DummyArch:
    """
    Minimal stand-in for CircuitArchitecture for testing ExactGramsRetriever.

    It only needs:
      - num_qubits: int
      - blocks: iterable of objects with a `.qubits` attribute
    """
    num_qubits: int
    blocks: List[DummyBlock]


def make_dummy_arch(n_qubits: int) -> DummyArch:
    # One 1-qubit block per qubit: {0}, {1}, ..., {n-1}
    blocks = [DummyBlock(qubits=(q,)) for q in range(n_qubits)]
    return DummyArch(num_qubits=n_qubits, blocks=blocks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_exact_grams_parallel_matches_sequential():
    """
    Check that using multiple workers (n_jobs > 1) gives the same
    Gram tensors as the sequential implementation (n_jobs = 1).
    """
    rng = np.random.default_rng(123)

    n_qubits = 3
    dim = 1 << n_qubits  # 2**n_qubits
    R = 4                # number of draws
    M = 5                # number of inputs per draw

    # Random complex statevectors of shape (R, M, 2**n)
    states_real = rng.normal(size=(R, M, dim))
    states_imag = rng.normal(size=(R, M, dim))
    states = states_real + 1j * states_imag

    arch = make_dummy_arch(n_qubits)
    results = ExactResults(states=states, arch=arch)

    retriever = ExactGramsRetriever(arch=arch, exact_grams=None)

    # Sequential path
    grams_seq = retriever.get_exact_grams(results, n_jobs=1)

    # Parallel path (ensure n_jobs > 1 and <= R)
    grams_par = retriever.get_exact_grams(results, n_jobs=3)

    assert grams_seq.shape == grams_par.shape
    assert np.allclose(grams_seq, grams_par, atol=1e-10)


def test_exact_grams_retriever_save_and_load_roundtrip(tmp_path: pytest.TempPathFactory):
    """
    Check that saving and loading ExactGramsRetriever preserves:
      - the architecture
      - the last computed exact_grams tensor
      - behaviour (get_exact_grams still yields the same result)
    """
    rng = np.random.default_rng(42)

    n_qubits = 2
    dim = 1 << n_qubits
    R = 3
    M = 4

    states_real = rng.normal(size=(R, M, dim))
    states_imag = rng.normal(size=(R, M, dim))
    states = states_real + 1j * states_imag

    arch = make_dummy_arch(n_qubits)
    results = ExactResults(states=states, arch=arch)

    retriever = ExactGramsRetriever(arch=arch, exact_grams=None)

    # Reference Gram tensors (and populate retriever.exact_grams)
    grams_ref = retriever.get_exact_grams(results, n_jobs=1)
    assert retriever.exact_grams is not None
    np.testing.assert_allclose(retriever.exact_grams, grams_ref, atol=1e-10)

    # --- Test the official save/load API (dict payload) ---
    path = tmp_path / "exact_grams_retriever.pkl"
    retriever.save(path)

    loaded = ExactGramsRetriever.load(path)

    assert isinstance(loaded, ExactGramsRetriever)
    # arch should round-trip
    assert loaded.arch == arch

    # exact_grams should be restored
    assert loaded.exact_grams is not None
    np.testing.assert_allclose(loaded.exact_grams, grams_ref, atol=1e-10)

    # Behaviour after load: recomputing should match as well
    grams_loaded = loaded.get_exact_grams(results, n_jobs=1)
    np.testing.assert_allclose(grams_loaded, grams_ref, atol=1e-10)

    # --- Also test backward-compatible whole-object pickle path ---
    direct_path = tmp_path / "exact_grams_retriever_direct.pkl"
    with direct_path.open("wb") as f:
        pickle.dump(retriever, f)

    loaded_direct = ExactGramsRetriever.load(direct_path)
    assert isinstance(loaded_direct, ExactGramsRetriever)

    # direct pickle preserves attribute state automatically
    assert loaded_direct.exact_grams is not None
    np.testing.assert_allclose(loaded_direct.exact_grams, grams_ref, atol=1e-10)

    grams_direct = loaded_direct.get_exact_grams(results, n_jobs=1)
    np.testing.assert_allclose(grams_direct, grams_ref, atol=1e-10)

