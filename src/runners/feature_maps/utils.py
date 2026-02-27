from __future__ import annotations
from typing import List, Sequence, Tuple
from itertools import combinations, product
from qiskit.quantum_info import SparsePauliOp


def generate_k_local_paulis(locality: int, num_qubits: int) -> List[SparsePauliOp]:
    """
    Generate all Pauli strings on `num_qubits` whose locality is <= `locality`,
    returned as `SparsePauliOp` objects.

    - Locality = number of non-identity positions (X, Y, or Z). Qubit 0 is the **rightmost** character of the string.

    The all-identity operator (0-local) is *not* included.
    """

    if locality < 1:
        return []

    locality = min(locality, num_qubits)

    paulis: List[SparsePauliOp] = []
    letters = ("X", "Y", "Z")

    # choose locality ℓ = 1..k
    for l in range(1, locality + 1):
        # choose which qubits are non-identity
        for qubit_indices in combinations(range(num_qubits), l):
            # for those qubits, choose which Pauli (X/Y/Z) acts
            for pauli_choice in product(letters, repeat=l):
                chars = ["I"] * num_qubits

                for q, p in zip(qubit_indices, pauli_choice):
                    # map qubit index -> string index (qubit 0 is rightmost)
                    str_idx = num_qubits - 1 - q
                    chars[str_idx] = p

                pauli_str = "".join(chars)
                paulis.append(SparsePauliOp(pauli_str))

    return paulis


def generate_edges_correlator_paulis(edges: Sequence[Tuple[int, int]], num_qubits: int) -> List[SparsePauliOp]:
    """
    Generate all Pauli strings on the global register that correspond to

      • single-qubit Paulis X, Y, Z on each qubit in `layer.qubits`
      • two-qubit “correlators” XX, YY, ZZ on each edge in `layer.edges`

    Conventions:
      - Qubit 0 is the **rightmost** character of the Pauli string.
      - The all-identity operator is not included.
    """

    # Deduplicate qubits and edges, keep them sorted for determinism
    qubits = [i for i in range(num_qubits)]

    unique_edges = sorted({tuple(sorted(e)) for e in edges})

    pauli_letters = ("X", "Y", "Z")
    ops: List[SparsePauliOp] = []

    def make_label(active_qubits, letter: str) -> str:
        """Return Pauli label with `letter` on active_qubits, I elsewhere."""
        chars = ["I"] * num_qubits
        for q in active_qubits:
            # map qubit index q -> string index (q=0 is rightmost)
            idx = num_qubits - 1 - q
            chars[idx] = letter
        return "".join(chars)

    # 1) Single-qubit Paulis
    for q in qubits:
        for letter in pauli_letters:
            label = make_label([q], letter)
            ops.append(SparsePauliOp(label))

    # 2) Two-qubit correlators along edges (XX, YY, ZZ)
    for q1, q2 in unique_edges:
        # Optionally enforce that both endpoints are in this layer
        if q1 not in qubits or q2 not in qubits:
            continue
        for letter in pauli_letters:
            label = make_label([q1, q2], letter)
            ops.append(SparsePauliOp(label))

    return ops
