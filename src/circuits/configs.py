from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class BlockSpec:
    """Specification of a block Q_b and its intra-block edge set E_b.

    - qubits: indices of qubits in this block, e.g. (0, 1, 2)
    - edges: undirected edges between qubits in this block, e.g. ((0, 1), (1, 2))
    """
    qubits: Sequence[int]
    edges: Sequence[Tuple[int, int]]

    # @property
    # def __len__(self):
    #     return len(self.qubits)


@dataclass(frozen=True)
class CircuitArchitecture:
    num_qubits: int                 # n
    blocks: Sequence[BlockSpec]     # (Q_b, E_b)_b
    input_dim: int                  # zeta
    num_layers: int                 # L

    def __post_init__(self):
        self.validate_blocks()

    def validate_blocks(self) -> None:
        """Validate that blocks form a disjoint partition of [0, num_qubits)
        and that edges are well-formed intra-block couplings.
        """
        if self.num_qubits <= 0:
            raise ValueError(f"num_qubits must be positive, got {self.num_qubits}.")

        if not self.blocks:
            raise ValueError("At least one block must be specified (blocks is empty).")

        seen_qubits = set()

        for b_idx, block in enumerate(self.blocks):
            # --- qubits checks ---
            qubits = tuple(block.qubits)
            if not qubits:
                raise ValueError(f"Block {b_idx} has an empty qubit set.")

            block_qubits = set(qubits)
            if len(block_qubits) != len(qubits):
                raise ValueError(
                    f"Block {b_idx} has duplicate qubit indices: {qubits}."
                )

            # Check range of qubits
            out_of_range = [q for q in block_qubits if not (0 <= q < self.num_qubits)]
            if out_of_range:
                raise ValueError(
                    f"Block {b_idx} has qubit indices out of range [0, {self.num_qubits}): "
                    f"{sorted(out_of_range)}."
                )

            # Enforce disjointness between blocks
            clash = seen_qubits & block_qubits
            if clash:
                raise ValueError(
                    f"Qubits {sorted(clash)} appear in multiple blocks "
                    f"(problem in block {b_idx}, qubits={qubits})."
                )

            seen_qubits |= block_qubits

            # --- edges checks ---
            edges = tuple(block.edges)
            seen_edges = set()

            for e_idx, edge in enumerate(edges):
                if len(edge) != 2:
                    raise ValueError(
                        f"Edge {edge} in block {b_idx} (index {e_idx}) must "
                        f"contain exactly two qubit indices."
                    )

                u, v = edge

                # No self-loops
                if u == v:
                    raise ValueError(
                        f"Self-loop edge ({u}, {v}) found in block {b_idx} "
                        f"(edge index {e_idx})."
                    )

                # Edge endpoints must be in this block
                if u not in block_qubits or v not in block_qubits:
                    raise ValueError(
                        f"Edge ({u}, {v}) in block {b_idx} connects qubits not "
                        f"both in the block's qubit set {sorted(block_qubits)}."
                    )

                # Treat edges as undirected: normalize ordering for duplicate checks
                normalized = (u, v) if u < v else (v, u)

                if normalized in seen_edges:
                    raise ValueError(
                        f"Duplicate edge {normalized} found in block {b_idx}."
                    )

                seen_edges.add(normalized)

        # Enforce that blocks form a full partition of all qubits
        missing = set(range(self.num_qubits)) - seen_qubits
        if missing:
            raise ValueError(
                "Blocks do not cover all qubits in [0, num_qubits). "
                f"Missing qubits: {sorted(missing)}."
            )
