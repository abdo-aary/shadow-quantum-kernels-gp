from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class LayerSpec:
    """Specification of a layer ell and its intra-layer edge set E_ell.

    - qubits: indices of qubits in this layer, e.g. (0, 1, 2)
    - edges: undirected edges between qubits in this layer, e.g. ((0, 1), (1, 2))
    """
    qubits: Sequence[int]
    edges: Sequence[Tuple[int, int]]


@dataclass(frozen=True)
class CircuitArchitecture:
    num_qubits: int  # n
    layers: Sequence[LayerSpec]  # [(V, E_ell)]
    input_dim: int  # zeta

    def __post_init__(self):
        self.validate_layers()

    def validate_layers(self) -> None:
        """Validate that layers form a disjoint partition of [0, num_qubits)
        and that edges are well-formed intra-layer couplings.
        """
        if self.num_qubits <= 0:
            raise ValueError(f"num_qubits must be positive, got {self.num_qubits}.")

        if not self.layers:
            raise ValueError("At least one layer must be specified (layers is empty).")

        seen_qubits = set()

        for layer_idx, layer in enumerate(self.layers):
            # --- qubits checks ---
            qubits = tuple(layer.qubits)
            if not qubits:
                raise ValueError(f"Layer {layer_idx} has an empty qubit set.")

            layer_qubits = set(qubits)
            if len(layer_qubits) != len(qubits):
                raise ValueError(
                    f"Layers {layer_idx} has duplicate qubit indices: {qubits}."
                )

            # Check range of qubits
            out_of_range = [q for q in layer_qubits if not (0 <= q < self.num_qubits)]
            if out_of_range:
                raise ValueError(
                    f"Layer {layer_idx} has qubit indices out of range [0, {self.num_qubits}): "
                    f"{sorted(out_of_range)}."
                )

            seen_qubits |= layer_qubits

            # --- edges checks ---
            edges = tuple(layer.edges)
            seen_edges = set()

            for e_idx, edge in enumerate(edges):
                if len(edge) != 2:
                    raise ValueError(
                        f"Edge {edge} in layer {layer_idx} (index {e_idx}) must "
                        f"contain exactly two qubit indices."
                    )

                u, v = edge

                # No self-loops
                if u == v:
                    raise ValueError(
                        f"Self-loop edge ({u}, {v}) found in layer {layer_idx} "
                        f"(edge index {e_idx})."
                    )

                # Edge endpoints must be in this layer
                if u not in layer_qubits or v not in layer_qubits:
                    raise ValueError(
                        f"Edge ({u}, {v}) in layer {layer_idx} connects qubits not "
                        f"both in the layer's qubit set {sorted(layer_qubits)}."
                    )

                # Treat edges as undirected: normalize ordering for duplicate checks
                normalized = (u, v) if u < v else (v, u)

                if normalized in seen_edges:
                    raise ValueError(
                        f"Duplicate edge {normalized} found in layer {layer_idx}."
                    )

                seen_edges.add(normalized)

        # Enforce that layers form a full partition of all qubits
        missing = set(range(self.num_qubits)) - seen_qubits
        if missing:
            raise ValueError(
                "Layers do not cover all qubits in [0, num_qubits). "
                f"Missing qubits: {sorted(missing)}."
            )

    @property
    def num_layers(self):
        return len(self.layers)


def ring_layer(qubits: Sequence[int]) -> LayerSpec:
    # Order the sequence of qubits by indices
    # Run a (i, i+1) edges and finish by (n-1, 0) edge
    qubits = sorted(qubits)
    edges = []
    n = len(qubits)
    for i in range(n - 1):
        edges.append((qubits[i], qubits[i + 1]))
    if n > 2:
        edges.append((qubits[n - 1], qubits[0]))

    return LayerSpec(qubits=tuple(qubits), edges=tuple(edges))
