from typing import Optional, List, Dict
from math import pi, log2, ceil

import numpy as np
from scipy.stats import qmc  # Sobol engine

from qiskit import QuantumCircuit

from src.circuits.param_sampling.base_sampling import ParameterSamplingStrategy


def _map_point_to_assignment(
        qc: QuantumCircuit,
        layout: Dict,
        u: np.ndarray,
) -> Dict[str, float]:
    """
    Map a single Sobol point u ∈ [0, 1)^dim to a dict {param_name: value}.

    Uses the layout returned by _build_layout and applies the
    reparameterizations described in Sec. IV-B:
      - ξ_ℓ: direction from ζ coords, radius from one coord.
      - δ_* ∈ (−1, 1), others ∈ (−π, π).
    """
    assignment: Dict[str, float] = {}

    zeta = layout["zeta"]
    xi_by_layer = layout["xi_by_layer"]
    layer_ids = layout["layer_ids"]
    other_params = layout["other_params"]
    dim = layout["dim"]

    if len(u) != dim:
        raise ValueError(
            f"Sobol point dimension {len(u)} does not match layout dim={dim}."
        )

    idx = 0

    # --- 1) Handle ξ_ℓ blocks: (ζ coords for direction, 1 coord for radius) ---
    for ell in layer_ids:
        # Direction in ℝ^ζ from ζ coordinates in (0, 1)
        if zeta > 0:
            v_raw = u[idx: idx + zeta]
            idx += zeta

            # Map to [-1, 1]^ζ so v has both signs
            v = 2.0 * v_raw - 1.0
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                v = np.ones_like(v)
                norm_v = np.linalg.norm(v)
            unit_dir = v / norm_v
        else:
            unit_dir = np.array([])

        # Radius r_ℓ ∈ (0, π), using one extra coordinate
        u_radius = u[idx]
        idx += 1
        # Sobol coordinate is in [0,1); map to (0, π) and slightly
        # shrink to avoid hitting exactly π in any implementation.
        radius = pi * (u_radius ** (1.0 / max(1, zeta)))  # (0, π)

        xi_vec = radius * unit_dir  # ‖ξ_ℓ‖ < π

        for name, j in xi_by_layer[ell]:
            if j < 0 or j >= zeta:
                raise ValueError(
                    f"xi index j={j} out of bounds for input_dim={zeta}"
                )
            assignment[name] = float(xi_vec[j])

    # --- 2) Handle all other (non-ξ, non-data) parameters ---
    for name in other_params:
        u_val = u[idx]
        idx += 1

        if name.startswith("delta_"):
            # δ ∈ (−1, 1)
            assignment[name] = float(2.0 * u_val - 1.0)
        else:
            # α, β, ω, and other non-data scalars ∈ (−π, π)
            assignment[name] = float((2.0 * u_val - 1.0) * pi)

    if idx != dim:
        raise RuntimeError(
            f"Sobol mapping consumed {idx} dimensions, expected dim={dim}."
        )

    return assignment


class SobolParameterSamplingStrategy(ParameterSamplingStrategy):
    """
    Sobol space-filling sampling strategy on the parameter hypercube, as in
    Sec. IV-B of the paper:

      • We build a global parameter vector ϑ ∈ ℝ^η by stacking:
          - generator parameters {δ_q^(ℓ)} with |δ_q^(ℓ)| < 1,
          - projection vectors {ξ_ℓ ∈ ℝ^ζ : ‖ξ_ℓ‖ < π},
          - unitary angles {α_q^(ℓ), β_{q,q′}^(ℓ), ω_q^(ℓ)} ∈ (−π, π).

      • For a given circuit, we construct a layout of all non-data parameters,
        define a Sobol dimension d, and draw points u ∈ [0, 1)^d.

      • Component-wise reparameterization:
          - ξ_ℓ: use ζ coordinates for a direction in ℝ^ζ, one extra
            coordinate for a radius r_ℓ ∈ (0, π), and set
                  ξ_ℓ = r_ℓ * v / ‖v‖
            so that ‖ξ_ℓ‖ < π.
          - δ_*: map u ↦ 2u − 1 ∈ (−1, 1).
          - α_*, β_*, ω_* and all other non-data params: map
            u ↦ (2u − 1)π ∈ (−π, π).

      • x[*] data parameters are *not* included; they’re bound separately.
    """

    def __post_init__(self):
        # We don't need a persistent RNG here, but we keep the seed
        # from the base dataclass for reproducibility.
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_layout(self, qc: QuantumCircuit) -> Dict:
        """
        Inspect the circuit parameters and build a deterministic layout
        for sampling:

          - Group xi_* parameters layer-wise: {ℓ -> [(name, j), ...]}
          - Collect all other non-data parameters in a fixed order.
          - Compute the Sobol dimension:
                dim = #layers_with_xi * (ζ + 1) + #other_params
        """
        zeta = self.circuit_cfg.input_dim

        xi_by_layer: Dict[int, List[tuple[str, int]]] = {}
        other_params: List[str] = []

        for p in qc.parameters:
            name = p.name

            # Skip data parameters x[*]: user will bind actual inputs later
            if name.startswith("x["):
                continue

            # Group ξ parameters by layer ℓ, based on "xi_<ell>[j]"
            if name.startswith("xi_"):
                prefix, rest = name.split("[", 1)  # "xi_<ell>", "j]"
                ell_str = prefix.split("_", 1)[1]  # part after "xi_"
                ell = int(ell_str)
                j_str = rest.split("]", 1)[0]
                j = int(j_str)
                xi_by_layer.setdefault(ell, []).append((name, j))
            else:
                # All other non-data parameters: deltas, α, β, ω, ...
                other_params.append(name)

        # Sort layers so mapping is deterministic
        layer_ids = sorted(xi_by_layer.keys())

        # Sanity check: each layer that has ξ parameters should have ζ entries
        for ell in layer_ids:
            if len(xi_by_layer[ell]) != zeta:
                raise ValueError(
                    f"Layer ℓ={ell} has {len(xi_by_layer[ell])} xi-params, "
                    f"but input_dim=ζ={zeta}."
                )

        num_layers_with_xi = len(layer_ids)

        # For each layer with ξ_ℓ we use (ζ coords for direction + 1 coord for radius)
        dim = num_layers_with_xi * (zeta + 1) + len(other_params)

        return {
            "zeta": zeta,
            "xi_by_layer": xi_by_layer,
            "layer_ids": layer_ids,
            "other_params": other_params,
            "dim": dim,
        }

    # ------------------------------------------------------------------
    # Public API (mirrors UniformParameterSamplingStrategy)
    # ------------------------------------------------------------------
    def sample_random_assignment(
            self,
            qc: QuantumCircuit,
            seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Sample a single parameter assignment using one Sobol point.

        This simply calls sample_R_random_assignments with num_draws=1.
        """
        return self.sample_R_random_assignments(qc, num_draws=1, seed=seed)[0]

    def sample_R_random_assignments(
            self,
            qc: QuantumCircuit,
            num_draws: int,
            seed: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Sample `num_draws` parameter assignments using a Sobol design
        on [0, 1)^dim, then reparameterize to the constrained domain.

        - We build a Sobol engine in dimension = dim(layout) for this circuit.
        - To preserve good net properties, we draw 2^m points for
          m = ceil(log2(num_draws)) and only keep the first num_draws.
        """
        if num_draws <= 0:
            raise ValueError(f"num_draws must be positive, got {num_draws}.")

        layout = self._build_layout(qc)
        dim = layout["dim"]

        # If no non-data parameters, just return empty dicts
        if dim == 0:
            return [{} for _ in range(num_draws)]

        # Use the sampler's own seed if none is provided
        if seed is None:
            seed = self.seed

        # Construct a fresh Sobol engine for this (dim, seed) pair
        engine = qmc.Sobol(d=dim, scramble=True, seed=seed)

        # Use 2^m points to retain low-discrepancy guarantees
        m = int(ceil(log2(num_draws)))
        sobol_points = engine.random_base2(m=m)
        sobol_points = sobol_points[:num_draws, :]

        assignments: List[Dict[str, float]] = []
        for u in sobol_points:
            assignments.append(_map_point_to_assignment(qc, layout, u))

        return assignments
