from typing import Union

from torch.nn import ModuleList, Parameter
import torch
import torch.nn.functional as F
from torch import Tensor
from gpytorch.kernels import Kernel
from linear_operator import LinearOperator

from src.kernels.hybrid_kernel import HybridKernelQuantumFMP


class MixtureHybridKernel(Kernel):
    """
    Mixture kernel:
        k_mix(x, x') = sum_r w_r k_r(x, x')

    where each k_r is a HybridKernelQuantumFMP built from the r-th draw of FMPs.

    Args
    ----
    input_dataset : Tensor of shape (M, input_dim)
    fmps          : Tensor of shape (R, M, num_obs)
                    R = number of draws of the quantum feature map.
    """

    def __init__(self, input_dataset: Tensor, fmps: Tensor, type_kernel: str = 'linear', **kwargs) -> None:
        super().__init__(**kwargs)

        assert input_dataset.ndim == 2 and fmps.ndim == 3, (
            f"input_dataset must be (M, d), fmps must be (R, M, num_obs), "
            f"got {input_dataset.shape} and {fmps.shape}"
        )

        self.input_dataset = input_dataset
        self.fmps_all = fmps  # (R, M, num_obs)
        self.num_draws, self.num_points, self.num_obs = fmps.shape

        # Shared lookup: x ↦ index in input_dataset (same for all draws)
        keys = [tuple(row.tolist()) for row in input_dataset]
        self._idx_by_key = {key: i for i, key in enumerate(keys)}

        # One HybridKernelQuantumFMP per draw – we'll *bypass* its get_fmps and
        # call .classical_kernel directly with already-indexed fmps.
        self.base_kernels = ModuleList(
            [
                HybridKernelQuantumFMP(
                    input_dataset=input_dataset,
                    fmps=fmps[r],
                    type_kernel=type_kernel,
                    **kwargs,
                )
                for r in range(self.num_draws)
            ]
        )

        # Mixture weights on the simplex via softmax
        self.register_parameter(
            name="raw_weights",
            parameter=Parameter(torch.zeros(self.num_draws, device=input_dataset.device)),
        )

    @property
    def weights(self) -> Tensor:
        # shape: (R,)
        return F.softmax(self.raw_weights, dim=0)

    def _indices(self, x: Tensor) -> Tensor:
        # Compute indices *once* for all draws
        idx_list = [self._idx_by_key[tuple(row.tolist())] for row in x]
        return torch.tensor(idx_list, device=self.fmps_all.device, dtype=torch.long)

    def forward(
            self,
            x1: Tensor,
            x2: Tensor,
            diag: bool = False,
            last_dim_is_batch: bool = False,
            **params,
    ) -> Union[Tensor, LinearOperator]:

        # 1. Index the precomputed FMPs once
        idx1 = self._indices(x1)  # (B1,)
        idx2 = self._indices(x2)  # (B2,)

        fmps1_all = self.fmps_all[:, idx1, :]  # (R, B1, num_obs)
        fmps2_all = self.fmps_all[:, idx2, :]  # (R, B2, num_obs)

        w = self.weights  # (R,)

        # 2. Sum weighted component kernels
        cov = None
        for r, base in enumerate(self.base_kernels):
            # Call the *classical* kernel directly; no Python dict here.
            k_r = base.classical_kernel(
                fmps1_all[r],
                fmps2_all[r],
                diag=diag,
                last_dim_is_batch=last_dim_is_batch,
                **params,
            )

            # Weight this component
            if diag:
                k_r = w[r] * k_r  # Tensor * scalar
            else:
                k_r = k_r * w[r]  # LinearOperator * scalar

            cov = k_r if cov is None else cov + k_r

        return cov

    def num_outputs_per_input(self, x1, x2):
        # Same as any component kernel
        return self.base_kernels[0].num_outputs_per_input(x1, x2)
