from typing import Union, Dict

from gpytorch.constraints import GreaterThan, Positive
from gpytorch.kernels import Kernel, LinearKernel, RBFKernel, ScaleKernel, ConstantKernel
from linear_operator import LinearOperator
from torch import Tensor
import torch


class HybridKernelQuantumFMP(Kernel):
    """
    We use a mapping between an input of shape (zeta,) and its precompute_fmp of shape (num_observables,)
    """
    input_dataset: Tensor  # A tensor of shape (M, input_dim)
    fmps: Tensor  # A tensor of shape (M, num_obs)
    idx_by_key: Dict

    def __init__(self, input_dataset: Tensor, fmps: Tensor, type_kernel: str = 'linear', **kwargs):
        super().__init__(**kwargs)
        self.input_dataset = input_dataset
        self.fmps = fmps
        self.idx_by_key = {
            tuple(row.tolist()): i
            for i, row in enumerate(self.input_dataset.cpu())
        }
        if type_kernel == 'linear':
            self.classical_kernel = LinearKernel(batch_shape=self.batch_shape, lengthscale_constraint=GreaterThan(1e-6))
        elif type_kernel == 'rbf':
            self.classical_kernel = RBFKernel(batch_shape=self.batch_shape, lengthscale_constraint=GreaterThan(1e-6))
        else:
            raise NotImplementedError(f"The kernel type requested is not implemented! Got {type_kernel}")

        # self.classical_kernel = (
        #                         ScaleKernel(
        #                                 RBFKernel(batch_shape=self.batch_shape, lengthscale_constraint=GreaterThan(0)),
        #                                 batch_shape=self.batch_shape,
        #                                 outputscale_constraint=GreaterThan(0),
        #                             )
        #                         +
        #                         ScaleKernel(
        #                                 LinearKernel(batch_shape=self.batch_shape),
        #                                 batch_shape=self.batch_shape,
        #                                 outputscale_constraint=GreaterThan(0),
        #                             )
        #                         +
        #                         ConstantKernel()
        #                      )

    def get_fmps(self, x: Tensor) -> Tensor:
        # x: (B, input_dim)
        # Build list of indices via Python dict lookups
        idx_list = [self.idx_by_key[tuple(row.tolist())] for row in x]
        idx = torch.tensor(idx_list, device=self.fmps.device, dtype=torch.long)
        return self.fmps[idx]  # (B, num_obs)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params) -> Union[
        Tensor, LinearOperator]:
        # Asser that
        fmps1 = self.get_fmps(x1)
        fmps2 = self.get_fmps(x2)

        return self.classical_kernel(fmps1, fmps2, diag, last_dim_is_batch, **params)
