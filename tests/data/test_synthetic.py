# tests/data/test_synthetic.py

import numpy as np

from data.synthetic import (
    KernelConfig,
    make_dataset,
    _kernel_matrix,
)


def _kernel_configs():
    return [
        KernelConfig(
            name="rbf",
            kernel_type="rbf",
            params={"lengthscale": 0.3},
        ),
        KernelConfig(
            name="matern32",
            kernel_type="matern32",
            params={"lengthscale": 0.3},
        ),
        KernelConfig(
            name="nonstat_amp",
            kernel_type="nonstat_amp",
            params={
                "base_lengthscale": 0.3,
                "a": 0.3,
                "b": 0.6,
                "c": 2.0,
            },
        ),
        KernelConfig(
            name="nonstat_ls",
            kernel_type="nonstat_ls",
            params={
                "ell_min": 0.1,
                "ell_max": 0.8,
                "c": 2.0,
                "ell_rest": 0.3,
            },
        ),
    ]


def test_make_dataset_shapes_and_lengths():
    M = 150
    input_dim = 2
    cfgs = _kernel_configs()

    data = make_dataset(
        M=M,
        input_dim=input_dim,
        sampling="uniform",
        kernel_cfgs=cfgs,
        seed=123,
    )

    X = data["inputs"]
    labels = data["labels"]

    assert X.shape == (M, input_dim)
    assert isinstance(labels, list)
    assert len(labels) == len(cfgs)

    for y in labels:
        assert isinstance(y, np.ndarray)
        assert y.shape == (M,)


def test_make_dataset_reproducibility_with_seed():
    M = 50
    input_dim = 1
    cfgs = _kernel_configs()

    data1 = make_dataset(
        M=M,
        input_dim=input_dim,
        sampling="uniform",
        kernel_cfgs=cfgs,
        seed=42,
    )
    data2 = make_dataset(
        M=M,
        input_dim=input_dim,
        sampling="uniform",
        kernel_cfgs=cfgs,
        seed=42,
    )

    X1, labels1 = data1["inputs"], data1["labels"]
    X2, labels2 = data2["inputs"], data2["labels"]

    assert np.allclose(X1, X2)
    assert len(labels1) == len(labels2)

    for y1, y2 in zip(labels1, labels2):
        assert np.allclose(y1, y2)


def test_kernel_matrix_in_unit_range_and_diag_one():
    M = 40
    input_dim = 1
    cfgs = _kernel_configs()

    # Use a fixed X so we can reuse across kernels
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(M, input_dim))

    for cfg in cfgs:
        K = _kernel_matrix(X, cfg)

        # Shape
        assert K.shape == (M, M)

        # Symmetry
        assert np.allclose(K, K.T, atol=1e-8)

        # Values in [0,1]
        assert np.all(K >= -1e-8)
        assert np.all(K <= 1.0 + 1e-8)

        # Diagonal exactly 1
        assert np.allclose(np.diag(K), 1.0, atol=1e-8)
