from synthetic import KernelConfig

def default_kernel_configs() -> list[KernelConfig]:
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
