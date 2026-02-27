from typing import Dict, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def draw_from_inference(
    data: Dict[str, np.ndarray],
    means: np.ndarray,
    vars_: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    kernel_cfgs: Sequence = None,
) -> Figure:
    """
    Visualise GP regression results from an MKL / SoftmaxMKL learner
    in 1D, with x-axis = input value (sorted), and predictions only
    on the test region (right part in x if split is contiguous).

    Parameters
    ----------
    data : dict
        Output of `make_dataset`, must contain:
            - "inputs": (M, 1)
            - "labels": (num_gps, M)
    means : np.ndarray
        Predictive means on the *test* set, shape (num_gps, M_test),
        ordered like `test_idx`.
    vars_ : np.ndarray
        Predictive variances on the *test* set, same shape as `means`.
    train_idx : np.ndarray
        1D array of training indices (w.r.t. the original dataset).
    test_idx : np.ndarray
        1D array of test indices (w.r.t. the original dataset).
    kernel_cfgs : sequence, optional
        If provided, used for subplot titles via `kernel_cfgs[g].name`.

    Returns
    -------
    fig : Figure
        One subplot per GP, showing:
        - blue curve: true GP sample (sorted by x)
        - blue dots: training observations
        - pale blue dots: true test observations
        - orange curve: predictive mean on test inputs
        - grey band: ±2σ on test inputs.
    """
    X = np.asarray(data["inputs"])
    Y = np.asarray(data["labels"])
    num_gps, M = Y.shape

    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError(
            f"Expected X of shape (M, 1) for 1D plotting, got {X.shape}"
        )

    x = X[:, 0]
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    M_test = test_idx.shape[0]
    if means.shape != (num_gps, M_test):
        raise ValueError(
            f"`means` shape should be (num_gps, M_test)=({num_gps}, {M_test}), "
            f"got {means.shape}"
        )
    if vars_.shape != (num_gps, M_test):
        raise ValueError(
            f"`vars_` shape should be (num_gps, M_test)=({num_gps}, {M_test}), "
            f"got {vars_.shape}"
        )

    # ---- Global sort by x so curves look like your GP plots ----
    order_all = np.argsort(x)
    x_sorted = x[order_all]

    # Map original index -> position in sorted array
    pos_in_sorted = np.empty(M, dtype=int)
    pos_in_sorted[order_all] = np.arange(M)

    # Boolean mask (in sorted order) of which points are test
    is_test_sorted = np.zeros(M, dtype=bool)
    is_test_sorted[pos_in_sorted[test_idx]] = True

    # For predictions we also want test inputs sorted by x
    x_test = x[test_idx]
    order_test = np.argsort(x_test)             # order within test set
    x_test_sorted = x_test[order_test]          # (M_test,)

    fig, axes = plt.subplots(
        nrows=num_gps,
        ncols=1,
        figsize=(6, 2.8 * num_gps),
        sharex=True,
    )
    if num_gps == 1:
        axes = [axes]

    for g in range(num_gps):
        ax = axes[g]

        y_full = Y[g]

        # full GP sample, sorted by x
        y_sorted = y_full[order_all]
        ax.plot(
            x_sorted,
            y_sorted,
            color="C0",
            alpha=0.4,
            lw=1.0,
            label="true GP",
        )

        # training points (blue dots), in sorted x
        train_sorted_mask = ~is_test_sorted
        ax.scatter(
            x_sorted[train_sorted_mask],
            y_sorted[train_sorted_mask],
            color="C0",
            s=15,
            alpha=0.9,
            label="train obs",
        )

        # true test points (faint blue dots), in sorted test x
        y_test = y_full[test_idx]
        y_test_sorted = y_test[order_test]
        ax.scatter(
            x_test_sorted,
            y_test_sorted,
            color="C0",
            s=15,
            alpha=0.3,
            label="test obs (true)",
        )

        # predictive mean + ±2σ, aligned with sorted test x
        mean_test_sorted = means[g, order_test]
        std_test_sorted = np.sqrt(np.maximum(vars_[g, order_test], 0.0))

        ax.plot(
            x_test_sorted,
            mean_test_sorted,
            color="tab:orange",
            lw=2.0,
            label="predictive mean",
        )
        ax.fill_between(
            x_test_sorted,
            mean_test_sorted - 2.0 * std_test_sorted,
            mean_test_sorted + 2.0 * std_test_sorted,
            color="0.7",
            alpha=0.4,
            label="±2σ",
        )

        # Title
        if kernel_cfgs is not None and len(kernel_cfgs) > g:
            title = getattr(kernel_cfgs[g], "name", f"GP {g}")
        else:
            title = f"GP {g}"
        ax.set_title(title)
        ax.set_ylabel("y")

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(
            uniq.values(), uniq.keys(), fontsize=8, loc="best"
        )

    axes[-1].set_xlabel("x₁ (sorted)")
    fig.tight_layout()
    return fig


def draw_test_only_predictions(
    data: Dict[str, np.ndarray],
    means: np.ndarray,
    vars_: np.ndarray,
    test_idx: np.ndarray,
    kernel_names: Sequence = None,
) -> Figure:
    """
    Plot test-set predictions only, for 1D inputs, arranged in a
    rectangular grid of subplots (one subplot per GP/kernel).

    Parameters
    ----------
    data : dict
        Output of `make_dataset`, with:
          - "inputs": shape (M, 1)
          - "labels": shape (num_gps, M)
    means : np.ndarray
        Predictive means on the *test* set, shape (num_gps, M_test),
        where M_test = len(test_idx). Order must match `test_idx`.
    vars_ : np.ndarray
        Predictive variances on the *test* set, same shape as `means`.
    test_idx : np.ndarray
        1D array of indices (w.r.t. the original dataset) used as test points.
    kernel_names : sequence, optional
        If provided, used for subplot titles via `kernel_names[g]`.

    Returns
    -------
    fig : Figure
        Rectangular grid of subplots, one per GP:
          - blue dots: true test observations
          - orange curve: predictive mean
          - grey band: ±2σ confidence interval
    """
    X = np.asarray(data["inputs"])
    Y = np.asarray(data["labels"])

    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError(f"Expected X of shape (M, 1), got {X.shape}")

    num_gps, M = Y.shape

    test_idx = np.asarray(test_idx, dtype=int)
    M_test = test_idx.shape[0]

    if means.shape != (num_gps, M_test):
        raise ValueError(
            f"`means` must have shape (num_gps, M_test)=({num_gps}, {M_test}), "
            f"got {means.shape}"
        )
    if vars_.shape != (num_gps, M_test):
        raise ValueError(
            f"`vars_` must have shape (num_gps, M_test)=({num_gps}, {M_test}), "
            f"got {vars_.shape}"
        )

    # Extract test inputs/outputs
    x_test = X[test_idx, 0]       # (M_test,)
    y_test = Y[:, test_idx]       # (num_gps, M_test)

    # Sort test points by x for nice curves
    order_test = np.argsort(x_test)
    x_test_sorted = x_test[order_test]

    # ---- choose a "rectangular" grid layout ----
    n_cols = int(np.ceil(np.sqrt(num_gps)))
    n_rows = int(np.ceil(num_gps / n_cols))

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4.0 * n_cols, 3.0 * n_rows),
        sharex=False,
    )

    # axes can be scalar, 1D, or 2D depending on n_rows/n_cols
    axes = np.atleast_1d(axes).reshape(-1)

    for g in range(num_gps):
        ax = axes[g]

        y_test_sorted = y_test[g, order_test]
        mean_sorted = means[g, order_test]
        std_sorted = np.sqrt(np.maximum(vars_[g, order_test], 0.0))

        # True test labels (blue dots)
        ax.scatter(
            x_test_sorted,
            y_test_sorted,
            color="C0",
            s=18,
            alpha=0.8,
            label="test obs (true)",
        )

        # Predictive mean (orange)
        ax.plot(
            x_test_sorted,
            mean_sorted,
            color="tab:orange",
            lw=2.0,
            label="predictive mean",
        )

        # ±2σ band (grey)
        ax.fill_between(
            x_test_sorted,
            mean_sorted - 2.0 * std_sorted,
            mean_sorted + 2.0 * std_sorted,
            color="0.7",
            alpha=0.4,
            label="±2σ",
        )

        # Title
        if kernel_names is not None and len(kernel_names) > g:
            title = str(kernel_names[g])
        else:
            title = f"GP {g}"
        ax.set_title(title)
        ax.set_xlabel("x (test points)")
        ax.set_ylabel("y")

        # Deduplicated legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(
            uniq.values(),
            uniq.keys(),
            fontsize=8,
            loc="best",
        )

    # Hide any unused axes (e.g. for 3 GPs in a 2×2 grid)
    for k in range(num_gps, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    return fig



def draw_test_only_mean_predictions(
    data: Dict[str, np.ndarray],
    means: np.ndarray,
    test_idx: np.ndarray,
    kernel_cfgs: Sequence = None,
) -> Figure:
    """
    Plot TEST-ONLY predictions for 1D inputs, without confidence intervals,
    arranged in a rectangular grid of subplots (one subplot per GP/kernel).

    Parameters
    ----------
    data : dict
        Output of `make_dataset`, with:
          - "inputs": shape (M, 1)
          - "labels": shape (num_gps, M)
    means : np.ndarray
        Predictive means on the *test* set, shape (num_gps, M_test),
        where M_test = len(test_idx). Order must match `test_idx`.
    test_idx : np.ndarray
        1D array of indices (w.r.t. the original dataset) used as test points.
    kernel_cfgs : sequence, optional
        If provided, used for subplot titles via `kernel_cfgs[g].name`.

    Returns
    -------
    fig : Figure
        Rectangular grid of subplots, one per GP:
          - blue dots: true test observations
          - orange line: predictive mean
    """
    X = np.asarray(data["inputs"])
    Y = np.asarray(data["labels"])

    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError(f"Expected X of shape (M, 1), got {X.shape}")

    num_gps, M = Y.shape

    test_idx = np.asarray(test_idx, dtype=int)
    M_test = test_idx.shape[0]

    if means.shape != (num_gps, M_test):
        raise ValueError(
            f"`means` must have shape (num_gps, M_test)=({num_gps}, {M_test}), "
            f"got {means.shape}"
        )

    # Extract test inputs/outputs
    x_test = X[test_idx, 0]         # (M_test,)
    y_test = Y[:, test_idx]         # (num_gps, M_test)

    # Sort test points by x for nice curves
    order_test = np.argsort(x_test)
    x_test_sorted = x_test[order_test]

    # ---- choose a "rectangular" grid layout ----
    n_cols = int(np.ceil(np.sqrt(num_gps)))
    n_rows = int(np.ceil(num_gps / n_cols))

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4.0 * n_cols, 3.0 * n_rows),
        sharex=False,
    )

    # axes can be scalar, 1D, or 2D depending on n_rows/n_cols
    axes = np.atleast_1d(axes).reshape(-1)

    for g in range(num_gps):
        ax = axes[g]

        y_test_sorted = y_test[g, order_test]
        mean_sorted = means[g, order_test]

        # True test labels (blue dots)
        ax.scatter(
            x_test_sorted,
            y_test_sorted,
            color="C0",
            s=18,
            alpha=0.9,
            label="test obs (true)",
        )

        # Predictive mean (orange)
        ax.plot(
            x_test_sorted,
            mean_sorted,
            color="tab:orange",
            lw=2.0,
            label="predictive mean",
        )

        # Title
        if kernel_cfgs is not None and len(kernel_cfgs) > g:
            title = getattr(kernel_cfgs[g], "name", f"GP {g}")
        else:
            title = f"GP {g}"
        ax.set_title(title)
        ax.set_xlabel("x (test points)")
        ax.set_ylabel("y")

        # Deduplicated legend
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(
            uniq.values(),
            uniq.keys(),
            fontsize=8,
            loc="best",
        )

    # Hide any unused axes (e.g. 2×2 grid with 3 GPs)
    for k in range(num_gps, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    return fig


def draw_kernel_heatmap_comparison(
    true_K: np.ndarray,
    learned_K: np.ndarray,
    gp_name: str = "rbf",
    cmap: str = "viridis",
    show_axes_ticks: bool = False,
) -> Figure:
    """
    Plot a 2-column heatmap comparison between a true kernel matrix and a
    learned kernel matrix.

    Parameters
    ----------
    true_K : np.ndarray
        Reference (ground-truth) kernel matrix of shape (M, M).
    learned_K : np.ndarray
        Learned kernel matrix of shape (M, M).
    gp_name : str, optional
        Name of the GP/kernel to use in titles.
    cmap : str, optional
        Matplotlib colormap name.
    show_axes_ticks : bool, optional
        If False (default), hide x/y ticks for a cleaner look.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    assert true_K.shape == learned_K.shape, "true_K and learned_K must have same shape"
    assert true_K.ndim == 2 and true_K.shape[0] == true_K.shape[1], "K must be square"

    M = true_K.shape[0]

    # Use common color scale so visual differences are meaningful
    vmin = float(min(true_K.min(), learned_K.min()))
    vmax = float(max(true_K.max(), learned_K.max()))

    fig, axes = plt.subplots(
        1, 2, figsize=(8, 3.2), constrained_layout=True
    )  # 2 columns

    # Left: true kernel
    im0 = axes[0].imshow(
        true_K,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
    )
    axes[0].set_title(f"Ground truth {gp_name}")
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("index")

    # Right: learned kernel
    im1 = axes[1].imshow(
        learned_K,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
    )
    axes[1].set_title(f"Learned {gp_name}")
    axes[1].set_xlabel("index")
    axes[1].set_ylabel("index")

    if not show_axes_ticks:
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

    # Single colorbar for both
    cbar = fig.colorbar(
        im1,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        label="kernel value",
    )

    return fig