import numpy as np

from src.models.mk_learner import MKLearner


# -------------------------------------------------------------------
# Dummy architecture for MKLearner tests
# -------------------------------------------------------------------


class DummyBlock:
    def __init__(self):
        pass


class DummyArch:
    """
    Minimal stand-in for CircuitArchitecture.

    MKLearner only needs `len(arch.blocks)` in __post_init__,
    so this is enough for testing.
    """

    def __init__(self, num_blocks: int):
        self.blocks = [DummyBlock() for _ in range(num_blocks)]


def make_random_psd_grams(R: int, B: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate random symmetric PSD Gram matrices of shape (R, B, M, M).
    """
    grams = np.zeros((R, B, M, M), dtype=float)
    for r in range(R):
        for b in range(B):
            A = rng.normal(size=(M, M))
            G = A @ A.T  # PSD
            G += 1e-3 * np.eye(M)  # make strictly PD
            grams[r, b] = G
    return grams


# -------------------------------------------------------------------
# Existing tests (unchanged)
# -------------------------------------------------------------------


def test_mk_learner_parallel_matches_sequential():
    """
    Check that parallel fit (n_jobs > 1) yields the same weights,
    log-marginals and predictions as sequential fit.
    """
    rng = np.random.default_rng(0)

    R = 3
    B = 2
    M = 8
    num_gps = 4
    input_dim = 2

    grams = make_random_psd_grams(R, B, M, rng)
    X = rng.normal(size=(M, input_dim))
    Y = rng.normal(size=(num_gps, M))

    arch = DummyArch(num_blocks=B)

    mkl_seq = MKLearner(
        arch=arch,
        grams=grams,
        X=X,
        Y=Y,
        train_test_split=0.6,
        # force exact gradient backend so this test only probes
        # the parallelism and not approximation noise
        use_approx_grad=False,
    )

    mkl_par = MKLearner(
        arch=arch,
        grams=grams,
        X=X,
        Y=Y,
        train_test_split=0.6,
        use_approx_grad=False,
    )

    # Sequential fit
    mkl_seq.fit(max_iter=10, tol=1e-6, verbose=False, n_jobs=1)

    # Parallel fit with 2 workers
    mkl_par.fit(max_iter=10, tol=1e-6, verbose=False, n_jobs=2)

    # Weights and log-marginals should match
    np.testing.assert_allclose(mkl_seq.weights, mkl_par.weights, atol=1e-8)
    np.testing.assert_allclose(
        mkl_seq.train_log_marginals_, mkl_par.train_log_marginals_, atol=1e-8
    )

    # Posterior means / variances should match as well
    means_seq, vars_seq = mkl_seq.infer()
    means_par, vars_par = mkl_par.infer()

    np.testing.assert_allclose(means_seq, means_par, atol=1e-8)
    np.testing.assert_allclose(vars_seq, vars_par, atol=1e-8)


def test_mk_learner_parallel_verbose_no_mixing(capsys):
    """
    Check that verbose output in parallel mode is grouped per-GP, i.e.
    log lines for GP 0 come first, then GP 1, etc., with no interleaving.
    """
    rng = np.random.default_rng(123)

    R = 2
    B = 2
    M = 6
    num_gps = 3
    input_dim = 1

    grams = make_random_psd_grams(R, B, M, rng)
    X = rng.normal(size=(M, input_dim))
    Y = rng.normal(size=(num_gps, M))

    arch = DummyArch(num_blocks=B)

    mkl = MKLearner(
        arch=arch,
        grams=grams,
        X=X,
        Y=Y,
        train_test_split=0.5,
        use_approx_grad=False,
    )

    mkl.fit(max_iter=5, tol=1e-4, verbose=True, n_jobs=3)

    out = capsys.readouterr().out
    lines = out.splitlines()

    # Extract GP indices from lines of the form "[MKL] [GP k] ..."
    gp_sequence = []
    for line in lines:
        if "[GP" in line:
            start = line.find("[GP") + 4
            end = line.find("]", start)
            if end == -1:
                continue
            gp_str = line[start:end].strip()
            try:
                gp_idx = int(gp_str)
            except ValueError:
                continue
            gp_sequence.append(gp_idx)

    # We should have seen all GPs at least once
    assert set(gp_sequence) == set(range(num_gps))

    # The sequence of GPs in logs should be non-decreasing, i.e. blocks
    # of GP0 lines, then GP1 lines, then GP2 lines, but never 0,1,0, ...
    for i in range(1, len(gp_sequence)):
        assert gp_sequence[i] >= gp_sequence[i - 1]


# -------------------------------------------------------------------
# New tests: approximate gradients vs exact backend
# -------------------------------------------------------------------


def test_log_marginal_and_grad_approx_close_to_exact_for_large_rank_and_probes():
    """
    Directly probe the gradient backend:

    - Build a tiny problem.
    - Compute exact log-marginal + gradient at a fixed weight vector.
    - Compute approximate log-marginal + gradient with:
        * low-rank factors of full rank on the training set
        * a reasonably large number of Hutchinson probes
    - Check that the log-marginal is identical (within numerical noise)
      and that the approximate gradient is close to the exact one.
    """
    rng = np.random.default_rng(42)

    R = 2
    B = 2
    M = 10
    num_gps = 1
    input_dim = 1

    grams = make_random_psd_grams(R, B, M, rng)
    X = rng.normal(size=(M, input_dim))
    Y = rng.normal(size=(num_gps, M))

    arch = DummyArch(num_blocks=B)

    # Exact backend
    mkl_exact = MKLearner(
        arch=arch,
        grams=grams,
        X=X,
        Y=Y,
        train_test_split=0.7,
        use_approx_grad=False,
    )

    # Determine training size for full-rank approximation
    M_tr = mkl_exact.train_idx.size
    E = mkl_exact.R * mkl_exact.B

    # Approximate backend with full rank and many probes
    mkl_approx = MKLearner(
        arch=arch,
        grams=grams,
        X=X,
        Y=Y,
        train_test_split=0.7,
        use_approx_grad=True,
        approx_rank=M_tr,            # full rank
        n_hutchinson_probes=64,      # many probes
        approx_seed=1234,
    )

    g = 0
    y_train = mkl_exact.Y[g, mkl_exact.train_idx]
    K_tr = mkl_exact._K_components_train
    # slightly random, but projected, simplex weight vector
    v = rng.normal(size=E)
    w = mkl_exact._project_simplex(v)

    # exact vs approximate backends at the same point
    L_exact, grad_exact, _, _ = mkl_exact._log_marginal_and_grad_exact(
        y_train, K_tr, w
    )
    L_approx, grad_approx, _, _ = mkl_approx._log_marginal_and_grad_approx(
        y_train, K_tr, w
    )

    # log-marginal is computed from the exact K_w in both backends
    assert np.allclose(L_exact, L_approx, atol=1e-8)

    # Gradients should be close in relative norm (Hutchinson noise only)
    grad_norm = np.linalg.norm(grad_exact)
    rel_err = np.linalg.norm(grad_approx - grad_exact) / (grad_norm + 1e-12)

    # This threshold is intentionally not ultra-tight to avoid flakiness,
    # but should detect major issues in the implementation.
    assert rel_err < 0.25, f"Relative gradient error too large: {rel_err:.3f}"


def test_fit_with_large_rank_and_probes_matches_exact_solution():
    """
    End-to-end check at the *optimization* level:

    - Fit MKLearner with exact gradients.
    - Fit MKLearner with approximate gradients using full-rank factors
      and a reasonably large number of Hutchinson probes.
    - Compare:
        * train log-marginals per GP,
        * learned weights (in a soft sense).

    We do **not** enforce that mixture kernels or predictions match
    pointwise: because the log-marginal can be fairly flat over the
    simplex, different weight vectors can yield very similar objectives
    and predictions. This test is only meant to ensure that the
    approximate-gradient backend is good enough to drive the optimizer
    towards a comparable region of the weight simplex.
    """
    rng = np.random.default_rng(123)

    R = 2
    B = 2
    M = 12
    num_gps = 2
    input_dim = 1

    grams = make_random_psd_grams(R, B, M, rng)
    X = rng.normal(size=(M, input_dim))
    Y = rng.normal(size=(num_gps, M))

    arch = DummyArch(num_blocks=B)

    # Exact learner
    mkl_exact = MKLearner(
        arch=arch,
        grams=grams,
        X=X,
        Y=Y,
        train_test_split=0.75,
        use_approx_grad=False,
    )

    # Approximate learner with "large" approximations
    M_tr = mkl_exact.train_idx.size
    E = mkl_exact.R * mkl_exact.B

    mkl_approx = MKLearner(
        arch=arch,
        grams=grams,
        X=X,
        Y=Y,
        train_test_split=0.75,
        use_approx_grad=True,
        approx_rank=M_tr,      # full rank on train set
        n_hutchinson_probes=64,
        approx_seed=2024,
    )

    # Fit both (limit iterations to keep tests reasonably fast)
    mkl_exact.fit(max_iter=50, tol=1e-5, verbose=False, n_jobs=1)
    mkl_approx.fit(max_iter=50, tol=1e-5, verbose=False, n_jobs=1)

    # ---- Compare train log-marginals ----
    ll_exact = mkl_exact.train_log_marginals_
    ll_approx = mkl_approx.train_log_marginals_
    rel_ll = np.abs(ll_exact - ll_approx) / (np.abs(ll_exact) + 1e-12)
    max_rel_ll = float(np.max(rel_ll))

    # Allow up to 10% max relative difference in log-marginals.
    assert max_rel_ll < 0.10, (
        f"Train log-marginals differ too much: "
        f"max rel diff = {max_rel_ll:.3f}, per GP = {rel_ll}"
    )

    # ---- Compare weights (very soft check) ----
    w_exact = mkl_exact.weights.reshape(num_gps, E)
    w_approx = mkl_approx.weights.reshape(num_gps, E)

    weight_l1 = np.sum(np.abs(w_exact - w_approx), axis=1)
    min_weight_l1 = float(np.min(weight_l1))

    # We only require that at least one GP has reasonably close weights;
    # for the others, degeneracy of the optimum can lead to very different
    # weight vectors with similar objectives.
    assert min_weight_l1 < 0.5, (
        f"No GP has closely matching weights; L1 distances: {weight_l1}"
    )


# -------------------------------------------------------------------
# New tests: persistence API
# -------------------------------------------------------------------


def test_mk_learner_persistence_roundtrip(tmp_path):
    """
    Check that MKLearner.save / MKLearner.load preserve:

      - learned weights
      - train log-marginals
      - approximation structures (when use_approx_grad=True)
      - posterior predictions

    We use the approximate backend here on purpose to ensure that the
    precomputed Phi / Hutchinson structures are correctly pickled.
    """
    rng = np.random.default_rng(999)

    R = 2
    B = 2
    M = 10
    num_gps = 3
    input_dim = 2

    grams = make_random_psd_grams(R, B, M, rng)
    X = rng.normal(size=(M, input_dim))
    Y = rng.normal(size=(num_gps, M))

    arch = DummyArch(num_blocks=B)

    learner = MKLearner(
        arch=arch,
        grams=grams,
        X=X,
        Y=Y,
        train_test_split=0.6,
        use_approx_grad=True,
        approx_rank=None,         # let it auto-choose
        n_hutchinson_probes=16,
        approx_seed=321,
    )

    learner.fit(max_iter=30, tol=1e-5, verbose=False, n_jobs=2)

    weights_before = learner.weights.copy()
    train_ll_before = learner.train_log_marginals_.copy()
    means_before, vars_before = learner.infer()

    # Also record basic approximation metadata
    approx_rank_eff_before = learner.approx_rank_effective
    has_phi_before = learner._Phi_components_train is not None
    hutch_shape_before = (
        None if learner._hutchinson_Z is None else learner._hutchinson_Z.shape
    )

    # Save to a temporary file
    path = tmp_path / "mkl_learner.pkl"
    learner.save(str(path))

    # Load back
    loaded = MKLearner.load(str(path))

    assert isinstance(loaded, MKLearner)

    # Check core quantities are preserved
    np.testing.assert_allclose(weights_before, loaded.weights, atol=1e-12)
    np.testing.assert_allclose(train_ll_before, loaded.train_log_marginals_, atol=1e-12)

    means_after, vars_after = loaded.infer()
    np.testing.assert_allclose(means_before, means_after, atol=1e-12)
    np.testing.assert_allclose(vars_before, vars_after, atol=1e-12)

    # Approximation metadata also preserved
    assert loaded.approx_rank_effective == approx_rank_eff_before
    assert (loaded._Phi_components_train is not None) == has_phi_before

    if hutch_shape_before is None:
        assert loaded._hutchinson_Z is None
    else:
        assert loaded._hutchinson_Z is not None
        assert loaded._hutchinson_Z.shape == hutch_shape_before


def test_prepare_approximations_parallel_matches_sequential():
    """
    Check that prepare_approximations() produces identical results when
    run with a single worker (sequential) and with multiple workers
    (ProcessPoolExecutor), provided the same random seed and settings.

    We compare:
      - effective approx_rank
      - Phi_components_train (low-rank factors)
      - Hutchinson probes Z
      - KEZ = K_e Z approximations
    """
    rng = np.random.default_rng(2025)

    R = 3
    B = 2
    M = 10
    num_gps = 1
    input_dim = 1

    grams = make_random_psd_grams(R, B, M, rng)
    X = rng.normal(size=(M, input_dim))
    Y = rng.normal(size=(num_gps, M))

    arch = DummyArch(num_blocks=B)

    approx_rank = 5
    n_probes = 8
    seed = 1234

    # --- Sequential learner: n_workers = 1 ---
    mkl_seq = MKLearner(
        arch=arch,
        grams=grams,
        X=X,
        Y=Y,
        train_test_split=0.7,
        use_approx_grad=True,
        approx_rank=approx_rank,
        n_hutchinson_probes=n_probes,
        approx_seed=seed,
        n_workers=1,   # force sequential in prepare_approximations
    )

    # --- Parallel learner: n_workers > 1 ---
    mkl_par = MKLearner(
        arch=arch,
        grams=grams,
        X=X,
        Y=Y,
        train_test_split=0.7,
        use_approx_grad=True,
        approx_rank=approx_rank,
        n_hutchinson_probes=n_probes,
        approx_seed=seed,
        n_workers=4,   # allow parallel path (capped by E)
    )

    # Basic sanity
    assert mkl_seq.approx_rank_effective == mkl_par.approx_rank_effective
    assert mkl_seq._Phi_components_train is not None
    assert mkl_par._Phi_components_train is not None

    # Same shapes
    assert mkl_seq._Phi_components_train.shape == mkl_par._Phi_components_train.shape
    assert mkl_seq._hutchinson_Z.shape == mkl_par._hutchinson_Z.shape
    assert mkl_seq._KEZ.shape == mkl_par._KEZ.shape

    # Phi factors must match
    np.testing.assert_allclose(
        mkl_seq._Phi_components_train,
        mkl_par._Phi_components_train,
        atol=1e-10,
    )

    # Hutchinson probes must be identical (same approx_seed)
    np.testing.assert_allclose(
        mkl_seq._hutchinson_Z,
        mkl_par._hutchinson_Z,
        atol=0.0,
    )

    # KEZ = K_e Z approximations must match
    np.testing.assert_allclose(
        mkl_seq._KEZ,
        mkl_par._KEZ,
        atol=1e-10,
    )
