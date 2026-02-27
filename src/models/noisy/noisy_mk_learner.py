from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import numpy as np

from src.circuits.configs import CircuitArchitecture
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os

import math


def _fit_single_gp_worker(args):
    """
    Worker to fit MKL weights for a single GP index g.

    This is designed to be used with ThreadPoolExecutor. It does NOT
    modify the learner instance; it only reads from it and returns the
    results (including a list of log lines if verbose).
    """
    learner, g, max_iter, tol, verbose = args

    # Basic shapes / helpers
    E = learner.R * learner.B
    y_train = learner.Y[g, learner.train_idx]
    K_tr = learner._K_components_train  # (E, M_tr, M_tr)

    # Uniform initialization on simplex
    w_init = np.full(E, 1.0 / E, dtype=float)

    # Collect logs in-memory; the main thread will print them later.
    logs: list[str] = []

    w_opt, L_opt, alpha_opt, L_chol_opt = learner._optimize_weights_for_gp(
        y_train=y_train,
        K_components_train=K_tr,
        w_init=w_init,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        logger=logs.append,
        gp_index=g,
    )

    return g, w_opt, L_opt, alpha_opt, L_chol_opt, logs


def default_approx_rank(M_tr: int) -> int:
    # Keep at least 8 directions, at most 64, and about 20% of M_tr
    r_frac = int(0.2 * M_tr)
    return max(8, min(64, r_frac, M_tr))


def default_n_hutchinson_probes(M_tr: int) -> int:
    # Between 8 and 32, growing (slowly) with M_tr
    base = int(4 * math.log2(M_tr + 1))
    return max(8, min(32, base))


def _prepare_component_worker(args):
    """
    Worker used by MKLearner.prepare_approximations to compute low-rank
    factor Phi_e (and optionally KEZ_e) for a single component index e.

    Parameters
    ----------
    args : tuple
        (K_e, approx_rank, Z)

        - K_e : np.ndarray, shape (M_tr, M_tr)
            Training Gram matrix for component e.
        - approx_rank : int
            Truncation rank r.
        - Z : np.ndarray or None, shape (M_tr, S)
            Hutchinson probe matrix. If None, KEZ_e is not computed.

    Returns
    -------
    Phi_e : np.ndarray, shape (M_tr, r)
        Low-rank factor so that K_e ≈ Phi_e Phi_e^T.
    KEZ_e : np.ndarray or None, shape (M_tr, S)
        Approximation of K_e Z = Phi_e Phi_e^T Z, or None if Z is None.
    """
    K_e, approx_rank, Z = args

    # Eigen-decomposition (symmetric PSD)
    vals, vecs = np.linalg.eigh(K_e)
    idx = np.argsort(vals)[::-1]
    vals_r = np.clip(vals[idx[:approx_rank]], 0.0, None)
    vecs_r = vecs[:, idx[:approx_rank]]

    # Phi_e = V_r diag(sqrt(lambda_r))
    sqrt_vals = np.sqrt(vals_r, dtype=float)
    Phi_e = vecs_r * sqrt_vals[None, :]

    KEZ_e = None
    if Z is not None:
        # K_e Z ≈ Phi_e (Phi_e^T Z)
        KEZ_e = Phi_e @ (Phi_e.T @ Z)

    return Phi_e, KEZ_e


@dataclass
class NoisyMKLearner:
    """
    Noisy Multiple Kernel Learner on top of block-local quantum Gram matrices.

    Parameters
    ----------
    arch : CircuitArchitecture
        Circuit architecture used to generate the block kernels (only used
        for consistency checks at the moment).
    grams : np.ndarray
        Array of shape (R, B, M, M) with per-draw, per-block Gram matrices.
        Here:
            R : number of random parameter draws
            B : number of blocks (must equal len(arch.blocks))
            M : number of datapoints
    X : np.ndarray
        Inputs, shape (M, input_dim).
    Y : np.ndarray
        Zero-mean GP samples, shape (num_gps, M).
        Each row is one function draw from a GP prior.
    train_test_split : float
        Fraction of points used for training (0 < split < 1).
    jitter : float
        Small diagonal jitter added to K_w for numerical stability.
        Does not affect the objective (it’s w-independent).
    psd_tol : float
        Tolerance for PSD and symmetry checks on the Gram matrices.

    Attributes
    ----------
    weights : np.ndarray
        Array of shape (num_gps, R, B) with non-negative weights that sum
        to 1 along axis (1,2) for each GP.
    """

    arch: CircuitArchitecture
    grams: np.ndarray  # (R, B, M, M)
    X: np.ndarray  # (M, input_dim)
    Y: np.ndarray  # (num_gps, M)
    train_test_split: float = 2.0 / 3.0
    shuffle_train_test: bool = False
    jitter: float = 1e-6
    psd_tol: float = 1e-10
    check_psd: bool = False

    # Observation noise variance σ^2 (can be 0.0 for noiseless GP)
    noise_variance: float = 0.0
    # If True, learn a separate noise variance σ_g^2 per GP (shape: (num_gps,))
    fit_noise_variance: bool = False

    # approximation / optimization backend options
    use_approx_grad: bool = False
    approx_rank: Optional[int] = None
    n_hutchinson_probes: int = 8
    approx_seed: Optional[int] = None
    n_workers_prepare_approx: Optional[int] = None

    # internal fields initialized in __post_init__
    R: int = field(init=False)
    B: int = field(init=False)
    M: int = field(init=False)
    num_gps: int = field(init=False)
    input_dim: int = field(init=False)
    train_idx: np.ndarray = field(init=False)
    test_idx: np.ndarray = field(init=False)

    _weights: np.ndarray = field(init=False, repr=False)
    _K_components_full: np.ndarray = field(init=False, repr=False)  # (E, M, M)
    _K_components_train: np.ndarray = field(init=False, repr=False)  # (E, M_tr, M_tr)
    _K_components_test_train: np.ndarray = field(init=False, repr=False)  # (E, M_te, M_tr)
    _K_components_test_test: np.ndarray = field(init=False, repr=False)  # (E, M_te, M_te)

    # logs and cached factors per GP
    train_log_marginals_: Optional[np.ndarray] = field(init=False, default=None)
    _L_factors: Optional[list] = field(init=False, default=None)  # list of Cholesky L per GP
    _alpha: Optional[list] = field(init=False, default=None)  # list of alpha per GP

    # Per-GP noise variances σ_g^2 if fit_noise_variance=True
    noise_variances_: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    # precomputed low-rank + Hutchinson structures for approximate gradients
    approx_rank_effective: Optional[int] = field(init=False, default=None)
    _Phi_components_train: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _hutchinson_Z: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _KEZ: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        # --- basic shape checks ---
        grams = np.asarray(self.grams, dtype=float)
        if grams.ndim != 4:
            raise ValueError(f"`grams` must have 4 dims (R,B,M,M). Got {grams.shape}.")

        self.R, self.B, self.M, M2 = grams.shape
        if self.M != M2:
            raise ValueError(f"`grams` last two dimensions must be MxM. Got {grams.shape}.")

        X = np.asarray(self.X, dtype=float)
        Y = np.asarray(self.Y, dtype=float)

        if X.shape[0] != self.M:
            raise ValueError(f"X has M={X.shape[0]} points but grams has M={self.M}.")
        if Y.shape[1] != self.M:
            raise ValueError(f"Y has M={Y.shape[1]} points but grams has M={self.M}.")

        self.num_gps = Y.shape[0]
        self.input_dim = X.shape[1]

        # train/test split
        if not (0.0 < self.train_test_split < 1.0):
            raise ValueError("train_test_split must be in (0,1).")
        M_train = int(np.floor(self.train_test_split * self.M))
        if not (1 <= M_train < self.M):
            raise ValueError(
                f"Invalid split leads to M_train={M_train}, M={self.M}."
            )

        # for now: deterministic split (first M_train are train, rest test)
        if not self.shuffle_train_test:
            self.train_idx = np.arange(M_train, dtype=int)
            self.test_idx = np.arange(M_train, self.M, dtype=int)
        else:
            raise NotImplementedError(f"Shuffling train and test is not yet supported. "
                                      f"Got shuffle_train_test={self.shuffle_train_test}")

        # PSD / symmetry checks
        if self.check_psd:
            self._check_grams_psd_and_sym(grams)

        # Precompute component stack: e = r*B + b
        E = self.R * self.B
        self._K_components_full = grams.reshape(E, self.M, self.M)

        # Precompute restricted versions for speed
        tr = self.train_idx
        te = self.test_idx
        self._K_components_train = self._K_components_full[:, tr][:, :, tr]  # (E, M_tr, M_tr)
        self._K_components_test_train = self._K_components_full[:, te][:, :, tr]  # (E, M_te, M_tr)
        self._K_components_test_test = self._K_components_full[:, te][:, :, te]  # (E, M_te, M_te)

        # Initialize weights uniformly on simplex for each GP
        self._weights = np.full(
            (self.num_gps, self.R, self.B),
            fill_value=1.0 / E,
            dtype=float,
        )

        # buffers for fit results
        self.train_log_marginals_ = np.full(self.num_gps, np.nan, dtype=float)
        self._L_factors = [None] * self.num_gps
        self._alpha = [None] * self.num_gps

        # Initialise per-GP noise variances if requested;
        # default all σ_g^2 to the global noise_variance.
        if self.fit_noise_variance:
            self.noise_variances_ = np.full(
                self.num_gps,
                float(self.noise_variance),
                dtype=float,
            )
        else:
            self.noise_variances_ = None

        # Prepare approximation structures if needed
        if self.use_approx_grad and self.n_hutchinson_probes > 0:
            self.prepare_approximations()

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def _noise_plus_jitter(self) -> float:
        return float(self.noise_variance + self.jitter)

    def _noise_plus_jitter_for_gp(self, gp_index: Optional[int]) -> float:
        """
        Return σ_g^2 + jitter for the given GP if per-GP noise is enabled,
        otherwise fall back to the global noise_variance.
        """
        if self.fit_noise_variance and self.noise_variances_ is not None:
            if gp_index is None:
                idx = 0
            else:
                idx = int(gp_index)
            return float(self.noise_variances_[idx] + self.jitter)
        return self._noise_plus_jitter
    @property
    def weights(self) -> np.ndarray:
        """Weights of shape (num_gps, R, B), each slice summing to 1."""
        return self._weights

    @weights.setter
    def weights(self, w: np.ndarray) -> None:
        w = np.asarray(w, dtype=float)
        if w.shape != (self.num_gps, self.R, self.B):
            raise ValueError(
                f"weights must have shape (num_gps,R,B)=({self.num_gps},{self.R},{self.B}), "
                f"got {w.shape}."
            )
        # non-negativity and simplex constraint per GP
        flat = w.reshape(self.num_gps, -1)
        if np.any(flat < -1e-12):
            raise ValueError("weights must be non-negative.")
        sums = flat.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-8):
            raise ValueError("Each GP's weights must sum to 1.")
        # clip tiny negatives arising from numeric noise
        flat = np.maximum(flat, 0.0)
        flat /= flat.sum(axis=1, keepdims=True)
        self._weights = flat.reshape(self.num_gps, self.R, self.B)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _check_grams_psd_and_sym(self, grams: np.ndarray) -> None:
        """
        Assert all (r,b) Gram matrices are symmetric and PSD to psd_tol.
        """
        for r in range(self.R):
            for b in range(self.B):
                G = grams[r, b]
                if not np.allclose(G, G.T, atol=self.psd_tol):
                    raise ValueError(
                        f"Gram matrix at (r={r}, b={b}) is not symmetric "
                        f"within tol={self.psd_tol}."
                    )
                # eigenvalues
                eigvals = np.linalg.eigvalsh(G)
                if eigvals.min() < -self.psd_tol:
                    raise ValueError(
                        f"Gram matrix at (r={r}, b={b}) not PSD: "
                        f"min eigenvalue={eigvals.min()}."
                    )

    # ------------------------------------------------------------------
    # Approximation precomputation
    # ------------------------------------------------------------------
    def prepare_approximations(
            self,
            approx_rank: Optional[int] = None,
            n_hutchinson_probes: Optional[int] = None,
    ) -> None:
        """
        Precompute low-rank factors and Hutchinson probe vectors used to
        approximate traces in the GP log-marginal gradient.

        This only affects the optimization backend (gradient evaluation).
        Inference (posterior means / variances) still uses the exact kernels.

        Parameters
        ----------
        approx_rank : Optional[int]
            Truncation rank r for each component kernel K_e on the training
            subset. If None, defaults to ``self.approx_rank`` if set, else
            ``min(M_train, 32)``.
        n_hutchinson_probes : Optional[int]
            Number of Hutchinson probe vectors S. If None, defaults to
            ``self.n_hutchinson_probes``.
        """
        if not self.use_approx_grad:
            # Nothing to do if we are not in approximate-gradient mode.
            return

        K_tr = self._K_components_train  # (E, M_tr, M_tr)
        E, M_tr, _ = K_tr.shape

        # ----- resolve approx_rank -----
        if approx_rank is None:
            approx_rank = self.approx_rank if self.approx_rank is not None else min(M_tr, 32)
        approx_rank = max(1, min(approx_rank, M_tr))
        self.approx_rank_effective = approx_rank

        # ----- resolve n_hutchinson_probes -----
        if n_hutchinson_probes is None:
            n_hutchinson_probes = self.n_hutchinson_probes
        if n_hutchinson_probes < 0:
            raise ValueError(f"n_hutchinson_probes must be >= 0, got {n_hutchinson_probes}.")
        # n_hutchinson_probes == 0 means: no Hutchinson, but we may still
        # want low-rank factors Phi_e.

        # ----- Hutchinson probes Z (shared across components) -----
        Z = None
        if n_hutchinson_probes > 0:
            rng = np.random.default_rng(self.approx_seed)
            Z = rng.choice([-1.0, 1.0], size=(M_tr, n_hutchinson_probes)).astype(float)
            self._hutchinson_Z = Z
        else:
            self._hutchinson_Z = None

        # Allocate output arrays
        Phi = np.empty((E, M_tr, approx_rank), dtype=float)
        KEZ = None
        if Z is not None:
            KEZ = np.empty((E, M_tr, n_hutchinson_probes), dtype=float)

        # ----- choose worker count -----
        # Use self.n_workers_prepare_approx if provided, else all available cores.
        n_workers_prepare_approx = self.n_workers_prepare_approx
        if n_workers_prepare_approx is None:
            n_workers_prepare_approx = os.cpu_count() or 1
        n_workers_prepare_approx = max(1, min(n_workers_prepare_approx, E))

        # ----- sequential path (original behaviour) -----
        if n_workers_prepare_approx == 1:
            for e in range(E):
                Phi_e, KEZ_e = _prepare_component_worker((K_tr[e], approx_rank, Z))
                Phi[e] = Phi_e
                if KEZ is not None:
                    KEZ[e] = KEZ_e
        else:
            # ----- parallel path: one process per component (up to n_workers) -----
            tasks = [(K_tr[e], approx_rank, Z) for e in range(E)]
            with ProcessPoolExecutor(max_workers=n_workers_prepare_approx) as executor:
                for e, (Phi_e, KEZ_e) in enumerate(executor.map(_prepare_component_worker, tasks)):
                    Phi[e] = Phi_e
                    if KEZ is not None:
                        KEZ[e] = KEZ_e

        self._Phi_components_train = Phi
        self._KEZ = KEZ

    @staticmethod
    def _project_simplex(v: np.ndarray) -> np.ndarray:
        """
        Euclidean projection of a vector v onto the probability simplex
            Delta = {w : w_i >= 0, sum_i w_i = 1}.

        Implementation: sorting-based algorithm (Duchi et al. 2008).
        """
        v = np.asarray(v, dtype=float)
        n = v.shape[0]

        if n == 1:
            return np.array([1.0])

        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
        if len(rho) == 0:
            # v is already feasible up to numerical noise
            w = np.maximum(v, 0.0)
            s = w.sum()
            if s > 0:
                w /= s
            else:
                w[:] = 1.0 / n
            return w
        rho = rho[-1]
        theta = (cssv[rho] - 1) / float(rho + 1)
        w = np.maximum(v - theta, 0.0)
        return w

    # ------------------------------------------------------------------
    # GP log-marginal and gradients
    # ------------------------------------------------------------------
    def _log_marginal_and_grad_exact(
            self,
            y_train: np.ndarray,
            K_components_train: np.ndarray,  # (E, M_tr, M_tr)
            w_flat: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Exact GP log-marginal log p(y | X, w) and gradient wrt w.

        This is the original exact computation used both for optimization
        (when ``use_approx_grad=False``) and to recompute the final
        posterior factors even in approximate mode.
        """
        E, M_tr, _ = K_components_train.shape
        if w_flat.shape != (E,):
            raise ValueError(f"w_flat must have shape (E,) = ({E},), got {w_flat.shape}.")

        # mixture kernel K_w = sum_e w_e K_e on the training subset
        K_e = K_components_train  # alias
        K_w = np.tensordot(w_flat, K_e, axes=(0, 0))  # (M_tr, M_tr)
        K_y = K_w + self._noise_plus_jitter * np.eye(M_tr)

        # Cholesky factor of noisy kernel
        L = np.linalg.cholesky(K_y)

        # Solve alpha = K_w^{-1} y via two triangular solves
        tmp = np.linalg.solve(L, y_train)
        alpha = np.linalg.solve(L.T, tmp)

        # y^T K^{-1} y
        quad = float(y_train @ alpha)

        # log det K_w from Cholesky
        logdet = 2.0 * np.sum(np.log(np.diag(L)))

        # log marginal likelihood (constant term dropped)
        L_val = -0.5 * quad - 0.5 * logdet

        # K_inv via Cholesky solves (used only here for exact traces)
        I = np.eye(M_tr)
        tmp = np.linalg.solve(L, I)
        K_inv = np.linalg.solve(L.T, tmp)  # (M_tr, M_tr)

        # s_e = alpha^T K_e alpha
        # t_e = Tr(K_inv K_e)
        s = np.einsum("i,eij,j->e", alpha, K_e, alpha)
        t = np.einsum("ij,eij->e", K_inv, K_e)

        grad = 0.5 * (s - t)

        return L_val, grad, alpha, L

    def _log_marginal_and_grad_approx(
            self,
            y_train: np.ndarray,
            K_components_train: np.ndarray,  # (E, M_tr, M_tr)
            w_flat: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Approximate GP log-marginal and gradient using low-rank factors
        and a Hutchinson estimator for the trace term.

        - K_w is still built exactly (from full K_e) so the log-marginal
          itself is exact for the current weights.
        - The gradient components s_e, t_e are approximated as:

            s_e ≈ || Phi_e^T alpha ||^2
            t_e ≈ 1/S sum_s u_s^T (K_e Z_s),

          where Phi_e are low-rank factors K_e ≈ Phi_e Phi_e^T on train
          set, Z_s are Hutchinson probes and u_s = K_w^{-1} Z_s.
        """
        E, M_tr, _ = K_components_train.shape
        if w_flat.shape != (E,):
            raise ValueError(f"w_flat must have shape (E,) = ({E},), got {w_flat.shape}.")

        if self._Phi_components_train is None:
            raise RuntimeError("prepare_approximations() must be called before using approximate gradients.")

        # mixture kernel on train set
        K_e = K_components_train
        K_w = np.tensordot(w_flat, K_e, axes=(0, 0))
        K_y = K_w + self._noise_plus_jitter * np.eye(M_tr)

        L = np.linalg.cholesky(K_y)
        tmp = np.linalg.solve(L, y_train)
        alpha = np.linalg.solve(L.T, tmp)

        quad = float(y_train @ alpha)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        L_val = -0.5 * quad - 0.5 * logdet

        # ---- approximate s_e = alpha^T K_e alpha via low-rank Phi_e ----
        Phi = self._Phi_components_train  # (E, M_tr, r)
        # tmp_e = Phi_e^T alpha, s_e = ||tmp_e||^2
        tmp_alpha = np.einsum("eij,i->ej", Phi, alpha)
        s = np.sum(tmp_alpha ** 2, axis=1)

        # ---- approximate t_e = Tr(K^{-1} K_e) via Hutchinson ----
        if self._hutchinson_Z is None or self._KEZ is None:
            # fall back to exact traces if probes are not available
            I = np.eye(M_tr)
            tmp_I = np.linalg.solve(L, I)
            K_inv = np.linalg.solve(L.T, tmp_I)
            t = np.einsum("ij,eij->e", K_inv, K_e)
        else:
            Z = self._hutchinson_Z  # (M_tr, S)
            KEZ = self._KEZ  # (E, M_tr, S)
            S = Z.shape[1]

            # U = K_w^{-1} Z via triangular solves
            tmp_Z = np.linalg.solve(L, Z)
            U = np.linalg.solve(L.T, tmp_Z)  # (M_tr, S)

            # t_e ≈ 1/S sum_s u_s^T (K_e Z_s)
            t = (U[None, :, :] * KEZ).sum(axis=(1, 2)) / float(S)

        grad = 0.5 * (s - t)

        return L_val, grad, alpha, L

    def _log_marginal_and_grad(
            self,
            y_train: np.ndarray,
            K_components_train: np.ndarray,  # (E, M_tr, M_tr)
            w_flat: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Wrapper selecting exact or approximate backend for the
        log-marginal and its gradient.
        """
        if self.use_approx_grad and self.n_hutchinson_probes > 0:
            return self._log_marginal_and_grad_approx(y_train, K_components_train, w_flat)
        else:
            return self._log_marginal_and_grad_exact(y_train, K_components_train, w_flat)

    def _optimize_weights_for_gp(
            self,
            y_train: np.ndarray,
            K_components_train: np.ndarray,
            w_init: np.ndarray,
            max_iter: int = 200,
            tol: float = 1e-6,
            verbose: bool = False,
            logger=None,
            gp_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Projected gradient ascent on simplex for one GP.

        In exact mode we use a standard Armijo condition.
        In approximate-gradient mode we relax Armijo and only require
        non-decrease in the *true* log-marginal (as computed from K_w).

        Parameters
        ----------
        y_train : np.ndarray
            Training outputs for this GP, shape (M_tr,).
        K_components_train : np.ndarray
            All component kernels on training points, shape (E, M_tr, M_tr).
        w_init : np.ndarray
            Initial weights (unnormalized), shape (E,).
        max_iter : int
            Maximum number of projected gradient iterations.
        tol : float
            Relative improvement tolerance in log-marginal likelihood.
        verbose : bool
            If True, logs progress via `logger`.
        logger : callable or None
            Logging function taking a single string. If None, uses `print`.
            In parallel mode we pass `logs.append` so logs are collected and
            printed from the main thread.
        gp_index : int or None
            Optional GP index for nicer log messages (e.g. [GP 0]).

        Returns
        -------
        w_opt_flat : np.ndarray, shape (E,)
        L_opt : float
        alpha_opt : np.ndarray, shape (M_tr,)
        L_chol_opt : np.ndarray, shape (M_tr, M_tr)
        """
        if logger is None:
            logger = print

        prefix = "[MKL]" if gp_index is None else f"[MKL] [GP {gp_index + 1}]"

        approx_mode = self.use_approx_grad and self.n_hutchinson_probes > 0
        grad_tol = 1e-12

        # Start from simplex-projected initialization
        w = self._project_simplex(w_init)
        L_val, grad, alpha, L_chol = self._log_marginal_and_grad(
            y_train, K_components_train, w
        )

        E = w.shape[0]
        step0 = 1.0
        beta = 0.5
        c = 1e-4
        min_step = 1e-6

        for it in range(max_iter):
            prev_L = L_val
            prev_w = w.copy()
            prev_grad = grad.copy()

            grad_norm2 = float(np.dot(prev_grad, prev_grad))
            if grad_norm2 < grad_tol:
                if verbose:
                    logger(f"{prefix} gradient near zero at iter {it}. Stopping.")
                break

            step = step0
            directional = grad_norm2
            improved = False

            while step >= min_step:
                # projected step on the simplex
                w_trial = self._project_simplex(prev_w + step * prev_grad)
                if np.linalg.norm(w_trial - prev_w, ord=1) < 1e-12:
                    # projection kills the movement → no point shrinking further
                    break

                L_trial, _, alpha_trial, L_chol_trial = self._log_marginal_and_grad(
                    y_train, K_components_train, w_trial
                )

                # Armijo target: strict in exact mode, relaxed in approx mode
                if approx_mode:
                    target = prev_L  # any non-decrease is fine
                else:
                    target = prev_L + c * step * directional

                if L_trial >= target:
                    w = w_trial
                    L_val = L_trial
                    alpha = alpha_trial
                    L_chol = L_chol_trial
                    improved = True
                    break

                step *= beta

            if not improved:
                if verbose:
                    logger(f"{prefix} no further improvement at iter {it}.")
                break

            # New gradient at updated w
            L_new, grad, alpha, L_chol = self._log_marginal_and_grad(
                y_train, K_components_train, w
            )

            rel_impr = abs(L_new - prev_L) / (abs(prev_L) + 1e-12)
            L_val = L_new

            if verbose:
                logger(
                    f"{prefix} iter {it:03d}  L={L_val:.6f}  rel_impr={rel_impr:.3e}"
                )

            if rel_impr < tol:
                break

        return w, L_val, alpha, L_chol

    def _optimize_noise_for_gp(
        self,
        y_train: np.ndarray,
        K_components_train: np.ndarray,
        w_flat: np.ndarray,
        *,
        gp_index: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Simple 1D search over σ^2 for a single GP, keeping mixture
        weights w fixed.

        Returns
        -------
        L_opt : float
            Best log-marginal value.
        alpha_opt : np.ndarray
            Corresponding K_y^{-1} y vector.
        L_chol_opt : np.ndarray
            Corresponding Cholesky factor of K_y.
        """
        prefix = "[MKL]" if gp_index is None else f"[MKL] [GP {gp_index + 1}]"

        # Base scale: current per-GP value if available, else global noise_variance.
        if self.fit_noise_variance and self.noise_variances_ is not None and gp_index is not None:
            base = float(self.noise_variances_[gp_index])
        else:
            base = float(self.noise_variance)

        if not np.isfinite(base) or base <= 0.0:
            base = max(self.jitter * 10.0, 1e-3)

        # Log-spaced grid around base (two orders of magnitude each side)
        log10_base = float(np.log10(base))
        log10_min = log10_base - 2.0
        log10_max = log10_base + 2.0
        sigma_grid = np.logspace(log10_min, log10_max, num=9)
        # Ensure base itself is included
        sigma_grid = np.unique(np.concatenate(([base], sigma_grid)))

        best_L = -np.inf
        best_sigma = base
        best_alpha = None
        best_L_chol = None

        for sigma2 in sigma_grid:
            # Temporarily set global noise for this evaluation
            self.noise_variance = float(sigma2)
            L_val, _, alpha, L_chol = self._log_marginal_and_grad(
                y_train, K_components_train, w_flat
            )
            if verbose:
                print(f"{prefix}   σ²={sigma2:.3e}  L={L_val:.6f}")
            if L_val > best_L:
                best_L = float(L_val)
                best_sigma = float(sigma2)
                best_alpha = alpha
                best_L_chol = L_chol

        # Store best σ² for this GP
        self.noise_variance = best_sigma
        if self.fit_noise_variance and self.noise_variances_ is not None and gp_index is not None:
            self.noise_variances_[gp_index] = best_sigma

        if verbose:
            print(f"{prefix}   selected σ²={best_sigma:.3e}  L={best_L:.6f}")

        return best_L, best_alpha, best_L_chol

    # ------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------
    def fit(
            self,
            max_iter: int = 200,
            tol: float = 1e-6,
            verbose: bool = False,
            n_jobs: Optional[int] = None,
    ):
        """
        Fit MKL weights for each GP sample in Y by maximizing the GP
        log-marginal likelihood over the simplex of (R,B) block mixtures.

        We proceed in two stages:

          1) For each GP g, learn its mixture weights w_g (possibly in
             parallel across GPs via n_jobs > 1), using a fixed noise
             variance.

          2) If fit_noise_variance=True, refine a separate noise
             variance σ_g^2 for each GP g by maximizing that GP's
             log-marginal w.r.t. σ^2, keeping w_g fixed. This stage is
             done sequentially (cheap 1D search per GP).

        Parameters
        ----------
        max_iter : int
            Maximum number of projected gradient steps per GP.
        tol : float
            Relative improvement tolerance on log-marginal likelihood.
        verbose : bool
            If True, prints progress for each GP.
        n_jobs : int or None
            Number of worker threads to use across GPs for the weight
            optimisation step.
              - If None, uses 1 (sequential).
              - If > 1, uses ThreadPoolExecutor with at most `num_gps`
                workers.

        Returns
        -------
        self
        """
        y_all = self.Y
        K_tr = self._K_components_train
        E = self.R * self.B
        M_tr = self.train_idx.size

        if M_tr == 0:
            raise RuntimeError("No training points to fit on.")

        # reset buffers
        self.train_log_marginals_ = np.full(self.num_gps, np.nan, dtype=float)
        self._L_factors = [None] * self.num_gps
        self._alpha = [None] * self.num_gps

        # decide number of jobs for weight optimisation
        if n_jobs is None:
            n_jobs = 1
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {n_jobs}.")
        n_jobs = min(n_jobs, self.num_gps)

        # --------------------------------------------------------------
        # 1) Optimise weights w_g (possibly in parallel across GPs)
        # --------------------------------------------------------------
        if n_jobs == 1 or self.num_gps == 1:
            # sequential path
            for g in range(self.num_gps):
                if verbose:
                    print(f"\n[MKL] === GP {g + 1} / {self.num_gps} ===")

                y_train = y_all[g, self.train_idx]
                w_init = np.full(E, 1.0 / E, dtype=float)  # uniform simplex

                w_opt, L_opt, alpha_opt, L_chol_opt = self._optimize_weights_for_gp(
                    y_train=y_train,
                    K_components_train=K_tr,
                    w_init=w_init,
                    max_iter=max_iter,
                    tol=tol,
                    verbose=verbose,
                    logger=print,
                    gp_index=g,
                )

                self._weights[g] = w_opt.reshape(self.R, self.B)
                self.train_log_marginals_[g] = L_opt
                self._alpha[g] = alpha_opt
                self._L_factors[g] = L_chol_opt
        else:
            # parallel path: one job per GP (up to n_jobs)
            if verbose:
                print(f"[MKL] Using {n_jobs} parallel worker(s) for {self.num_gps} GPs.")

            tasks = [
                (self, g, max_iter, tol, verbose) for g in range(self.num_gps)
            ]

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # executor.map preserves order of `tasks`
                for g, w_opt, L_opt, alpha_opt, L_chol_opt, logs in executor.map(
                        _fit_single_gp_worker, tasks
                ):
                    self._weights[g] = w_opt.reshape(self.R, self.B)
                    self.train_log_marginals_[g] = L_opt
                    self._alpha[g] = alpha_opt
                    self._L_factors[g] = L_chol_opt

                    if verbose:
                        print(f"\n[MKL] === GP {g} / {self.num_gps - 1} ===")
                        for line in logs:
                            print(line)

        # --------------------------------------------------------------
        # 2) Refine per-GP noise variances σ_g^2 (sequential, cheap)
        # --------------------------------------------------------------
        if self.fit_noise_variance:
            if verbose:
                print("\n[MKL] Refining per-GP noise variances σ_g^2...")

            for g in range(self.num_gps):
                y_train = y_all[g, self.train_idx]
                w_flat = self._weights[g].reshape(E)

                L_opt, alpha_opt, L_chol_opt = self._optimize_noise_for_gp(
                    y_train=y_train,
                    K_components_train=K_tr,
                    w_flat=w_flat,
                    gp_index=g,
                    verbose=verbose,
                )

                # overwrite caches with noise-optimised values
                self.train_log_marginals_[g] = L_opt
                self._alpha[g] = alpha_opt
                self._L_factors[g] = L_chol_opt

        return self

    def infer(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior mean and variance at TEST points for each GP
        using the learned MKL weights.

        Returns
        -------
        means : np.ndarray
            Shape (num_gps, M_test), posterior means at test inputs.
        variances : np.ndarray
            Shape (num_gps, M_test), posterior variances at test inputs
            (latent function variance, i.e., without adding σ_g^2).
        """
        if self._L_factors is None or any(L is None for L in self._L_factors):
            raise RuntimeError("Call fit() before infer().")

        te = self.test_idx
        tr = self.train_idx
        M_te = te.size
        if M_te == 0:
            raise RuntimeError("No test points to infer on.")

        E = self.R * self.B

        means = np.zeros((self.num_gps, M_te), dtype=float)
        variances = np.zeros((self.num_gps, M_te), dtype=float)

        # pre-stacked components
        K_tr = self._K_components_train  # (E, M_tr, M_tr)
        K_te_tr = self._K_components_test_train  # (E, M_te, M_tr)
        K_te_te = self._K_components_test_test  # (E, M_te, M_te)

        for g in range(self.num_gps):
            w_flat = self._weights[g].reshape(E)

            # mixture kernels
            K_train = np.tensordot(w_flat, K_tr, axes=(0, 0))  # (M_tr, M_tr)
            K_test_train = np.tensordot(w_flat, K_te_tr, axes=(0, 0))  # (M_te, M_tr)
            K_test_test = np.tensordot(w_flat, K_te_te, axes=(0, 0))  # (M_te, M_te)

            # Per-GP noise for this GP (same convention as training)
            noise_g = self._noise_plus_jitter_for_gp(g)
            K_y = K_train + noise_g * np.eye(K_train.shape[0])

            # Use cached factors if available, else recompute
            L = self._L_factors[g]
            alpha = self._alpha[g]
            if L is None or alpha is None:
                L = np.linalg.cholesky(K_y)
                y_train = self.Y[g, tr]
                tmp = np.linalg.solve(L, y_train)
                alpha = np.linalg.solve(L.T, tmp)

            # predictive mean: k_*^T K^{-1} y = K_test_train @ alpha
            mu = K_test_train @ alpha

            # predictive variance of latent function:
            # var[f(x_*)] = diag(K_ss - k_* K^{-1} k_*^T)
            v = np.linalg.solve(L, K_test_train.T)  # (M_tr, M_te)
            var = np.diag(K_test_test) - np.sum(v * v, axis=0)

            means[g] = mu
            variances[g] = np.maximum(var, 0.0)  # numerical safety

        return means, variances

    def get_train_test_metrics(self) -> Dict[str, np.ndarray]:
        """
        Compute goodness-of-fit metrics on train / test:

        - train_log_marginal_likelihood           : sum log p(y_train | X_train, w*)  per GP
        - train_log_marginal_likelihood_per_point: same but divided by #train points
        - test_log_predictive_density            : sum_n log p(y_test_n | X_train, y_train) per GP
        - test_avg_nlpd                          : average negative log predictive density on test
        - rmse_test                              : root mean squared error on test
        - mae_test                               : mean absolute error on test
        - z_mean, z_var                          : mean / variance of standardized residuals z
        - coverage_1sigma, coverage_2sigma       : fraction of |z| <= 1 and <= 2 on test

        Returns
        -------
        metrics : dict[str, np.ndarray]
        """
        if self.train_log_marginals_ is None:
            raise RuntimeError("Call fit() before get_train_test_metrics().")

        # Posterior means / variances at test points
        means, vars_ = self.infer()
        te = self.test_idx
        tr = self.train_idx

        y_all = self.Y
        num_gps = self.num_gps
        M_te = te.size
        M_tr = tr.size

        log_pred = np.zeros(num_gps, dtype=float)
        avg_nlpd = np.zeros(num_gps, dtype=float)
        rmse = np.zeros(num_gps, dtype=float)
        mae = np.zeros(num_gps, dtype=float)
        z_mean = np.zeros(num_gps, dtype=float)
        z_var = np.zeros(num_gps, dtype=float)
        cov1 = np.zeros(num_gps, dtype=float)
        cov2 = np.zeros(num_gps, dtype=float)

        const = 0.5 * np.log(2.0 * np.pi)

        for g in range(num_gps):
            y_test = y_all[g, te]
            mu = means[g]
            var = vars_[g] + self._noise_plus_jitter_for_gp(g)  # predictive variance + small noise

            # log predictive density under N(mu, var) per test point
            logp_per_point = -0.5 * ((y_test - mu) ** 2 / var + np.log(var)) - const
            log_pred[g] = np.sum(logp_per_point)
            avg_nlpd[g] = -np.mean(logp_per_point)  # average NEGATIVE log-predictive density

            # pointwise errors
            err = y_test - mu
            rmse[g] = float(np.sqrt(np.mean(err ** 2)))
            mae[g] = float(np.mean(np.abs(err)))

            # standardized residuals (for calibration)
            z = err / np.sqrt(var)
            z_mean[g] = float(np.mean(z))
            z_var[g] = float(np.var(z))
            cov1[g] = float(np.mean(np.abs(z) <= 1.0))  # within ±1σ
            cov2[g] = float(np.mean(np.abs(z) <= 2.0))  # within ±2σ

        train_ll = self.train_log_marginals_.copy()
        train_ll_per_point = train_ll / float(M_tr)

        return dict(
            train_log_marginal_likelihood=train_ll,
            train_log_marginal_likelihood_per_point=train_ll_per_point,
            test_log_predictive_density=log_pred,
            test_avg_nlpd=avg_nlpd,
            rmse_test=rmse,
            mae_test=mae,
            z_mean=z_mean,
            z_var=z_var,
            coverage_1sigma=cov1,
            coverage_2sigma=cov2,
        )

    # ------------------------------------------------------------------
    # Kernel–kernel similarity metrics
    # ------------------------------------------------------------------
    def get_kernel_similarity_metrics(
            self,
            reference_kernels: np.ndarray,
            subset: str = "test",
            *,
            zero_one_scale=True,
            center_alignment: bool = True,
            center_frobenius: bool = False,
    ) -> dict:
        """
        Compare the learned mixed kernels to reference (true) kernels.

        Parameters
        ----------
        reference_kernels : np.ndarray
            Array of shape (num_gps, m, m) on the same subset as `subset`.
        subset : {"train", "test", "full"}
            Which subset to build the mixed kernels on.

        Returns
        -------
        metrics : dict
            {
              "kernel_frobenius_rel_error": np.ndarray shape (num_gps,),
              "kernel_alignment": np.ndarray shape (num_gps,),
            }
        """
        from results.metrics.kernel_similarity import (
            frobenius_relative_error,
            kernel_alignment,
        )

        K_mix = self.get_mixed_kernels(subset=subset, zero_one_scale=zero_one_scale)
        K_ref = np.asarray(reference_kernels, dtype=float)

        if K_mix.shape != K_ref.shape:
            raise ValueError(
                f"Shape mismatch: mixed {K_mix.shape} vs reference {K_ref.shape}"
            )

        num_gps = self.num_gps
        frob_err = np.empty(num_gps, dtype=float)
        align = np.empty(num_gps, dtype=float)

        for g in range(num_gps):
            frob_err[g] = frobenius_relative_error(
                K_mix[g], K_ref[g], center=center_frobenius
            )
            align[g] = kernel_alignment(
                K_mix[g], K_ref[g], center=center_alignment
            )

        return {
            "kernel_frobenius_rel_error": frob_err,
            "kernel_alignment": align,
        }

    # ------------------------------------------------------------------
    # Mixed kernel extraction
    # ------------------------------------------------------------------
    def get_mixed_kernels(self, subset: str = "test", zero_one_scale: bool=True) -> np.ndarray:
        """
        Return the learned mixed kernels for each GP on a chosen subset.

        Parameters
        ----------
        subset : {"train", "test", "full"}
            Which indices to use.
        zero_one_scale
            Zero one Scales or not the gram matrices

        Returns
        -------
        K_mix : np.ndarray
            Array of shape (num_gps, m, m), where m is the number of points
            in the chosen subset.
        """
        if self.weights is None:
            raise RuntimeError("Weights have not been learned yet. Call .fit() first.")

        if subset == "train":
            idx = self.train_idx
        elif subset == "test":
            idx = self.test_idx
        elif subset == "full":
            idx = np.arange(self.M)
        else:
            raise ValueError(f"Unknown subset: {subset!r}")

        # self.grams: (R, B, M, M)
        # restrict to the chosen indices: (R, B, m, m)
        grams_sub = self.grams[:, :, idx, :][:, :, :, idx]  # (R, B, m, m)

        # weights: (num_gps, R, B)
        # tensordot over (R, B) to obtain (num_gps, m, m)
        K_mix = np.tensordot(self.weights, grams_sub, axes=([1, 2], [0, 1]))
        # np.tensordot will put the GP axis first already: (num_gps, m, m)

        if zero_one_scale:

            mins = K_mix.min(axis=(1, 2), keepdims=True)  # shape (n, 1, 1)
            maxs = K_mix.max(axis=(1, 2), keepdims=True)  # shape (n, 1, 1)

            denom = maxs - mins

            # Safe division in case some slice is constant (max == min)
            K_mix = np.where(
                denom > 0,
                (K_mix - mins) / denom,
                0.0
            )

        return K_mix

    # ------------------------------------------------------------------
    # Persistence API
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save the MKLearner (including any precomputed approximations) to disk.

        Parameters
        ----------
        path : str
            Destination path for the pickle file.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "NoisyMKLearner":
        """
        Load a previously saved MKLearner from disk.

        Parameters
        ----------
        path : str
            Path to the pickle file created by :meth:`save`.

        Returns
        -------
        MKLearner
            The reconstructed learner instance.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, NoisyMKLearner):
            raise TypeError(f"Expected a MKLearner instance in {path!r}, got {type(obj)!r}.")
        return obj
