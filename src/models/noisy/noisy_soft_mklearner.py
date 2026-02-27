from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.models.noisy.noisy_mk_learner import NoisyMKLearner


def _fit_single_gp_softmax_worker(args):
    """
    Worker to fit softmax MKL weights for a single GP index g.

    This is designed to be used with ThreadPoolExecutor. It does NOT
    modify the learner instance; it only reads from it and returns the
    results.
    """
    learner, g, max_iter, tol, verbose = args

    E = learner.R * learner.B
    y_train = learner.Y[g, learner.train_idx]
    K_tr = learner._K_components_train  # (E, M_tr, M_tr)

    u_init = np.zeros(E, dtype=float)

    u_opt, w_opt_flat, L_opt, alpha_opt, L_chol_opt = learner._optimize_logits_for_gp(
        y_train=y_train,
        K_components_train=K_tr,
        u_init=u_init,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        gp_index=g,
    )

    return g, u_opt, w_opt_flat, L_opt, alpha_opt, L_chol_opt

@dataclass
class NoisySoftmaxMKLearner(NoisyMKLearner):
    """
    MKL variant that parameterises each GP's mixture weights via logits
    and a softmax, w = softmax(u).

    All approximation options are inherited from MKLearner:

        - use_approx_grad : bool
        - approx_rank : Optional[int]
        - n_hutchinson_probes : int
        - approx_seed : Optional[int]

    If use_approx_grad=True and n_hutchinson_probes>0, gradients of the
    log-marginal w.r.t *linear* weights w are computed using the
    low-rank + Hutchinson scheme; inference (posterior means/vars) is
    still exact.
    """

    # one logit per weight; same shape as self._weights
    logits: np.ndarray = field(init=False, repr=False)

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Let MKLearner do all shape checks, train/test split and,
        # in approximate mode, call prepare_approximations().
        super().__post_init__()

        # Initialise logits so that softmax(logits) is uniform.
        self.logits = np.zeros_like(self._weights, dtype=float)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(u_flat: np.ndarray) -> np.ndarray:
        """Numerically stable softmax on a 1D array."""
        u_shift = u_flat - np.max(u_flat)
        exp_u = np.exp(u_shift)
        Z = np.sum(exp_u)
        if not np.isfinite(Z) or Z <= 0.0:
            n = u_flat.size
            return np.full(n, 1.0 / n, dtype=float)
        return exp_u / Z

    def _log_marginal_and_grad_w(
        self,
        y_train: np.ndarray,
        K_components_train: np.ndarray,
        w_flat: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return log-marginal and gradient wrt *linear* weights w.

        The backend (exact vs approximate) is selected by MKLearner via
        self._log_marginal_and_grad().
        """
        return self._log_marginal_and_grad(y_train, K_components_train, w_flat)

    @staticmethod
    def _grad_logits_from_grad_w(
        grad_w: np.ndarray,
        w_flat: np.ndarray,
    ) -> np.ndarray:
        """
        Chain rule for u -> w = softmax(u).

        For scalar L(w):
            dL/du_j = w_j ( dL/dw_j - <dL/dw, w> ).
        """
        dot = float(np.dot(grad_w, w_flat))
        return w_flat * (grad_w - dot)

    # ------------------------------------------------------------------
    # optimisation for a single GP
    # ------------------------------------------------------------------
    def _optimize_logits_for_gp(
        self,
        y_train: np.ndarray,
        K_components_train: np.ndarray,
        u_init: np.ndarray,
        max_iter: int = 200,
        tol: float = 1e-6,
        verbose: bool = False,
        gp_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Gradient-ascent in logit space for a single GP.

        Returns
        -------
        u_opt_flat, w_opt_flat, L_opt, alpha_opt, L_chol_opt
        """
        prefix = "[SoftMKL]" if gp_index is None else f"[SoftMKL] [GP {gp_index}]"

        u = np.asarray(u_init, dtype=float).reshape(-1)

        # initial weights / objective / gradient (w.r.t weights)
        w = self._softmax(u)
        L_val, grad_w, alpha, L_chol = self._log_marginal_and_grad_w(
            y_train, K_components_train, w
        )

        step0 = 1.0
        beta = 0.5
        c = 1e-4
        min_step = 1e-6

        approx_mode = self.use_approx_grad and self.n_hutchinson_probes > 0

        for it in range(max_iter):
            prev_L = L_val
            prev_u = u.copy()
            prev_w = w.copy()
            prev_grad_w = grad_w.copy()

            # gradient in logit space
            grad_u = self._grad_logits_from_grad_w(prev_grad_w, prev_w)
            directional = float(np.dot(grad_u, grad_u))
            if directional < 1e-16:
                if verbose:
                    print(f"{prefix} gradient too small at iter {it}.")
                break

            # Backtracking line search in logit space
            step = step0
            improved = False

            while step >= min_step:
                u_trial = prev_u + step * grad_u
                w_trial = self._softmax(u_trial)

                L_trial, grad_w_trial, alpha_trial, L_chol_trial = (
                    self._log_marginal_and_grad_w(
                        y_train, K_components_train, w_trial
                    )
                )

                # In approximate-gradient mode we only require the *exact*
                # objective not to decrease; otherwise we use true Armijo.
                if approx_mode:
                    condition = L_trial >= prev_L - 1e-10
                else:
                    condition = L_trial >= prev_L + c * step * directional

                if condition:
                    u = u_trial
                    w = w_trial
                    L_val = L_trial
                    grad_w = grad_w_trial
                    alpha = alpha_trial
                    L_chol = L_chol_trial
                    improved = True
                    break

                step *= beta

            if not improved:
                if verbose:
                    print(f"{prefix} no further improvement at iter {it}.")
                break

            rel_impr = abs(L_val - prev_L) / (abs(prev_L) + 1e-12)
            if verbose:
                print(f"{prefix} iter {it:03d}  L={L_val:.6f}  rel_impr={rel_impr:.3e}")

            if rel_impr < tol:
                break

        return u, w, L_val, alpha, L_chol

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(
            self,
            max_iter: int = 200,
            tol: float = 1e-6,
            verbose: bool = False,
            n_jobs: Optional[int] = None,
            n_coord_steps: int = 3,
    ):
        """
        Fit softmax weights for each GP by maximising the log-marginal.

        Coordinate-ascent flavour:

          for step in 1..n_coord_steps:
              (A) optimise logits/weights w_g for all GPs (possibly
                  in parallel across GPs),
              (B) if fit_noise_variance=True, refine a separate noise
                  variance σ_g^2 for each GP (sequential, cheap).

        If `use_approx_grad=True` and `n_hutchinson_probes>0`, the
        gradients wrt w are computed using the low-rank + Hutchinson
        approximation prepared in NoisyMKLearner.__post_init__().

        Parameters
        ----------
        max_iter : int
            Maximum number of gradient steps in logit space per GP
            *per coordinate step*.
        tol : float
            Relative improvement tolerance on log-marginal likelihood.
        verbose : bool
            If True, prints progress for each GP and each coord step.
        n_jobs : int or None
            Number of worker threads to use across GPs for the logits
            optimisation step. If None, uses 1 (sequential).
        n_coord_steps : int
            Number of outer coordinate-ascent steps:
                - n_coord_steps = 1 reproduces the previous
                  "weights-then-noise-once" behaviour.
                - n_coord_steps > 1 alternates between updating
                  weights and noise a few times.
        """
        y_all = self.Y
        K_tr = self._K_components_train
        E = self.R * self.B

        if n_coord_steps < 1:
            raise ValueError(f"n_coord_steps must be >= 1, got {n_coord_steps}.")

        # decide number of jobs for logits optimisation
        if n_jobs is None:
            n_jobs = 1
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {n_jobs}.")
        n_jobs = min(n_jobs, self.num_gps)

        # Main coordinate-ascent loop
        for step in range(n_coord_steps):
            if verbose:
                print(f"\n[SoftMKL] ==== Coordinate step {step + 1} / {n_coord_steps} ====")

            # reset caches at each coordinate step
            self.train_log_marginals_ = np.full(self.num_gps, np.nan, dtype=float)
            self._L_factors = [None] * self.num_gps
            self._alpha = [None] * self.num_gps

            # ----------------------------------------------------------
            # (A) Optimise logits / weights per GP (possibly in parallel)
            # ----------------------------------------------------------
            if n_jobs == 1 or self.num_gps == 1:
                # sequential path
                for g in range(self.num_gps):
                    if verbose:
                        print(f"\n[SoftMKL] [coord {step + 1}] === GP {g} / {self.num_gps - 1} ===")

                    y_train = y_all[g, self.train_idx]

                    # Warm start from current logits; on the very first
                    # call these are zeros (uniform softmax).
                    u_init = self.logits[g].reshape(-1)

                    u_opt, w_opt_flat, L_opt, alpha_opt, L_chol_opt = (
                        self._optimize_logits_for_gp(
                            y_train=y_train,
                            K_components_train=K_tr,
                            u_init=u_init,
                            max_iter=max_iter,
                            tol=tol,
                            verbose=verbose,
                            gp_index=g,
                        )
                    )

                    self.logits[g] = u_opt.reshape(self.R, self.B)
                    self._weights[g] = w_opt_flat.reshape(self.R, self.B)
                    self.train_log_marginals_[g] = L_opt
                    self._alpha[g] = alpha_opt
                    self._L_factors[g] = L_chol_opt
            else:
                # parallel path: one job per GP
                if verbose:
                    print(f"[SoftMKL] Using {n_jobs} parallel worker(s) for {self.num_gps} GPs.")

                # In each coord step we warm-start from current logits.
                tasks = [
                    (self, g, max_iter, tol, verbose) for g in range(self.num_gps)
                ]

                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    for g, u_opt, w_opt_flat, L_opt, alpha_opt, L_chol_opt in executor.map(
                            _fit_single_gp_softmax_worker, tasks
                    ):
                        self.logits[g] = u_opt.reshape(self.R, self.B)
                        self._weights[g] = w_opt_flat.reshape(self.R, self.B)
                        self.train_log_marginals_[g] = L_opt
                        self._alpha[g] = alpha_opt
                        self._L_factors[g] = L_chol_opt

            # ----------------------------------------------------------
            # (B) Refine per-GP noise variances σ_g^2 (sequential, cheap)
            # ----------------------------------------------------------
            if self.fit_noise_variance:
                if verbose:
                    print("\n[SoftMKL] Refining per-GP noise variances σ_g^2...")

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

                    self.train_log_marginals_[g] = L_opt
                    self._alpha[g] = alpha_opt
                    self._L_factors[g] = L_chol_opt

        return self
