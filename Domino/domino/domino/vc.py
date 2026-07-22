"""Variance-component estimation in a shared LOCO eigenbasis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


def _apply_eigen_operator(
    x,
    U,
    s,
    h2,
    scale=1.0,
    power=-0.5,
    tol=1e-10,
    residual_eigenvalue=0.0,
):
    """Apply ``[scale * (h2 G + (1-h2) I)]**power`` to arrays."""
    x = np.asarray(x, dtype=np.float64)
    was_vector = x.ndim == 1
    if was_vector:
        x = x[:, None]
    U = np.asarray(U, dtype=np.float64)
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    d_top = np.maximum(scale * (h2 * s + (1.0 - h2)), tol)
    d_res = max(
        scale * (h2 * float(residual_eigenvalue) + (1.0 - h2)), tol
    )
    utx = U.T @ x
    result = U @ (utx * ((d_top ** power) - d_res ** power)[:, None])
    result += x * (d_res ** power)
    return result[:, 0] if was_vector else result


def covariance_matrix(
    U,
    s,
    h2,
    scale=1.0,
    power=-0.5,
    tol=1e-10,
    residual_eigenvalue=0.0,
):
    """Materialize an eigen-operator for compatibility and small test cases."""
    identity = np.eye(np.asarray(U).shape[0], dtype=np.float64)
    return _apply_eigen_operator(
        identity,
        U,
        s,
        h2,
        scale,
        power,
        tol,
        residual_eigenvalue=residual_eigenvalue,
    )


def inverse_covariance_weights(
    s, h2, yvar, residual_eigenvalue=0.0, tol=1e-10
):
    """Return complement weight and retained-eigenspace corrections for V^-1."""
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    top = np.maximum(yvar * (h2 * s + 1.0 - h2), tol)
    residual = max(
        yvar * (h2 * float(residual_eigenvalue) + 1.0 - h2), tol
    )
    w_residual = 1.0 / residual
    return w_residual, 1.0 / top - w_residual


@dataclass
class ProfileREMLContext:
    """Cached profile ML/REML algebra for one phenotype and fixed design."""

    y: np.ndarray
    U: np.ndarray
    s: np.ndarray
    C: np.ndarray
    method: str = "REML"
    tol: float = 1e-10
    residual_eigenvalue: float = 0.0
    precomputed: object = None

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=np.float64).ravel()
        self.U = np.asarray(self.U, dtype=np.float64)
        self.s = np.maximum(np.asarray(self.s, dtype=np.float64), 0.0)
        self.C = np.asarray(self.C, dtype=np.float64)
        if self.C.ndim == 1:
            self.C = self.C[:, None]
        self.method = self.method.upper()
        n = self.y.size
        if self.method not in {"ML", "REML"}:
            raise ValueError("method must be 'ML' or 'REML'")
        if self.C.shape[0] != n or self.U.shape != (n, len(self.s)):
            raise ValueError("y, U, s, and C dimensions do not agree")
        if not np.isfinite(self.y).all() or not np.isfinite(self.C).all():
            raise ValueError("y and covar must be finite")
        self.rank_c = int(np.linalg.matrix_rank(self.C))
        if self.rank_c == 0 or self.rank_c >= n:
            raise ValueError("fixed-effect design must have rank between 1 and n-1")
        self.n = n
        self.df = n - self.rank_c if self.method == "REML" else n
        self.k_residual = max(n - len(self.s), 0)
        if self.precomputed is None:
            self.uty = self.U.T @ self.y
            self.utc = self.U.T @ self.C
            self.yty = float(self.y @ self.y)
            self.cty = self.C.T @ self.y
            self.ctc = self.C.T @ self.C
        else:
            self.uty = self.precomputed["uty"]
            self.utc = self.precomputed["utc"]
            self.yty = float(self.precomputed["yty"])
            self.cty = self.precomputed["cty"]
            self.ctc = self.precomputed["ctc"]
        self.n_evaluations = 0

    def products(self, h2):
        d_top = np.maximum(h2 * self.s + 1.0 - h2, self.tol)
        d_residual = max(
            h2 * float(self.residual_eigenvalue) + 1.0 - h2, self.tol
        )
        w_residual = 1.0 / d_residual
        delta = 1.0 / d_top - w_residual
        ywy = w_residual * self.yty + float(np.sum(delta * self.uty ** 2))
        cwy = w_residual * self.cty + self.utc.T @ (delta * self.uty)
        cwc = w_residual * self.ctc + self.utc.T @ (delta[:, None] * self.utc)
        return d_top, d_residual, ywy, cwy, cwc

    def objective(self, h2):
        self.n_evaluations += 1
        d_top, d_residual, ywy, cwy, cwc = self.products(h2)
        sign, logdet_information = np.linalg.slogdet(cwc)
        if sign <= 0:
            return np.inf
        beta = np.linalg.solve(cwc, cwy)
        q = max(float(ywy - cwy @ beta), self.tol)
        sigma2 = max(q / self.df, self.tol)
        logdet = float(
            np.log(d_top).sum() + self.k_residual * np.log(d_residual)
        )
        value = self.df * np.log(sigma2) + logdet
        if self.method == "REML":
            value += logdet_information
        return 0.5 * value

    def fit(self):
        result = minimize_scalar(
            self.objective, bounds=(1e-4, 1.0 - 1e-4), method="bounded"
        )
        h2 = float(result.x)
        _, _, ywy, cwy, cwc = self.products(h2)
        beta = np.linalg.solve(cwc, cwy)
        sigma2 = max(float(ywy - cwy @ beta) / self.df, self.tol)
        return {
            "h2": h2,
            "yvar": sigma2,
            "converged": bool(result.success),
            "objective": float(result.fun),
            "n_evaluations": int(self.n_evaluations),
            "estimator": f"profile_{self.method.lower()}",
        }


def estimate_h2(
    y,
    U,
    s,
    method="REML",
    tol=1e-10,
    covar=None,
    residual_eigenvalue=0.0,
):
    """Profile ML/REML estimate with all eigenspace products cached."""
    y = np.asarray(y, dtype=np.float64).ravel()
    C = np.ones((len(y), 1), dtype=np.float64) if covar is None else covar
    return ProfileREMLContext(
        y,
        U,
        s,
        C,
        method=method,
        tol=tol,
        residual_eigenvalue=residual_eigenvalue,
    ).fit()


def estimate_h2_many(
    Y,
    U,
    s,
    covar=None,
    method="REML",
    tol=1e-10,
    residual_eigenvalue=0.0,
):
    """Fit independent profile models sharing one LOCO eigenbasis."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    C = np.ones((len(Y), 1), dtype=np.float64) if covar is None else np.asarray(covar, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    U = np.asarray(U, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    UTY = U.T @ Y
    UTC = U.T @ C
    YTY = np.sum(Y * Y, axis=0)
    CTY = C.T @ Y
    CTC = C.T @ C
    fits = []
    for column in range(Y.shape[1]):
        context = ProfileREMLContext(
            Y[:, column],
            U,
            s,
            C,
            method=method,
            tol=tol,
            residual_eigenvalue=residual_eigenvalue,
            precomputed={
                "uty": UTY[:, column],
                "utc": UTC,
                "yty": YTY[column],
                "cty": CTY[:, column],
                "ctc": CTC,
            },
        )
        fits.append(context.fit())
    return fits


def _covariate_residual_basis(C):
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    rank = int(np.linalg.matrix_rank(C))
    if rank < 1 or rank >= len(C):
        raise ValueError("fixed-effect design must have rank between 1 and n-1")
    left, _, _ = np.linalg.svd(C, full_matrices=False)
    return left[:, :rank], rank


def _nearest_psd(matrix, floor=0.0):
    matrix = (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix).T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return (vectors * values) @ vectors.T


def score_variance_components(
    Y,
    U,
    s,
    covar=None,
    residual_eigenvalue=0.0,
    full_covariance=False,
    tol=1e-10,
):
    """Covariate-adjusted two-component SCORE/moment estimator.

    The equations are evaluated in the residual subspace of ``covar``.  The
    independent path is O(ntr); ``full_covariance=True`` additionally computes
    the t by t genetic and residual covariance matrices and projects both to
    the positive-semidefinite cone.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, n_traits = Y.shape
    U = np.asarray(U, dtype=np.float64)
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    if U.shape != (n, len(s)):
        raise ValueError("Y and eigensystem dimensions do not agree")
    C = np.ones((n, 1), dtype=np.float64) if covar is None else covar
    Qc, rank_c = _covariate_residual_basis(C)
    residual_df = n - rank_c
    Yp = Y - Qc @ (Qc.T @ Y)
    utc = U.T @ Qc
    H = np.eye(len(s), dtype=np.float64) - utc @ utc.T
    bulk = float(residual_eigenvalue)
    spectral_difference = s - bulk
    trace_a = float(np.sum(spectral_difference * np.diag(H)))
    trace_a2 = float(
        np.sum(
            (spectral_difference[:, None] * spectral_difference[None, :])
            * (H * H)
        )
    )
    trace_k = trace_a + bulk * residual_df
    trace_k2 = trace_a2 + 2.0 * bulk * trace_a + bulk * bulk * residual_df
    information = np.array(
        [[trace_k2, trace_k], [trace_k, float(residual_df)]], dtype=np.float64
    )
    determinant = float(np.linalg.det(information))
    if not np.isfinite(determinant) or determinant <= tol:
        raise np.linalg.LinAlgError("SCORE variance-component information is singular")
    uty = U.T @ Yp

    if full_covariance:
        B2 = Yp.T @ Yp
        B1 = uty.T @ (spectral_difference[:, None] * uty) + bulk * B2
        genetic = (information[1, 1] * B1 - information[0, 1] * B2) / determinant
        residual = (information[0, 0] * B2 - information[1, 0] * B1) / determinant
        genetic = _nearest_psd(genetic)
        residual = _nearest_psd(residual, floor=tol)
        genetic_diag = np.diag(genetic)
        residual_diag = np.diag(residual)
    else:
        B2 = np.sum(Yp * Yp, axis=0)
        B1 = np.sum(spectral_difference[:, None] * uty * uty, axis=0) + bulk * B2
        genetic_diag = (information[1, 1] * B1 - information[0, 1] * B2) / determinant
        residual_diag = (information[0, 0] * B2 - information[1, 0] * B1) / determinant
        genetic_diag = np.maximum(genetic_diag, 0.0)
        residual_diag = np.maximum(residual_diag, tol)
        genetic = residual = None

    total = np.maximum(genetic_diag + residual_diag, tol)
    h2 = np.clip(genetic_diag / total, 0.0, 1.0 - 1e-8)
    fits = [
        {
            "h2": float(h2[index]),
            "yvar": float(total[index]),
            "sigma_g2": float(genetic_diag[index]),
            "sigma_e2": float(residual_diag[index]),
            "converged": True,
            "estimator": "score",
        }
        for index in range(n_traits)
    ]
    return {
        "fits": fits,
        "genetic_covariance": genetic,
        "residual_covariance": residual,
        "information": information,
        "fixed_effect_rank": rank_c,
    }


def multivariate_score_transform(
    Y, U, s, covar=None, residual_eigenvalue=0.0, tol=1e-10
):
    """Diagonalize SCORE genetic/residual trait covariance components."""
    score = score_variance_components(
        Y,
        U,
        s,
        covar=covar,
        residual_eigenvalue=residual_eigenvalue,
        full_covariance=True,
        tol=tol,
    )
    genetic = score["genetic_covariance"]
    residual = score["residual_covariance"]
    original_fits = score["fits"]
    scale = max(float(np.trace(residual)) / max(len(residual), 1), 1.0)
    residual_regularized = residual + np.eye(len(residual)) * (tol * scale)
    eigenvalues, transform = eigh(genetic, residual_regularized, check_finite=False)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    inverse_transform = np.linalg.inv(transform)
    transformed = np.asarray(Y, dtype=np.float64) @ transform
    fits = [
        {
            "h2": float(value / (value + 1.0)),
            "yvar": float(value + 1.0),
            "sigma_g2": float(value),
            "sigma_e2": 1.0,
            "converged": True,
            "estimator": "score_multivariate_component",
        }
        for value in eigenvalues
    ]
    return {
        **score,
        "Y": transformed,
        "transform": transform,
        "inverse_transform": inverse_transform,
        "generalized_eigenvalues": eigenvalues,
        "original_fits": original_fits,
        "fits": fits,
    }


def whiten_y(
    y, U, s, h2, yvar, tol=1e-10, residual_eigenvalue=0.0
):
    return _apply_eigen_operator(
        y,
        U,
        s,
        h2,
        yvar,
        power=-0.5,
        tol=tol,
        residual_eigenvalue=residual_eigenvalue,
    )


def whiten_matrix(
    x, U, s, h2, yvar, tol=1e-10, residual_eigenvalue=0.0
):
    return _apply_eigen_operator(
        x,
        U,
        s,
        h2,
        yvar,
        power=-0.5,
        tol=tol,
        residual_eigenvalue=residual_eigenvalue,
    )


def blup_resid(
    y, U, s, h2, yvar, tol=1e-10, residual_eigenvalue=0.0
):
    """GRAMMAR residual retained for compatibility."""
    y = np.asarray(y, dtype=np.float64).ravel()
    centered = y - y.mean()
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    denominator = h2 * s + 1.0 - h2
    weights = (h2 * (s - residual_eigenvalue)) / np.maximum(denominator, tol)
    return centered - np.asarray(U) @ (weights * (np.asarray(U).T @ centered))
