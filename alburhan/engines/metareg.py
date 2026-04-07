"""
Frequentist Meta-Regression Engine — WLS with Knapp-Hartung correction.

Fits a single-covariate weighted least-squares meta-regression:
    yi = b0 + b1*xi + ei

Steps:
  1. Compute DL tau2_total (without covariate) as baseline heterogeneity.
  2. WLS with weights wi = 1 / (sei^2 + tau2_total).
  3. Solve beta = (X'WX)^-1 X'Wy via np.linalg.solve.
  4. Residual tau2_res via DL on regression residuals.
  5. R2_analog = max(0, 1 - tau2_res / tau2_total).
  6. HKSJ correction: scale SE by sqrt(max(1, QE/(k-2))).
  7. F-test (Knapp-Hartung): F = (b1/se_b1)^2, df1=1, df2=k-2.
"""

import logging
import numpy as np
from scipy.stats import f as f_dist

logger = logging.getLogger(__name__)


class MetaRegressionEngine:
    name = "MetaRegression"

    def evaluate(self, claim_data):
        yi = np.array(claim_data.get("yi", []), dtype=float)
        sei = np.array(claim_data.get("sei", []), dtype=float)
        logger.info("%s: evaluating k=%d studies", self.name, len(yi))
        covariate = claim_data.get("years")

        # Require covariate
        if covariate is None:
            return {
                "status": "skipped",
                "message": "No covariate supplied (expected 'years' in claim_data).",
            }

        xi = np.array(covariate, dtype=float)
        k = len(yi)

        # Minimum k=3 (need k-2 df for HKSJ)
        if k < 3:
            return {
                "status": "skipped",
                "message": "Need k>=3 for meta-regression.",
            }

        if len(xi) != k or len(sei) != k:
            return {
                "status": "skipped",
                "message": "yi, sei and covariate must have the same length.",
            }

        # ── Step 1: DL tau2_total (no covariate, baseline) ─────────────────
        tau2_total = self._dl_tau2(yi, sei)

        # ── Step 2: WLS weights ─────────────────────────────────────────────
        wi = 1.0 / (sei ** 2 + tau2_total)

        # ── Step 3: Solve WLS beta ──────────────────────────────────────────
        # Design matrix X: [intercept, covariate]
        X = np.column_stack([np.ones(k), xi])   # shape (k, 2)
        W = np.diag(wi)
        XtWX = X.T @ W @ X
        XtWy = X.T @ (wi * yi)

        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            return {
                "status": "error",
                "message": "Singular matrix in WLS — covariate may be constant.",
            }

        b0, b1 = float(beta[0]), float(beta[1])

        # ── Step 4: Residual tau2 via DL on regression residuals ────────────
        residuals = yi - X @ beta
        tau2_res = self._dl_tau2(residuals, sei)

        # ── Step 5: R2_analog ───────────────────────────────────────────────
        if tau2_total > 0:
            r2 = float(max(0.0, 1.0 - tau2_res / tau2_total))
        else:
            r2 = 0.0
        r2 = min(r2, 1.0)  # clamp to [0,1]

        # ── Step 6: HKSJ-corrected standard errors ──────────────────────────
        # QE = weighted sum of squared residuals
        QE = float(np.sum(wi * residuals ** 2))
        df_res = k - 2  # k observations minus 2 parameters

        # Wald (unadjusted) SE
        try:
            cov_beta = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            cov_beta = np.full((2, 2), np.nan)

        se_b1_wald = float(np.sqrt(max(0.0, cov_beta[1, 1])))

        # HKSJ multiplicative correction: scale by sqrt(max(1, QE/df_res))
        hksj_scale = float(np.sqrt(max(1.0, QE / df_res)))
        se_b1 = se_b1_wald * hksj_scale
        se_b0 = float(np.sqrt(max(0.0, cov_beta[0, 0]))) * hksj_scale

        # ── Step 7: F-test (Knapp-Hartung) ──────────────────────────────────
        if se_b1 > 0:
            F_stat = float((b1 / se_b1) ** 2)
            p_value = float(1.0 - f_dist.cdf(F_stat, dfn=1, dfd=df_res))
        else:
            F_stat = 0.0
            p_value = 1.0

        # QM = Q_model (moderator test statistic, chi2 scale = F * df_num)
        QM = F_stat  # with df=1, QM ≈ F statistic

        return {
            "status": "evaluated",
            "n_studies": k,
            "slope": round(b1, 4),
            "intercept": round(b0, 4),
            "se_slope": round(se_b1, 4),
            "se_slope_wald": round(se_b1_wald, 4),
            "r2_analog": round(r2, 4),
            "tau2_total": round(float(tau2_total), 4),
            "tau2_residual": round(float(tau2_res), 4),
            "QE": round(QE, 4),
            "QM": round(QM, 4),
            "F_stat": round(F_stat, 4),
            "p_value": round(p_value, 4),
            "df_residual": df_res,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _dl_tau2(yi, sei):
        """DerSimonian-Laird tau2 estimator."""
        k = len(yi)
        wi0 = 1.0 / sei ** 2
        theta0 = np.sum(wi0 * yi) / np.sum(wi0)
        Q = float(np.sum(wi0 * (yi - theta0) ** 2))
        c = float(np.sum(wi0) - np.sum(wi0 ** 2) / np.sum(wi0))
        if c <= 0:
            return 0.0
        return max(0.0, (Q - (k - 1)) / c)
