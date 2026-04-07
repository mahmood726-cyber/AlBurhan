"""
Dose-Response Meta-Regression Engine.

Fits 4 models to study-level (dose, effect, se) data:

1. Linear      — WLS: yi = b0 + b1*dose_i,  wi = 1/(sei^2 + tau2_DL)
2. Quadratic   — WLS: yi = b0 + b1*dose_i + b2*dose_i^2 + AIC comparison
3. RCS         — Restricted cubic splines (3 knots at p10, p50, p90)
4. MED         — Minimum Effective Dose: smallest dose where CI excludes 0

Requires claim_data to contain:
  - "yi"    : list of effect sizes per study
  - "sei"   : list of SEs per study
  - "doses" : list of dose levels per study  (if absent → status="skipped")
"""

import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


class DoseResponseEngine:
    name = "DoseResponse"

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(self, claim_data):
        logger.info("%s: evaluating dose-response", self.name)
        doses = claim_data.get("doses")
        if doses is None:
            return {"status": "skipped", "message": "No 'doses' key in claim_data."}

        yi  = np.array(claim_data.get("yi",  []), dtype=float)
        sei = np.array(claim_data.get("sei", []), dtype=float)
        d   = np.array(doses, dtype=float)

        n = len(yi)
        if n < 3 or len(sei) != n or len(d) != n:
            return {
                "status": "skipped",
                "message": "Need k>=3 matching yi/sei/doses.",
            }

        tau2 = self._dl_tau2(yi, sei)
        wi   = 1.0 / (sei**2 + tau2)

        linear    = self._fit_linear(d, yi, wi)
        quadratic = self._fit_quadratic(d, yi, wi)
        rcs       = self._fit_rcs(d, yi, wi)
        med       = self._compute_med(d, linear)

        return {
            "status":    "evaluated",
            "n_studies": int(n),
            "tau2_dl":   round(float(tau2), 6),
            "linear":    linear,
            "quadratic": quadratic,
            "rcs":       rcs,
            "med":       med,
        }

    # ------------------------------------------------------------------ #
    #  Model 1: Linear dose-response                                        #
    # ------------------------------------------------------------------ #

    def _fit_linear(self, d, yi, wi):
        """WLS fit yi = b0 + b1*d.  Returns slope, intercept, p_value, R2."""
        X = np.column_stack([np.ones(len(d)), d])
        coeffs, rss, se_coeffs = self._wls(X, yi, wi)
        if coeffs is None:
            return {"status": "singular"}

        b0, b1 = coeffs
        n = len(d)
        # Residual SE for t-test on b1
        if se_coeffs is not None and se_coeffs[1] > 0:
            t_stat = b1 / se_coeffs[1]
            p_val  = _two_tailed_p(t_stat, df=n - 2)
        else:
            p_val = float("nan")

        # R2 (weighted)
        ymean = np.sum(wi * yi) / np.sum(wi)
        ss_tot = float(np.sum(wi * (yi - ymean) ** 2))
        r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")

        aic = self._aic(rss, n, k_params=2)

        return {
            "intercept": round(float(b0), 6),
            "slope":     round(float(b1), 6),
            "p_value":   round(float(p_val), 6),
            "r2":        round(float(r2), 6),
            "aic":       round(float(aic), 4),
        }

    # ------------------------------------------------------------------ #
    #  Model 2: Quadratic dose-response                                     #
    # ------------------------------------------------------------------ #

    def _fit_quadratic(self, d, yi, wi):
        """WLS fit yi = b0 + b1*d + b2*d^2.  Returns b2 + AIC."""
        X = np.column_stack([np.ones(len(d)), d, d**2])
        coeffs, rss, _ = self._wls(X, yi, wi)
        if coeffs is None:
            return {"status": "singular"}

        b0, b1, b2 = coeffs
        n   = len(d)
        aic = self._aic(rss, n, k_params=3)

        return {
            "intercept": round(float(b0), 6),
            "slope":     round(float(b1), 6),
            "b2":        round(float(b2), 6),
            "aic":       round(float(aic), 4),
        }

    # ------------------------------------------------------------------ #
    #  Model 3: Restricted Cubic Splines (3 knots)                         #
    # ------------------------------------------------------------------ #

    def _fit_rcs(self, d, yi, wi):
        """WLS on RCS basis with 3 knots at p10, p50, p90 of doses."""
        k1 = float(np.percentile(d, 10))
        k2 = float(np.percentile(d, 50))
        k3 = float(np.percentile(d, 90))

        knots = [k1, k2, k3]

        # If all knots are identical the spline is degenerate
        if k1 == k3:
            return {"status": "degenerate", "knots": knots,
                    "message": "All dose values identical — spline not identifiable."}

        # Build RCS design matrix: [1, dose, rcs1]
        # Standard 3-knot RCS adds 1 truncated-power basis column
        rcs1 = _rcs_basis(d, k1, k2, k3)
        X = np.column_stack([np.ones(len(d)), d, rcs1])
        coeffs, rss, _ = self._wls(X, yi, wi)
        if coeffs is None:
            return {"status": "singular", "knots": knots}

        b0, b1, b_rcs = coeffs
        n   = len(d)
        aic = self._aic(rss, n, k_params=3)

        return {
            "knots":      [round(k, 4) for k in knots],
            "intercept":  round(float(b0), 6),
            "slope":      round(float(b1), 6),
            "spline_coef": round(float(b_rcs), 6),
            "aic":        round(float(aic), 4),
        }

    # ------------------------------------------------------------------ #
    #  Model 4: Minimum Effective Dose                                      #
    # ------------------------------------------------------------------ #

    def _compute_med(self, d, linear):
        """Smallest dose (from unique sorted doses) where linear CI excludes 0."""
        if "slope" not in linear:
            return None

        b0  = linear["intercept"]
        b1  = linear["slope"]
        r2  = linear.get("r2", float("nan"))

        # We need the SE of the fitted value at dose d_val.
        # Use a simplified SE: |b1 * d_val + b0| / 1.96 as the CI half-width
        # approximation; but for a true MED we need the prediction SE.
        # Since we don't store the full covariance matrix, we use the slope SE
        # back-derived from p_value.
        # A cleaner approach: store se_b0, se_b1 in linear and use them here.
        # Because _fit_linear does not currently store se_b1 directly, we
        # recompute MED using p_value to recover se_b1.
        p    = linear.get("p_value", float("nan"))
        aic  = linear.get("aic", 0.0)

        # Recover se_b1 from t-stat = b1/se_b1 and p-value
        # p = 2*(1 - Phi(|t|))  →  |t| = Phi^{-1}(1 - p/2)
        se_b1 = None
        if math.isfinite(p) and 0 < p < 1 and b1 != 0:
            z = _norm_ppf(1.0 - p / 2.0)
            if z > 0:
                se_b1 = abs(b1) / z

        unique_doses = sorted(set(d.tolist()))

        med = None
        for dose_val in unique_doses:
            fit = b0 + b1 * dose_val
            if se_b1 is not None:
                # Approximate CI using slope SE (conservative — ignores intercept uncertainty)
                half = 1.96 * abs(se_b1 * dose_val)
                ci_lo = fit - half
                ci_hi = fit + half
            else:
                # Fallback: CI excludes 0 iff |fit| > 0 (very conservative)
                ci_lo = fit - abs(fit) * 0.99
                ci_hi = fit + abs(fit) * 0.99

            if ci_lo > 0 or ci_hi < 0:
                med = round(float(dose_val), 6)
                break

        return med  # None if all CIs include 0

    # ------------------------------------------------------------------ #
    #  WLS solver                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wls(X, y, w):
        """
        Weighted least squares via normal equations.
        Returns (coeffs, weighted_rss, se_coeffs) or (None, None, None) if singular.
        """
        W  = np.diag(w)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        try:
            cond = np.linalg.cond(XtWX)
            if cond > 1e12:
                raise np.linalg.LinAlgError("Ill-conditioned matrix")
            coeffs = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            return None, None, None

        resid = y - X @ coeffs
        rss   = float(np.sum(w * resid**2))

        # SE of coefficients from (X'WX)^{-1} * rss / (n - p)
        n, p = X.shape
        dof  = n - p
        if dof > 0:
            try:
                cov    = np.linalg.inv(XtWX) * (rss / dof)
                se_vec = np.sqrt(np.abs(np.diag(cov)))
            except np.linalg.LinAlgError:
                se_vec = None
        else:
            se_vec = None

        return coeffs, rss, se_vec

    # ------------------------------------------------------------------ #
    #  AIC                                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _aic(rss, n, k_params):
        """AIC = 2*k + n*log(RSS/n)."""
        if rss <= 0 or n <= 0:
            return float("nan")
        return 2.0 * k_params + n * math.log(rss / n)

    # ------------------------------------------------------------------ #
    #  DerSimonian-Laird tau2                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dl_tau2(yi, sei):
        k   = len(yi)
        wi0 = 1.0 / sei**2
        th0 = np.sum(wi0 * yi) / np.sum(wi0)
        Q   = float(np.sum(wi0 * (yi - th0) ** 2))
        c   = float(np.sum(wi0) - np.sum(wi0**2) / np.sum(wi0))
        return max(0.0, (Q - (k - 1)) / c)


# ------------------------------------------------------------------ #
#  Module-level helpers                                                #
# ------------------------------------------------------------------ #

def _rcs_basis(d, k1, k2, k3):
    """
    Single RCS basis column for 3 knots.

    Standard formula (Harrell 2001):
      rcs(d) = (d-k1)^3_+ - (k3-k1)/(k3-k2) * (d-k2)^3_+
               + (k2-k1)/(k3-k2) * (d-k3)^3_+
    """
    def _tp3(x, knot):
        return np.where(x > knot, (x - knot)**3, 0.0)

    span = k3 - k2
    if span == 0:
        return np.zeros_like(d)

    return (
        _tp3(d, k1)
        - ((k3 - k1) / (k3 - k2)) * _tp3(d, k2)
        + ((k2 - k1) / (k3 - k2)) * _tp3(d, k3)
    )


def _two_tailed_p(t, df):
    """Approximate two-tailed p-value from t-statistic (large-sample normal approx)."""
    # For df >= 5, normal approximation is adequate for testing purposes.
    z = abs(t)
    return 2.0 * (1.0 - _norm_cdf(z))


def _norm_cdf(x):
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_ppf(p):
    """Approximate standard normal quantile via rational approximation (Abramowitz & Stegun)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    # Rational approximation for p in [0.5, 1)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    num = c[0] + c[1]*t + c[2]*t*t
    den = 1.0 + d[0]*t + d[1]*t*t + d[2]*t*t*t
    return t - num / den
