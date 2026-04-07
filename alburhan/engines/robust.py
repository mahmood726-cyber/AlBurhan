"""
Robust Meta-Analysis Engine — 3 outlier-resistant estimators.

1. Paule-Mandel (PM): Iterative tau2 that solves Q(tau2) = k-1 via brentq.
2. Weighted Median: Weighted median of study effects.
3. Winsorized Mean: Trim extreme 10% of effects before weighted average.

Each method returns: theta, se, tau2, ci_lo, ci_hi (95% Wald CI).
"""

import logging
import numpy as np
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


class RobustMAEngine:
    name = "RobustMA"

    def evaluate(self, claim_data):
        yi = np.array(claim_data.get("yi", []), dtype=float)
        sei = np.array(claim_data.get("sei", []), dtype=float)
        k = len(yi)
        logger.info("%s: evaluating k=%d studies", self.name, k)

        if k < 3:
            return {
                "status": "skipped",
                "message": "Need k>=3 for robust meta-analysis.",
            }

        pm = self._paule_mandel(yi, sei)
        wmed = self._weighted_median(yi, sei)
        wins = self._winsorized_mean(yi, sei)

        return {
            "status": "evaluated",
            "n_studies": k,
            "paule_mandel": pm,
            "weighted_median": wmed,
            "winsorized_mean": wins,
        }

    # ------------------------------------------------------------------ #
    #  Method 1: Paule-Mandel                                              #
    # ------------------------------------------------------------------ #

    def _paule_mandel(self, yi, sei):
        """Paule-Mandel tau2 via brentq on Q(tau2) = k-1."""
        k = len(yi)

        def _theta(tau2):
            wi = 1.0 / (sei**2 + tau2)
            return np.sum(wi * yi) / np.sum(wi)

        def _Q(tau2):
            wi = 1.0 / (sei**2 + tau2)
            theta = np.sum(wi * yi) / np.sum(wi)
            return float(np.sum(wi * (yi - theta) ** 2))

        target = k - 1

        # If Q(0) < k-1 there is no heterogeneity
        if _Q(0.0) <= target:
            tau2 = 0.0
            converged = True
        else:
            try:
                tau2 = brentq(_Q_minus_target := lambda t: _Q(t) - target, 0.0, 100.0)
                converged = True
            except ValueError:
                tau2 = 0.0
                converged = False

        theta = _theta(tau2)
        wi = 1.0 / (sei**2 + tau2)
        se = float(np.sqrt(1.0 / np.sum(wi)))
        ci_lo = theta - 1.96 * se
        ci_hi = theta + 1.96 * se

        return {
            "theta": round(float(theta), 4),
            "se": round(se, 4),
            "tau2": round(float(tau2), 4),
            "ci_lo": round(float(ci_lo), 4),
            "ci_hi": round(float(ci_hi), 4),
            "converged": converged,
        }

    # ------------------------------------------------------------------ #
    #  Method 2: Weighted Median                                           #
    # ------------------------------------------------------------------ #

    def _weighted_median(self, yi, sei):
        """Weighted median using DerSimonian-Laird tau2 for weights."""
        tau2_dl = self._dl_tau2(yi, sei)
        wi = 1.0 / (sei**2 + tau2_dl)

        # Sort by effect size
        order = np.argsort(yi)
        yi_s = yi[order]
        wi_s = wi[order]

        # Cumulative weights normalised to [0, 1]
        cumw = np.cumsum(wi_s)
        half = cumw[-1] / 2.0

        # Weighted median: first yi where cumulative weight >= half
        idx = int(np.searchsorted(cumw, half))
        idx = min(idx, len(yi_s) - 1)
        theta = float(yi_s[idx])

        # SE: use DL se for interval
        se_dl = float(np.sqrt(1.0 / np.sum(wi)))
        ci_lo = theta - 1.96 * se_dl
        ci_hi = theta + 1.96 * se_dl

        return {
            "theta": round(theta, 4),
            "se": round(se_dl, 4),
            "tau2": round(float(tau2_dl), 4),
            "ci_lo": round(float(ci_lo), 4),
            "ci_hi": round(float(ci_hi), 4),
        }

    # ------------------------------------------------------------------ #
    #  Method 3: Winsorized Mean                                           #
    # ------------------------------------------------------------------ #

    def _winsorized_mean(self, yi, sei):
        """Winsorized mean — replace bottom/top 10% effects then reweight."""
        tau2_dl = self._dl_tau2(yi, sei)
        wi = 1.0 / (sei**2 + tau2_dl)

        # Compute 10th / 90th percentiles of yi
        p10 = float(np.percentile(yi, 10))
        p90 = float(np.percentile(yi, 90))

        # Winsorise
        yi_w = np.clip(yi, p10, p90)

        theta = float(np.sum(wi * yi_w) / np.sum(wi))
        se = float(np.sqrt(1.0 / np.sum(wi)))
        ci_lo = theta - 1.96 * se
        ci_hi = theta + 1.96 * se

        return {
            "theta": round(theta, 4),
            "se": round(se, 4),
            "tau2": round(float(tau2_dl), 4),
            "ci_lo": round(float(ci_lo), 4),
            "ci_hi": round(float(ci_hi), 4),
        }

    # ------------------------------------------------------------------ #
    #  Helper: DerSimonian-Laird tau2                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dl_tau2(yi, sei):
        k = len(yi)
        wi0 = 1.0 / sei**2
        theta0 = np.sum(wi0 * yi) / np.sum(wi0)
        Q = float(np.sum(wi0 * (yi - theta0) ** 2))
        c = float(np.sum(wi0) - np.sum(wi0**2) / np.sum(wi0))
        tau2 = max(0.0, (Q - (k - 1)) / c)
        return tau2
