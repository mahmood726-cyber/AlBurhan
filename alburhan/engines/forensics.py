"""
Registry Forensics Engine — real statistical forensic analysis.

Methods:
  1. Terminal Digit Analysis (TDA): chi-squared test on last digit uniformity
  2. GRIM Test: checks if reported proportions are consistent with integer N
  3. SE Homogeneity: Cochran's C test for suspiciously similar standard errors
  4. Normality: Shapiro-Wilk test on standardized residuals
  5. Shannon Entropy: information-theoretic measure of effect size dispersion
"""

import math
import numpy as np
from scipy import stats as sp_stats


class RegistryForensicsEngine:
    name = "RegistryForensics"

    def evaluate(self, claim_data):
        yi = np.array(claim_data.get('yi', []), dtype=float)
        sei = np.array(claim_data.get('sei', []), dtype=float)
        treat_events = claim_data.get('treat_events')
        treat_total = claim_data.get('treat_total')

        k = len(yi)
        if k < 3:
            return {"status": "skipped", "message": "Need k>=3 for forensic analysis."}

        results = {"status": "evaluated"}

        # 1. Terminal Digit Analysis on effect sizes
        results["terminal_digit"] = self._terminal_digit_test(yi)

        # 2. GRIM test (only if we have count data)
        if treat_events is not None and treat_total is not None:
            results["grim"] = self._grim_test(treat_events, treat_total)
        else:
            results["grim"] = {"status": "skipped", "reason": "No count data provided"}

        # 3. SE Homogeneity (Cochran's C)
        results["se_homogeneity"] = self._se_homogeneity_test(sei)

        # 4. Shapiro-Wilk on standardized residuals
        results["normality"] = self._normality_test(yi, sei)

        # 5. Shannon Entropy of effect distribution
        results["scientific_entropy"] = self._shannon_entropy(yi)

        # Composite anomaly count
        flags = sum(1 for key in ("terminal_digit", "se_homogeneity", "normality")
                    if results.get(key, {}).get("flagged", False))
        if isinstance(results["grim"], dict) and results["grim"].get("flagged", False):
            flags += 1

        results["anomaly_flags"] = flags
        results["total_tests"] = 4 if treat_events is not None else 3
        results["forensic_status"] = "Anomalies Detected" if flags >= 2 else "Nominal"

        return results

    def _terminal_digit_test(self, yi, alpha=0.05):
        """Chi-squared goodness-of-fit for uniform distribution of last digits."""
        # Extract last significant digit from each effect size
        digits = []
        for val in yi:
            if val == 0:
                digits.append(0)
                continue
            # Get the last non-zero decimal digit (up to 4 decimal places)
            s = f"{abs(val):.4f}"
            # Strip trailing zeros and get last digit
            s = s.rstrip('0')
            if s.endswith('.'):
                digits.append(0)
            else:
                digits.append(int(s[-1]))

        if len(digits) < 5:
            return {"status": "skipped", "reason": "Need >=5 studies for TDA"}

        observed = np.zeros(10)
        for d in digits:
            observed[d] += 1

        expected = np.full(10, len(digits) / 10)
        # Merge bins with expected < 1
        mask = expected >= 1
        if mask.sum() < 2:
            return {"status": "skipped", "reason": "Insufficient digit variation"}

        chi2, p_value = sp_stats.chisquare(observed[mask], expected[mask])

        return {
            "chi2": round(float(chi2), 3),
            "p_value": round(float(p_value), 4),
            "df": int(mask.sum() - 1),
            "flagged": p_value < alpha
        }

    def _grim_test(self, events, totals, tolerance=0.01):
        """GRIM consistency: check if proportions are achievable with integer N."""
        inconsistent = 0
        tested = 0
        for e, n in zip(events, totals):
            if n <= 0:
                continue
            tested += 1
            proportion = e / n
            # For integer e and n, proportion must be a multiple of 1/n
            granularity = 1.0 / n
            remainder = proportion % granularity
            # Check if proportion is consistent (within floating point tolerance)
            if remainder > tolerance and (granularity - remainder) > tolerance:
                inconsistent += 1

        return {
            "inconsistent_count": inconsistent,
            "tested": tested,
            "proportion_inconsistent": round(inconsistent / tested, 3) if tested > 0 else 0,
            "flagged": inconsistent > 0
        }

    def _se_homogeneity_test(self, sei, alpha=0.05):
        """Cochran's C test: is the largest SE suspiciously dominant or are SEs too similar?"""
        k = len(sei)
        if k < 3:
            return {"status": "skipped", "reason": "Need k>=3"}

        variances = sei ** 2
        total_var = np.sum(variances)
        if total_var == 0:
            return {"cochrans_c": 0.0, "p_value": 1.0, "flagged": False}

        c_stat = float(np.max(variances) / total_var)

        # Under H0 (all variances equal), expected C = 1/k
        # Approximate critical value using F-distribution
        # C_crit ~ 1 / (1 + (k-1)/F_{alpha, df1=k-1, df2=(k-1)*(n-1)})
        # Simplified: check if C is much smaller than expected (too uniform)
        cv = float(np.std(sei) / np.mean(sei)) if np.mean(sei) > 0 else 0

        # SEs suspiciously uniform if CV < 0.1 with k >= 5
        too_uniform = cv < 0.10 and k >= 5

        return {
            "cochrans_c": round(c_stat, 4),
            "se_cv": round(cv, 4),
            "flagged": too_uniform
        }

    def _normality_test(self, yi, sei, alpha=0.05):
        """Shapiro-Wilk test on standardized residuals."""
        k = len(yi)
        if k < 3:
            return {"status": "skipped", "reason": "Need k>=3"}

        # Compute RE model residuals
        wi = 1.0 / sei ** 2
        theta_fe = np.sum(wi * yi) / np.sum(wi)
        residuals = (yi - theta_fe) / sei  # Standardized residuals

        if k > 5000:
            # Shapiro-Wilk limited to 5000
            return {"status": "skipped", "reason": "k > 5000"}

        stat, p_value = sp_stats.shapiro(residuals)
        return {
            "shapiro_w": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "flagged": p_value < alpha
        }

    def _shannon_entropy(self, yi):
        """Shannon entropy of effect size distribution (binned)."""
        k = len(yi)
        if k < 3:
            return 0.0

        bins = max(3, min(k // 2, 10))
        hist, _ = np.histogram(yi, bins=bins)
        hist = hist[hist > 0].astype(float)
        if len(hist) == 0:
            return 0.0
        probs = hist / hist.sum()
        entropy = float(-np.sum(probs * np.log2(probs)))
        return round(entropy, 3)
