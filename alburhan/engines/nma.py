"""
Network Consistency Engine — real influence and outlier analysis.

Without treatment network identifiers, full NMA (Bucher indirect comparison)
is not possible. This engine performs what IS computable from pairwise data:

  1. Leave-One-Out influence analysis: which study removal changes significance?
  2. Externally studentized residuals for outlier detection
  3. Cook's distance analog for meta-analysis
  4. Heterogeneity influence: each study's contribution to Q
"""

import numpy as np
from scipy import stats as sp_stats


class NetworkMetaEngine:
    name = "NetworkMeta"

    def evaluate(self, claim_data):
        yi = np.array(claim_data.get('yi', []), dtype=float)
        sei = np.array(claim_data.get('sei', []), dtype=float)

        k = len(yi)
        if k < 3:
            return {"status": "skipped", "message": "Need k>=3 for influence analysis."}

        # Full-data DL estimate
        full = self._dl_meta(yi, sei)

        # Leave-one-out analysis
        loo_results = self._leave_one_out(yi, sei, full)

        # Externally studentized residuals
        residuals = self._studentized_residuals(yi, sei, full)

        # Heterogeneity contribution per study
        q_contributions = self._q_contributions(yi, sei, full)

        # Cook's distance analog per study
        cooks_distance = self._cooks_distance(yi, sei, full, loo_results)

        # Galbraith radial plot statistics
        galbraith = self._galbraith_radial(yi, sei)

        # Identify influential studies
        influential = []
        for i in range(k):
            reasons = []
            if loo_results[i]["significance_changes"]:
                reasons.append("changes_significance")
            if abs(residuals[i]) > 2.5:
                reasons.append("outlier_residual")
            if q_contributions[i] > full["Q"] / k * 3:
                reasons.append("high_Q_contribution")
            if reasons:
                influential.append({"study_index": i, "reasons": reasons})

        return {
            "status": "evaluated",
            "full_model": {
                "theta": round(full["theta"], 4),
                "se": round(full["se"], 4),
                "Q": round(full["Q"], 3),
                "tau2": round(full["tau2"], 6),
                "I2": round(full["I2"], 1),
                "p_value": round(full["p_value"], 4),
            },
            "influential_studies": influential,
            "n_influential": len(influential),
            "max_studentized_residual": round(float(np.max(np.abs(residuals))), 3),
            "leave_one_out_range": [
                round(float(min(r["theta"] for r in loo_results)), 4),
                round(float(max(r["theta"] for r in loo_results)), 4),
            ],
            "consistency_status": "Consistent" if len(influential) == 0 else "Influential Studies Detected",
            "cooks_distance": cooks_distance,
            "galbraith": galbraith,
        }

    def _dl_meta(self, yi, sei):
        """DerSimonian-Laird random-effects meta-analysis."""
        k = len(yi)
        wi = 1.0 / sei ** 2
        sum_w = np.sum(wi)
        theta_fe = float(np.sum(wi * yi) / sum_w)
        Q = float(np.sum(wi * (yi - theta_fe) ** 2))
        C = float(sum_w - np.sum(wi ** 2) / sum_w)
        tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0

        wi_star = 1.0 / (sei ** 2 + tau2)
        theta = float(np.sum(wi_star * yi) / np.sum(wi_star))
        se = float(1.0 / np.sqrt(np.sum(wi_star)))
        z = theta / se if se > 0 else 0.0
        p_value = float(2 * (1 - sp_stats.norm.cdf(abs(z))))
        I2 = max(0.0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0.0

        return {
            "theta": theta, "se": se, "Q": Q, "tau2": tau2,
            "I2": I2, "p_value": p_value, "significant": p_value < 0.05,
        }

    def _leave_one_out(self, yi, sei, full_result):
        """Leave-one-out meta-analysis: re-fit dropping each study."""
        k = len(yi)
        results = []
        for i in range(k):
            mask = np.ones(k, dtype=bool)
            mask[i] = False
            loo = self._dl_meta(yi[mask], sei[mask])
            results.append({
                "study_dropped": i,
                "theta": loo["theta"],
                "p_value": loo["p_value"],
                "significance_changes": loo["significant"] != full_result["significant"],
            })
        return results

    def _studentized_residuals(self, yi, sei, full_result):
        """Externally studentized residuals (Viechtbauer & Cheung, 2010)."""
        tau2 = full_result["tau2"]
        theta = full_result["theta"]
        vi = sei ** 2
        wi = 1.0 / (vi + tau2)
        hat_i = wi / np.sum(wi)  # leverage
        residuals = yi - theta
        # Variance of residual_i = (vi + tau2)(1 - hat_i)
        var_ri = (vi + tau2) * (1 - hat_i)
        var_ri = np.maximum(var_ri, 1e-12)
        rstudent = residuals / np.sqrt(var_ri)
        return rstudent

    def _q_contributions(self, yi, sei, full_result):
        """Each study's contribution to Cochran's Q."""
        theta_fe = full_result["theta"]
        wi = 1.0 / sei ** 2
        return wi * (yi - theta_fe) ** 2

    def _cooks_distance(self, yi, sei, full_result, loo_results):
        """Cook's distance analog: D_i = (theta_full - theta_loo_i)^2 / var(theta_full)."""
        theta_full = full_result["theta"]
        var_full = full_result["se"] ** 2
        if var_full <= 0:
            return [0.0] * len(yi)
        cooks = []
        for loo in loo_results:
            d = (theta_full - loo["theta"]) ** 2 / var_full
            cooks.append(round(float(d), 6))
        return cooks

    def _galbraith_radial(self, yi, sei):
        """Galbraith radial plot: regress z/se on 1/se; flag studies outside ±2 residuals."""
        k = len(yi)
        x = 1.0 / sei          # precision (1/se)
        z = yi / sei           # standardised effect (z-score)

        # OLS: z = intercept + slope * x  (slope ≈ theta, intercept ≈ 0 under homogeneity)
        x_bar = float(np.mean(x))
        z_bar = float(np.mean(z))
        ssxx = float(np.sum((x - x_bar) ** 2))
        ssxz = float(np.sum((x - x_bar) * (z - z_bar)))

        if ssxx == 0:
            slope = 0.0
            intercept = z_bar
        else:
            slope = ssxz / ssxx
            intercept = z_bar - slope * x_bar

        # Residuals from the regression line
        fitted = intercept + slope * x
        residuals = z - fitted

        # Outlier indices: |residual| > 2
        outlier_indices = [int(i) for i in range(k) if abs(residuals[i]) > 2.0]

        return {
            "intercept": round(float(intercept), 4),
            "slope": round(float(slope), 4),
            "outlier_indices": outlier_indices,
        }
