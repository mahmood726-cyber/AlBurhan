import numpy as np

class CausalSynthEngine:
    name = "CausalSynth"

    def evaluate(self, claim_data):
        """
        Evaluate causal robustness and confounding sensitivity.
        Uses E-value (VanderWeele & Ding, 2017) to quantify how much
        unmeasured confounding would be needed to nullify the observed effect.

        Reports both point-estimate E-value and CI-based E-value (STAT-P0-5).
        Returns status="skipped" if no valid theta available (STAT-P1-3).
        """
        theta = claim_data.get('theta', None)
        se = claim_data.get('se', None)

        # Skip if no valid upstream estimate (STAT-P1-3)
        if theta is None:
            return {"status": "skipped", "message": "No valid effect estimate available."}

        # Point estimate E-value
        rr = np.exp(abs(float(theta)))  # abs() makes E-value symmetric
        e_value_point = self._compute_e_value(rr)

        # CI-based E-value: E-value for the CI bound closest to null (STAT-P0-5)
        e_value_ci = 1.0
        if se is not None and float(se) > 0:
            # CI bound closest to null
            lower_rr = np.exp(abs(float(theta)) - 1.96 * float(se))
            e_value_ci = self._compute_e_value(lower_rr)

        # Bias-adjusted E-value: E-value after adjusting RR for measured confounders
        # rr_adjusted = rr * bias_factor (default 0.9 = 10% reduction)
        bias_factor = float(claim_data.get('bias_factor', 0.9))
        rr_adjusted = rr * bias_factor
        e_value_bias_adjusted = self._compute_e_value(rr_adjusted)

        return {
            "status": "evaluated",
            "e_value": round(float(e_value_point), 3),
            "e_value_ci": round(float(e_value_ci), 3),
            "e_value_bias_adjusted": round(float(e_value_bias_adjusted), 3),
            "causal_path_stability": "High" if e_value_ci > 2.0 else "Fragile",
            "confounding_resistance": "Robust" if e_value_ci > 3.0 else "Moderate"
        }

    @staticmethod
    def _compute_e_value(rr):
        """E-value formula per VanderWeele & Ding (2017)."""
        if rr <= 1.0:
            return 1.0
        return float(rr + np.sqrt(rr * (rr - 1)))
