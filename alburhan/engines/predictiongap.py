import math
import numpy as np
from scipy import stats

class PredictionGapEngine:
    name = "PredictionGap"
    
    def evaluate(self, claim_data):
        """
        Evaluate a claim based on CI/PI discordance.
        claim_data should contain 'yi' (effect sizes) and 'sei' (standard errors).
        """
        yi = np.array(claim_data.get('yi', []))
        sei = np.array(claim_data.get('sei', []))
        
        if len(yi) < 3:
            return {"status": "error", "message": "At least 3 studies required for PI calculation."}
            
        stats_result = self.compute_prediction_interval(yi, sei)
        discordance = self.classify_discordance(stats_result)
        
        return {
            "status": "evaluated",
            "metrics": stats_result,
            "discordance": discordance,
            "is_hollow": discordance == "FALSE_REASSURANCE"
        }

    def compute_prediction_interval(self, yi, sei, conf_level=0.95):
        k = len(yi)
        wi = 1.0 / sei ** 2
        sum_w = np.sum(wi)
        theta_fe = np.sum(wi * yi) / sum_w
        Q = float(np.sum(wi * (yi - theta_fe) ** 2))
        C = float(sum_w - np.sum(wi ** 2) / sum_w)
        tau2 = max(0, (Q - (k - 1)) / C) if C > 0 else 0

        wi_star = 1.0 / (sei ** 2 + tau2)
        theta = float(np.sum(wi_star * yi) / np.sum(wi_star))
        se_theta = float(1.0 / math.sqrt(np.sum(wi_star)))

        alpha = 1 - conf_level
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lo = theta - z_crit * se_theta
        ci_hi = theta + z_crit * se_theta

        t_crit = stats.t.ppf(1 - alpha / 2, k - 2)
        pi_se = math.sqrt(tau2 + se_theta ** 2)
        pi_lo = theta - t_crit * pi_se
        pi_hi = theta + t_crit * pi_se

        I2 = max(0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(theta / se_theta))) if se_theta > 0 else 1

        # Knapp-Hartung adjusted CI (HKSJ variant)
        # q = sum(wi*(yi-theta)^2)/(k-1), se_kh = se * sqrt(max(1, q))
        q_kh = float(np.sum(wi_star * (yi - theta) ** 2) / (k - 1)) if k > 1 else 1.0
        se_kh = se_theta * math.sqrt(max(1.0, q_kh))
        t_kh = stats.t.ppf(1 - alpha / 2, k - 1)
        kh_ci_lo = theta - t_kh * se_kh
        kh_ci_hi = theta + t_kh * se_kh

        return {
            'theta': theta,
            'se': se_theta,
            'tau2': tau2,
            'I2': I2,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'pi_lo': pi_lo,
            'pi_hi': pi_hi,
            'pi_width': pi_hi - pi_lo,
            'ci_width': ci_hi - ci_lo,
            'pi_ci_ratio': (pi_hi - pi_lo) / (ci_hi - ci_lo) if (ci_hi - ci_lo) > 0 else float('inf'),
            'p_value': p_value,
            'k': k,
            'kh_ci_lo': kh_ci_lo,
            'kh_ci_hi': kh_ci_hi,
        }

    def classify_discordance(self, result):
        null = 0.0
        ci_excludes_null = (result['ci_lo'] > null) or (result['ci_hi'] < null)
        pi_excludes_null = (result['pi_lo'] > null) or (result['pi_hi'] < null)
        
        if ci_excludes_null and not pi_excludes_null:
            return 'FALSE_REASSURANCE'
        elif ci_excludes_null and pi_excludes_null:
            return 'CONCORDANT_SIG'
        elif not ci_excludes_null:
            return 'CONCORDANT_NS'
        return 'UNKNOWN'
