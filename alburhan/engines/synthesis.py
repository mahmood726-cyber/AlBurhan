"""
Synthesis Loss Engine — real information-theoretic evidence loss quantification.

Computes from actual study data:
  1. Design Effect: effective sample size reduction from heterogeneity
  2. Information Loss Ratio: FE vs RE precision gap
  3. Effective N: how many homogeneous studies is this body worth?
  4. Redundancy: proportion of total variance due to between-study heterogeneity
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class SynthesisLossEngine:
    name = "SynthesisLoss"

    def evaluate(self, claim_data):
        yi = np.array(claim_data.get('yi', []), dtype=float)
        sei = np.array(claim_data.get('sei', []), dtype=float)
        logger.info("%s: evaluating k=%d studies", self.name, len(yi))
        n_per_study = claim_data.get('n_per_study')

        k = len(yi)
        if k < 2:
            return {"status": "skipped", "message": "Need k>=2 for synthesis loss computation."}

        vi = sei ** 2
        wi_fe = 1.0 / vi

        # DL tau2
        theta_fe = float(np.sum(wi_fe * yi) / np.sum(wi_fe))
        Q = float(np.sum(wi_fe * (yi - theta_fe) ** 2))
        C = float(np.sum(wi_fe) - np.sum(wi_fe ** 2) / np.sum(wi_fe))
        tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0

        # 1. Information Loss Ratio: how much precision is lost to heterogeneity
        se_fe = float(1.0 / np.sqrt(np.sum(wi_fe)))
        wi_re = 1.0 / (vi + tau2)
        se_re = float(1.0 / np.sqrt(np.sum(wi_re)))
        # Loss = 1 - (FE_variance / RE_variance) = fraction of RE variance due to tau2
        info_loss = 1.0 - (se_fe ** 2 / se_re ** 2) if se_re > 0 else 0.0
        info_loss = max(0.0, info_loss)

        # 2. Design Effect: DEFF = 1 + (k-1) * I2_proportion
        I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
        deff = 1.0 + (k - 1) * I2

        # 3. Effective N: actual total N / DEFF
        if n_per_study is not None:
            total_n = sum(n_per_study)
        else:
            # Estimate from SE: for log-OR, N ~ 4/SE^2
            total_n = sum(max(20, int(4.0 / se_val ** 2)) for se_val in sei)
        effective_n = total_n / deff if deff > 0 else total_n

        # 4. Redundancy (proportion of total variance from between-study heterogeneity)
        avg_vi = float(np.mean(vi))
        redundancy = tau2 / (tau2 + avg_vi) if (tau2 + avg_vi) > 0 else 0.0

        # 5. Uplift potential: if heterogeneity were eliminated
        uplift_se_ratio = se_fe / se_re if se_re > 0 else 1.0  # <1 means FE is more precise

        return {
            "status": "evaluated",
            "information_loss_ratio": round(float(info_loss), 3),
            "design_effect": round(float(deff), 3),
            "total_n": int(total_n),
            "effective_n": round(float(effective_n), 1),
            "redundancy": round(float(redundancy), 3),
            "I2_proportion": round(float(I2), 3),
            "uplift_se_ratio": round(float(uplift_se_ratio), 3),
            "is_wasteful": info_loss > 0.5,
        }
