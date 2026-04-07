"""
SequentialTSAEngine — Proper alpha-spending Trial Sequential Analysis.

Implements:
- Lan-DeMets O'Brien-Fleming spending function
- Lan-DeMets Pocock spending function
- Beta-spending for futility (symmetric)
- Heterogeneity-corrected RIS (Required Information Size)
- Conditional power at each interim look
- Cumulative Z-statistic trajectory
"""

import math
import logging
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class SequentialTSAEngine:
    """Proper alpha-spending TSA engine (Lan-DeMets OBF + Pocock)."""

    name = "SequentialTSA"

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def evaluate(self, claim_data):
        """
        Evaluate TSA with alpha-spending boundaries.

        Expects: yi, sei, years (for ordering), n_per_study (optional),
                 alpha (default 0.05), power (default 0.80),
                 spending (default 'obf'; or 'pocock').
        Returns dict with status, boundaries, z_trajectory, ris,
                conditional_powers, boundary_crossed.
        """
        yi = np.array(claim_data.get("yi", []), dtype=float)
        sei = np.array(claim_data.get("sei", []), dtype=float)
        logger.info("%s: evaluating k=%d studies", self.name, len(yi))
        years = list(claim_data.get("years", []))
        n_per_study = claim_data.get("n_per_study", None)
        alpha = float(claim_data.get("alpha", 0.05))
        power = float(claim_data.get("power", 0.80))
        spending = claim_data.get("spending", "obf").lower()

        if len(yi) < 2:
            return {
                "status": "skipped",
                "message": "Need k>=2 studies for sequential analysis.",
            }

        # Sort by year
        idx = np.argsort(years)
        yi_s = yi[idx]
        sei_s = sei[idx]

        if n_per_study is not None:
            n_s = [n_per_study[i] for i in idx]
        else:
            # Estimate N from SE for log-OR: N ~ 4/SE^2
            n_s = [max(20, int(4.0 / (se ** 2))) for se in sei_s]

        # ---- RIS (heterogeneity-corrected) --------------------------------
        ris = self._compute_ris(yi_s, sei_s, n_s, alpha, power)

        # ---- Cumulative meta-analysis at each look -------------------------
        looks = self._cumulative_meta(yi_s, sei_s, n_s)

        # ---- Alpha-spending boundaries ------------------------------------
        obf_alpha_boundaries = []
        obf_futility_boundaries = []
        pocock_alpha_boundaries = []
        pocock_futility_boundaries = []
        z_trajectory = []
        conditional_powers = []
        boundary_crossed = False

        z_alpha_half = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        # Final OBF boundary (at t=1): z_alpha_half
        z_final_obf = z_alpha_half

        for look in looks:
            t = look["cumN"] / ris if ris > 0 else 1.0
            t = min(t, 1.0)  # cap at 1

            z_trajectory.append(look["z"])

            # -- OBF efficacy boundary at this look --
            alpha_spent_obf = self._obf_spending(t, alpha)
            z_obf = self._alpha_spent_to_boundary(alpha_spent_obf)
            obf_alpha_boundaries.append(z_obf)

            # -- OBF futility boundary (beta-spending, symmetric) --
            beta = 1.0 - power
            beta_spent_obf = self._obf_spending(t, beta)
            z_fut_obf = self._alpha_spent_to_boundary(beta_spent_obf)
            obf_futility_boundaries.append(z_fut_obf)

            # -- Pocock efficacy boundary at this look --
            alpha_spent_poc = self._pocock_spending(t, alpha)
            z_poc = self._alpha_spent_to_boundary(alpha_spent_poc)
            pocock_alpha_boundaries.append(z_poc)

            # -- Pocock futility boundary --
            beta_spent_poc = self._pocock_spending(t, beta)
            z_fut_poc = self._alpha_spent_to_boundary(beta_spent_poc)
            pocock_futility_boundaries.append(z_fut_poc)

            # -- Conditional power (Brownian motion under H1) --
            cp = self._conditional_power(
                look["z"], t, z_final_obf, z_alpha_half, z_beta
            )
            conditional_powers.append(cp)

            # -- Check OBF boundary crossing --
            if abs(look["z"]) >= z_obf:
                boundary_crossed = True

        return {
            "status": "evaluated",
            "ris": float(ris),
            "z_trajectory": z_trajectory,
            "information_fractions": [
                min(lk["cumN"] / ris, 1.0) for lk in looks
            ] if ris > 0 else [],
            "obf_alpha_boundaries": obf_alpha_boundaries,
            "obf_futility_boundaries": obf_futility_boundaries,
            "pocock_alpha_boundaries": pocock_alpha_boundaries,
            "pocock_futility_boundaries": pocock_futility_boundaries,
            "conditional_powers": conditional_powers,
            "boundary_crossed": boundary_crossed,
            "final_I2": looks[-1]["I2"] if looks else None,
            "n_looks": len(looks),
        }

    # ------------------------------------------------------------------ #
    #  Spending functions                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _obf_spending(t: float, alpha: float) -> float:
        """
        Lan-DeMets O'Brien-Fleming spending function.
        alpha*(t) = 2 - 2*Phi(z_{alpha/2} / sqrt(t))
        where z_{alpha/2} = Phi^{-1}(1 - alpha/2).
        """
        if t <= 0.0:
            return 0.0
        t = min(t, 1.0)
        z_half = stats.norm.ppf(1.0 - alpha / 2.0)
        return 2.0 - 2.0 * stats.norm.cdf(z_half / math.sqrt(t))

    @staticmethod
    def _pocock_spending(t: float, alpha: float) -> float:
        """
        Lan-DeMets Pocock spending function.
        alpha*(t) = alpha * ln(1 + (e-1)*t)
        """
        if t <= 0.0:
            return 0.0
        t = min(t, 1.0)
        return alpha * math.log(1.0 + (math.e - 1.0) * t)

    @staticmethod
    def _alpha_spent_to_boundary(alpha_spent: float) -> float:
        """
        Convert cumulative alpha spent to two-sided Z boundary.
        z_k = Phi^{-1}(1 - alpha_spent/2)
        """
        alpha_spent = max(alpha_spent, 1e-12)
        alpha_spent = min(alpha_spent, 1.0 - 1e-12)
        return stats.norm.ppf(1.0 - alpha_spent / 2.0)

    # ------------------------------------------------------------------ #
    #  RIS calculation                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_ris(yi_s, sei_s, n_s, alpha: float, power: float) -> float:
        """
        Heterogeneity-adjusted Required Information Size.

        Uses the actual meta-analytic variance:
          FE: var_pooled = 1/sum(1/sei^2)
          RE: var_pooled = 1/sum(1/(sei^2 + tau2))

        RIS = (z_alpha + z_beta)^2 * var_pooled / delta^2 * (1 + D2)
        D2 = min(I2/(1-I2), 10)
        delta = observed pooled effect (actual target)
        """
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
        z_beta = stats.norm.ppf(power)

        k = len(yi_s)
        wi = 1.0 / sei_s ** 2
        theta_fe = float(np.sum(wi * yi_s) / np.sum(wi))
        Q = float(np.sum(wi * (yi_s - theta_fe) ** 2))
        C = float(np.sum(wi) - np.sum(wi ** 2) / np.sum(wi))
        tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0

        # RE weights and pooled estimate
        wi_re = 1.0 / (sei_s ** 2 + tau2)
        theta = float(np.sum(wi_re * yi_s) / np.sum(wi_re))

        # Actual meta-analytic variance (RE): 1/sum_weights
        var_pooled = float(1.0 / np.sum(wi_re))

        # Target delta: use the observed pooled effect size
        delta = abs(theta) if abs(theta) > 1e-6 else 0.1

        ris_base = (z_alpha + z_beta) ** 2 * var_pooled / (delta ** 2)
        ris_base = max(ris_base, 50.0)

        # Heterogeneity correction
        I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
        D2 = min(I2 / (1.0 - I2), 10.0) if I2 < 1.0 else 10.0
        ris_adj = ris_base * (1.0 + D2)

        return float(ris_adj)

    # ------------------------------------------------------------------ #
    #  Cumulative meta-analysis (DerSimonian-Laird)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cumulative_meta(yi_s, sei_s, n_s):
        """DL cumulative meta-analysis; starts at look k=2."""
        results = []
        for k in range(2, len(yi_s) + 1):
            sub_yi = yi_s[:k]
            sub_sei = sei_s[:k]

            wi = 1.0 / sub_sei ** 2
            theta_fe = float(np.sum(wi * sub_yi) / np.sum(wi))
            Q = float(np.sum(wi * (sub_yi - theta_fe) ** 2))
            C = float(np.sum(wi) - np.sum(wi ** 2) / np.sum(wi))
            tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0

            wi_re = 1.0 / (sub_sei ** 2 + tau2)
            theta = float(np.sum(wi_re * sub_yi) / np.sum(wi_re))
            se = float(1.0 / math.sqrt(np.sum(wi_re)))
            z = theta / se if se > 0 else 0.0

            I2 = max(0.0, (Q - (k - 1)) / Q * 100.0) if Q > 0 else 0.0
            cum_n = sum(n_s[:k])

            results.append(
                {"z": z, "theta": theta, "se": se, "I2": I2, "cumN": cum_n}
            )
        return results

    # ------------------------------------------------------------------ #
    #  Conditional power                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _conditional_power(
        z_k: float,
        t_k: float,
        z_final: float,
        z_alpha_half: float,
        z_beta: float,
    ) -> float:
        """
        Conditional power at interim look k given current Z-statistic.

        Under Brownian motion with drift theta*sqrt(n):
        CP = Phi( (z_k*sqrt(t_k) - z_final) / sqrt(1 - t_k)
                 + drift_correction )

        Simplified: drift estimated from current z_k / sqrt(t_k) * sqrt(1-t_k).
        """
        if t_k <= 0.0 or t_k >= 1.0:
            return float(abs(z_k) >= z_alpha_half)

        # Estimated drift (theta_hat * sqrt(N)) from current Z
        drift_remaining = z_k * math.sqrt(t_k) / math.sqrt(t_k)  # = z_k
        # Contribution from remaining information
        sqrt_rem = math.sqrt(1.0 - t_k)

        # Expected Z contribution from future data under current drift
        # CP = P(Z_final > z_final | Z_k = z_k)
        # Z_final | Z_k ~ N(z_k * sqrt(t_k) + drift*(1-t_k)/..., 1-t_k)
        # Using standard Brownian bridge formula:
        # CP = Phi( (z_k*sqrt(t_k) - z_final*sqrt(1) ) / sqrt(1-t_k) + delta_hat*sqrt(1-t_k) )
        # where delta_hat = (z_alpha_half + z_beta) as unit drift
        delta_hat = (z_alpha_half + z_beta) * (z_k / (z_alpha_half + z_beta + 1e-9))
        numerator = (z_k * math.sqrt(t_k) - z_final) / sqrt_rem + delta_hat * sqrt_rem
        cp = float(stats.norm.cdf(numerator))
        return max(0.0, min(1.0, cp))
