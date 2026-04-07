import logging
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, brentq

logger = logging.getLogger(__name__)


class FragilityEngine:
    name = "FragilityAtlas"

    def evaluate(self, claim_data):
        """
        Multiverse fragility analysis.
        Checks robustness across variance estimators and CI methods.

        Grid: 5 estimators (FE, DL, REML, PM, SJ) x 2 CI methods (Wald, HKSJ)
        = 10 unique specifications.
        """
        yi = np.array(claim_data.get('yi', []))
        logger.info("%s: evaluating k=%d studies", self.name, len(yi))
        sei = np.array(claim_data.get('sei', []))

        if len(yi) < 3:
            return {"status": "skipped", "message": "Need k>=3 for multiverse."}

        results = []
        reference_sig = self._is_sig(yi, sei, method="DL", ci="Wald")

        for estimator in ["FE", "DL", "REML", "PM", "SJ"]:
            for ci_method in ["Wald", "HKSJ"]:
                sig = self._is_sig(yi, sei, method=estimator, ci=ci_method)
                results.append(sig == reference_sig)

        robustness_score = sum(results) / len(results) * 100

        classification = "Robust"
        if robustness_score < 50:
            classification = "Unstable"
        elif robustness_score < 70:
            classification = "Fragile"
        elif robustness_score < 90:
            classification = "Moderately Robust"

        return {
            "status": "evaluated",
            "robustness_score": round(robustness_score, 1),
            "classification": classification,
            "reference_agreement": f"{sum(results)}/{len(results)}",
            "is_fragile": robustness_score < 70
        }

    def _estimate_tau2(self, yi, sei, method):
        """Estimate between-study variance tau2."""
        k = len(yi)
        wi = 1.0 / sei**2

        if method == "FE":
            return 0.0

        # DerSimonian-Laird
        theta_fe = np.sum(wi * yi) / np.sum(wi)
        Q = float(np.sum(wi * (yi - theta_fe)**2))
        C = float(np.sum(wi) - np.sum(wi**2) / np.sum(wi))
        tau2_dl = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0

        if method == "DL":
            return tau2_dl

        # REML via iterative optimization (STAT-P0-1)
        if method == "REML":
            return self._reml_tau2(yi, sei, tau2_init=tau2_dl)

        # Paule-Mandel: solve Q(tau2) = k-1 via brentq
        if method == "PM":
            return self._pm_tau2(yi, sei, k, wi)

        # Sidik-Jonkman: unweighted moment estimator
        if method == "SJ":
            theta_fe = np.sum(wi * yi) / np.sum(wi)
            tau2_sj = float(np.sum((yi - theta_fe) ** 2 - sei ** 2) / (k - 1))
            return max(0.0, tau2_sj)

        return tau2_dl

    def _pm_tau2(self, yi, sei, k, wi):
        """Paule-Mandel estimator: solve Q(tau2) = k-1 via brentq."""
        vi = sei ** 2

        def q_minus_df(tau2):
            w = 1.0 / (vi + tau2)
            theta = np.sum(w * yi) / np.sum(w)
            return float(np.sum(w * (yi - theta) ** 2)) - (k - 1)

        # If Q(0) <= k-1, tau2=0 is the solution
        if q_minus_df(0.0) <= 0.0:
            return 0.0

        # Find upper bound where Q(tau2) crosses k-1
        upper = 10.0
        while q_minus_df(upper) > 0 and upper < 1e6:
            upper *= 10.0
        if q_minus_df(upper) > 0:
            return 0.0

        try:
            tau2_pm = brentq(q_minus_df, 0.0, upper, xtol=1e-8)
        except ValueError:
            tau2_pm = 0.0
        return max(0.0, float(tau2_pm))

    def _reml_tau2(self, yi, sei, tau2_init=0.0):
        """REML estimator via negative restricted log-likelihood minimization."""
        k = len(yi)
        vi = sei**2

        def neg_reml_ll(log_tau2):
            tau2 = np.exp(log_tau2)
            w = 1.0 / (vi + tau2)
            theta = np.sum(w * yi) / np.sum(w)
            # Restricted log-likelihood (Viechtbauer 2005)
            ll = -0.5 * (np.sum(np.log(vi + tau2))
                         + np.log(np.sum(w))
                         + np.sum(w * (yi - theta)**2))
            return -ll

        # Search over log(tau2) for numerical stability
        if tau2_init > 0:
            x0_log = np.log(tau2_init)
        else:
            x0_log = np.log(0.01)

        result = minimize_scalar(neg_reml_ll, bounds=(np.log(1e-10), np.log(100)),
                                 method='bounded')
        tau2 = np.exp(result.x)
        return max(0.0, float(tau2))

    def _is_sig(self, yi, sei, method="DL", ci="Wald"):
        k = len(yi)
        tau2 = self._estimate_tau2(yi, sei, method)

        wi_star = 1.0 / (sei**2 + tau2)
        theta = float(np.sum(wi_star * yi) / np.sum(wi_star))
        se = float(1.0 / np.sqrt(np.sum(wi_star)))

        if ci == "Wald":
            z = theta / se if se > 0 else 0.0
            p = 2 * (1 - stats.norm.cdf(abs(z)))
        elif ci == "HKSJ":
            # Hartung-Knapp-Sidik-Jonkman with Rover truncation (documented)
            # max(1.0, q) is the conservative variant per Rover et al. (2015)
            # that prevents HKSJ CIs from being narrower than Wald CIs
            q = float(np.sum(wi_star * (yi - theta)**2) / (k - 1)) if k > 1 else 1.0
            se_hksj = se * np.sqrt(max(1.0, q))  # Rover truncation
            t_val = theta / se_hksj if se_hksj > 0 else 0.0
            p = 2 * (1 - stats.t.cdf(abs(t_val), df=max(1, k - 1)))
        else:
            p = 1.0

        return p < 0.05
