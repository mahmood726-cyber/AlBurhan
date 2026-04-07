"""
Bayesian Normal-Normal Hierarchical Meta-Analysis Engine.

Model:
  Likelihood:  yi | mu, tau2  ~ N(mu, sei^2 + tau2)
  Prior:        mu ~ N(0, 10^2)
  Prior:        tau ~ HalfCauchy(0.5)   [scale = 0.5]

Posterior via grid approximation over log(tau2), analytic integration of mu.

For each tau2 on the grid:
  - Conditional posterior of mu is normal (conjugate update)
  - Marginal likelihood: integral of p(data|mu,tau2) * p(mu) dmu  (analytic)
  - Log marginal posterior: log p(tau2|data) ∝ log p(tau2) + log m(data|tau2)

Monte Carlo draws (n_mc=5000, seed=42) from the resulting joint posterior for
credible intervals and posterior predictive intervals.

Bayes Factor (H0: mu=0 vs H1: mu≠0) via Savage-Dickey density ratio:
  BF01 = posterior_density_at_0 / prior_density_at_0

Evidence classification: Kass & Raftery (1995).
"""

import numpy as np
from scipy import stats


class BayesianMAEngine:
    name = "BayesianMA"

    # Grid settings
    _LOG_TAU2_MIN = -10.0
    _LOG_TAU2_MAX = 5.0
    _N_GRID = 500

    # Prior parameters
    _MU_PRIOR_MEAN = 0.0
    _MU_PRIOR_VAR = 100.0   # 10^2
    _TAU_HC_SCALE = 0.5     # HalfCauchy scale for tau (not tau2)

    # Monte Carlo
    _N_MC = 5000
    _SEED = 42

    def evaluate(self, claim_data):
        yi = np.array(claim_data.get("yi", []), dtype=float)
        sei = np.array(claim_data.get("sei", []), dtype=float)
        k = len(yi)

        if k < 3:
            return {
                "status": "skipped",
                "message": "Need k>=3 for Bayesian MA (grid approximation).",
            }

        # --- Grid over log(tau2) ---
        log_tau2_grid = np.linspace(self._LOG_TAU2_MIN, self._LOG_TAU2_MAX, self._N_GRID)
        tau2_grid = np.exp(log_tau2_grid)

        log_marg_post = np.zeros(self._N_GRID)
        # Store conditional posterior params for each grid point
        cond_mu_mean = np.zeros(self._N_GRID)
        cond_mu_var = np.zeros(self._N_GRID)

        for i, tau2 in enumerate(tau2_grid):
            vi = sei**2 + tau2
            # Conjugate update: mu | data, tau2 ~ N(mu_n, V_n)
            # Prior: mu ~ N(mu0, V0)
            V0 = self._MU_PRIOR_VAR
            mu0 = self._MU_PRIOR_MEAN
            precision_prior = 1.0 / V0
            precision_data = np.sum(1.0 / vi)
            V_n = 1.0 / (precision_prior + precision_data)
            mu_n = V_n * (mu0 / V0 + np.sum(yi / vi))
            cond_mu_mean[i] = mu_n
            cond_mu_var[i] = V_n

            # Log marginal likelihood: integrate out mu analytically
            # p(data | tau2) = integral p(data | mu, tau2) p(mu) dmu
            # = prod_i N(yi; mu, vi) * N(mu; mu0, V0)  integrated over mu
            # Residual variance of yi around mu0 in the marginal:
            #   Var(yi - yj | tau2) involves the joint distribution
            # Standard conjugate result:
            #   log p(data | tau2) = -0.5 * [sum log(vi) + log(V0 * sum(1/vi) + 1)
            #                         + (yi - mu0)^T * [diag(vi) + mu0*11^T*V0]^{-1} * (yi-mu0)]
            # Using the Woodbury / matrix-determinant lemma for scalar rank-1 update:
            # More directly: marginal covariance is diag(vi) + V0 * 11^T
            # log det = sum log(vi) + log(1 + V0 * sum(1/vi))
            # quadratic form = sum((yi-mu0)^2/vi) - V_n/V0^2 * (sum((yi-mu0)/vi))^2 * V0^2
            # Simplified:
            r = yi - mu0
            sum_r_over_vi = np.sum(r / vi)
            log_det = np.sum(np.log(vi)) + np.log(V0 * precision_data + 1.0)
            quad = np.sum(r**2 / vi) - V_n * sum_r_over_vi**2
            log_marg_lik = -0.5 * (k * np.log(2 * np.pi) + log_det + quad)

            # Log prior on tau: HalfCauchy(scale=s) → p(tau) = 2/(pi*s*(1+(tau/s)^2))
            # tau = sqrt(tau2), dtau/dtau2 = 1/(2*sqrt(tau2))
            # log p(tau2) = log p(tau) + log|dtau/dtau2|
            #             = log(2) - log(pi*s) - log(1+(tau2/s^2)) - 0.5*log(tau2)
            tau = np.sqrt(tau2)
            s = self._TAU_HC_SCALE
            log_prior_tau2 = (np.log(2.0) - np.log(np.pi * s)
                              - np.log(1.0 + (tau / s)**2)
                              - 0.5 * np.log(tau2))

            log_marg_post[i] = log_marg_lik + log_prior_tau2

        # Normalize in log space → weights for tau2 grid
        log_marg_post -= np.max(log_marg_post)  # numerical stability
        weights = np.exp(log_marg_post)
        weights /= np.sum(weights)

        # --- Point estimates for tau2 and mu ---
        posterior_tau2 = float(np.sum(weights * tau2_grid))
        posterior_mu = float(np.sum(weights * cond_mu_mean))

        # --- Monte Carlo for credible and predictive intervals ---
        rng = np.random.default_rng(self._SEED)
        # 1. Sample tau2 indices by weight
        idx_samples = rng.choice(self._N_GRID, size=self._N_MC, replace=True, p=weights)
        # 2. For each sampled tau2, draw mu from its conditional posterior
        mu_samples_mean = cond_mu_mean[idx_samples]
        mu_samples_std = np.sqrt(cond_mu_var[idx_samples])
        mu_samples = rng.normal(mu_samples_mean, mu_samples_std)
        # 3. Posterior predictive: new study draws from N(mu, tau2)
        tau2_samples = tau2_grid[idx_samples]
        tau_samples = np.sqrt(tau2_samples)
        y_pred = rng.normal(mu_samples, tau_samples)

        cri_lo = float(np.percentile(mu_samples, 2.5))
        cri_hi = float(np.percentile(mu_samples, 97.5))
        ppi_lo = float(np.percentile(y_pred, 2.5))
        ppi_hi = float(np.percentile(y_pred, 97.5))

        # --- Bayes Factor via Savage-Dickey density ratio ---
        # BF01 = p(mu=0 | data) / p(mu=0 | prior)
        # Prior density at mu=0: N(0, V0)
        prior_density_at_0 = stats.norm.pdf(0.0, loc=self._MU_PRIOR_MEAN,
                                             scale=np.sqrt(self._MU_PRIOR_VAR))
        # Posterior density at mu=0: weighted mixture of conditional Gaussians
        posterior_density_at_0 = float(
            np.sum(weights * stats.norm.pdf(0.0, loc=cond_mu_mean,
                                            scale=np.sqrt(cond_mu_var)))
        )
        bf01 = float(posterior_density_at_0 / prior_density_at_0) if prior_density_at_0 > 0 else np.nan

        # Kass & Raftery (1995) classification on BF10 = 1/BF01
        bf10 = 1.0 / bf01 if bf01 > 0 else np.inf
        evidence_label = self._kass_raftery(bf10)

        # --- Posterior SD of mu ---
        mu_samples_var = float(np.sum(weights * (cond_mu_var + cond_mu_mean**2))
                               - posterior_mu**2)
        posterior_mu_sd = float(np.sqrt(max(0.0, mu_samples_var)))

        return {
            "status": "evaluated",
            "posterior_mu": round(posterior_mu, 4),
            "posterior_mu_sd": round(posterior_mu_sd, 4),
            "posterior_tau2": round(posterior_tau2, 4),
            "cri_lo": round(cri_lo, 4),
            "cri_hi": round(cri_hi, 4),
            "ppi_lo": round(ppi_lo, 4),
            "ppi_hi": round(ppi_hi, 4),
            "bf01": round(bf01, 4),
            "bf10": round(bf10, 4),
            "evidence_label": evidence_label,
            "n_studies": k,
        }

    @staticmethod
    def _kass_raftery(bf10):
        """Kass & Raftery (1995) evidence classification for BF10."""
        if bf10 < 1.0:
            return "Evidence for H0"
        elif bf10 < 3.0:
            return "Barely worth mentioning"
        elif bf10 < 20.0:
            return "Positive"
        elif bf10 < 150.0:
            return "Strong"
        else:
            return "Very strong"
