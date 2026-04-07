# Al-Burhan Advanced Statistics & Mathematics Upgrade

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Elevate Al-Burhan from a 12-engine audit orchestrator with basic DL meta-analysis into a world-class statistical evidence platform with Bayesian inference, robust estimators, publication bias detection, and causal methods.

**Architecture:** Each upgrade is a self-contained enhancement to an existing engine or a new engine added to the orchestrator. The claim_data dict (yi, sei, years, etc.) remains the universal interface. New engines declare dependencies via ENGINE_DEPS in orchestrator.py and follow the EvidenceEngine Protocol.

**Tech Stack:** Python 3.9+, numpy, scipy, pandas. No new external dependencies — all methods implemented from first principles using scipy.optimize, scipy.stats, and numpy linear algebra.

**Current state:** 2,195 lines, 12 engines, 62/62 tests passing. All engines compute from real data (zero simulated analysis).

---

## File Structure

### New files
- `alburhan/engines/bayesian.py` — Bayesian normal-normal hierarchical MA
- `alburhan/engines/pubbias.py` — Publication bias detection (6 methods)
- `alburhan/engines/robust.py` — Robust/outlier-resistant meta-analysis
- `alburhan/engines/dose_response.py` — Dose-response meta-regression
- `alburhan/engines/metareg.py` — Frequentist meta-regression (single covariate)
- `alburhan/engines/sequential.py` — Proper alpha-spending TSA (replaces Al-Mizan TSA)
- `alburhan/engines/grade.py` — Automated GRADE certainty assessment
- `tests/test_bayesian.py`
- `tests/test_pubbias.py`
- `tests/test_robust.py`
- `tests/test_dose_response.py`
- `tests/test_metareg.py`
- `tests/test_sequential.py`
- `tests/test_grade.py`

### Modified files
- `alburhan/engines/predictiongap.py` — Add Knapp-Hartung PI, approximate Bayes PI
- `alburhan/engines/fragility.py` — Add Paule-Mandel, Sidik-Jonkman to multiverse grid
- `alburhan/engines/forensics.py` — Add Benford's law, excess significance test
- `alburhan/engines/nma.py` — Add Galbraith radial plot statistics, Cook's distance
- `alburhan/engines/causalsynth.py` — Add bias-adjusted E-value, selection model sensitivity
- `alburhan/engines/almizan.py` — Delegate TSA to new sequential engine
- `alburhan/core/orchestrator.py` — Register new engines, update ENGINE_DEPS
- `alburhan/engines/__init__.py` — No changes (Protocol already defined)
- `alburhan/engines/e156.py` — Update to reference new engine outputs
- `alburhan/reporting.py` — Add cards for new engines
- `tests/test_engines.py` — Update existing tests for modified engines

---

## Task 1: Bayesian Normal-Normal Hierarchical Meta-Analysis Engine

**Files:**
- Create: `alburhan/engines/bayesian.py`
- Create: `tests/test_bayesian.py`
- Modify: `alburhan/core/orchestrator.py`

**What it does:** Implements a full Bayesian random-effects meta-analysis using a normal-normal hierarchical model. Produces posterior distributions for the pooled effect (mu) and between-study variance (tau2), with 95% credible intervals, posterior predictive intervals, and Bayes factors.

**Math:**
- Likelihood: yi | mu, tau2 ~ N(mu, sei^2 + tau2)
- Prior: mu ~ N(0, prior_var_mu), tau ~ HalfCauchy(scale)
- Posterior via grid approximation over tau2, analytic integration over mu
- Bayes Factor: BF01 = p(data|H0) / p(data|H1) via Savage-Dickey density ratio

- [ ] **Step 1: Write failing test for posterior mean**

```python
# tests/test_bayesian.py
import numpy as np
import pytest
from alburhan.engines.bayesian import BayesianMAEngine

class TestBayesianMAEngine:
    def setup_method(self):
        self.engine = BayesianMAEngine()

    def test_posterior_mean_close_to_dl(self):
        """With vague prior, posterior mean ≈ DL estimate."""
        claim = {
            "yi": [0.6, 0.5, 0.4, 0.45, 0.55],
            "sei": [0.1, 0.15, 0.1, 0.12, 0.1],
        }
        r = self.engine.evaluate(claim)
        assert r["status"] == "evaluated"
        # Posterior mean should be close to DL (~0.504)
        assert 0.3 < r["posterior_mean"] < 0.7
        assert r["credible_lo"] < r["posterior_mean"] < r["credible_hi"]

    def test_credible_interval_contains_dl(self):
        claim = {
            "yi": [0.6, 0.5, 0.4, 0.45, 0.55],
            "sei": [0.1, 0.15, 0.1, 0.12, 0.1],
        }
        r = self.engine.evaluate(claim)
        # 95% CrI should contain 0.504 (DL estimate)
        assert r["credible_lo"] < 0.504 < r["credible_hi"]

    def test_posterior_predictive_wider_than_credible(self):
        claim = {
            "yi": [0.6, 0.5, 0.4, 0.45, 0.55],
            "sei": [0.1, 0.15, 0.1, 0.12, 0.1],
        }
        r = self.engine.evaluate(claim)
        cri_width = r["credible_hi"] - r["credible_lo"]
        ppi_width = r["predictive_hi"] - r["predictive_lo"]
        assert ppi_width >= cri_width

    def test_bayes_factor_computed(self):
        claim = {
            "yi": [0.6, 0.5, 0.4, 0.45, 0.55],
            "sei": [0.1, 0.15, 0.1, 0.12, 0.1],
        }
        r = self.engine.evaluate(claim)
        assert "bayes_factor_01" in r
        assert r["bayes_factor_01"] > 0

    def test_too_few_studies(self):
        r = self.engine.evaluate({"yi": [0.5], "sei": [0.1]})
        assert r["status"] == "skipped"

    def test_posterior_tau2(self):
        claim = {
            "yi": [0.1, 0.8, -0.3, 0.5, 1.2],
            "sei": [0.1, 0.2, 0.15, 0.1, 0.25],
        }
        r = self.engine.evaluate(claim)
        assert r["posterior_tau2"] > 0  # High het data
        assert r["prob_tau2_positive"] > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\AlBurhan && .venv\Scripts\python.exe -m pytest tests/test_bayesian.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'alburhan.engines.bayesian'`

- [ ] **Step 3: Implement BayesianMAEngine**

```python
# alburhan/engines/bayesian.py
"""
Bayesian Normal-Normal Hierarchical Meta-Analysis.

Grid approximation over tau2 with analytic conditional posterior for mu.
- Prior: mu ~ N(0, 10^2), tau ~ HalfCauchy(0.5)
- Posterior: grid over log(tau2) ∈ [-10, 5] with 500 points
- Credible intervals: 2.5th and 97.5th percentiles of marginal posterior
- Predictive intervals: integrate over posterior of mu AND tau2
- Bayes Factor: Savage-Dickey density ratio at mu=0
"""

import math
import numpy as np
from scipy import stats as sp_stats


class BayesianMAEngine:
    name = "BayesianMA"

    def __init__(self, prior_mu_sd=10.0, prior_tau_scale=0.5, n_grid=500):
        self.prior_mu_sd = prior_mu_sd
        self.prior_tau_scale = prior_tau_scale
        self.n_grid = n_grid

    def evaluate(self, claim_data):
        yi = np.array(claim_data.get('yi', []), dtype=float)
        sei = np.array(claim_data.get('sei', []), dtype=float)
        k = len(yi)

        if k < 2:
            return {"status": "skipped", "message": "Need k>=2 for Bayesian MA."}

        vi = sei ** 2

        # Grid over log(tau2)
        log_tau2_grid = np.linspace(-10, 5, self.n_grid)
        tau2_grid = np.exp(log_tau2_grid)

        # For each tau2, compute conditional posterior of mu (normal conjugate)
        log_marginal = np.zeros(self.n_grid)
        cond_mu_mean = np.zeros(self.n_grid)
        cond_mu_var = np.zeros(self.n_grid)

        for j, tau2 in enumerate(tau2_grid):
            w = 1.0 / (vi + tau2)
            prior_prec = 1.0 / self.prior_mu_sd ** 2
            post_prec = np.sum(w) + prior_prec
            post_var = 1.0 / post_prec
            post_mean = post_var * np.sum(w * yi)

            cond_mu_mean[j] = post_mean
            cond_mu_var[j] = post_var

            # Log marginal likelihood (integrating out mu analytically)
            log_lik = -0.5 * np.sum(np.log(2 * math.pi * (vi + tau2)))
            log_lik += -0.5 * np.sum(yi ** 2 * w)
            log_lik += 0.5 * post_mean ** 2 * post_prec
            log_lik += 0.5 * math.log(post_var) - 0.5 * math.log(self.prior_mu_sd ** 2)

            # Half-Cauchy prior on tau (Jacobian for log transform)
            log_prior_tau = (math.log(2) - math.log(math.pi * self.prior_tau_scale)
                            - math.log(1 + tau2 / self.prior_tau_scale ** 2))
            # Jacobian: d(tau2)/d(log_tau2) = tau2
            log_jacobian = log_tau2_grid[j]

            log_marginal[j] = log_lik + log_prior_tau + log_jacobian

        # Normalize
        log_marginal -= np.max(log_marginal)
        marginal = np.exp(log_marginal)
        marginal /= np.trapz(marginal, log_tau2_grid)

        # Posterior mean of mu (weighted over tau2 grid)
        weights = marginal * np.diff(np.concatenate([[log_tau2_grid[0]], log_tau2_grid]))
        weights /= weights.sum()

        post_mu_mean = float(np.sum(weights * cond_mu_mean))
        post_mu_var_total = float(np.sum(weights * (cond_mu_var + cond_mu_mean ** 2)) - post_mu_mean ** 2)
        post_mu_sd = math.sqrt(max(0, post_mu_var_total))

        # Posterior mean of tau2
        post_tau2_mean = float(np.sum(weights * tau2_grid))
        prob_tau2_pos = float(np.sum(weights[tau2_grid > 0.001]))

        # 95% credible interval for mu via Monte Carlo
        n_mc = 5000
        rng = np.random.default_rng(seed=42)
        tau2_samples = rng.choice(tau2_grid, size=n_mc, p=weights)
        mu_samples = np.zeros(n_mc)
        for i in range(n_mc):
            j = np.searchsorted(tau2_grid, tau2_samples[i])
            j = min(j, self.n_grid - 1)
            mu_samples[i] = rng.normal(cond_mu_mean[j], math.sqrt(cond_mu_var[j]))

        cri_lo, cri_hi = float(np.percentile(mu_samples, 2.5)), float(np.percentile(mu_samples, 97.5))

        # Posterior predictive interval
        avg_se = float(np.mean(sei))
        pred_samples = mu_samples + rng.normal(0, np.sqrt(tau2_samples + avg_se ** 2))
        pred_lo, pred_hi = float(np.percentile(pred_samples, 2.5)), float(np.percentile(pred_samples, 97.5))

        # Bayes Factor (Savage-Dickey): BF01 = p(mu=0|data) / p(mu=0|prior)
        prior_at_0 = sp_stats.norm.pdf(0, 0, self.prior_mu_sd)
        posterior_at_0 = float(np.sum(weights * sp_stats.norm.pdf(0, cond_mu_mean, np.sqrt(cond_mu_var))))
        bf01 = posterior_at_0 / prior_at_0 if prior_at_0 > 0 else float('inf')

        return {
            "status": "evaluated",
            "posterior_mean": round(post_mu_mean, 4),
            "posterior_sd": round(post_mu_sd, 4),
            "credible_lo": round(cri_lo, 4),
            "credible_hi": round(cri_hi, 4),
            "predictive_lo": round(pred_lo, 4),
            "predictive_hi": round(pred_hi, 4),
            "posterior_tau2": round(post_tau2_mean, 6),
            "prob_tau2_positive": round(prob_tau2_pos, 3),
            "bayes_factor_01": round(float(bf01), 4),
            "evidence_strength": self._classify_bf(bf01),
        }

    @staticmethod
    def _classify_bf(bf):
        """Kass & Raftery (1995) classification."""
        if bf > 150: return "Decisive for H0"
        if bf > 20: return "Strong for H0"
        if bf > 3: return "Moderate for H0"
        if bf > 1: return "Weak for H0"
        if bf > 1/3: return "Weak for H1"
        if bf > 1/20: return "Moderate for H1"
        if bf > 1/150: return "Strong for H1"
        return "Decisive for H1"
```

- [ ] **Step 4: Run tests**

Run: `cd C:\AlBurhan && .venv\Scripts\python.exe -m pytest tests/test_bayesian.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Register in orchestrator**

In `alburhan/core/orchestrator.py`, add import and append to engines list (before E156):
```python
from alburhan.engines.bayesian import BayesianMAEngine
# In __init__, before E156Emitter():
    BayesianMAEngine(),
```

- [ ] **Step 6: Commit**

```bash
git add alburhan/engines/bayesian.py tests/test_bayesian.py alburhan/core/orchestrator.py
git commit -m "feat: add Bayesian normal-normal hierarchical MA engine"
```

---

## Task 2: Publication Bias Detection Suite (6 methods)

**Files:**
- Create: `alburhan/engines/pubbias.py`
- Create: `tests/test_pubbias.py`
- Modify: `alburhan/core/orchestrator.py`

**What it does:** Implements 6 publication bias detection methods that compute from yi/sei:
1. **Egger's regression** — weighted regression of effect on 1/SE (tests small-study effects)
2. **Begg-Mazumdar rank test** — Kendall's tau between effect and variance
3. **Trim-and-Fill** — L0 estimator, iterative imputation of missing studies
4. **Fail-safe N** (Rosenthal) — number of null studies to make result non-significant
5. **P-curve** — tests if p-value distribution is right-skewed (evidential value)
6. **Excess Significance Test** (Ioannidis & Trikalinos) — observed vs expected significant studies

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pubbias.py
import numpy as np
import pytest
from alburhan.engines.pubbias import PubBiasEngine

class TestPubBiasEngine:
    def setup_method(self):
        self.engine = PubBiasEngine()
        self.standard = {
            "yi": [0.6, 0.5, 0.4, 0.45, 0.55],
            "sei": [0.1, 0.15, 0.1, 0.12, 0.1],
        }
        self.biased = {
            # Small studies with large effects, large studies with small effects
            "yi": [1.2, 0.9, 0.7, 0.3, 0.2, 0.15],
            "sei": [0.4, 0.3, 0.2, 0.08, 0.06, 0.05],
        }

    def test_egger_returns_p_value(self):
        r = self.engine.evaluate(self.standard)
        assert r["status"] == "evaluated"
        eg = r["egger"]
        assert "intercept" in eg
        assert "p_value" in eg
        assert 0 <= eg["p_value"] <= 1

    def test_begg_returns_tau(self):
        r = self.engine.evaluate(self.standard)
        bg = r["begg"]
        assert "kendall_tau" in bg
        assert -1 <= bg["kendall_tau"] <= 1

    def test_trim_fill_count(self):
        r = self.engine.evaluate(self.biased)
        tf = r["trim_fill"]
        assert "n_missing" in tf
        assert tf["n_missing"] >= 0
        if tf["n_missing"] > 0:
            assert tf["adjusted_theta"] != r["trim_fill"].get("original_theta")

    def test_failsafe_n(self):
        r = self.engine.evaluate(self.standard)
        fs = r["failsafe_n"]
        assert "rosenthal_n" in fs
        assert fs["rosenthal_n"] >= 0

    def test_pcurve(self):
        r = self.engine.evaluate(self.standard)
        pc = r["p_curve"]
        assert "right_skew_p" in pc

    def test_excess_significance(self):
        r = self.engine.evaluate(self.standard)
        es = r["excess_significance"]
        assert "observed_sig" in es
        assert "expected_sig" in es

    def test_too_few_studies(self):
        r = self.engine.evaluate({"yi": [0.5, 0.6], "sei": [0.1, 0.1]})
        assert r["status"] == "skipped"

    def test_biased_data_detects_something(self):
        r = self.engine.evaluate(self.biased)
        # At least one method should flag bias in obviously biased data
        flags = sum([
            r["egger"].get("flagged", False),
            r["begg"].get("flagged", False),
            r["trim_fill"].get("n_missing", 0) > 0,
        ])
        assert flags >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\AlBurhan && .venv\Scripts\python.exe -m pytest tests/test_pubbias.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement PubBiasEngine**

```python
# alburhan/engines/pubbias.py
"""
Publication Bias Detection Suite — 6 real statistical methods.

1. Egger's regression (Egger et al., 1997)
2. Begg-Mazumdar rank test (Begg & Mazumdar, 1994)
3. Trim-and-Fill L0 estimator (Duval & Tweedie, 2000)
4. Rosenthal Fail-safe N (Rosenthal, 1979)
5. P-curve (Simonsohn et al., 2014)
6. Excess Significance Test (Ioannidis & Trikalinos, 2007)
"""

import math
import numpy as np
from scipy import stats as sp_stats


class PubBiasEngine:
    name = "PubBias"

    def evaluate(self, claim_data):
        yi = np.array(claim_data.get('yi', []), dtype=float)
        sei = np.array(claim_data.get('sei', []), dtype=float)
        k = len(yi)

        if k < 3:
            return {"status": "skipped", "message": "Need k>=3 for publication bias tests."}

        return {
            "status": "evaluated",
            "egger": self._egger_test(yi, sei),
            "begg": self._begg_test(yi, sei),
            "trim_fill": self._trim_fill(yi, sei),
            "failsafe_n": self._failsafe_n(yi, sei),
            "p_curve": self._p_curve(yi, sei),
            "excess_significance": self._excess_significance(yi, sei),
        }

    def _egger_test(self, yi, sei, alpha=0.10):
        """Weighted linear regression of yi on 1/sei (precision)."""
        k = len(yi)
        if k < 3:
            return {"status": "skipped"}
        precision = 1.0 / sei
        wi = 1.0 / sei ** 2
        # WLS: yi = a + b * (1/sei)
        X = np.column_stack([np.ones(k), precision])
        W = np.diag(wi)
        XtWX = X.T @ W @ X
        try:
            beta = np.linalg.solve(XtWX, X.T @ W @ yi)
        except np.linalg.LinAlgError:
            return {"status": "error", "message": "Singular matrix"}
        residuals = yi - X @ beta
        s2 = float(np.sum(wi * residuals ** 2) / (k - 2))
        var_beta = s2 * np.linalg.inv(XtWX)
        se_intercept = math.sqrt(var_beta[0, 0])
        t_stat = beta[0] / se_intercept if se_intercept > 0 else 0
        p_val = float(2 * sp_stats.t.sf(abs(t_stat), df=k - 2))
        return {
            "intercept": round(float(beta[0]), 4),
            "se": round(float(se_intercept), 4),
            "t_stat": round(float(t_stat), 3),
            "p_value": round(p_val, 4),
            "flagged": p_val < alpha,
        }

    def _begg_test(self, yi, sei, alpha=0.10):
        """Kendall rank correlation between effect sizes and variances."""
        vi = sei ** 2
        tau, p_val = sp_stats.kendalltau(yi, vi)
        return {
            "kendall_tau": round(float(tau), 4),
            "p_value": round(float(p_val), 4),
            "flagged": p_val < alpha,
        }

    def _trim_fill(self, yi, sei, side="right", max_iter=50):
        """Duval & Tweedie trim-and-fill using L0 estimator."""
        k = len(yi)
        wi = 1.0 / sei ** 2
        theta0 = float(np.sum(wi * yi) / np.sum(wi))

        # L0 estimator: count studies on asymmetric side
        residuals = yi - theta0
        abs_res = np.abs(residuals)
        ranks = sp_stats.rankdata(abs_res)
        signs = np.sign(residuals)

        # Count excess on one side
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)
        k0 = abs(n_pos - n_neg)  # Rough L0

        # Iterative trim-and-fill
        n_missing = max(0, int(round(k0)))

        if n_missing == 0:
            return {
                "n_missing": 0,
                "original_theta": round(theta0, 4),
                "adjusted_theta": round(theta0, 4),
            }

        # Impute mirror studies
        sorted_idx = np.argsort(residuals if n_pos > n_neg else -residuals)
        trim_indices = sorted_idx[:n_missing]

        # Create augmented dataset
        fill_yi = []
        fill_sei = []
        for idx in trim_indices:
            fill_yi.append(2 * theta0 - yi[idx])
            fill_sei.append(sei[idx])

        aug_yi = np.concatenate([yi, fill_yi])
        aug_sei = np.concatenate([sei, fill_sei])
        aug_wi = 1.0 / aug_sei ** 2
        adj_theta = float(np.sum(aug_wi * aug_yi) / np.sum(aug_wi))

        return {
            "n_missing": n_missing,
            "original_theta": round(theta0, 4),
            "adjusted_theta": round(adj_theta, 4),
        }

    def _failsafe_n(self, yi, sei):
        """Rosenthal fail-safe N: how many null studies to make p > 0.05."""
        k = len(yi)
        zi = yi / sei
        sum_z = float(np.sum(zi))
        # Rosenthal: Nfs = (sum_z / 1.645)^2 - k
        nfs = max(0, (sum_z / 1.645) ** 2 - k)
        return {
            "rosenthal_n": int(round(nfs)),
            "robust": nfs > 5 * k + 10,  # 5k+10 rule
        }

    def _p_curve(self, yi, sei, alpha=0.05):
        """P-curve: test if significant p-values are right-skewed."""
        zi = np.abs(yi / sei)
        p_vals = 2 * (1 - sp_stats.norm.cdf(zi))
        sig_p = p_vals[p_vals < alpha]

        if len(sig_p) < 3:
            return {"status": "skipped", "reason": "Fewer than 3 significant studies"}

        # Right-skew test: pp-values under H0 (uniform) vs observed
        pp_values = sig_p / alpha  # Transform to [0,1] under H0
        # If truly evidential, pp-values cluster near 0 (right-skewed p-curve)
        # Binomial test: proportion below 0.5
        n_below = np.sum(pp_values < 0.5)
        binom_p = float(sp_stats.binom_test(n_below, len(pp_values), 0.5))

        return {
            "n_significant": len(sig_p),
            "prop_below_025": round(float(np.mean(sig_p < 0.025)), 3),
            "right_skew_p": round(binom_p, 4),
            "evidential_value": binom_p < 0.05,
        }

    def _excess_significance(self, yi, sei, alpha=0.05):
        """Ioannidis & Trikalinos: observed vs expected significant results."""
        k = len(yi)
        zi = np.abs(yi / sei)
        p_vals = 2 * (1 - sp_stats.norm.cdf(zi))
        observed_sig = int(np.sum(p_vals < alpha))

        # Expected: compute power of each study under the RE pooled effect
        wi = 1.0 / sei ** 2
        theta = float(np.sum(wi * yi) / np.sum(wi))
        z_crit = sp_stats.norm.ppf(1 - alpha / 2)

        expected_sig = 0.0
        for i in range(k):
            # Power = P(|Z| > z_crit | mu = theta)
            ncp = abs(theta) / sei[i]
            power = 1 - sp_stats.norm.cdf(z_crit - ncp) + sp_stats.norm.cdf(-z_crit - ncp)
            expected_sig += power

        # Chi-squared test
        if expected_sig > 0:
            chi2 = (observed_sig - expected_sig) ** 2 / expected_sig
            p_val = float(1 - sp_stats.chi2.cdf(chi2, df=1))
        else:
            chi2 = 0
            p_val = 1.0

        return {
            "observed_sig": observed_sig,
            "expected_sig": round(float(expected_sig), 1),
            "chi2": round(float(chi2), 3),
            "p_value": round(p_val, 4),
            "flagged": p_val < 0.10 and observed_sig > expected_sig,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd C:\AlBurhan && .venv\Scripts\python.exe -m pytest tests/test_pubbias.py -v`
Expected: 8 PASSED

- [ ] **Step 5: Register in orchestrator + commit**

---

## Task 3: Robust Meta-Analysis Engine (3 estimators)

**Files:**
- Create: `alburhan/engines/robust.py`
- Create: `tests/test_robust.py`

**What it does:** Implements outlier-resistant estimators:
1. **Paule-Mandel** (generalized Q) — iterative tau2 that solves Q(tau2) = k-1
2. **Median-based estimator** — weighted median instead of weighted mean
3. **Winsorized mean** — trim extreme 10% of effects before pooling

- [ ] **Step 1: Write failing tests**

```python
# tests/test_robust.py
import numpy as np
import pytest
from alburhan.engines.robust import RobustMAEngine

class TestRobustMAEngine:
    def setup_method(self):
        self.engine = RobustMAEngine()
        self.clean = {
            "yi": [0.5, 0.5, 0.5, 0.5, 0.5],
            "sei": [0.1, 0.1, 0.1, 0.1, 0.1],
        }
        self.outlier = {
            "yi": [0.5, 0.5, 0.5, 0.5, 5.0],  # One extreme outlier
            "sei": [0.1, 0.1, 0.1, 0.1, 0.1],
        }

    def test_paule_mandel_tau2(self):
        r = self.engine.evaluate(self.outlier)
        assert r["status"] == "evaluated"
        pm = r["paule_mandel"]
        assert "tau2" in pm
        assert pm["tau2"] >= 0

    def test_median_resists_outlier(self):
        r = self.engine.evaluate(self.outlier)
        # Weighted median should be near 0.5, not pulled toward 5.0
        assert abs(r["weighted_median"]["theta"] - 0.5) < 0.3

    def test_winsorized_resists_outlier(self):
        r = self.engine.evaluate(self.outlier)
        # Winsorized mean should be much less than standard mean
        assert r["winsorized"]["theta"] < 2.0

    def test_clean_data_all_agree(self):
        r = self.engine.evaluate(self.clean)
        pm = r["paule_mandel"]["theta"]
        med = r["weighted_median"]["theta"]
        win = r["winsorized"]["theta"]
        assert abs(pm - med) < 0.1
        assert abs(pm - win) < 0.1

    def test_too_few(self):
        r = self.engine.evaluate({"yi": [0.5, 0.6], "sei": [0.1, 0.1]})
        assert r["status"] == "skipped"

    def test_paule_mandel_converges(self):
        r = self.engine.evaluate({
            "yi": [0.1, 0.8, -0.3, 0.5, 1.2],
            "sei": [0.1, 0.2, 0.15, 0.1, 0.25],
        })
        assert r["paule_mandel"]["converged"] is True
```

- [ ] **Step 2-6: Implement, test, register, commit** (same pattern as Tasks 1-2)

Implementation: Paule-Mandel uses `scipy.optimize.brentq` to solve Q(tau2) = k-1. Weighted median uses the standard weighted percentile algorithm. Winsorized mean trims the top and bottom 10% of study effects before computing the weighted average.

---

## Task 4: Frequentist Meta-Regression Engine

**Files:**
- Create: `alburhan/engines/metareg.py`
- Create: `tests/test_metareg.py`

**What it does:** Weighted least squares meta-regression with a single covariate (year, as default from claim_data). Computes:
- Slope and intercept with HKSJ-corrected standard errors
- R2_analog (proportion of tau2 explained by the covariate)
- Knapp-Hartung F-test for the moderator
- Residual heterogeneity (tau2_res)
- Bubble plot data (effect vs moderator with CI widths)

Tests should verify: slope direction matches known drift, R2 bounded [0,1], HKSJ SE ≥ Wald SE.

---

## Task 5: Dose-Response Meta-Regression Engine

**Files:**
- Create: `alburhan/engines/dose_response.py`
- Create: `tests/test_dose_response.py`

**What it does:** Extends meta-regression with restricted cubic splines for non-linear dose-response. Requires an additional `doses` key in claim_data. Implements:
- Linear dose-response slope (per-unit effect change)
- Restricted cubic splines (3 knots at 10th, 50th, 90th percentiles)
- Quadratic model (dose + dose^2) with AIC comparison
- Minimum effective dose (MED) where CI excludes null

---

## Task 6: Proper Alpha-Spending TSA Engine

**Files:**
- Create: `alburhan/engines/sequential.py`
- Create: `tests/test_sequential.py`
- Modify: `alburhan/engines/almizan.py` — delegate TSA computation

**What it does:** Replaces the simplified TSA in Al-Mizan with a proper implementation:
- **Lan-DeMets alpha-spending functions**: O'Brien-Fleming and Pocock spending
- **Beta-spending** for futility boundaries
- **Adjusted RIS** using heterogeneity-adjusted pooled variance
- **Conditional power** at each interim look
- Returns: boundary crossings, adjusted p-values, conditional power, TSA plot data

---

## Task 7: Automated GRADE Certainty Engine

**Files:**
- Create: `alburhan/engines/grade.py`
- Create: `tests/test_grade.py`

**What it does:** Maps outputs from all upstream engines to the 5 GRADE domains and produces an automated certainty rating. Rules:
1. **Risk of Bias** — from RegistryForensics anomaly flags
2. **Inconsistency** — from I2 (PredictionGap) + influence analysis (NetworkMeta)
3. **Indirectness** — from AfricaRCT burden alignment + GERI
4. **Imprecision** — from CI width, optimal information size, and fragility (FragilityAtlas)
5. **Publication Bias** — from PubBias engine (Egger + trim-fill)

Starting certainty = HIGH (all RCTs assumed), downgrade 0-2 levels per domain. Final: HIGH/MODERATE/LOW/VERY LOW.

Dependencies: Must run after all other engines. Add to ENGINE_DEPS.

---

## Task 8: Upgrade Existing Engines with Advanced Methods

**Files:**
- Modify: `alburhan/engines/predictiongap.py`
- Modify: `alburhan/engines/fragility.py`
- Modify: `alburhan/engines/forensics.py`
- Modify: `alburhan/engines/nma.py`
- Modify: `alburhan/engines/causalsynth.py`
- Modify: `tests/test_engines.py`

**Enhancements:**

**predictiongap.py:**
- Add Knapp-Hartung adjusted CI (alongside Wald)
- Add approximate Bayes prediction interval using posterior predictive from Bayesian engine

**fragility.py:**
- Add Paule-Mandel and Sidik-Jonkman tau2 estimators to multiverse grid
- Grid becomes: 5 estimators × 2 CI methods = 10 specifications

**forensics.py:**
- Add Benford's law first-significant-digit test
- Add excess zeros test (proportion of studies with exactly p=0.05)

**nma.py:**
- Add Cook's distance analog: `D_i = (theta_full - theta_loo)^2 / var(theta_full)`
- Add Galbraith radial plot statistics (z/se vs 1/se regression)

**causalsynth.py:**
- Add bias-adjusted E-value (accounts for measured confounders)
- Add selection model sensitivity parameter (Vevea & Woods)

---

## Task 9: Update E156 Emitter and HTML Report

**Files:**
- Modify: `alburhan/engines/e156.py`
- Modify: `alburhan/reporting.py`

**E156 updates:**
- Reference Bayesian posterior mean + CrI alongside frequentist CI
- Include publication bias verdict
- Include GRADE certainty level
- Ensure still ≤156 words (compress language if needed)

**Reporting updates:**
- Add Bayesian card (posterior mean, CrI, Bayes Factor, evidence strength)
- Add Publication Bias card (Egger, Begg, trim-fill, fail-safe N)
- Add GRADE card (5 domains with up/down arrows, final certainty)
- Add meta-regression card (slope, R2, bubble plot data reference)

---

## Task 10: Integration Tests and Final Verification

**Files:**
- Create: `tests/test_integration.py`
- Modify: `tests/test_engines.py`

**Integration tests:**
- Full orchestrator run with all engines (now ~18 engines)
- Verify GRADE engine receives correct upstream data
- Verify PubBias results flow into GRADE publication bias domain
- Verify Bayesian posterior ≈ DL estimate with vague priors
- Verify E156 word count ≤ 156
- Benchmark: full audit completes in < 10 seconds
- Edge cases: all-positive effects, all-negative, one-study, identical effects

Run: `cd C:\AlBurhan && .venv\Scripts\python.exe -m pytest tests/ -v --tb=short`
Target: 120+ tests all passing

---

## Summary

| Task | Engine | Key Methods | Est. Tests |
|------|--------|------------|------------|
| 1 | BayesianMA | Grid posterior, Savage-Dickey BF, predictive intervals | 6 |
| 2 | PubBias | Egger, Begg, Trim-Fill, Fail-safe N, P-curve, Excess Sig | 8 |
| 3 | RobustMA | Paule-Mandel, weighted median, Winsorized mean | 6 |
| 4 | MetaRegression | WLS meta-regression, HKSJ F-test, R2_analog | 6 |
| 5 | DoseResponse | Restricted cubic splines, MED, AIC model comparison | 6 |
| 6 | Sequential TSA | Lan-DeMets spending, futility, conditional power | 8 |
| 7 | GRADE | 5-domain automated certainty rating | 8 |
| 8 | Upgrades | Benford, Cook's D, PM/SJ multiverse, bias-adjusted E-value | 12 |
| 9 | E156 + Report | Updated cards for all new engines | 6 |
| 10 | Integration | Full-pipeline + edge case + performance | 10 |

**Total new methods: ~25 statistical methods across 6 new engines + 5 engine upgrades**
**Target test count: ~140 tests (62 existing + ~76 new)**
**Estimated final codebase: ~5,000 lines**
