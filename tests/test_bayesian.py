"""
Tests for BayesianMAEngine — Bayesian normal-normal hierarchical meta-analysis.

6 tests:
  1. Posterior mean close to DL estimate (~0.504) with vague prior
  2. 95% CrI contains the DL estimate
  3. Posterior predictive interval wider than credible interval
  4. Bayes factor computed and > 0
  5. Too-few-studies returns skipped
  6. High heterogeneity data gives posterior_tau2 > 0
"""

import numpy as np
import pytest
from alburhan.engines.bayesian import BayesianMAEngine


# ══════════════════════════════ FIXTURES ════════════════════════════════════

@pytest.fixture
def engine():
    return BayesianMAEngine()


@pytest.fixture
def standard_claim():
    """Standard 5-study fixture.  DL estimate ≈ 0.504."""
    return {
        "yi":  [0.6, 0.5, 0.4, 0.45, 0.55],
        "sei": [0.1, 0.15, 0.1, 0.12, 0.1],
    }


@pytest.fixture
def heterogeneous_claim():
    """High heterogeneity — forces tau2 >> 0."""
    return {
        "yi":  [0.1, 0.8, -0.3, 0.5, 1.2, -0.1, 0.9],
        "sei": [0.1, 0.2,  0.15, 0.1, 0.25, 0.12, 0.18],
    }


# ══════════════════════════════ TESTS ═══════════════════════════════════════

class TestBayesianMAEngine:

    # 1. Posterior mean close to DL estimate with vague prior
    def test_posterior_mean_close_to_dl(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        # DL estimate ≈ 0.504; vague prior (sigma=10) → posterior dominated by data
        assert abs(r["posterior_mu"] - 0.504) < 0.05, (
            f"posterior_mu={r['posterior_mu']} not close to DL estimate ~0.504"
        )

    # 2. 95% CrI contains the DL estimate
    def test_cri_contains_dl_estimate(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        dl_estimate = 0.504
        assert r["cri_lo"] <= dl_estimate <= r["cri_hi"], (
            f"95% CrI [{r['cri_lo']}, {r['cri_hi']}] does not contain DL estimate {dl_estimate}"
        )

    # 3. Posterior predictive interval wider than credible interval
    def test_ppi_wider_than_cri(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        cri_width = r["cri_hi"] - r["cri_lo"]
        ppi_width = r["ppi_hi"] - r["ppi_lo"]
        assert ppi_width > cri_width, (
            f"PPI width {ppi_width:.4f} should exceed CrI width {cri_width:.4f}"
        )

    # 4. Bayes factor computed and > 0
    def test_bayes_factor_positive(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        assert "bf01" in r, "bf01 key missing from result"
        assert r["bf01"] > 0, f"BF01={r['bf01']} should be > 0"
        assert "evidence_label" in r, "evidence_label key missing"
        assert isinstance(r["evidence_label"], str)

    # 5. Too-few-studies returns skipped
    def test_too_few_studies_skipped(self, engine):
        r = engine.evaluate({"yi": [0.5, 0.6], "sei": [0.1, 0.1]})
        assert r["status"] == "skipped", (
            f"Expected 'skipped' for k=2, got {r['status']!r}"
        )

    # 6. High heterogeneity data gives posterior_tau2 > 0
    def test_high_heterogeneity_tau2_positive(self, engine, heterogeneous_claim):
        r = engine.evaluate(heterogeneous_claim)
        assert r["status"] == "evaluated"
        assert r["posterior_tau2"] > 0, (
            f"posterior_tau2={r['posterior_tau2']} should be > 0 for high-heterogeneity data"
        )
