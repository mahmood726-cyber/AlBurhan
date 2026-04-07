"""
Tests for PubBiasEngine — Publication Bias Detection Suite (6 methods).

8 tests:
  1. Egger returns intercept + p_value in [0, 1]
  2. Begg returns kendall_tau in [-1, 1]
  3. Trim-fill n_missing >= 0; adjusted_theta differs when n_missing > 0
  4. Fail-safe N >= 0 with robust check (5k+10 rule)
  5. P-curve returns right_skew_p
  6. Excess significance returns observed_sig + expected_sig
  7. Too-few-studies (k<3) returns skipped
  8. Obviously biased data (small studies with large effects) triggers >= 1 flag
"""

import numpy as np
import pytest
from alburhan.engines.pubbias import PubBiasEngine


# ══════════════════════════════ FIXTURES ════════════════════════════════════

@pytest.fixture
def engine():
    return PubBiasEngine()


@pytest.fixture
def standard_claim():
    """Balanced 7-study meta-analysis — moderate heterogeneity, no obvious bias."""
    return {
        "yi":  [0.6, 0.5, 0.4, 0.45, 0.55, 0.50, 0.48],
        "sei": [0.1, 0.15, 0.1, 0.12, 0.1, 0.13, 0.11],
    }


@pytest.fixture
def biased_claim():
    """
    Simulated publication bias: small studies (large sei) report large effects;
    large studies (small sei) report modest effects — classic funnel asymmetry.
    """
    return {
        "yi":  [2.5, 2.0, 1.8, 0.6, 0.5, 0.4, 0.45, 0.50],
        "sei": [0.8, 0.7, 0.9, 0.1, 0.1, 0.12, 0.11, 0.10],
    }


# ══════════════════════════════ TESTS ═══════════════════════════════════════

class TestPubBiasEngine:

    # 1. Egger returns intercept + p_value in [0, 1]
    def test_egger_intercept_and_pvalue(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        egger = r["egger"]
        assert "intercept" in egger, "egger result missing 'intercept'"
        assert "p_value" in egger, "egger result missing 'p_value'"
        assert isinstance(egger["intercept"], float), "intercept should be float"
        assert 0.0 <= egger["p_value"] <= 1.0, (
            f"Egger p_value={egger['p_value']} not in [0, 1]"
        )

    # 2. Begg returns kendall_tau in [-1, 1]
    def test_begg_kendall_tau_range(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        begg = r["begg"]
        assert "kendall_tau" in begg, "begg result missing 'kendall_tau'"
        tau = begg["kendall_tau"]
        assert -1.0 <= tau <= 1.0, (
            f"Kendall's tau={tau} not in [-1, 1]"
        )

    # 3. Trim-fill n_missing >= 0; adjusted_theta differs when n_missing > 0
    def test_trim_fill_n_missing_and_adjustment(self, engine, standard_claim, biased_claim):
        r_balanced = engine.evaluate(standard_claim)
        tf_balanced = r_balanced["trim_fill"]
        assert tf_balanced["n_missing"] >= 0, "n_missing should be >= 0"
        assert "adjusted_theta" in tf_balanced, "trim_fill missing 'adjusted_theta'"

        r_biased = engine.evaluate(biased_claim)
        tf_biased = r_biased["trim_fill"]
        if tf_biased["n_missing"] > 0:
            assert tf_biased["adjusted_theta"] != tf_biased["original_theta"], (
                "When n_missing > 0, adjusted_theta should differ from original_theta"
            )

    # 4. Fail-safe N >= 0 with robust check
    def test_failsafe_n_nonnegative_and_robust(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        fsn = r["failsafe_n"]
        assert "failsafe_n" in fsn, "failsafe_n result missing 'failsafe_n' key"
        assert fsn["failsafe_n"] >= 0, (
            f"failsafe_n={fsn['failsafe_n']} should be >= 0"
        )
        assert "robust" in fsn, "failsafe_n result missing 'robust' key"
        assert isinstance(fsn["robust"], bool), "'robust' should be bool"
        # Verify threshold formula: 5k + 10
        k = len(standard_claim["yi"])
        assert fsn["robust_threshold"] == 5 * k + 10, (
            f"robust_threshold={fsn['robust_threshold']} != 5*{k}+10={5*k+10}"
        )

    # 5. P-curve returns right_skew_p
    def test_pcurve_returns_right_skew_p(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        pcurve = r["p_curve"]
        assert "right_skew_p" in pcurve, "p_curve result missing 'right_skew_p'"
        # right_skew_p is either a float in [0,1] or nan (if no significant studies)
        rsp = pcurve["right_skew_p"]
        if not (isinstance(rsp, float) and np.isnan(rsp)):
            assert 0.0 <= rsp <= 1.0, (
                f"right_skew_p={rsp} not in [0, 1]"
            )
        assert "n_significant" in pcurve, "p_curve result missing 'n_significant'"

    # 6. Excess significance returns observed_sig + expected_sig
    def test_excess_significance_observed_and_expected(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        es = r["excess_significance"]
        assert "observed_sig" in es, "excess_significance missing 'observed_sig'"
        assert "expected_sig" in es, "excess_significance missing 'expected_sig'"
        assert isinstance(es["observed_sig"], int), "observed_sig should be int"
        assert es["expected_sig"] >= 0.0, (
            f"expected_sig={es['expected_sig']} should be >= 0"
        )
        # observed_sig cannot exceed total k
        k = len(standard_claim["yi"])
        assert 0 <= es["observed_sig"] <= k, (
            f"observed_sig={es['observed_sig']} out of [0, {k}]"
        )

    # 7. Too-few-studies (k < 3) returns skipped
    def test_too_few_studies_skipped(self, engine):
        r = engine.evaluate({"yi": [0.5, 0.6], "sei": [0.1, 0.1]})
        assert r["status"] == "skipped", (
            f"Expected 'skipped' for k=2, got {r['status']!r}"
        )

        r_one = engine.evaluate({"yi": [0.5], "sei": [0.1]})
        assert r_one["status"] == "skipped", (
            f"Expected 'skipped' for k=1, got {r_one['status']!r}"
        )

    # 8. Obviously biased data triggers at least 1 flag
    def test_biased_data_triggers_flags(self, engine, biased_claim):
        r = engine.evaluate(biased_claim)
        assert r["status"] == "evaluated"
        flags = r["bias_flags"]
        assert isinstance(flags, list), "bias_flags should be a list"
        assert len(flags) >= 1, (
            f"Expected >= 1 bias flag for obviously asymmetric data, got 0. "
            f"Full result: {r}"
        )
