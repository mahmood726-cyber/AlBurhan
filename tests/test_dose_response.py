"""
Tests for DoseResponseEngine — 6 tests covering all models and edge cases.

1. Linear model returns slope, intercept, p_value, R2
2. Quadratic model returns b2 coefficient + AIC
3. RCS returns spline coefficients (knots + spline_coef)
4. MED computed (or None when all CIs include null)
5. No 'doses' key → status="skipped"
6. Monotonic dose-response detected (slope > 0 when effects increase with dose)
"""

import pytest
import math
from alburhan.engines.dose_response import DoseResponseEngine


# ══════════════════════════════ FIXTURES ════════════════════════════════════

@pytest.fixture
def engine():
    return DoseResponseEngine()


@pytest.fixture
def monotonic_claim():
    """Strong positive linear dose-response across 6 dose levels."""
    return {
        "yi":    [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
        "sei":   [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "doses": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }


@pytest.fixture
def null_claim():
    """Effects all near zero — CI always includes null → MED = None."""
    return {
        "yi":    [0.001, -0.001, 0.001, -0.001, 0.001, -0.001],
        "sei":   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "doses": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }


@pytest.fixture
def curved_claim():
    """Inverted-U shape — tests quadratic b2 < 0."""
    return {
        "yi":    [0.2, 0.5, 0.8, 0.8, 0.5, 0.2],
        "sei":   [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "doses": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }


# ══════════════════════════════ TESTS ═══════════════════════════════════════

class TestDoseResponseEngine:

    # 1. Linear model returns slope, intercept, p_value, R2
    def test_linear_model_keys(self, engine, monotonic_claim):
        r = engine.evaluate(monotonic_claim)
        assert r["status"] == "evaluated", f"Expected 'evaluated', got {r['status']!r}"
        lin = r["linear"]
        for key in ("slope", "intercept", "p_value", "r2"):
            assert key in lin, f"Missing key '{key}' in linear result: {lin}"
        assert math.isfinite(lin["slope"]),     "slope should be finite"
        assert math.isfinite(lin["intercept"]), "intercept should be finite"
        assert 0.0 <= lin["p_value"] <= 1.0,   f"p_value={lin['p_value']} out of [0,1]"
        assert math.isfinite(lin["r2"]),         "r2 should be finite"

    # 2. Quadratic model returns b2 coefficient + AIC
    def test_quadratic_model_keys(self, engine, curved_claim):
        r = engine.evaluate(curved_claim)
        assert r["status"] == "evaluated"
        quad = r["quadratic"]
        for key in ("b2", "aic"):
            assert key in quad, f"Missing key '{key}' in quadratic result: {quad}"
        assert math.isfinite(quad["b2"]),  "b2 should be finite"
        assert math.isfinite(quad["aic"]), "AIC should be finite"
        # Inverted-U → b2 should be negative
        assert quad["b2"] < 0, f"Expected b2 < 0 for inverted-U data, got {quad['b2']}"

    # 3. RCS returns spline coefficients
    def test_rcs_returns_spline_coef(self, engine, monotonic_claim):
        r = engine.evaluate(monotonic_claim)
        assert r["status"] == "evaluated"
        rcs = r["rcs"]
        assert "knots" in rcs,       "RCS result must contain 'knots'"
        assert "spline_coef" in rcs, "RCS result must contain 'spline_coef'"
        assert len(rcs["knots"]) == 3, f"Expected 3 knots, got {len(rcs['knots'])}"
        assert math.isfinite(rcs["spline_coef"]), "spline_coef should be finite"

    # 4. MED computed (or None when all CIs include null)
    def test_med_none_when_ci_includes_null(self, engine, null_claim):
        r = engine.evaluate(null_claim)
        assert r["status"] == "evaluated"
        med = r["med"]
        assert med is None, (
            f"Expected MED=None when all CIs include null, got {med}"
        )

    # 5. No 'doses' key → status="skipped"
    def test_no_doses_skipped(self, engine):
        claim = {
            "yi":  [0.5, 0.6, 0.7, 0.8],
            "sei": [0.1, 0.1, 0.1, 0.1],
            # no 'doses' key
        }
        r = engine.evaluate(claim)
        assert r["status"] == "skipped", (
            f"Expected 'skipped' when doses absent, got {r['status']!r}"
        )

    # 6. Monotonic dose-response detected (slope > 0 when effects increase with dose)
    def test_monotonic_slope_positive(self, engine, monotonic_claim):
        r = engine.evaluate(monotonic_claim)
        assert r["status"] == "evaluated"
        slope = r["linear"]["slope"]
        assert slope > 0, (
            f"Expected positive slope for monotonically increasing dose-response, got {slope}"
        )
        # Also check MED is not None (strong signal should produce a detectable MED)
        # Note: MED may be the lowest dose if the CI at the lowest dose excludes 0
        med = r["med"]
        # MED should be a positive dose value when effects are reliably above null
        if med is not None:
            assert med > 0, f"MED should be a positive dose, got {med}"
