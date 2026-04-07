"""
Tests for MetaRegressionEngine — WLS meta-regression with Knapp-Hartung correction.

6 tests:
  1. Basic evaluation returns slope, intercept, R2, p_value
  2. R2_analog bounded [0, 1]
  3. HKSJ SE >= Wald SE
  4. Slope direction matches known drift (monotonically decreasing effects → negative slope)
  5. Too-few-studies (k<3) returns skipped
  6. Homogeneous data gives R2 ≈ 0 (no variance to explain)
"""

import pytest
import numpy as np
from alburhan.engines.metareg import MetaRegressionEngine


# ══════════════════════════════ FIXTURES ════════════════════════════════════

@pytest.fixture
def engine():
    return MetaRegressionEngine()


@pytest.fixture
def standard_claim():
    """5-study fixture with mild year-effect relationship."""
    return {
        "yi":   [0.8, 0.7, 0.5, 0.4, 0.3],
        "sei":  [0.1, 0.12, 0.1, 0.11, 0.1],
        "years": [2000, 2005, 2010, 2015, 2020],
    }


@pytest.fixture
def decreasing_claim():
    """Studies with monotonically decreasing effects over years → slope < 0."""
    return {
        "yi":   [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
        "sei":  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "years": [2000, 2004, 2008, 2012, 2016, 2020],
    }


@pytest.fixture
def homogeneous_claim():
    """All effects identical — no variance for the covariate to explain."""
    return {
        "yi":   [0.5, 0.5, 0.5, 0.5, 0.5],
        "sei":  [0.1, 0.1, 0.1, 0.1, 0.1],
        "years": [2000, 2005, 2010, 2015, 2020],
    }


# ══════════════════════════════ TESTS ═══════════════════════════════════════

class TestMetaRegressionEngine:

    # 1. Basic evaluation returns slope, intercept, R2, p_value
    def test_basic_returns_required_keys(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        assert r["status"] == "evaluated", f"Expected 'evaluated', got {r['status']!r}"
        for key in ("slope", "intercept", "r2_analog", "p_value"):
            assert key in r, f"Missing key: {key!r}"
        # Values are finite numbers
        assert np.isfinite(r["slope"]), "slope is not finite"
        assert np.isfinite(r["intercept"]), "intercept is not finite"
        assert np.isfinite(r["p_value"]), "p_value is not finite"

    # 2. R2_analog bounded [0, 1]
    def test_r2_bounded(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        r2 = r["r2_analog"]
        assert 0.0 <= r2 <= 1.0, f"R2_analog={r2} is outside [0, 1]"

    # 3. HKSJ SE >= Wald SE
    def test_hksj_se_ge_wald_se(self, engine, standard_claim):
        r = engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        hksj = r["se_slope"]
        wald = r["se_slope_wald"]
        assert hksj >= wald - 1e-9, (
            f"HKSJ SE {hksj:.6f} should be >= Wald SE {wald:.6f}"
        )

    # 4. Slope direction matches known drift (monotonically decreasing → negative slope)
    def test_slope_direction_decreasing(self, engine, decreasing_claim):
        r = engine.evaluate(decreasing_claim)
        assert r["status"] == "evaluated"
        assert r["slope"] < 0, (
            f"Slope={r['slope']} should be negative for monotonically decreasing effects"
        )

    # 5. Too-few-studies (k<3) returns skipped
    def test_too_few_studies_skipped(self, engine):
        r = engine.evaluate({
            "yi":   [0.5, 0.6],
            "sei":  [0.1, 0.1],
            "years": [2010, 2015],
        })
        assert r["status"] == "skipped", (
            f"Expected 'skipped' for k=2, got {r['status']!r}"
        )

    # 6. Homogeneous data gives R2 ≈ 0 (no heterogeneity to explain)
    def test_homogeneous_data_r2_near_zero(self, engine, homogeneous_claim):
        r = engine.evaluate(homogeneous_claim)
        assert r["status"] == "evaluated"
        r2 = r["r2_analog"]
        assert r2 < 0.1, (
            f"R2_analog={r2} should be near 0 for perfectly homogeneous data"
        )
