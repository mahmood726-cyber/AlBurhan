"""
Tests for RobustMAEngine — 3 outlier-resistant meta-analysis estimators.

6 tests:
  1. Paule-Mandel returns tau2 >= 0
  2. Weighted median resists outlier (extreme value at 5.0 among 0.5s — median stays near 0.5)
  3. Winsorized mean resists outlier (result < 2.0 with the same outlier data)
  4. Clean data (all 0.5) — all 3 methods agree within 0.1
  5. Too-few (k<3) returns skipped
  6. Paule-Mandel converges flag is True for heterogeneous data
"""

import pytest
from alburhan.engines.robust import RobustMAEngine


# ══════════════════════════════ FIXTURES ════════════════════════════════════

@pytest.fixture
def engine():
    return RobustMAEngine()


@pytest.fixture
def clean_claim():
    """All effects at 0.5 — homogeneous."""
    return {
        "yi":  [0.5, 0.5, 0.5, 0.5, 0.5],
        "sei": [0.1, 0.1, 0.1, 0.1, 0.1],
    }


@pytest.fixture
def outlier_claim():
    """Four studies at 0.5, one extreme outlier at 5.0."""
    return {
        "yi":  [0.5, 0.5, 0.5, 0.5, 5.0],
        "sei": [0.1, 0.1, 0.1, 0.1, 0.1],
    }


@pytest.fixture
def heterogeneous_claim():
    """High heterogeneity — forces tau2 >> 0."""
    return {
        "yi":  [0.1, 0.8, -0.3, 0.5, 1.2, -0.1, 0.9],
        "sei": [0.1, 0.2,  0.15, 0.1, 0.25, 0.12, 0.18],
    }


# ══════════════════════════════ TESTS ═══════════════════════════════════════

class TestRobustMAEngine:

    # 1. Paule-Mandel returns tau2 >= 0
    def test_paule_mandel_tau2_nonnegative(self, engine, heterogeneous_claim):
        r = engine.evaluate(heterogeneous_claim)
        assert r["status"] == "evaluated"
        pm = r["paule_mandel"]
        assert pm["tau2"] >= 0, (
            f"Paule-Mandel tau2={pm['tau2']} should be >= 0"
        )

    # 2. Weighted median resists outlier
    def test_weighted_median_resists_outlier(self, engine, outlier_claim):
        r = engine.evaluate(outlier_claim)
        assert r["status"] == "evaluated"
        wmed_theta = r["weighted_median"]["theta"]
        assert abs(wmed_theta - 0.5) < 0.5, (
            f"Weighted median theta={wmed_theta} should stay near 0.5 despite outlier at 5.0"
        )

    # 3. Winsorized mean resists outlier
    def test_winsorized_mean_resists_outlier(self, engine, outlier_claim):
        r = engine.evaluate(outlier_claim)
        assert r["status"] == "evaluated"
        wins_theta = r["winsorized_mean"]["theta"]
        assert wins_theta < 2.0, (
            f"Winsorized mean theta={wins_theta} should be < 2.0 despite outlier at 5.0"
        )

    # 4. Clean data — all 3 methods agree within 0.1
    def test_clean_data_methods_agree(self, engine, clean_claim):
        r = engine.evaluate(clean_claim)
        assert r["status"] == "evaluated"
        pm_theta   = r["paule_mandel"]["theta"]
        wmed_theta = r["weighted_median"]["theta"]
        wins_theta = r["winsorized_mean"]["theta"]
        assert abs(pm_theta   - 0.5) < 0.1, f"PM theta={pm_theta} far from 0.5"
        assert abs(wmed_theta - 0.5) < 0.1, f"WMed theta={wmed_theta} far from 0.5"
        assert abs(wins_theta - 0.5) < 0.1, f"Wins theta={wins_theta} far from 0.5"

    # 5. Too-few studies returns skipped
    def test_too_few_studies_skipped(self, engine):
        r = engine.evaluate({"yi": [0.5, 0.6], "sei": [0.1, 0.1]})
        assert r["status"] == "skipped", (
            f"Expected 'skipped' for k=2, got {r['status']!r}"
        )

    # 6. Paule-Mandel converged flag is True for heterogeneous data
    def test_paule_mandel_converges_heterogeneous(self, engine, heterogeneous_claim):
        r = engine.evaluate(heterogeneous_claim)
        assert r["status"] == "evaluated"
        assert r["paule_mandel"]["converged"] is True, (
            "Paule-Mandel should converge for heterogeneous data"
        )
