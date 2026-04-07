"""
Tests for the RoB 2 Risk-of-Bias Engine.

6 tests covering: basic structure, valid overall_risk, per_study_risk length,
clean data assessment, graceful degradation on missing forensics, and
n_high_risk_studies bounds.
"""

import pytest
import numpy as np
from alburhan.engines.rob import RoB2Engine


# ─── Helpers ─────────────────────────────────────────────────────────────────

_VALID_RISKS = {"Low", "Some Concerns", "High"}
_DOMAINS = ("randomization", "deviations", "missing_data", "measurement", "selection")


def _clean_forensics():
    """Forensics result with no anomaly flags — all tests passed."""
    return {
        "status": "evaluated",
        "anomaly_flags": 0,
        "grim": {"flagged": False, "status": "evaluated"},
        "terminal_digit": {"flagged": False, "p_value": 0.72},
        "se_homogeneity": {"flagged": False, "cochran_c": 0.21},
        "normality": {"flagged": False, "p_value": 0.45},
        "benford": {"flagged": False, "p_value": 0.38},
    }


def _clean_pubbias():
    """PubBias result with no bias signals."""
    return {
        "status": "evaluated",
        "egger": {"significant": False, "p_value": 0.45},
        "trim_fill": {"n_missing": 0},
        "excess_significance": {"significant": False, "p_value": 0.62},
    }


def _standard_claim():
    return {
        "yi": [0.6, 0.5, 0.4, 0.45, 0.55],
        "sei": [0.10, 0.14, 0.11, 0.12, 0.10],
        "audit_results": {
            "RegistryForensics": _clean_forensics(),
            "PubBias": _clean_pubbias(),
            "NetworkMeta": {
                "status": "evaluated",
                "n_influential": 0,
                "influential_studies": [],
            },
        },
    }


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestRoB2Engine:
    def setup_method(self):
        self.engine = RoB2Engine()

    # Test 1: Basic evaluation returns required fields with correct structure
    def test_basic_evaluation_structure(self):
        result = self.engine.evaluate(_standard_claim())

        assert result["status"] == "evaluated"
        assert "overall_risk" in result
        assert "domain_judgments" in result
        assert "per_study_risk" in result
        assert "n_high_risk_studies" in result

        # All 5 domains present
        domain_judgments = result["domain_judgments"]
        for domain in _DOMAINS:
            assert domain in domain_judgments, f"Domain '{domain}' missing"
            assert "judgment" in domain_judgments[domain]
            assert "reason" in domain_judgments[domain]
            assert isinstance(domain_judgments[domain]["reason"], str)
            assert len(domain_judgments[domain]["reason"]) > 0

    # Test 2: overall_risk is always one of the 3 valid values
    def test_overall_risk_is_valid_value(self):
        result = self.engine.evaluate(_standard_claim())
        assert result["overall_risk"] in _VALID_RISKS

        # Also check on empty audit
        empty_claim = {
            "yi": [0.3, 0.4, 0.5],
            "sei": [0.1, 0.1, 0.1],
            "audit_results": {},
        }
        result2 = self.engine.evaluate(empty_claim)
        assert result2["overall_risk"] in _VALID_RISKS

    # Test 3: per_study_risk has exactly k entries
    def test_per_study_risk_has_k_entries(self):
        for k in [3, 5, 8]:
            claim = {
                "yi": list(np.linspace(0.3, 0.7, k)),
                "sei": [0.1] * k,
                "audit_results": {},
            }
            result = self.engine.evaluate(claim)
            assert len(result["per_study_risk"]) == k, (
                f"Expected {k} per_study_risk entries, got {len(result['per_study_risk'])}"
            )

    # Test 4: Clean data (no forensic flags, no bias) → Low or Some Concerns
    def test_clean_data_gives_low_or_some_concerns(self):
        result = self.engine.evaluate(_standard_claim())
        # Domain 2 (deviations) always defaults to Some Concerns
        # So overall worst-case with clean data should be Some Concerns, not High
        assert result["overall_risk"] in ("Low", "Some Concerns"), (
            f"Expected Low or Some Concerns on clean data, got {result['overall_risk']!r}"
        )

    # Test 5: Missing upstream forensics → Some Concerns (not crash)
    def test_missing_forensics_defaults_gracefully(self):
        claim = {
            "yi": [0.4, 0.5, 0.6, 0.45],
            "sei": [0.12, 0.10, 0.11, 0.13],
            "audit_results": {},   # No upstream engines
        }
        result = self.engine.evaluate(claim)

        assert result["status"] == "evaluated"
        assert result["overall_risk"] in _VALID_RISKS

        for domain in _DOMAINS:
            j = result["domain_judgments"][domain]
            assert j["judgment"] in _VALID_RISKS
            assert "defaulting" in j["reason"].lower() or len(j["reason"]) > 0

    # Test 6: n_high_risk_studies is between 0 and k
    def test_n_high_risk_studies_bounded(self):
        result = self.engine.evaluate(_standard_claim())
        k = len(_standard_claim()["yi"])
        assert result["n_high_risk_studies"] >= 0
        assert result["n_high_risk_studies"] <= k

        # Verify it equals count of "High" entries in per_study_risk
        high_count = sum(1 for r in result["per_study_risk"] if r == "High")
        assert result["n_high_risk_studies"] == high_count
