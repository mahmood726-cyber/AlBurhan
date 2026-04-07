"""
Tests for the GRADE Certainty Assessment Engine.

8 tests covering: basic evaluation, HIGH certainty, compound downgrades,
VERY LOW certainty, publication bias, unknown alignment, missing engines,
and score clamping.
"""

import pytest
from alburhan.engines.grade import GRADEEngine


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_audit_results(
    rob_flags=0,
    i2=10.0,
    n_influential=0,
    aligned=True,
    ci_width=0.3,
    is_fragile=False,
    egger_significant=False,
    n_missing=0,
):
    """Build a minimal audit_results dict with configurable parameters."""
    return {
        "RegistryForensics": {
            "status": "evaluated",
            "anomaly_flags": rob_flags,
        },
        "PredictionGap": {
            "status": "evaluated",
            "metrics": {
                "I2": i2,
                "ci_width": ci_width,
            },
        },
        "NetworkMeta": {
            "status": "evaluated",
            "n_influential": n_influential,
        },
        "AfricaRCT": {
            "status": "evaluated",
            "burden_alignment": {
                "aligned": aligned,
                "reason": "Condition aligned with burden" if aligned else "Not in top burden",
            },
        },
        "FragilityAtlas": {
            "status": "evaluated",
            "is_fragile": is_fragile,
        },
        "PubBias": {
            "status": "evaluated",
            "egger": {"significant": egger_significant},
            "trim_fill": {"n_missing": n_missing},
        },
    }


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestGRADEEngine:
    def setup_method(self):
        self.engine = GRADEEngine()

    # Test 1: Basic evaluation returns correct structure
    def test_basic_evaluation_structure(self):
        audit = _make_audit_results()
        result = self.engine.evaluate({"audit_results": audit})

        assert result["status"] == "evaluated"
        assert "certainty" in result
        assert "certainty_score" in result
        assert "domains" in result
        assert "total_downgrade" in result

        domains = result["domains"]
        for domain in ("risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias"):
            assert domain in domains
            assert "downgrade" in domains[domain]
            assert "reason" in domains[domain]
            assert isinstance(domains[domain]["reason"], str)
            assert len(domains[domain]["reason"]) > 0

        assert result["certainty"] in ("HIGH", "MODERATE", "LOW", "VERY LOW")
        assert result["certainty_score"] in (1, 2, 3, 4)

    # Test 2: No downgrades → HIGH certainty (score 4)
    def test_no_downgrades_gives_high_certainty(self):
        audit = _make_audit_results(
            rob_flags=0,
            i2=10.0,
            n_influential=0,
            aligned=True,
            ci_width=0.3,
            is_fragile=False,
            egger_significant=False,
            n_missing=0,
        )
        result = self.engine.evaluate({"audit_results": audit})

        assert result["certainty"] == "HIGH"
        assert result["certainty_score"] == 4
        assert result["total_downgrade"] == 0
        for domain, info in result["domains"].items():
            assert info["downgrade"] == 0, f"{domain} unexpectedly downgraded"

    # Test 3: High I2 + fragile → at least 2 downgrades (inconsistency + imprecision)
    def test_high_i2_and_fragile_gives_two_downgrades(self):
        audit = _make_audit_results(
            i2=80.0,           # > 75% → -2 inconsistency
            n_influential=0,
            is_fragile=True,   # fragile → at least -1 imprecision
            ci_width=0.3,      # < 0.5 but fragile → -1
        )
        result = self.engine.evaluate({"audit_results": audit})

        total = result["total_downgrade"]
        assert total <= -2, f"Expected at least 2 downgrades, got {total}"

        domains = result["domains"]
        assert domains["inconsistency"]["downgrade"] == -2
        assert domains["imprecision"]["downgrade"] == -1

    # Test 4: All domains bad → VERY LOW certainty
    def test_all_domains_bad_gives_very_low(self):
        audit = _make_audit_results(
            rob_flags=3,       # → -2
            i2=90.0,           # → -2
            n_influential=3,
            aligned=False,     # → -1
            ci_width=1.5,      # > 1.0 AND fragile → -2
            is_fragile=True,
            egger_significant=True,  # both flagged, n_missing>=2 → -2
            n_missing=3,
        )
        result = self.engine.evaluate({"audit_results": audit})

        assert result["certainty"] == "VERY LOW"
        assert result["certainty_score"] == 1

    # Test 5: Publication bias flagged → -1 in that domain
    def test_publication_bias_egger_flagged(self):
        audit = _make_audit_results(
            egger_significant=True,
            n_missing=0,
        )
        result = self.engine.evaluate({"audit_results": audit})

        pub_domain = result["domains"]["publication_bias"]
        assert pub_domain["downgrade"] == -1
        assert "egger" in pub_domain["reason"].lower() or "flagged" in pub_domain["reason"].lower()

    # Test 6: Unknown country → -1 indirectness
    def test_unknown_country_alignment_gives_downgrade(self):
        audit = _make_audit_results(aligned=None)
        # Override AfricaRCT to simulate unknown country (aligned=None)
        audit["AfricaRCT"]["burden_alignment"] = {
            "aligned": None,
            "reason": "No GBD data for Mars",
        }
        result = self.engine.evaluate({"audit_results": audit})

        ind_domain = result["domains"]["indirectness"]
        assert ind_domain["downgrade"] == -1

    # Test 7: Missing upstream engines handled gracefully (no crash, worst-case downgrade)
    def test_missing_upstream_engines_no_crash(self):
        # Provide completely empty audit_results
        result = self.engine.evaluate({"audit_results": {}})

        assert result["status"] == "evaluated"
        assert result["certainty"] in ("HIGH", "MODERATE", "LOW", "VERY LOW")
        assert result["certainty_score"] >= 1

        # Each domain must still have a downgrade value
        for domain_name, domain_info in result["domains"].items():
            assert "downgrade" in domain_info
            assert isinstance(domain_info["downgrade"], int)
            assert "reason" in domain_info

    # Test 8: Score clamped to [1, 4] range
    def test_score_clamped_to_valid_range(self):
        # All-bad case: raw score would be 4 + (-2-2-1-2-2) = 4-9 = -5 → clamp to 1
        audit = _make_audit_results(
            rob_flags=3,
            i2=90.0,
            n_influential=3,
            aligned=False,
            ci_width=1.5,
            is_fragile=True,
            egger_significant=True,
            n_missing=3,
        )
        result = self.engine.evaluate({"audit_results": audit})
        assert result["certainty_score"] >= 1
        assert result["certainty_score"] <= 4

        # No-downgrade case: raw score = 4 → no overshoot
        audit_clean = _make_audit_results()
        result_clean = self.engine.evaluate({"audit_results": audit_clean})
        assert result_clean["certainty_score"] == 4
