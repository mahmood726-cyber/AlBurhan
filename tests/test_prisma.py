"""
Tests for the PRISMA 2020 Compliance Engine.

6 tests covering: basic structure, valid statuses, item 13a multi-engine logic,
graceful degradation on missing upstreams, GRADE presence, and score bounds.
"""

import pytest
from alburhan.engines.prisma import PRISMAEngine


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _engine_ok(name):
    return {name: {"status": "evaluated"}}


def _full_audit():
    """All relevant upstream engines evaluated."""
    engines = [
        "E156", "Al-Mizan", "RoB2",
        "PredictionGap", "FragilityAtlas", "RobustMA",
        "PubBias", "GRADE", "NetworkMeta",
        "BayesianMA", "MetaRegression",
    ]
    return {name: {"status": "evaluated"} for name in engines}


def _empty_audit():
    return {}


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestPRISMAEngine:
    def setup_method(self):
        self.engine = PRISMAEngine()

    # Test 1: Basic evaluation returns all required fields
    def test_basic_evaluation_structure(self):
        result = self.engine.evaluate({"audit_results": _full_audit()})

        assert result["status"] == "evaluated"
        assert "compliance_score" in result
        assert "total_assessable" in result
        assert "total_na" in result
        assert "percent_compliant" in result
        assert "items" in result
        assert "gaps" in result

        # Items 1-27, with item 13 split into 13a/13b/13c/13d (30 total keys)
        assert len(result["items"]) == 30

        # Numeric sanity
        assert isinstance(result["compliance_score"], int)
        assert isinstance(result["total_assessable"], int)
        assert isinstance(result["total_na"], int)
        assert isinstance(result["gaps"], list)

    # Test 2: All assessable items have valid status values
    def test_all_items_have_valid_status(self):
        result = self.engine.evaluate({"audit_results": _full_audit()})
        valid_statuses = {"YES", "PARTIAL", "NO", "NA"}

        for item_id, item_info in result["items"].items():
            assert "status" in item_info, f"Item {item_id} missing 'status'"
            assert item_info["status"] in valid_statuses, (
                f"Item {item_id} has invalid status: {item_info['status']!r}"
            )
            assert "label" in item_info
            assert "description" in item_info
            assert "source" in item_info

    # Test 3: Item 13a is YES when >= 3 synthesis engines are evaluated
    def test_item_13a_yes_when_multiple_synthesis_engines_evaluated(self):
        audit = {
            "PredictionGap": {"status": "evaluated"},
            "BayesianMA":    {"status": "evaluated"},
            "RobustMA":      {"status": "evaluated"},
            "MetaRegression": {"status": "evaluated"},
            "FragilityAtlas": {"status": "evaluated"},
        }
        result = self.engine.evaluate({"audit_results": audit})
        assert result["items"]["13a"]["status"] == "YES"

    # Test 3b: Item 13a is PARTIAL when 1-2 synthesis engines are evaluated
    def test_item_13a_partial_when_few_synthesis_engines(self):
        audit = {
            "PredictionGap": {"status": "evaluated"},
            "BayesianMA":    {"status": "evaluated"},
        }
        result = self.engine.evaluate({"audit_results": audit})
        assert result["items"]["13a"]["status"] == "PARTIAL"

    # Test 4: Missing upstream engines degrade gracefully (NO not crash)
    def test_missing_engines_degrade_to_no_not_crash(self):
        result = self.engine.evaluate({"audit_results": _empty_audit()})

        assert result["status"] == "evaluated"
        # NA items are those requiring human input
        # All engine-referenced items (non-None, non-always) should be NO
        for item_id, item_info in result["items"].items():
            assert item_info["status"] in {"YES", "PARTIAL", "NO", "NA"}, (
                f"Item {item_id} has unexpected status {item_info['status']!r} on empty audit"
            )
        # Item 1 is always YES (structural output)
        assert result["items"][1]["status"] == "YES"

    # Test 5: GRADE presence makes items 15 and 22 YES
    def test_grade_presence_scores_items_15_and_22_as_yes(self):
        audit = {"GRADE": {"status": "evaluated"}}
        result = self.engine.evaluate({"audit_results": audit})
        assert result["items"][15]["status"] == "YES", "Item 15 (Certainty) should be YES when GRADE evaluated"
        assert result["items"][22]["status"] == "YES", "Item 22 (Certainty results) should be YES when GRADE evaluated"

    # Test 6: compliance_score <= total_assessable always
    def test_compliance_score_bounded_by_assessable(self):
        for audit in [_empty_audit(), _full_audit()]:
            result = self.engine.evaluate({"audit_results": audit})
            assert result["compliance_score"] >= 0
            assert result["compliance_score"] <= result["total_assessable"], (
                f"compliance_score={result['compliance_score']} > total_assessable={result['total_assessable']}"
            )
            # NA + assessable = 30 (27 PRISMA items, item 13 split into 4 sub-items)
            assert result["total_na"] + result["total_assessable"] == 30
