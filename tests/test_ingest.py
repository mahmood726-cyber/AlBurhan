"""Tests for CT.gov ingestion pipeline."""

import csv
import math
import pytest
from alburhan.ingest.parser import (
    hr_to_yi_sei,
    or_to_yi_sei,
    rr_to_yi_sei,
    md_to_yi_sei,
    counts_to_yi_sei,
    parse_effect,
)


class TestParser:
    def test_hr_to_log(self):
        """HR=0.75 (0.60, 0.94) -> negative log(HR)."""
        yi, sei = hr_to_yi_sei(0.75, 0.60, 0.94)
        assert yi == pytest.approx(math.log(0.75), abs=1e-6)
        assert sei > 0

    def test_or_to_log(self):
        yi, sei = or_to_yi_sei(2.0, 1.2, 3.3)
        assert yi == pytest.approx(math.log(2.0), abs=1e-6)
        assert sei > 0

    def test_md_stays_linear(self):
        """Mean difference is not log-transformed."""
        yi, sei = md_to_yi_sei(5.0, 2.0, 8.0)
        assert yi == 5.0
        assert sei > 0

    def test_counts_to_logor(self):
        """2x2 table: 30/100 vs 15/100."""
        yi, sei = counts_to_yi_sei(30, 100, 15, 100)
        # log(OR) = log(30*85 / 70*15) ~ log(2.43) ~ 0.888
        assert 0.5 < yi < 1.5
        assert sei > 0

    def test_counts_zero_cell(self):
        """Zero cell should not crash (continuity correction)."""
        yi, sei = counts_to_yi_sei(0, 50, 5, 50)
        assert math.isfinite(yi)
        assert sei > 0

    def test_parse_effect_dispatch(self):
        """parse_effect dispatches to correct converter."""
        yi_hr, _ = parse_effect("HR", 0.75, 0.60, 0.94)
        yi_or, _ = parse_effect("Odds Ratio", 2.0, 1.2, 3.3)
        assert yi_hr < 0  # Protective
        assert yi_or > 0  # Harmful

    def test_parse_unknown_type(self):
        """Unknown measure type returns None."""
        yi, sei = parse_effect("WEIRD_TYPE", 1.0, 0.5, 1.5)
        assert yi is None

    def test_se_from_ci_width(self):
        """SE should be proportional to CI width."""
        _, sei_narrow = hr_to_yi_sei(0.8, 0.75, 0.85)
        _, sei_wide = hr_to_yi_sei(0.8, 0.50, 1.28)
        assert sei_wide > sei_narrow * 2

    def test_rr_converter(self):
        yi, sei = rr_to_yi_sei(1.5, 1.1, 2.0)
        assert yi > 0
        assert sei > 0

    def test_parse_effect_rr_aliases(self):
        """Both 'RR' and 'RELATIVE RISK' should work."""
        yi1, sei1 = parse_effect("RR", 1.5, 1.1, 2.0)
        yi2, sei2 = parse_effect("RELATIVE RISK", 1.5, 1.1, 2.0)
        assert yi1 == pytest.approx(yi2, abs=1e-9)
        assert sei1 == pytest.approx(sei2, abs=1e-9)


class TestAACTClient:
    def test_missing_dir_returns_error(self, tmp_path):
        from alburhan.ingest.aact import AACTClient

        client = AACTClient(data_dir=str(tmp_path / "nonexistent"))
        result = client.build_claim_data("Heart Failure")
        assert result["status"] == "error"

    def test_empty_studies_returns_empty(self, tmp_path):
        """studies.csv present but no matching rows -> empty."""
        from alburhan.ingest.aact import AACTClient

        studies_path = tmp_path / "studies.csv"
        with open(studies_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "nct_id",
                    "brief_title",
                    "official_title",
                    "enrollment",
                    "completion_date",
                    "overall_status",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "nct_id": "NCT00000001",
                    "brief_title": "Diabetes Treatment Trial",
                    "official_title": "",
                    "enrollment": "200",
                    "completion_date": "2020-01-01",
                    "overall_status": "completed",
                }
            )

        client = AACTClient(data_dir=str(tmp_path))
        result = client.build_claim_data("Heart Failure")
        assert result["status"] == "empty"

    def test_matching_study_no_outcomes_returns_empty(self, tmp_path):
        """Matching study found but no outcome_analyses.csv -> empty."""
        from alburhan.ingest.aact import AACTClient

        studies_path = tmp_path / "studies.csv"
        with open(studies_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "nct_id",
                    "brief_title",
                    "official_title",
                    "enrollment",
                    "completion_date",
                    "overall_status",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "nct_id": "NCT00000002",
                    "brief_title": "Heart Failure Intervention Study",
                    "official_title": "",
                    "enrollment": "300",
                    "completion_date": "2019-06-01",
                    "overall_status": "completed",
                }
            )

        client = AACTClient(data_dir=str(tmp_path))
        result = client.build_claim_data("Heart Failure")
        assert result["status"] == "empty"

    def test_full_pipeline_with_mock_csv(self, tmp_path):
        """End-to-end: matching study + valid outcome -> claim_data returned."""
        from alburhan.ingest.aact import AACTClient

        # Write studies.csv
        studies_path = tmp_path / "studies.csv"
        with open(studies_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "nct_id",
                    "brief_title",
                    "official_title",
                    "enrollment",
                    "completion_date",
                    "overall_status",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "nct_id": "NCT12345678",
                    "brief_title": "Heart Failure Drug Trial",
                    "official_title": "",
                    "enrollment": "500",
                    "completion_date": "2021-03-15",
                    "overall_status": "completed",
                }
            )

        # Write outcome_analyses.csv with a valid HR row
        outcomes_path = tmp_path / "outcome_analyses.csv"
        with open(outcomes_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "nct_id",
                    "param_value",
                    "ci_lower_limit",
                    "ci_upper_limit",
                    "statistical_method",
                    "param_type",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "nct_id": "NCT12345678",
                    "param_value": "0.75",
                    "ci_lower_limit": "0.60",
                    "ci_upper_limit": "0.94",
                    "statistical_method": "Cox Proportional Hazard",
                    "param_type": "Hazard Ratio",
                }
            )

        client = AACTClient(data_dir=str(tmp_path))
        result = client.build_claim_data("Heart Failure")

        assert result.get("status") not in ("error", "empty")
        assert len(result["yi"]) == 1
        assert result["yi"][0] == pytest.approx(math.log(0.75), abs=1e-4)
        assert result["sei"][0] > 0
        assert result["nct_ids"][0] == "NCT12345678"
        assert result["source"] == "aact_local"

    def test_safe_float_nan_returns_none(self, tmp_path):
        """_safe_float should return None for NaN strings."""
        from alburhan.ingest.aact import AACTClient

        client = AACTClient(data_dir=str(tmp_path))
        assert client._safe_float("nan") is None
        assert client._safe_float(None) is None
        assert client._safe_float("1.23") == pytest.approx(1.23)
