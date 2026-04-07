"""
Local AACT (Aggregate Analysis of ClinicalTrials.gov) CSV parser.
Reads from downloaded AACT data files.
"""
import os
import logging
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List
from alburhan.ingest.parser import parse_effect, counts_to_yi_sei

logger = logging.getLogger(__name__)

# Default AACT data directory
DEFAULT_AACT_DIR = os.environ.get(
    "ALBURHAN_AACT_DIR", str(Path.home() / "ctgov_data")
)


class AACTClient:
    """Parse trial data from local AACT CSV files."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or DEFAULT_AACT_DIR)

    def build_claim_data(
        self,
        condition: str,
        intervention: Optional[str] = None,
        max_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Parse local AACT CSVs, extract results, build claim_data.

        Expected files in data_dir:
        - studies.csv       (nct_id, brief_title, enrollment, completion_date, overall_status)
        - outcome_analyses.csv  (effect estimates)
        """
        studies_path = self.data_dir / "studies.csv"
        outcomes_path = self.data_dir / "outcome_analyses.csv"

        if not studies_path.exists():
            return {
                "status": "error",
                "message": f"AACT studies.csv not found at {studies_path}",
            }

        matching_ncts = self._find_matching_studies(
            studies_path, condition, intervention, max_trials
        )

        if not matching_ncts:
            return {
                "status": "empty",
                "message": f"No matching studies for {condition}",
            }

        yi_list: List[float] = []
        sei_list: List[float] = []
        years_list: List[int] = []
        n_list: List[int] = []
        nct_ids: List[str] = []

        if outcomes_path.exists():
            effects = self._parse_outcome_analyses(outcomes_path, matching_ncts)
            for eff in effects:
                yi_list.append(eff["yi"])
                sei_list.append(eff["sei"])
                years_list.append(eff["year"])
                n_list.append(eff["n"])
                nct_ids.append(eff["nct_id"])

        if not yi_list:
            return {
                "status": "empty",
                "message": f"No extractable results for {condition} in AACT",
            }

        return {
            "yi": yi_list,
            "sei": sei_list,
            "years": years_list,
            "n_per_study": n_list,
            "condition": condition,
            "country": "Global",
            "source": "aact_local",
            "nct_ids": nct_ids,
        }

    def _find_matching_studies(
        self,
        studies_path: Path,
        condition: str,
        intervention: Optional[str],
        max_trials: int,
    ) -> Dict[str, Dict]:
        """Find NCT IDs matching condition (and optionally intervention) from studies.csv."""
        matches: Dict[str, Dict] = {}
        condition_lower = condition.lower()
        intervention_lower = intervention.lower() if intervention else None

        with open(studies_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = (row.get("overall_status") or "").lower()
                if status != "completed":
                    continue

                title = (
                    (row.get("brief_title") or "")
                    + " "
                    + (row.get("official_title") or "")
                ).lower()
                if condition_lower not in title:
                    continue

                if intervention_lower and intervention_lower not in title:
                    continue

                nct_id = row.get("nct_id", "")
                year_str = (
                    row.get("completion_date")
                    or row.get("study_first_posted_date")
                    or "2020"
                )
                try:
                    year = int(str(year_str)[:4])
                except (ValueError, IndexError):
                    year = 2020

                enrollment = int(row.get("enrollment") or 0) or 100

                matches[nct_id] = {"year": year, "n": enrollment}

                if len(matches) >= max_trials:
                    break

        return matches

    def _parse_outcome_analyses(
        self,
        outcomes_path: Path,
        matching_ncts: Dict[str, Dict],
    ) -> List[Dict]:
        """Extract effect sizes from outcome_analyses.csv for matching NCTs."""
        results: List[Dict] = []
        seen_ncts: set = set()

        with open(outcomes_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nct_id = row.get("nct_id", "")
                if nct_id not in matching_ncts or nct_id in seen_ncts:
                    continue

                estimate = self._safe_float(
                    row.get("param_value") or row.get("estimate")
                )
                ci_lo = self._safe_float(row.get("ci_lower_limit"))
                ci_hi = self._safe_float(row.get("ci_upper_limit"))

                if estimate is None or ci_lo is None or ci_hi is None:
                    continue
                if not (ci_lo < estimate < ci_hi):
                    continue

                method = (
                    row.get("statistical_method") or row.get("param_type") or ""
                ).lower()
                if "hazard" in method:
                    etype = "HR"
                elif "odds" in method:
                    etype = "OR"
                elif "risk" in method:
                    etype = "RR"
                else:
                    etype = "MD"

                yi, sei = parse_effect(etype, estimate, ci_lo, ci_hi)
                if yi is not None and sei is not None and sei > 0:
                    meta = matching_ncts[nct_id]
                    results.append(
                        {
                            "yi": yi,
                            "sei": sei,
                            "year": meta["year"],
                            "n": meta["n"],
                            "nct_id": nct_id,
                        }
                    )
                    seen_ncts.add(nct_id)

        return results

    @staticmethod
    def _safe_float(val):
        if val is None:
            return None
        try:
            v = float(str(val).strip())
            return v if v == v else None  # NaN check (NaN != NaN)
        except (ValueError, TypeError):
            return None
