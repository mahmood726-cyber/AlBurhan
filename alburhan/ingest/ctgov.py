"""
Live ClinicalTrials.gov API v2 client.
Searches for completed trials with results, extracts outcome measures.
"""
import json
import logging
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional
from alburhan.ingest.parser import parse_effect, counts_to_yi_sei

logger = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2"


class CTGovClient:
    """Fetch trial data from ClinicalTrials.gov API v2."""

    def search_trials(self, condition: str, intervention: Optional[str] = None,
                      max_results: int = 50) -> List[Dict]:
        """Search for completed trials with results."""
        params = {
            "query.cond": condition,
            "filter.overallStatus": "COMPLETED",
            "pageSize": min(max_results, 100),
            "fields": "NCTId,BriefTitle,EnrollmentCount,StartDate,CompletionDate,Phase",
        }
        if intervention:
            params["query.intr"] = intervention

        url = f"{BASE_URL}/studies?" + urllib.parse.urlencode(
            {k: v for k, v in params.items() if v}
        )
        logger.info("Searching CT.gov: %s", url)

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            studies = data.get("studies", [])
            return [self._parse_study_brief(s) for s in studies]
        except Exception as e:
            logger.error("CT.gov API error: %s", e)
            return []

    def get_trial_results(self, nct_id: str) -> Optional[Dict]:
        """Fetch detailed results for a specific trial."""
        url = (
            f"{BASE_URL}/studies/{nct_id}"
            "?fields=NCTId,BriefTitle,OutcomeMeasuresModule,"
            "EnrollmentInfo,ArmsInterventionsModule"
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return data
        except Exception as e:
            logger.error("Failed to fetch %s: %s", nct_id, e)
            return None

    def build_claim_data(self, condition: str, intervention: Optional[str] = None,
                         max_trials: int = 20) -> Dict[str, Any]:
        """
        Search CT.gov, extract results, build claim_data for orchestrator.
        Returns dict with yi, sei, years, n_per_study, condition, etc.
        """
        studies = self.search_trials(condition, intervention, max_results=max_trials)

        yi_list: List[float] = []
        sei_list: List[float] = []
        years_list: List[int] = []
        n_list: List[int] = []
        nct_ids: List[str] = []
        titles: List[str] = []

        for study in studies:
            nct_id = study.get("nct_id")
            if not nct_id:
                continue

            detail = self.get_trial_results(nct_id)
            if not detail:
                continue

            outcomes = self._extract_outcomes(detail)
            for outcome in outcomes:
                yi, sei = outcome.get("yi"), outcome.get("sei")
                if yi is not None and sei is not None and sei > 0:
                    yi_list.append(yi)
                    sei_list.append(sei)
                    years_list.append(study.get("year", 2020))
                    n_list.append(study.get("enrollment", 100))
                    nct_ids.append(nct_id)
                    titles.append(study.get("title", ""))
                    break  # One effect per trial (primary outcome)

        if not yi_list:
            return {
                "status": "empty",
                "message": f"No extractable results for {condition}",
            }

        return {
            "yi": yi_list,
            "sei": sei_list,
            "years": years_list,
            "n_per_study": n_list,
            "condition": condition,
            "country": "Global",
            "source": "ctgov_live",
            "nct_ids": nct_ids,
            "titles": titles,
        }

    def _parse_study_brief(self, study: Dict) -> Dict:
        """Parse brief study info from search results."""
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        design = proto.get("designModule", {})

        completion = status.get("completionDateStruct", {})
        year = (
            int(completion.get("date", "2020")[:4])
            if completion.get("date")
            else 2020
        )

        enrollment = design.get("enrollmentInfo", {}).get("count", 0)

        return {
            "nct_id": ident.get("nctId", ""),
            "title": ident.get("briefTitle", ""),
            "year": year,
            "enrollment": enrollment or 100,
        }

    def _extract_outcomes(self, detail: Dict) -> List[Dict]:
        """Extract effect sizes from trial results."""
        results: List[Dict] = []
        results_section = detail.get("resultsSection", {})
        outcome_module = results_section.get("outcomeMeasuresModule", {})

        if not outcome_module:
            return results

        for measure in outcome_module.get("outcomeMeasures", []):
            analyses = measure.get("analyses", [])
            for analysis in analyses:
                param_type = analysis.get("paramType", "")
                stat_value = analysis.get("statisticalMethod", "")

                value = self._safe_float(
                    analysis.get("estimateValue") or analysis.get("pValue")
                )
                ci_lo = self._safe_float(analysis.get("ciLowerLimit"))
                ci_hi = self._safe_float(analysis.get("ciUpperLimit"))

                if (
                    value is not None
                    and ci_lo is not None
                    and ci_hi is not None
                    and ci_lo < value < ci_hi
                ):
                    method = (stat_value + " " + param_type).lower()
                    if "hazard" in method:
                        etype = "HR"
                    elif "odds" in method:
                        etype = "OR"
                    elif "risk" in method or "relative" in method:
                        etype = "RR"
                    else:
                        etype = "MD"

                    yi, sei = parse_effect(etype, value, ci_lo, ci_hi)
                    if yi is not None:
                        results.append({"yi": yi, "sei": sei, "type": etype})

        return results

    @staticmethod
    def _safe_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
