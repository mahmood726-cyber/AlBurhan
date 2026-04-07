"""
PRISMA 2020 Compliance Engine.

Maps upstream engine outputs to the 27-item PRISMA 2020 checklist.
Each item is scored YES / PARTIAL / NO / NA.

Items that reference None → NA (requires human documentation).
Item 13a → YES if >=3 synthesis engines are evaluated.
All other items → check whether the referenced engine is evaluated.

Reference:
  Page MJ et al. The PRISMA 2020 statement: an updated guideline for reporting
  systematic reviews. BMJ. 2021;372:n71.
"""

import logging

logger = logging.getLogger(__name__)

# Items that require human documentation and cannot be assessed from engine outputs.
_NA_SOURCE = None

# Engines that contribute to item 13a "MULTI" synthesis methods check.
_SYNTHESIS_ENGINES = ["PredictionGap", "BayesianMA", "RobustMA", "MetaRegression", "FragilityAtlas"]
_SYNTHESIS_THRESHOLD = 3

# Mapping: item_id → (section_label, description, evidence_source)
# evidence_source values:
#   None         → NA (human input required)
#   "always"     → always YES (structural output)
#   engine_name  → YES if that engine has status == "evaluated"
#   "MULTI"      → special case for synthesis methods
_CHECKLIST = {
    1:   ("Title",                      "Identify the report as a systematic review",    "always"),
    2:   ("Abstract",                   "Structured summary",                             "E156"),
    3:   ("Rationale",                  "Describe rationale",                             _NA_SOURCE),
    4:   ("Objectives",                 "Provide explicit statement",                     _NA_SOURCE),
    5:   ("Eligibility",                "Specify eligibility criteria",                   _NA_SOURCE),
    6:   ("Info sources",               "Specify information sources",                    "Al-Mizan"),
    7:   ("Search strategy",            "Present full search strategy",                   _NA_SOURCE),
    8:   ("Selection",                  "State selection process",                        _NA_SOURCE),
    9:   ("Data collection",            "Describe data collection process",               _NA_SOURCE),
    10:  ("Data items",                 "List outcome data sought",                       _NA_SOURCE),
    11:  ("RoB assessment",             "Describe RoB assessment methods",                "RoB2"),
    12:  ("Effect measures",            "Specify effect measures used",                   "PredictionGap"),
    "13a": ("Synthesis methods",        "Describe synthesis methods",                     "MULTI"),
    "13b": ("Heterogeneity methods",    "Methods for heterogeneity exploration",          "FragilityAtlas"),
    "13c": ("Sensitivity analyses",     "Methods for sensitivity analyses",               "RobustMA"),
    "13d": ("Reporting bias methods",   "Methods for reporting bias assessment",          "PubBias"),
    14:  ("Reporting bias",             "Describe reporting bias assessment method",      "PubBias"),
    15:  ("Certainty",                  "Describe certainty assessment method",           "GRADE"),
    16:  ("Study selection results",    "Describe results of study selection",            _NA_SOURCE),
    17:  ("Study characteristics",      "Cite studies and present characteristics",       _NA_SOURCE),
    18:  ("RoB in studies",             "Present RoB assessments for included studies",   "RoB2"),
    19:  ("Individual results",         "Present individual study results",               "NetworkMeta"),
    20:  ("Synthesis results",          "Present results of each synthesis",              "PredictionGap"),
    21:  ("Reporting biases",           "Present results of reporting bias assessment",   "PubBias"),
    22:  ("Certainty results",          "Present certainty assessment results",           "GRADE"),
    23:  ("Discussion",                 "Provide general interpretation of results",      "E156"),
    24:  ("Limitations",                "Discuss limitations of evidence",                "E156"),
    25:  ("Conclusions",                "Provide conclusions",                            "E156"),
    26:  ("Registration",               "Provide registration information",               _NA_SOURCE),
    27:  ("Funding",                    "Describe funding sources and role",              _NA_SOURCE),
}


class PRISMAEngine:
    name = "PRISMA"

    def evaluate(self, claim_data):
        results = claim_data.get("audit_results", {})
        logger.info("%s: evaluating against 27-item PRISMA 2020 checklist", self.name)

        items = {}
        gaps = []
        yes_count = 0
        assessable_count = 0

        for item_id, (label, description, source) in _CHECKLIST.items():
            status, item_source = self._score_item(item_id, label, source, results)
            items[item_id] = {
                "label": label,
                "description": description,
                "status": status,
                "source": item_source,
            }

            if status == "NA":
                gaps.append(f"Item {item_id}: {label} — requires human documentation")
            else:
                assessable_count += 1
                if status in ("YES", "PARTIAL"):
                    yes_count += 1
                elif status == "NO":
                    gaps.append(f"Item {item_id}: {label} — upstream engine not evaluated")

        total_na = sum(1 for v in items.values() if v["status"] == "NA")
        percent_compliant = round(100.0 * yes_count / assessable_count, 1) if assessable_count > 0 else 0.0

        return {
            "status": "evaluated",
            "compliance_score": yes_count,
            "total_assessable": assessable_count,
            "total_na": total_na,
            "percent_compliant": percent_compliant,
            "items": items,
            "gaps": gaps,
        }

    # ─── Scoring helpers ─────────────────────────────────────────────────────

    def _score_item(self, item_id, label, source, results):
        """Return (status, source_description) for a single checklist item."""
        if source is None:
            return "NA", "requires human documentation"

        if source == "always":
            return "YES", "always — structured output produced"

        if source == "MULTI":
            return self._score_multi(results)

        # Single engine reference
        engine_result = results.get(source, {})
        if engine_result.get("status") == "evaluated":
            return "YES", f"engine:{source}"
        elif engine_result.get("status") in ("skipped", "error"):
            return "NO", f"engine:{source} (status={engine_result.get('status')})"
        else:
            # Engine not run or missing from results
            return "NO", f"engine:{source} (not found in audit_results)"

    def _score_multi(self, results):
        """Item 13a: YES if >=3 synthesis engines are evaluated."""
        evaluated = [e for e in _SYNTHESIS_ENGINES if results.get(e, {}).get("status") == "evaluated"]
        n_evaluated = len(evaluated)
        if n_evaluated >= _SYNTHESIS_THRESHOLD:
            return "YES", f"MULTI ({n_evaluated}/{len(_SYNTHESIS_ENGINES)} synthesis engines evaluated)"
        elif n_evaluated > 0:
            return "PARTIAL", f"MULTI ({n_evaluated}/{len(_SYNTHESIS_ENGINES)} synthesis engines evaluated)"
        else:
            return "NO", f"MULTI (0/{len(_SYNTHESIS_ENGINES)} synthesis engines evaluated)"
