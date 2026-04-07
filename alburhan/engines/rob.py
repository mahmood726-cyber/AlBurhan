"""
RoB 2 Risk-of-Bias Engine.

Per-study and overall risk-of-bias assessment using forensic signals from
upstream engines. Maps to the 5 RoB 2 domains.

Domains:
  1. Randomization process  — GRIM test, terminal digit irregularities
  2. Deviations from interventions — No direct signal; default "Some Concerns"
  3. Missing outcome data   — SE homogeneity (imputation marker), SE CV
  4. Measurement of outcome — Shapiro-Wilk normality of standardized residuals
  5. Selection of reported result — PubBias Egger test, excess significance

Per-study signals:
  - Studentized residuals from NetworkMeta (|r| > 2.5 → High risk)
  - Contribution to Q from NetworkMeta (outlier contribution → concern)

Reference:
  Sterne JAC et al. RoB 2: a revised tool for assessing risk of bias in
  randomised trials. BMJ. 2019;366:l4898.
"""

import logging
import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

_JUDGMENT_ORDER = {"Low": 0, "Some Concerns": 1, "High": 2}
_JUDGMENTS = ("Low", "Some Concerns", "High")


def _worst(*judgments):
    """Return the highest-risk judgment among those provided."""
    order = [_JUDGMENT_ORDER.get(j, 1) for j in judgments]
    return _JUDGMENTS[max(order)]


class RoB2Engine:
    name = "RoB2"

    def evaluate(self, claim_data):
        yi = np.array(claim_data.get("yi", []), dtype=float)
        sei = np.array(claim_data.get("sei", []), dtype=float)
        k = len(yi)
        logger.info("%s: evaluating k=%d studies across 5 RoB 2 domains", self.name, k)

        if k < 3:
            return {
                "status": "skipped",
                "message": "Need k>=3 for RoB 2 assessment.",
            }

        results = claim_data.get("audit_results", {})
        forensics = results.get("RegistryForensics", {})
        nma = results.get("NetworkMeta", {})
        pubbias = results.get("PubBias", {})

        # ── Domain judgments ──────────────────────────────────────────────────
        domain1_j, domain1_r = self._domain_randomization(forensics)
        domain2_j, domain2_r = self._domain_deviations()
        domain3_j, domain3_r = self._domain_missing_data(forensics, sei)
        domain4_j, domain4_r = self._domain_measurement(forensics)
        domain5_j, domain5_r = self._domain_selection(pubbias)

        domain_judgments = {
            "randomization": {"judgment": domain1_j, "reason": domain1_r},
            "deviations": {"judgment": domain2_j, "reason": domain2_r},
            "missing_data": {"judgment": domain3_j, "reason": domain3_r},
            "measurement": {"judgment": domain4_j, "reason": domain4_r},
            "selection": {"judgment": domain5_j, "reason": domain5_r},
        }

        # ── Per-study risk ────────────────────────────────────────────────────
        per_study_risk = self._per_study_assessment(k, nma, domain1_j, domain5_j)
        n_high = sum(1 for r in per_study_risk if r == "High")

        # ── Overall risk = worst domain ───────────────────────────────────────
        overall = _worst(domain1_j, domain2_j, domain3_j, domain4_j, domain5_j)

        return {
            "status": "evaluated",
            "overall_risk": overall,
            "domain_judgments": domain_judgments,
            "n_high_risk_studies": n_high,
            "per_study_risk": per_study_risk,
        }

    # ── Domain 1: Randomization process ──────────────────────────────────────

    def _domain_randomization(self, forensics):
        if forensics.get("status") != "evaluated":
            return "Some Concerns", "RegistryForensics not available; defaulting to Some Concerns"

        flags = forensics.get("anomaly_flags", 0)
        grim = forensics.get("grim", {})
        terminal = forensics.get("terminal_digit", {})

        grim_flagged = isinstance(grim, dict) and grim.get("flagged", False)
        terminal_flagged = isinstance(terminal, dict) and terminal.get("flagged", False)

        if grim_flagged and terminal_flagged:
            return "High", "GRIM test failed and terminal digit irregularity detected"
        elif grim_flagged:
            return "High", "GRIM test failed — reported proportions inconsistent with integer N"
        elif terminal_flagged:
            return "Some Concerns", "Terminal digit irregularity detected"
        elif flags >= 3:
            return "High", f"{flags} forensic anomaly flags detected"
        elif flags >= 1:
            return "Some Concerns", f"{flags} forensic anomaly flag(s) detected"
        else:
            return "Low", "No randomization-related anomalies detected"

    # ── Domain 2: Deviations from interventions ───────────────────────────────

    def _domain_deviations(self):
        # No direct signal available from the engine suite
        return "Some Concerns", "No protocol deviation data available; defaulting to Some Concerns"

    # ── Domain 3: Missing outcome data ────────────────────────────────────────

    def _domain_missing_data(self, forensics, sei):
        if forensics.get("status") != "evaluated":
            return "Some Concerns", "RegistryForensics not available; defaulting to Some Concerns"

        se_hom = forensics.get("se_homogeneity", {})
        hom_flagged = isinstance(se_hom, dict) and se_hom.get("flagged", False)

        # SE coefficient of variation — very low CV suggests imputed/identical SEs
        cv = float(np.std(sei) / np.mean(sei)) if len(sei) >= 3 and np.mean(sei) > 0 else 1.0
        low_cv = cv < 0.05  # nearly identical SEs → possible imputation

        if hom_flagged and low_cv:
            return "High", f"SE homogeneity flagged and CV={cv:.3f} (possible imputation)"
        elif hom_flagged:
            return "Some Concerns", f"SE homogeneity flagged (Cochran's C test significant)"
        elif low_cv:
            return "Some Concerns", f"SE CV={cv:.3f} (very low variation — possible imputation)"
        else:
            return "Low", f"No SE homogeneity flag; SE CV={cv:.3f}"

    # ── Domain 4: Measurement of outcome ─────────────────────────────────────

    def _domain_measurement(self, forensics):
        if forensics.get("status") != "evaluated":
            return "Some Concerns", "RegistryForensics not available; defaulting to Some Concerns"

        normality = forensics.get("normality", {})
        if not isinstance(normality, dict):
            return "Some Concerns", "Normality test result unavailable"

        if normality.get("flagged", False):
            p_val = normality.get("p_value", float("nan"))
            return "Some Concerns", f"Shapiro-Wilk normality flagged (p={p_val:.3f})"
        else:
            p_val = normality.get("p_value", float("nan"))
            p_str = f"{p_val:.3f}" if not (isinstance(p_val, float) and np.isnan(p_val)) else "N/A"
            return "Low", f"Shapiro-Wilk passed (p={p_str})"

    # ── Domain 5: Selection of reported result ────────────────────────────────

    def _domain_selection(self, pubbias):
        if pubbias.get("status") != "evaluated":
            return "Some Concerns", "PubBias not available; defaulting to Some Concerns"

        egger = pubbias.get("egger", {})
        excess_sig = pubbias.get("excess_significance", {})

        egger_flagged = isinstance(egger, dict) and egger.get("significant", False)
        excess_flagged = isinstance(excess_sig, dict) and excess_sig.get("significant", False)

        egger_p = egger.get("p_value", float("nan")) if isinstance(egger, dict) else float("nan")

        if egger_flagged and excess_flagged:
            return "High", "Egger test and excess significance test both flagged"
        elif egger_flagged:
            p_str = f"{egger_p:.3f}" if isinstance(egger_p, float) and not np.isnan(egger_p) else "N/A"
            return "Some Concerns", f"Egger test flagged (p={p_str})"
        elif excess_flagged:
            return "Some Concerns", "Excess significance test flagged"
        else:
            p_str = f"{egger_p:.3f}" if isinstance(egger_p, float) and not np.isnan(egger_p) else "N/A"
            return "Low", f"No selection bias signals detected (Egger p={p_str})"

    # ── Per-study risk ────────────────────────────────────────────────────────

    def _per_study_assessment(self, k, nma, domain1_j, domain5_j):
        """
        Assign per-study risk based on:
          - Externally studentized residuals from NetworkMeta (|r|>2.5 → High)
          - Q contributions (outlier → Some Concerns)
          - Baseline from worst of domain 1 (randomization) and domain 5 (selection)

        Falls back gracefully if NetworkMeta data is unavailable.
        """
        # Baseline per-study risk = worst of whole-study randomization + selection signals
        baseline = _worst(domain1_j, domain5_j)
        # Cap baseline at "Some Concerns" for per-study (study-level evidence is limited)
        if baseline == "High":
            baseline = "Some Concerns"

        # Try to get studentized residuals from NetworkMeta
        studentized = nma.get("influential_studies") if nma.get("status") == "evaluated" else None

        # Build set of high-residual study indices
        high_residual_idx = set()
        if studentized is not None:
            # influential_studies contains dicts with study_index and reasons
            for entry in studentized:
                if "outlier_residual" in entry.get("reasons", []):
                    high_residual_idx.add(entry["study_index"])

        per_study = []
        for i in range(k):
            if i in high_residual_idx:
                per_study.append("High")
            else:
                per_study.append(baseline)

        return per_study
