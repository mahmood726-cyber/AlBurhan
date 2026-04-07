"""
Evidence Maturity Engine — real temporal analysis of the evidence base.

Computes from actual study data:
  1. Evidence accumulation rate (studies per year)
  2. Precision growth: how fast is the pooled SE shrinking?
  3. Phase detection: when did the cumulative effect stabilize?
  4. Maturity index: information saturation measure
"""

import numpy as np
from scipy import stats as sp_stats


class EvolutionEngine:
    name = "TreatmentEvolution"

    def evaluate(self, claim_data):
        yi = np.array(claim_data.get('yi', []), dtype=float)
        sei = np.array(claim_data.get('sei', []), dtype=float)
        years = claim_data.get('years', [])

        k = len(yi)
        if k < 3 or len(years) < 3:
            return {"status": "skipped", "message": "Need k>=3 with years for maturity analysis."}

        years = np.array(years, dtype=float)
        sort_idx = np.argsort(years)
        yi_sorted = yi[sort_idx]
        sei_sorted = sei[sort_idx]
        years_sorted = years[sort_idx]

        # 1. Evidence accumulation rate
        span = float(years_sorted[-1] - years_sorted[0])
        accumulation_rate = k / span if span > 0 else float(k)

        # 2. Precision growth — cumulative SE at each step
        cum_se = []
        for j in range(2, k + 1):
            sub_sei = sei_sorted[:j]
            vi = sub_sei ** 2
            w = 1.0 / vi
            se_j = 1.0 / np.sqrt(np.sum(w))
            cum_se.append(se_j)

        # Precision doubling: how many studies to halve the SE?
        if len(cum_se) >= 2 and cum_se[0] > 0:
            half_se = cum_se[0] / 2
            precision_doubling_k = None
            for j, se_val in enumerate(cum_se):
                if se_val <= half_se:
                    precision_doubling_k = j + 2  # +2 because cum_se starts at k=2
                    break
        else:
            precision_doubling_k = None

        # 3. Stability detection — when cumulative theta stops changing by >5%
        cum_theta = []
        for j in range(2, k + 1):
            sub_yi = yi_sorted[:j]
            sub_sei = sei_sorted[:j]
            wi = 1.0 / sub_sei ** 2
            theta_j = float(np.sum(wi * sub_yi) / np.sum(wi))
            cum_theta.append(theta_j)

        stabilization_index = None
        stabilization_year = None
        if len(cum_theta) >= 3:
            for j in range(2, len(cum_theta)):
                prev = cum_theta[j - 1]
                curr = cum_theta[j]
                if prev != 0:
                    pct_change = abs((curr - prev) / prev)
                    if pct_change < 0.05:
                        # Check if it stays stable for remaining studies
                        all_stable = True
                        for m in range(j, len(cum_theta)):
                            if cum_theta[m - 1] != 0:
                                if abs((cum_theta[m] - cum_theta[m - 1]) / cum_theta[m - 1]) >= 0.05:
                                    all_stable = False
                                    break
                        if all_stable:
                            stabilization_index = j + 2  # k value where it stabilized
                            stabilization_year = int(years_sorted[j + 1]) if j + 1 < len(years_sorted) else None
                            break

        # 4. Maturity index: information saturation
        # Ratio of current precision to "ideal" precision if all studies had min SE
        min_se = float(np.min(sei))
        ideal_se = min_se / np.sqrt(k)
        current_se = cum_se[-1] if cum_se else float(np.mean(sei))
        maturity_index = min(1.0, ideal_se / current_se) if current_se > 0 else 0.0

        # 5. Evidence age
        evidence_age = int(span) if span > 0 else 0

        # Phase classification from data
        if stabilization_index is not None and maturity_index > 0.6:
            phase = "Mature"
        elif stabilization_index is not None:
            phase = "Stabilizing"
        elif k >= 5:
            phase = "Accumulating"
        else:
            phase = "Early"

        return {
            "status": "evaluated",
            "evidence_age_years": evidence_age,
            "accumulation_rate": round(accumulation_rate, 2),
            "precision_doubling_k": precision_doubling_k,
            "stabilization_year": stabilization_year,
            "stabilization_k": stabilization_index,
            "maturity_index": round(float(maturity_index), 3),
            "phase": phase,
            "cumulative_theta_range": [
                round(float(min(cum_theta)), 4),
                round(float(max(cum_theta)), 4),
            ] if cum_theta else None,
        }
