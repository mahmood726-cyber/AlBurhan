"""
Automated GRADE Certainty Assessment Engine.

Maps outputs from all upstream engines to the 5 GRADE domains and
produces an automated certainty rating.

Domains:
  1. Risk of Bias     — from RegistryForensics anomaly_flags
  2. Inconsistency    — from PredictionGap I2 + NetworkMeta n_influential
  3. Indirectness     — from AfricaRCT burden_alignment
  4. Imprecision      — from PredictionGap ci_width + FragilityAtlas is_fragile
  5. Publication Bias — from PubBias egger + trim_fill

Starting certainty = 4 (HIGH). Subtract total downgrades.
Score map: 4=HIGH, 3=MODERATE, 2=LOW, <=1=VERY LOW.
"""

import logging

logger = logging.getLogger(__name__)

_CERTAINTY_MAP = {4: "HIGH", 3: "MODERATE", 2: "LOW", 1: "VERY LOW", 0: "VERY LOW"}


class GRADEEngine:
    name = "GRADE"

    def evaluate(self, claim_data):
        results = claim_data.get("audit_results", {})
        logger.info("%s: evaluating %d upstream engines", self.name, len(results))

        rob, rob_reason = self._risk_of_bias(results)
        inc, inc_reason = self._inconsistency(results)
        ind, ind_reason = self._indirectness(results)
        imp, imp_reason = self._imprecision(results)
        pub, pub_reason = self._publication_bias(results)

        total = rob + inc + ind + imp + pub
        raw_score = 4 + total  # total is negative
        score = max(1, min(4, raw_score))
        certainty = _CERTAINTY_MAP[score]

        return {
            "status": "evaluated",
            "certainty": certainty,
            "certainty_score": score,
            "domains": {
                "risk_of_bias": {"downgrade": rob, "reason": rob_reason},
                "inconsistency": {"downgrade": inc, "reason": inc_reason},
                "indirectness": {"downgrade": ind, "reason": ind_reason},
                "imprecision": {"downgrade": imp, "reason": imp_reason},
                "publication_bias": {"downgrade": pub, "reason": pub_reason},
            },
            "total_downgrade": total,
        }

    # ── Domain 1: Risk of Bias ────────────────────────────────────────────────

    def _risk_of_bias(self, results):
        rf = results.get("RegistryForensics", {})
        if rf.get("status") != "evaluated":
            # Missing upstream engine — worst-case downgrade
            return -2, "RegistryForensics unavailable; worst-case downgrade applied"

        flags = rf.get("anomaly_flags", 0)
        if flags == 0:
            return 0, "No anomaly flags"
        elif flags == 1:
            return -1, f"{flags} anomaly flag detected (serious)"
        else:
            return -2, f"{flags} anomaly flags detected (very serious)"

    # ── Domain 2: Inconsistency ───────────────────────────────────────────────

    def _inconsistency(self, results):
        pg = results.get("PredictionGap", {})
        nm = results.get("NetworkMeta", {})

        # Missing both — worst case
        if pg.get("status") != "evaluated" and nm.get("status") != "evaluated":
            return -2, "PredictionGap and NetworkMeta unavailable; worst-case downgrade applied"

        i2 = pg.get("metrics", {}).get("I2", None) if pg.get("status") == "evaluated" else None
        n_influential = nm.get("n_influential", 0) if nm.get("status") == "evaluated" else 0

        # Determine downgrade
        if i2 is None:
            # Only have NetworkMeta
            if n_influential == 0:
                return 0, f"No influential studies detected (I2 unavailable)"
            elif n_influential == 1:
                return -1, f"1 influential study detected (I2 unavailable)"
            else:
                return -2, f"{n_influential} influential studies detected (I2 unavailable)"

        i2_label = f"I2={i2:.0f}%"
        inf_label = f"{n_influential} influential study" if n_influential == 1 else f"{n_influential} influential studies"

        if i2 > 75 or n_influential >= 2:
            return -2, f"{i2_label}, {inf_label} (very serious)"
        elif i2 >= 25 or n_influential == 1:
            return -1, f"{i2_label}, {inf_label} (serious)"
        else:
            return 0, f"{i2_label}, {inf_label} — no concern"

    # ── Domain 3: Indirectness ────────────────────────────────────────────────

    def _indirectness(self, results):
        arct = results.get("AfricaRCT", {})
        if arct.get("status") != "evaluated":
            return -1, "AfricaRCT unavailable; downgrade applied"

        ba = arct.get("burden_alignment", {})
        aligned = ba.get("aligned")  # True, False, or None

        if aligned is True:
            return 0, "Condition aligned with burden"
        elif aligned is False:
            return -1, "Condition not aligned with regional burden"
        else:
            # None = unknown country
            return -1, ba.get("reason", "Burden alignment data unavailable")

    # ── Domain 4: Imprecision ─────────────────────────────────────────────────

    def _imprecision(self, results):
        pg = results.get("PredictionGap", {})
        fa = results.get("FragilityAtlas", {})

        pg_ok = pg.get("status") == "evaluated"
        fa_ok = fa.get("status") == "evaluated"

        if not pg_ok and not fa_ok:
            return -2, "PredictionGap and FragilityAtlas unavailable; worst-case downgrade applied"

        ci_width = pg.get("metrics", {}).get("ci_width", None) if pg_ok else None
        is_fragile = fa.get("is_fragile", True) if fa_ok else True  # assume fragile if missing

        if ci_width is None:
            # Only fragility data
            if is_fragile:
                return -1, "FragilityAtlas: fragile (CI width unavailable)"
            else:
                return 0, "FragilityAtlas: not fragile (CI width unavailable)"

        width_label = f"CI width {ci_width:.2f}"
        fragile_label = "fragile" if is_fragile else "not fragile"

        if ci_width > 1.0 and is_fragile:
            return -2, f"{width_label}, {fragile_label} (very serious)"
        elif ci_width >= 0.5 or is_fragile:
            return -1, f"{width_label}, {fragile_label} (serious)"
        else:
            return 0, f"{width_label}, {fragile_label} — no concern"

    # ── Domain 5: Publication Bias ────────────────────────────────────────────

    def _publication_bias(self, results):
        pb = results.get("PubBias", {})
        if pb.get("status") != "evaluated":
            return -1, "PubBias unavailable; downgrade applied"

        egger = pb.get("egger", {})
        trim_fill = pb.get("trim_fill", {})

        egger_flagged = egger.get("significant", False)
        n_missing = trim_fill.get("n_missing", 0)
        trim_flagged = n_missing > 0

        if egger_flagged and trim_flagged and n_missing >= 2:
            return -2, f"Egger flagged and trim-fill missing {n_missing} studies (very serious)"
        elif egger_flagged or trim_flagged:
            reasons = []
            if egger_flagged:
                reasons.append("Egger flagged")
            if trim_flagged:
                reasons.append(f"trim-fill missing {n_missing} studies")
            return -1, "; ".join(reasons)
        else:
            return 0, "No bias detected"
