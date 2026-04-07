"""
Publication Bias Detection Suite.

Six methods for detecting publication bias from yi/sei pairs:

1. Egger's regression (Egger et al., 1997)
   WLS regression of yi on 1/sei; test intercept ≠ 0 (t, k-2 df, alpha=0.10)

2. Begg-Mazumdar rank test (Begg & Mazumdar, 1994)
   Kendall's tau between effect sizes and variances (sei^2), alpha=0.10

3. Trim-and-Fill (Duval & Tweedie, 2000)
   L0 estimator: iterative imputation of missing studies on the asymmetric side;
   reports n_missing and adjusted pooled estimate.

4. Fail-safe N (Rosenthal, 1979)
   Nfs = (sum(zi) / 1.645)^2 - k  where zi = yi / sei
   Robust if Nfs > 5k + 10.

5. P-curve (Simonsohn et al., 2014)
   Among p < 0.05 studies, transform pp = p / 0.05, binomial test on prop(pp < 0.5) > 0.5
   (right skew = evidence against p-hacking; left skew = concern).

6. Excess Significance Test (Ioannidis & Trikalinos, 2007)
   Per-study power = P(|Z| > z_crit | mu = theta_pooled);
   chi2 = (O - E)^2 / E with df=1.
"""

import numpy as np
from scipy import stats


class PubBiasEngine:
    name = "PubBias"

    # Significance threshold for flagging
    _ALPHA = 0.10

    def evaluate(self, claim_data):
        yi = np.array(claim_data.get("yi", []), dtype=float)
        sei = np.array(claim_data.get("sei", []), dtype=float)
        k = len(yi)

        if k < 3:
            return {
                "status": "skipped",
                "message": "Need k>=3 for publication bias tests.",
            }

        egger = self._egger(yi, sei)
        begg = self._begg(yi, sei)
        trimfill = self._trim_fill(yi, sei)
        failsafe = self._failsafe_n(yi, sei)
        pcurve = self._pcurve(yi, sei)
        excess_sig = self._excess_significance(yi, sei)

        # Flag if any method's p-value is below alpha
        flags = []
        if egger.get("p_value", 1.0) < self._ALPHA:
            flags.append("Egger")
        if begg.get("p_value", 1.0) < self._ALPHA:
            flags.append("Begg-Mazumdar")
        if pcurve.get("right_skew_p", 1.0) < self._ALPHA:
            flags.append("P-curve (right-skew concern)")
        if excess_sig.get("p_value", 1.0) < self._ALPHA:
            flags.append("ExcessSignificance")
        if trimfill.get("n_missing", 0) > 0:
            flags.append("TrimFill (missing studies)")
        if not failsafe.get("robust", True):
            flags.append("FailsafeN (not robust)")

        return {
            "status": "evaluated",
            "n_studies": k,
            "egger": egger,
            "begg": begg,
            "trim_fill": trimfill,
            "failsafe_n": failsafe,
            "p_curve": pcurve,
            "excess_significance": excess_sig,
            "bias_flags": flags,
            "n_flags": len(flags),
        }

    # ─── Method 1: Egger's Regression ────────────────────────────────────────

    def _egger(self, yi, sei):
        """
        WLS regression: yi = a + b*(1/sei)  with weights = 1/sei^2.
        Test H0: a = 0 (intercept) using t with k-2 df.
        """
        k = len(yi)
        if k < 3:
            return {"intercept": float("nan"), "p_value": float("nan")}

        # Design matrix: [intercept, precision]
        precision = 1.0 / sei
        X = np.column_stack([np.ones(k), precision])
        W = np.diag(1.0 / sei ** 2)

        # WLS: beta = (X'WX)^{-1} X'Wy
        XtW = X.T @ W
        XtWX = XtW @ X
        XtWy = XtW @ yi
        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            return {"intercept": float("nan"), "p_value": float("nan")}

        # Residuals & variance of intercept
        fitted = X @ beta
        residuals = yi - fitted
        # Weighted RSS
        wrss = float(residuals @ W @ residuals)
        sigma2 = wrss / (k - 2)
        cov_beta = sigma2 * np.linalg.inv(XtWX)
        se_intercept = float(np.sqrt(cov_beta[0, 0]))

        intercept = float(beta[0])
        t_stat = intercept / se_intercept if se_intercept > 0 else float("nan")
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=k - 2)) if not np.isnan(t_stat) else float("nan")

        return {
            "intercept": round(intercept, 4),
            "se_intercept": round(se_intercept, 4),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "df": k - 2,
            "significant": p_value < self._ALPHA,
        }

    # ─── Method 2: Begg-Mazumdar Rank Test ───────────────────────────────────

    def _begg(self, yi, sei):
        """
        Kendall's tau between yi and sei^2; test H0: tau = 0, alpha = 0.10.
        """
        tau, p_value = stats.kendalltau(yi, sei ** 2)
        return {
            "kendall_tau": round(float(tau), 4),
            "p_value": round(float(p_value), 4),
            "significant": float(p_value) < self._ALPHA,
        }

    # ─── Method 3: Trim-and-Fill (L0 estimator) ──────────────────────────────

    def _trim_fill(self, yi, sei):
        """
        Duval & Tweedie (2000) L0 estimator.

        Algorithm:
          1. Estimate pooled theta_0 via FE.
          2. Centre studies: d_i = yi - theta_0.
          3. Rank |d_i| from largest (rank 1) to smallest.
          4. L0 = max(0, round((4*T_n / k) - 1))
             where T_n = sum of ranks assigned to the negative (left) deviations.
          5. Trim the L0 most extreme studies on the side with more outliers
             (the "positive" side for right-asymmetric funnel bias), recompute
             theta_0, repeat until convergence.
          6. Impute L0 mirror studies reflected around theta_final, recompute
             pooled estimate.

        The convention: trim from the side with the LARGEST extreme deviations
        (usually the right/positive side when small studies show larger effects).
        """
        yi = np.array(yi, dtype=float)
        sei = np.array(sei, dtype=float)
        k = len(yi)

        def fe_pool(y, s):
            w = 1.0 / s ** 2
            return float(np.sum(w * y) / np.sum(w))

        original_theta = fe_pool(yi, sei)
        theta0 = original_theta
        L0 = 0
        max_iter = 50

        for _ in range(max_iter):
            d = yi - theta0
            abs_d = np.abs(d)
            # Rank: rank 1 = largest |d|, rank k = smallest |d|
            order = np.argsort(abs_d)[::-1]   # indices sorted by descending |d|
            ranks_arr = np.empty(k, dtype=int)
            for rank_pos, orig_idx in enumerate(order):
                ranks_arr[orig_idx] = rank_pos + 1  # rank 1-based

            # T_n = sum of ranks for negative deviations
            neg_mask = d < 0
            T_n = float(np.sum(ranks_arr[neg_mask]))

            L0_new = max(0, int(round((4.0 * T_n / k) - 1.0)))
            # Cap at k-2 so we always keep at least 2 studies
            L0_new = min(L0_new, k - 2)

            if L0_new == L0:
                break
            L0 = L0_new

            if L0 == 0:
                break

            # Trim L0 most extreme studies (largest |d|) from the positive side
            # (right asymmetry: studies far right of theta are excess)
            pos_sorted = np.argsort(yi)[::-1]   # descending yi; rightmost first
            trim_idx = set(pos_sorted[:L0])
            keep_mask = np.array([i not in trim_idx for i in range(k)])

            if keep_mask.sum() < 2:
                break

            theta0 = fe_pool(yi[keep_mask], sei[keep_mask])

        n_missing = L0

        if n_missing == 0:
            return {
                "n_missing": 0,
                "adjusted_theta": round(original_theta, 4),
                "original_theta": round(original_theta, 4),
            }

        # Final trimmed theta for mirror imputation
        pos_sorted = np.argsort(yi)[::-1]
        trim_idx = set(pos_sorted[:n_missing])
        keep_mask = np.array([i not in trim_idx for i in range(k)])
        theta_trimmed = fe_pool(yi[keep_mask], sei[keep_mask]) if keep_mask.sum() >= 2 else theta0

        # Impute mirror studies reflected around theta_trimmed
        # The missing studies are the "excess" rightmost ones
        excess_idx = pos_sorted[:n_missing]
        mirror_yi = 2.0 * theta_trimmed - yi[excess_idx]
        mirror_sei = sei[excess_idx]

        yi_aug = np.concatenate([yi, mirror_yi])
        sei_aug = np.concatenate([sei, mirror_sei])
        theta_adj = fe_pool(yi_aug, sei_aug)

        return {
            "n_missing": n_missing,
            "adjusted_theta": round(float(theta_adj), 4),
            "original_theta": round(float(original_theta), 4),
        }

    # ─── Method 4: Fail-safe N (Rosenthal 1979) ──────────────────────────────

    def _failsafe_n(self, yi, sei):
        """
        Nfs = (sum(zi) / 1.645)^2 - k  where zi = yi / sei.
        Robust if Nfs > 5k + 10.
        """
        k = len(yi)
        zi = yi / sei
        sum_z = float(np.sum(zi))
        nfs = max(0.0, (sum_z / 1.645) ** 2 - k)
        robust_threshold = 5 * k + 10
        return {
            "failsafe_n": round(nfs, 1),
            "robust_threshold": robust_threshold,
            "robust": nfs > robust_threshold,
        }

    # ─── Method 5: P-curve ───────────────────────────────────────────────────

    def _pcurve(self, yi, sei):
        """
        Among studies with p < 0.05 (two-sided), compute pp = p / 0.05.
        Test H0: prop(pp < 0.5) = 0.5 using exact binomial.
        right_skew_p: p-value for testing that pp distribution is right-skewed
        (i.e., pp < 0.5 is MORE common than expected — genuine effect).
        A small p_value means the p-curve IS right-skewed (evidence of real effect).
        """
        zi = np.abs(yi / sei)
        p_vals = 2.0 * stats.norm.sf(zi)

        sig_mask = p_vals < 0.05
        sig_p = p_vals[sig_mask]
        n_sig = len(sig_p)

        if n_sig == 0:
            return {
                "n_significant": 0,
                "right_skew_p": float("nan"),
                "skew_direction": "undefined (no significant studies)",
            }

        pp = sig_p / 0.05
        n_below_half = int(np.sum(pp < 0.5))

        # Binomial test: H0 prop=0.5 vs H1 prop>0.5 (right skew)
        result = stats.binomtest(n_below_half, n_sig, p=0.5, alternative="greater")
        right_skew_p = float(result.pvalue)

        skew_direction = "right" if right_skew_p < self._ALPHA else "flat/left"

        return {
            "n_significant": n_sig,
            "n_below_half": n_below_half,
            "right_skew_p": round(right_skew_p, 4),
            "skew_direction": skew_direction,
        }

    # ─── Method 6: Excess Significance Test ──────────────────────────────────

    def _excess_significance(self, yi, sei):
        """
        Ioannidis & Trikalinos (2007).
        Pooled theta via FE; per-study power = P(|Z| > z_crit | mu = theta_pooled).
        chi2 = (O - E)^2 / E, df = 1.
        """
        k = len(yi)
        wi = 1.0 / sei ** 2
        theta_pooled = float(np.sum(wi * yi) / np.sum(wi))

        z_crit = stats.norm.ppf(0.975)  # 1.96 for alpha=0.05 two-sided

        # Per-study power
        ncp = np.abs(theta_pooled) / sei
        power = stats.norm.sf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
        power = np.clip(power, 1e-6, 1.0 - 1e-6)

        # Observed significant studies
        zi = np.abs(yi / sei)
        observed_sig = int(np.sum(2.0 * stats.norm.sf(zi) < 0.05))
        expected_sig = float(np.sum(power))

        chi2 = (observed_sig - expected_sig) ** 2 / expected_sig if expected_sig > 0 else float("nan")
        p_value = float(stats.chi2.sf(chi2, df=1)) if not np.isnan(chi2) else float("nan")

        return {
            "observed_sig": observed_sig,
            "expected_sig": round(expected_sig, 2),
            "chi2": round(float(chi2), 4),
            "p_value": round(p_value, 4),
            "significant": p_value < self._ALPHA if not np.isnan(p_value) else False,
        }
