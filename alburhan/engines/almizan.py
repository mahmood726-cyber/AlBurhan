import os
import numpy as np
import pandas as pd
import math
import logging
from scipy import stats
from pathlib import Path

logger = logging.getLogger(__name__)

class AlMizanEngine:
    name = "Al-Mizan"

    def __init__(self):
        csv_path = os.environ.get(
            'ALBURHAN_MOONSHOT_CSV',
            str(Path.home() / 'ctgov_moonshot' / 'output' / 'condition_rankings.csv')
        )
        self.moonshot_path = Path(csv_path)
        self.moonshot_df = None
        if self.moonshot_path.exists():
            try:
                df = pd.read_csv(self.moonshot_path, nrows=10000)
                # Validate expected columns (SEC-P1-3)
                if 'condition' in df.columns and 'active_count' in df.columns:
                    self.moonshot_df = df
                else:
                    logger.warning("Moonshot CSV missing required columns (condition, active_count)")
            except Exception as e:
                logger.warning("Failed to load moonshot CSV: %s", e)

    def evaluate(self, claim_data):
        """
        Evaluate tipping point and waste sentinel status.
        Expects 'yi', 'sei', 'years', 'condition'.
        Optionally accepts 'n_per_study' (list of sample sizes per study).
        """
        yi = np.array(claim_data.get('yi', []))
        sei = np.array(claim_data.get('sei', []))
        years = claim_data.get('years', [])
        condition = claim_data.get('condition', 'Unknown')
        n_per_study = claim_data.get('n_per_study', None)

        cum_results = self._compute_cumulative_meta(yi, sei, years, n_per_study)
        if not cum_results:
            return {"status": "error", "message": "Insufficient data for TSA."}

        alpha = 0.05
        power = 0.80

        tsa_result = self._compute_tsa(cum_results, alpha, power, yi, sei)

        # Waste Sentinel: Cross-reference with ctgov_moonshot
        moonshot_active = 0
        if self.moonshot_df is not None:
            # Literal string match to prevent ReDoS (SEC-P1-2)
            match = self.moonshot_df[
                self.moonshot_df['condition'].str.contains(condition, case=False, na=False, regex=False)
            ]
            if not match.empty:
                moonshot_active = int(match.iloc[0]['active_count'])

        waste_momentum = 0
        if tsa_result['tipping_index'] >= 0:
            waste_momentum = moonshot_active

        return {
            "status": "evaluated",
            "verdict": tsa_result['verdict'],
            "tipping_index": int(tsa_result['tipping_index']),
            "tipping_year": tsa_result['tipping_year'],
            "ris": int(tsa_result['ris']),
            "waste_momentum": waste_momentum,
            "moonshot_active_trials": moonshot_active,
            "is_wasteful": waste_momentum > 0
        }

    def _compute_cumulative_meta(self, yi, sei, years, n_per_study):
        if len(yi) < 2:
            return None
        indices = np.argsort(years)
        yi_sorted = yi[indices]
        sei_sorted = sei[indices]
        years_sorted = [years[i] for i in indices]

        # Use real sample sizes if available, otherwise estimate from SE (STAT-P0-4)
        if n_per_study is not None:
            n_sorted = [n_per_study[i] for i in indices]
        else:
            # Estimate N from SE: for log-OR, SE ~ sqrt(4/N), so N ~ 4/SE^2
            n_sorted = [max(20, int(4.0 / (se**2))) for se in sei_sorted]

        results = []
        for k in range(2, len(yi_sorted) + 1):  # Start from k=2 (STAT-P1-1)
            sub_yi = yi_sorted[:k]
            sub_sei = sei_sorted[:k]

            # DL Meta-analysis
            wi = 1.0 / sub_sei**2
            theta_fe = np.sum(wi * sub_yi) / np.sum(wi)
            Q = float(np.sum(wi * (sub_yi - theta_fe)**2))
            C = float(np.sum(wi) - np.sum(wi**2) / np.sum(wi))
            tau2 = max(0, (Q - (k - 1)) / C) if C > 0 else 0

            wi_star = 1.0 / (sub_sei**2 + tau2)
            theta = float(np.sum(wi_star * sub_yi) / np.sum(wi_star))
            se = float(1.0 / np.sqrt(np.sum(wi_star)))
            z = theta / se if se > 0 else 0.0
            p = 2 * (1 - stats.norm.cdf(abs(z)))

            I2 = max(0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0

            cum_n = sum(n_sorted[:k])

            results.append({
                'z': z,
                'p': p,
                'theta': theta,
                'se': se,
                'I2': I2,
                'cumN': cum_n,
                'year': years_sorted[k - 1]
            })
        return results

    def _compute_tsa(self, cum_results, alpha, power, yi, sei):
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Meta-analytic RIS using pooled variance (STAT-P0-4)
        # Use the variance from the most recent meta-analysis step
        last = cum_results[-1]
        pooled_var = last['se']**2 * len(yi)  # Approximate pooled variance
        delta = abs(last['theta']) if last['theta'] != 0 else 0.1

        ris = (z_alpha + z_beta)**2 * pooled_var / (delta**2)
        ris = max(ris, 50)  # Floor at reasonable minimum

        # Adjust for heterogeneity via D2
        I2 = last['I2'] / 100
        # D2 cap at 10 to prevent extreme RIS inflation (STAT-P1-4)
        D2 = min(I2 / (1 - I2), 10.0) if I2 < 1 else 10.0
        ris_adj = ris * (1 + D2)

        tipping_index = -1
        tipping_year = None

        for i, cr in enumerate(cum_results):
            info_frac = cr['cumN'] / ris_adj if ris_adj > 0 else 0
            if info_frac > 0:
                z_bound = z_alpha / math.sqrt(info_frac)
                # Floor boundary at z_alpha (STAT-P0-3)
                z_bound = max(z_bound, z_alpha)
            else:
                z_bound = float('inf')

            crossed = abs(cr['z']) >= z_bound
            if crossed and tipping_index == -1:
                tipping_index = i
                tipping_year = cr['year']

        verdict = 'GREEN'
        if tipping_index >= 0:
            verdict = 'RED'
        elif cum_results[-1]['p'] < 0.05:
            verdict = 'AMBER'

        return {
            'verdict': verdict,
            'tipping_index': tipping_index,
            'tipping_year': tipping_year,
            'ris': ris_adj
        }
