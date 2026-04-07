import numpy as np

class EvidenceDriftEngine:
    name = "EvidenceDrift"

    def evaluate(self, claim_data):
        """
        Detect temporal drift in evidence.
        Checks if newer studies contradict older ones.
        """
        yi = np.array(claim_data.get('yi', []))
        years = np.array(claim_data.get('years', []))

        if len(yi) < 4:
            return {"status": "skipped", "message": "Need >=4 studies for drift analysis."}

        # Correlation between year and effect size
        # Handle constant arrays that produce NaN (STAT-P1-7)
        if np.std(yi) == 0 or np.std(years) == 0:
            correlation = 0.0
            stability = "Constant"
        else:
            correlation = float(np.corrcoef(years, yi)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
                stability = "Indeterminate"
            else:
                stability = "Drifting" if abs(correlation) > 0.5 else "Stable"

        # Cumulative drift (absolute change in rolling mean)
        convolved = np.convolve(yi, np.ones(3)/3, mode='valid')
        if len(convolved) >= 2:
            drift_velocity = float(np.abs(np.diff(convolved)).mean())
        else:
            drift_velocity = 0.0

        return {
            "status": "evaluated",
            "drift_correlation": round(correlation, 3),
            "drift_velocity": round(drift_velocity, 3),
            "stability_status": stability
        }
