import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Path configurable via environment variable (ENG-P1-4)
METAFRONTIER_PATH = os.environ.get('ALBURHAN_METAFRONTIER_PATH', r"C:\MetaFrontierLab")


class MetaFrontierEngine:
    name = "MetaFrontierLab"

    def __init__(self):
        self.analyzer = None
        self._import_error = None

    def _lazy_init(self):
        """Deferred import — only runs when evaluate() is called (ENG-P0-1)."""
        if self.analyzer is not None or self._import_error is not None:
            return

        import sys
        if METAFRONTIER_PATH not in sys.path:
            sys.path.append(METAFRONTIER_PATH)
        try:
            from metafrontier.core import FrontierMetaAnalyzer
            self.analyzer = FrontierMetaAnalyzer(
                quadrature_points=11,
                ridge_grid=(0.1, 0.5, 1.0),
                selection_temperature=0.01,
                model_dispersion_scale=1.2
            )
        except ImportError as e:
            self._import_error = str(e)
            logger.warning("MetaFrontierLab not available: %s", e)

    def evaluate(self, claim_data):
        """
        Advanced evaluation using Exact Likelihoods and Model Ensembling.
        Expects binary data (treat_events/total, control_events/total)
        or continuous data (yi, sei).
        """
        self._lazy_init()

        if self._import_error is not None:
            return {
                "status": "skipped",
                "message": "MetaFrontierLab library not available. Install or set ALBURHAN_METAFRONTIER_PATH."
            }

        df = self._prepare_dataframe(claim_data)
        if df is None:
            return {"status": "skipped", "message": "Insufficient data for advanced TBEMA."}

        try:
            result = self.analyzer.fit(
                data=df,
                effect_col="yi" if "yi" in df.columns else None,
                se_col="sei" if "sei" in df.columns else None,
                treat_events_col="treat_events",
                treat_total_col="treat_total",
                control_events_col="control_events",
                control_total_col="control_total"
            )

            return {
                "status": "evaluated",
                "method": "TBEMA (Exact Sparse + Ensemble)",
                "estimate": round(result.estimate, 4),
                "std_error": round(result.std_error, 4),
                "ci": [round(result.ci_low, 4), round(result.ci_high, 4)],
                "tau": round(result.tau, 4),
                "ensemble_weights": result.submodel_weights,
                "n_studies": len(df),
                "likelihood": result.likelihood
            }
        except Exception as e:
            logger.error("MetaFrontierLab analysis failed: %s", e)
            return {"status": "error", "message": "MetaFrontier analysis failed. Check logs for details."}

    def _prepare_dataframe(self, claim_data):
        if "treat_events" in claim_data:
            return pd.DataFrame({
                "treat_events": claim_data["treat_events"],
                "treat_total": claim_data["treat_total"],
                "control_events": claim_data["control_events"],
                "control_total": claim_data["control_total"]
            })
        elif "yi" in claim_data:
            return pd.DataFrame({
                "yi": claim_data["yi"],
                "sei": claim_data["sei"]
            })
        return None
