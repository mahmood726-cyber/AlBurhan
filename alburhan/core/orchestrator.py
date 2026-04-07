import logging
from typing import Any, Dict, List

from alburhan.engines.predictiongap import PredictionGapEngine
from alburhan.engines.metafrontier import MetaFrontierEngine
from alburhan.engines.almizan import AlMizanEngine
from alburhan.engines.africarct import AfricaRCTEngine
from alburhan.engines.fragility import FragilityEngine
from alburhan.engines.evolution import EvolutionEngine
from alburhan.engines.synthesis import SynthesisLossEngine
from alburhan.engines.causalsynth import CausalSynthEngine
from alburhan.engines.drift import EvidenceDriftEngine
from alburhan.engines.forensics import RegistryForensicsEngine
from alburhan.engines.nma import NetworkMetaEngine
from alburhan.engines.bayesian import BayesianMAEngine
from alburhan.engines.e156 import E156Emitter

logger = logging.getLogger(__name__)

# Explicit dependency declarations (ENG-P1-2, ENG-P1-3)
# engine_name -> list of engine names that must run before it
ENGINE_DEPS: Dict[str, List[str]] = {
    "AfricaRCT": ["PredictionGap"],
    "CausalSynth": ["MetaFrontierLab", "PredictionGap"],
    "SynthesisLoss": ["Al-Mizan"],
    "E156": [],  # E156 depends on ALL others; handled specially
}


class EvidenceOrchestrator:
    def __init__(self):
        self.engines = [
            PredictionGapEngine(),
            MetaFrontierEngine(),
            FragilityEngine(),
            EvolutionEngine(),
            SynthesisLossEngine(),
            CausalSynthEngine(),
            EvidenceDriftEngine(),
            RegistryForensicsEngine(),
            NetworkMetaEngine(),
            AlMizanEngine(),
            AfricaRCTEngine(),
            BayesianMAEngine(),
            E156Emitter()
        ]

    def run_audit(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        for engine in self.engines:
            # Build engine-specific context (copy to avoid mutation — STAT-P1-5)
            ctx = dict(claim_data)

            # Inject cross-engine data based on declared dependencies
            self._inject_dependencies(engine.name, ctx, results)

            # E156 gets the full results so far
            if engine.name == "E156":
                ctx["audit_results"] = dict(results)

            # Wrap each engine in try/except (ENG-P0-3)
            try:
                results[engine.name] = engine.evaluate(ctx)
            except Exception as e:
                logger.error("Engine %s failed: %s", engine.name, e)
                results[engine.name] = {
                    "status": "error",
                    "message": f"Engine {engine.name} failed. Check logs for details."
                }

        return results

    def _inject_dependencies(self, engine_name: str, ctx: Dict, results: Dict):
        """Inject upstream results into engine context based on declared deps."""
        # PredictionGap → AfricaRCT, CausalSynth (theta)
        if engine_name in ("AfricaRCT", "CausalSynth"):
            pg_metrics = results.get("PredictionGap", {}).get("metrics", {})
            if "theta" in pg_metrics:
                ctx["theta"] = pg_metrics["theta"]
                ctx["se"] = pg_metrics.get("se")

        # MetaFrontierLab → CausalSynth (overrides PredictionGap theta)
        if engine_name == "CausalSynth":
            mf_res = results.get("MetaFrontierLab", {})
            if mf_res.get("status") == "evaluated" and "estimate" in mf_res:
                ctx["theta"] = mf_res["estimate"]
                ctx["se"] = mf_res.get("std_error")

        # Al-Mizan → SynthesisLoss (momentum)
        if engine_name == "SynthesisLoss":
            am = results.get("Al-Mizan", {})
            active = am.get("moonshot_active_trials", 0)
            ctx["momentum"] = active / 100 if active > 0 else 0.5
