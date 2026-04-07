# Al-Burhan (The Proof)

**Al-Burhan** is the Universal Evidence Orchestrator. It acts as the nervous system connecting the specialized clinical evidence engines across the `C:` drive. 

While individual tools like *Al-Mizan*, *PredictionGap*, *MetaFrontierLab*, and *GapFrontier* identify specific failure modes in the clinical translation pipeline, **Al-Burhan** unifies them into a single, multi-dimensional audit of medical claims.

## The Mission: Evidence Zero-Waste

Al-Burhan bridges the gap between *statistical truth* (average effects with asymptotic approximations) and *clinical reality* (predictive intervals, transportability to target populations, and trial equipoise).

It performs four key operations on any given medical claim or condition:

1. **The Truth Engine:** Re-evaluates claims using Transport-Bias Exact Meta-Analysis (TBEMA) from `MetaFrontierLab` and checks for the "Hollow Evidence" phenomenon using `PredictionGap`.
2. **The Waste Sentinel:** Monitors trial momentum via `ctgov_moonshot` and cross-references it with `Al-Mizan` tipping points. If a trial is randomizing patients to an answered question, Al-Burhan flags it as a violation of clinical equipoise.
3. **The Decolonization Filter:** Utilizes `AfricaRCT` insights and `transportability_ma` to apply Gaussian-kernel relevance weights, projecting whether a drug proven in the Global North is safe/effective for the Global South.
4. **The E156 Emitter:** Compresses the findings of this multi-dimensional audit into a dense, 156-word micro-paper via the `E156-framework`, ready for submission to ethics committees or guideline panels.

## Architecture

- `alburhan/core/`: The central orchestrator that manages the audit lifecycle.
- `alburhan/engines/`: Adapters that interface with the various C-drive legacy systems (PredictionGap, Al-Mizan, MetaFrontierLab, etc.).
- `data/`: Local storage for unified audit ledgers and output JSONs.

## Roadmap

- [ ] Connect the `PredictionGap` logic to compute PI/CI discordance dynamically.
- [ ] Connect `MetaFrontierLab` TBEMA estimator for robust, sparse-data recalculations.
- [ ] Integrate `Al-Mizan` tipping point logic to flag ongoing trials in `ctgov_moonshot`.
- [ ] Output the final synthesis through the `E156-framework` pipeline.
