# Al-Burhan (The Proof)

[![codeql](https://github.com/mahmood726-cyber/AlBurhan/actions/workflows/codeql.yml/badge.svg?branch=master)](https://github.com/mahmood726-cyber/AlBurhan/actions/workflows/codeql.yml) [![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

**Al-Burhan** is an evidence orchestrator. It runs a set of self-contained clinical-evidence engines over a medical claim and combines their outputs into one audit.

Engines named after standalone tools (*Al-Mizan*, *PredictionGap*, *MetaFrontierLab*) are reimplemented here as in-package Python modules under `alburhan/engines/`; they each target a specific failure mode in the clinical translation pipeline, and Al-Burhan runs them together as a single multi-engine audit of a medical claim.

## The Mission: Evidence Zero-Waste

Al-Burhan bridges the gap between *statistical truth* (average effects with asymptotic approximations) and *clinical reality* (predictive intervals, transportability to target populations, and trial equipoise).

It performs four key operations on any given medical claim or condition:

1. **The Truth Engine:** Re-evaluates claims using Transport-Bias Exact Meta-Analysis (TBEMA) from `MetaFrontierLab` and checks for the "Hollow Evidence" phenomenon using `PredictionGap`.
2. **The Waste Sentinel:** Monitors trial momentum via `ctgov_moonshot` and cross-references it with `Al-Mizan` tipping points. If a trial is randomizing patients to an answered question, Al-Burhan flags it as a violation of clinical equipoise.
3. **The Decolonization Filter:** Utilizes `AfricaRCT` insights and `transportability_ma` to apply Gaussian-kernel relevance weights, projecting whether a drug proven in the Global North is safe/effective for the Global South.
4. **The E156 Emitter:** Compresses the findings of this multi-dimensional audit into a dense, 156-word micro-paper via the `E156-framework`, ready for submission to ethics committees or guideline panels.

## Architecture

- `alburhan/core/`: The central orchestrator that manages the audit lifecycle and engine dependency ordering.
- `alburhan/engines/`: Self-contained Python engines (PredictionGap, Al-Mizan, MetaFrontier, Fragility, PubBias, GRADE, RoB2, NMA, Bayesian MA, dose-response, meta-regression, sequential TSA, PRISMA, E156 emitter, and more).
- `alburhan/ingest/`: AACT and CT.gov CSV ingestion plus effect parsing.

## Status

The orchestrator and its engines run as in-package Python modules with a test suite under `tests/`. Engines are reimplementations targeting each named failure mode, not live bridges to external tools.

## Roadmap

- [ ] Wire optional live data backends (`ctgov_moonshot`) behind the existing ingest layer.
- [ ] Expand E156 emitter output formatting for direct submission.
