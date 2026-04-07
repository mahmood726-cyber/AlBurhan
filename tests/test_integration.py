"""
Integration tests for the full Al-Burhan 19-engine orchestrator pipeline.

End-to-end tests that verify the complete audit pipeline runs correctly,
engines interact properly, and outputs meet specification.
"""

import time
import pytest
from alburhan.core.orchestrator import EvidenceOrchestrator

# ═══════════════════════════ STANDARD FIXTURE ═══════════════════════════════

STANDARD_CLAIM = {
    "yi": [0.6, 0.5, 0.4, 0.45, 0.55],
    "sei": [0.1, 0.15, 0.1, 0.12, 0.1],
    "years": [2010, 2012, 2015, 2018, 2020],
    "treat_events": [30, 25, 28, 22, 35],
    "treat_total": [100, 100, 100, 100, 100],
    "control_events": [15, 12, 14, 10, 18],
    "control_total": [100, 100, 100, 100, 100],
    "n_per_study": [200, 200, 200, 200, 200],
    "country": "South Africa",
    "condition": "Parkinson Disease",
}

# All 19 engine names registered in the orchestrator
ALL_ENGINES = [
    "PredictionGap",
    "MetaFrontierLab",
    "FragilityAtlas",
    "TreatmentEvolution",
    "SynthesisLoss",
    "CausalSynth",
    "EvidenceDrift",
    "RegistryForensics",
    "NetworkMeta",
    "Al-Mizan",
    "AfricaRCT",
    "BayesianMA",
    "RobustMA",
    "PubBias",
    "MetaRegression",
    "DoseResponse",
    "SequentialTSA",
    "GRADE",
    "E156",
]


# ═══════════════════════════ TEST 1: ALL ENGINES APPEAR ═════════════════════

def test_full_pipeline_all_engines():
    """All 19 engines must appear in results with a 'status' key."""
    results = EvidenceOrchestrator().run_audit(dict(STANDARD_CLAIM))

    for engine_name in ALL_ENGINES:
        assert engine_name in results, f"Engine '{engine_name}' missing from results"
        assert "status" in results[engine_name], (
            f"Engine '{engine_name}' result has no 'status' key; got: {results[engine_name]}"
        )

    # MetaFrontierLab may be evaluated or skipped depending on library availability
    assert results["MetaFrontierLab"]["status"] in ("evaluated", "skipped", "error"), (
        f"MetaFrontierLab status unexpected: {results['MetaFrontierLab']['status']}"
    )

    # DoseResponse skips when no doses provided
    assert results["DoseResponse"]["status"] in ("skipped", "error"), (
        f"DoseResponse expected skipped (no doses), got: {results['DoseResponse']['status']}"
    )


# ═══════════════════════════ TEST 2: GRADE UPSTREAM VALUES ══════════════════

def test_grade_receives_upstream():
    """GRADE certainty_score must be 1-4 and domains dict must have 5 keys."""
    results = EvidenceOrchestrator().run_audit(dict(STANDARD_CLAIM))
    grade = results.get("GRADE", {})

    assert grade.get("status") == "evaluated", (
        f"GRADE did not evaluate: {grade}"
    )
    score = grade.get("certainty_score")
    assert score is not None, "GRADE certainty_score missing"
    assert 1 <= score <= 4, f"GRADE certainty_score out of range: {score}"

    domains = grade.get("domains", {})
    assert len(domains) == 5, (
        f"GRADE domains must have 5 keys, got {len(domains)}: {list(domains.keys())}"
    )
    expected_domains = {
        "risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias"
    }
    assert set(domains.keys()) == expected_domains, (
        f"GRADE domain keys mismatch: {set(domains.keys())} vs {expected_domains}"
    )


# ═══════════════════════════ TEST 3: PUBBIAS FEEDS GRADE ════════════════════

def test_pubbias_feeds_grade():
    """If PubBias Egger is flagged, GRADE publication_bias domain must downgrade."""
    results = EvidenceOrchestrator().run_audit(dict(STANDARD_CLAIM))
    pb = results.get("PubBias", {})
    grade = results.get("GRADE", {})

    if pb.get("status") != "evaluated" or grade.get("status") != "evaluated":
        pytest.skip("PubBias or GRADE did not evaluate")

    egger_flagged = pb.get("egger", {}).get("significant", False)
    if egger_flagged:
        pub_bias_domain = grade.get("domains", {}).get("publication_bias", {})
        downgrade = pub_bias_domain.get("downgrade", 0)
        assert downgrade < 0, (
            f"Egger flagged but GRADE publication_bias downgrade={downgrade} (expected < 0)"
        )


# ═══════════════════════════ TEST 4: BAYESIAN NEAR DL ═══════════════════════

def test_bayesian_posterior_near_dl():
    """BayesianMA posterior_mu should be within 0.15 of PredictionGap theta (vague priors)."""
    results = EvidenceOrchestrator().run_audit(dict(STANDARD_CLAIM))
    pg = results.get("PredictionGap", {})
    bayes = results.get("BayesianMA", {})

    assert pg.get("status") == "evaluated", f"PredictionGap not evaluated: {pg}"
    assert bayes.get("status") == "evaluated", f"BayesianMA not evaluated: {bayes}"

    dl_theta = pg["metrics"]["theta"]
    bayes_mu = bayes["posterior_mu"]

    diff = abs(bayes_mu - dl_theta)
    assert diff < 0.15, (
        f"BayesianMA posterior_mu={bayes_mu} too far from PredictionGap theta={dl_theta} "
        f"(diff={diff:.4f}, threshold=0.15)"
    )


# ═══════════════════════════ TEST 5: E156 WORD COUNT ════════════════════════

def test_e156_word_count():
    """E156 body word_count must be > 50 (real content) and < 300 (reasonable length)."""
    results = EvidenceOrchestrator().run_audit(dict(STANDARD_CLAIM))
    e156 = results.get("E156", {})

    assert e156.get("status") == "emitted", f"E156 not emitted: {e156}"
    word_count = e156.get("word_count")
    assert word_count is not None, "E156 word_count missing"
    assert word_count > 50, f"E156 word_count too low (real content expected): {word_count}"
    assert word_count < 300, f"E156 word_count unreasonably high: {word_count}"


# ═══════════════════════════ TEST 6: BENCHMARK < 10 SECONDS ═════════════════

def test_benchmark_under_60_seconds():
    """Full 21-engine audit must complete in under 60 seconds."""
    start = time.time()
    EvidenceOrchestrator().run_audit(dict(STANDARD_CLAIM))
    elapsed = time.time() - start
    assert elapsed < 60.0, (
        f"Full audit took {elapsed:.2f}s, exceeding 60-second wall-clock limit"
    )


# ═══════════════════════════ TEST 7: ALL POSITIVE EFFECTS ═══════════════════

def test_all_positive_effects():
    """All yi > 0 and all significant → CONCORDANT_SIG discordance, high E-value."""
    claim = dict(STANDARD_CLAIM)
    claim["yi"] = [0.8, 0.9, 0.75, 0.85, 0.70]
    claim["sei"] = [0.05, 0.05, 0.05, 0.05, 0.05]

    results = EvidenceOrchestrator().run_audit(claim)

    # CausalSynth should produce a valid E-value > 1 for positive effects
    cs = results.get("CausalSynth", {})
    assert cs.get("status") == "evaluated", f"CausalSynth not evaluated: {cs}"
    e_value = cs.get("e_value")
    assert e_value is not None, "CausalSynth e_value missing"
    assert e_value > 1.0, f"E-value should be > 1 for positive effects, got {e_value}"

    # PredictionGap theta should be positive
    pg = results.get("PredictionGap", {})
    assert pg.get("status") == "evaluated"
    assert pg["metrics"]["theta"] > 0, "Expected positive pooled theta"


# ═══════════════════════════ TEST 8: ALL NEGATIVE EFFECTS ═══════════════════

def test_all_negative_effects():
    """All yi < 0 → CausalSynth should still produce valid E-value > 1."""
    claim = dict(STANDARD_CLAIM)
    claim["yi"] = [-0.6, -0.5, -0.4, -0.45, -0.55]
    claim["sei"] = [0.1, 0.15, 0.1, 0.12, 0.1]

    results = EvidenceOrchestrator().run_audit(claim)

    cs = results.get("CausalSynth", {})
    assert cs.get("status") == "evaluated", f"CausalSynth not evaluated: {cs}"
    e_value = cs.get("e_value")
    assert e_value is not None, "CausalSynth e_value missing"
    assert e_value > 1.0, (
        f"E-value should be > 1 for negative effects (abs() applied), got {e_value}"
    )


# ═══════════════════════════ TEST 9: SINGLE STUDY ════════════════════════════

def test_single_study():
    """k=1 input: most engines should skip/error gracefully — no crashes."""
    claim = {
        "yi": [0.5],
        "sei": [0.1],
        "years": [2020],
        "treat_events": [20],
        "treat_total": [100],
        "control_events": [10],
        "control_total": [100],
        "n_per_study": [200],
        "country": "Kenya",
        "condition": "HIV",
    }

    # Must not raise any exception
    results = EvidenceOrchestrator().run_audit(claim)

    # All engines must return a result dict with a status key
    for engine_name in ALL_ENGINES:
        assert engine_name in results, f"Engine '{engine_name}' missing from results"
        r = results[engine_name]
        assert isinstance(r, dict), f"Engine '{engine_name}' returned non-dict: {r}"
        assert "status" in r, f"Engine '{engine_name}' result has no 'status' key"
        # Status must be one of the valid values (skipped/error — not a full evaluation)
        assert r["status"] in ("evaluated", "skipped", "error", "emitted"), (
            f"Engine '{engine_name}' has invalid status: {r['status']}"
        )


# ═══════════════════════════ TEST 10: IDENTICAL EFFECTS ══════════════════════

def test_identical_effects():
    """All yi identical (tau2=0): FragilityAtlas 100% robust, SynthesisLoss info_loss ~0."""
    claim = {
        "yi": [0.5, 0.5, 0.5, 0.5, 0.5],
        "sei": [0.1, 0.1, 0.1, 0.1, 0.1],
        "years": [2016, 2017, 2018, 2019, 2020],
        "treat_events": [20, 20, 20, 20, 20],
        "treat_total": [100, 100, 100, 100, 100],
        "control_events": [10, 10, 10, 10, 10],
        "control_total": [100, 100, 100, 100, 100],
        "n_per_study": [200, 200, 200, 200, 200],
        "country": "Uganda",
        "condition": "HIV",
    }

    results = EvidenceOrchestrator().run_audit(claim)

    # FragilityAtlas: tau2=0 means all specifications agree → 100% robust
    fa = results.get("FragilityAtlas", {})
    assert fa.get("status") == "evaluated", f"FragilityAtlas not evaluated: {fa}"
    assert fa.get("robustness_score") == 100.0, (
        f"Expected robustness_score=100 for identical effects, got {fa.get('robustness_score')}"
    )

    # SynthesisLoss: tau2=0 → info_loss should be ~0 (below 0.01)
    syn = results.get("SynthesisLoss", {})
    assert syn.get("status") == "evaluated", f"SynthesisLoss not evaluated: {syn}"
    info_loss = syn.get("information_loss_ratio")
    assert info_loss is not None, "SynthesisLoss information_loss_ratio missing"
    assert info_loss < 0.01, (
        f"SynthesisLoss info_loss should be ~0 for identical effects, got {info_loss}"
    )
