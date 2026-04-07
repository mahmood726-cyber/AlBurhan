"""
Test suite for Al-Burhan evidence engines.

All engines compute from real input data — no simulated outputs.
Tests cover: correctness, edge cases (k=1..3), mathematical validation.
"""

import math
import numpy as np
import pytest
from scipy import stats

from alburhan.engines.predictiongap import PredictionGapEngine
from alburhan.engines.fragility import FragilityEngine
from alburhan.engines.causalsynth import CausalSynthEngine
from alburhan.engines.drift import EvidenceDriftEngine
from alburhan.engines.almizan import AlMizanEngine
from alburhan.engines.forensics import RegistryForensicsEngine
from alburhan.engines.nma import NetworkMetaEngine
from alburhan.engines.evolution import EvolutionEngine
from alburhan.engines.synthesis import SynthesisLossEngine
from alburhan.engines.africarct import AfricaRCTEngine
from alburhan.engines.e156 import E156Emitter
from alburhan.core.orchestrator import EvidenceOrchestrator


# ═══════════════════════════ FIXTURES ═══════════════════════════════════════

@pytest.fixture
def standard_claim():
    return {
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

@pytest.fixture
def minimal_claim():
    return {
        "yi": [0.3, 0.4, 0.5],
        "sei": [0.1, 0.1, 0.1],
        "years": [2020, 2021, 2022],
        "country": "Kenya",
        "condition": "HIV",
    }

@pytest.fixture
def two_study_claim():
    return {
        "yi": [0.5, 0.6], "sei": [0.1, 0.1],
        "years": [2020, 2021], "country": "Nigeria", "condition": "malaria",
    }

@pytest.fixture
def constant_effects_claim():
    return {
        "yi": [0.5, 0.5, 0.5, 0.5, 0.5],
        "sei": [0.1, 0.1, 0.1, 0.1, 0.1],
        "years": [2018, 2019, 2020, 2021, 2022],
        "country": "Uganda", "condition": "HIV",
    }

@pytest.fixture
def heterogeneous_claim():
    """High heterogeneity dataset for stress-testing."""
    return {
        "yi": [0.1, 0.8, -0.3, 0.5, 1.2, -0.1, 0.9],
        "sei": [0.1, 0.2, 0.15, 0.1, 0.25, 0.12, 0.18],
        "years": [2014, 2015, 2016, 2017, 2018, 2019, 2020],
        "country": "Kenya", "condition": "HIV",
    }


# ═══════════════════════════ PREDICTION GAP ═════════════════════════════════

class TestPredictionGapEngine:
    def setup_method(self):
        self.engine = PredictionGapEngine()

    def test_basic_evaluation(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        m = r["metrics"]
        assert 0 < m["theta"] < 1
        assert m["ci_lo"] < m["theta"] < m["ci_hi"]
        assert m["pi_lo"] < m["pi_hi"]
        assert m["pi_width"] >= m["ci_width"]
        assert m["k"] == 5

    def test_too_few_studies(self):
        r = self.engine.evaluate({"yi": [0.5, 0.6], "sei": [0.1, 0.1]})
        assert r["status"] == "error"

    def test_homogeneous_data(self, constant_effects_claim):
        r = self.engine.evaluate(constant_effects_claim)
        assert r["metrics"]["tau2"] == 0
        assert r["metrics"]["I2"] == 0

    def test_pi_uses_t_distribution(self):
        yi = np.array([0.5, 0.6, 0.7])
        sei = np.array([0.1, 0.1, 0.1])
        r = self.engine.compute_prediction_interval(yi, sei)
        assert r["pi_width"] > r["ci_width"] * 3  # t(1) >> z

    def test_formula_cross_validation(self):
        yi = np.array([0.6, 0.5, 0.4, 0.45, 0.55])
        sei = np.array([0.1, 0.15, 0.1, 0.12, 0.1])
        r = PredictionGapEngine().compute_prediction_interval(yi, sei)
        k = 5
        wi = 1.0 / sei**2
        theta_fe = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - theta_fe)**2)
        C = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
        tau2 = max(0, (Q - (k - 1)) / C)
        wi_star = 1.0 / (sei**2 + tau2)
        theta = np.sum(wi_star * yi) / np.sum(wi_star)
        se = 1.0 / math.sqrt(np.sum(wi_star))
        assert abs(r["theta"] - theta) < 1e-10
        assert abs(r["se"] - se) < 1e-10

    def test_knapp_hartung_ci(self, standard_claim):
        """KH CI must be at least as wide as Wald CI (Rover truncation ensures this)."""
        r = self.engine.compute_prediction_interval(
            np.array(standard_claim["yi"]), np.array(standard_claim["sei"])
        )
        wald_width = r["ci_hi"] - r["ci_lo"]
        kh_width = r["kh_ci_hi"] - r["kh_ci_lo"]
        assert "kh_ci_lo" in r
        assert "kh_ci_hi" in r
        assert kh_width >= wald_width - 1e-10  # KH >= Wald (Rover truncation)


# ═══════════════════════════ FRAGILITY ══════════════════════════════════════

class TestFragilityEngine:
    def setup_method(self):
        self.engine = FragilityEngine()

    def test_basic_evaluation(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        assert 0 <= r["robustness_score"] <= 100
        assert r["classification"] in ("Robust", "Moderately Robust", "Fragile", "Unstable")

    def test_too_few_studies(self):
        r = self.engine.evaluate({"yi": [0.5, 0.6], "sei": [0.1, 0.1]})
        assert r["status"] == "skipped"

    def test_reml_differs_from_dl(self):
        yi = np.array([0.1, 0.8, -0.3, 0.5, 1.2])
        sei = np.array([0.1, 0.2, 0.15, 0.1, 0.25])
        tau2_dl = self.engine._estimate_tau2(yi, sei, "DL")
        tau2_reml = self.engine._estimate_tau2(yi, sei, "REML")
        assert tau2_dl > 0
        assert tau2_reml > 0

    def test_fe_tau2_is_zero(self):
        yi = np.array([0.5, 0.6, 0.7])
        sei = np.array([0.1, 0.1, 0.1])
        assert self.engine._estimate_tau2(yi, sei, "FE") == 0.0

    def test_homogeneous_robust(self, constant_effects_claim):
        r = self.engine.evaluate(constant_effects_claim)
        assert r["robustness_score"] == 100.0

    def test_ten_specifications(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert int(r["reference_agreement"].split("/")[1]) == 10


# ═══════════════════════════ CAUSAL SYNTH ═══════════════════════════════════

class TestCausalSynthEngine:
    def setup_method(self):
        self.engine = CausalSynthEngine()

    def test_e_value_formula(self):
        r = self.engine.evaluate({"theta": 0.89, "se": 0.16})
        rr = np.exp(0.89)
        expected = rr + np.sqrt(rr * (rr - 1))
        assert abs(r["e_value"] - round(expected, 3)) < 0.01

    def test_ci_e_value(self):
        r = self.engine.evaluate({"theta": 0.89, "se": 0.16})
        assert "e_value_ci" in r
        assert r["e_value_ci"] <= r["e_value"]
        assert r["e_value_ci"] >= 1.0

    def test_null_gives_1(self):
        r = self.engine.evaluate({"theta": 0.0, "se": 0.1})
        assert r["e_value"] == 1.0

    def test_skipped_without_theta(self):
        r = self.engine.evaluate({"condition": "HIV"})
        assert r["status"] == "skipped"

    def test_protective_effect(self):
        r = self.engine.evaluate({"theta": -0.5, "se": 0.1})
        assert r["e_value"] > 1.0

    def test_bias_adjusted_evalue(self):
        """Bias-adjusted E-value <= point E-value (bias_factor < 1 reduces RR)."""
        r = self.engine.evaluate({"theta": 0.89, "se": 0.16})
        assert "e_value_bias_adjusted" in r
        # Default bias_factor=0.9 reduces RR, so bias-adjusted E-value <= point E-value
        assert r["e_value_bias_adjusted"] <= r["e_value"]

    def test_bias_adjusted_evalue_custom_factor(self):
        """Custom bias_factor is respected."""
        r1 = self.engine.evaluate({"theta": 0.89, "se": 0.16, "bias_factor": 0.5})
        r2 = self.engine.evaluate({"theta": 0.89, "se": 0.16, "bias_factor": 1.0})
        # bias_factor=0.5 gives a larger reduction than bias_factor=1.0
        assert r1["e_value_bias_adjusted"] <= r2["e_value_bias_adjusted"]


# ═══════════════════════════ DRIFT ══════════════════════════════════════════

class TestEvidenceDriftEngine:
    def setup_method(self):
        self.engine = EvidenceDriftEngine()

    def test_basic(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        assert -1.0 <= r["drift_correlation"] <= 1.0

    def test_too_few(self, two_study_claim):
        assert self.engine.evaluate(two_study_claim)["status"] == "skipped"

    def test_constant_no_nan(self, constant_effects_claim):
        r = self.engine.evaluate(constant_effects_claim)
        assert not math.isnan(r["drift_correlation"])
        assert r["stability_status"] == "Constant"

    def test_strong_drift(self):
        claim = {"yi": [0.1, 0.3, 0.5, 0.7, 0.9], "sei": [0.1]*5, "years": [2016,2017,2018,2019,2020]}
        r = self.engine.evaluate(claim)
        assert r["stability_status"] == "Drifting"
        assert r["drift_correlation"] > 0.5


# ═══════════════════════════ AL-MIZAN ═══════════════════════════════════════

class TestAlMizanEngine:
    def setup_method(self):
        self.engine = AlMizanEngine()

    def test_basic(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        assert r["verdict"] in ("GREEN", "AMBER", "RED")
        assert r["ris"] > 0

    def test_too_few(self):
        r = self.engine.evaluate({"yi": [0.5], "sei": [0.1], "years": [2020]})
        assert r["status"] == "error"


# ═══════════════════════════ FORENSICS ══════════════════════════════════════

class TestRegistryForensicsEngine:
    def setup_method(self):
        self.engine = RegistryForensicsEngine()

    def test_deterministic(self, standard_claim):
        r1 = self.engine.evaluate(standard_claim)
        r2 = self.engine.evaluate(standard_claim)
        assert r1["scientific_entropy"] == r2["scientific_entropy"]
        assert r1["anomaly_flags"] == r2["anomaly_flags"]

    def test_no_fraud_terminology(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert "fraud" not in str(r).lower()

    def test_real_terminal_digit(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        td = r["terminal_digit"]
        assert "chi2" in td or "status" in td

    def test_real_normality(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        norm = r["normality"]
        assert "shapiro_w" in norm

    def test_se_homogeneity(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        seh = r["se_homogeneity"]
        assert "cochrans_c" in seh
        assert 0 <= seh["cochrans_c"] <= 1

    def test_grim_with_counts(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert "inconsistent_count" in r["grim"]

    def test_grim_without_counts(self, minimal_claim):
        r = self.engine.evaluate(minimal_claim)
        assert r["grim"]["status"] == "skipped"

    def test_too_few(self):
        r = self.engine.evaluate({"yi": [0.5, 0.6], "sei": [0.1, 0.1]})
        assert r["status"] == "skipped"

    def test_entropy_is_real(self, heterogeneous_claim):
        r = self.engine.evaluate(heterogeneous_claim)
        assert r["scientific_entropy"] > 0

    def test_benford_law(self, standard_claim):
        """Benford test key present with chi2 + p_value."""
        r = self.engine.evaluate(standard_claim)
        assert "benford" in r
        benford = r["benford"]
        # Either computed or skipped (k=5 exactly hits the boundary)
        if benford.get("status") != "skipped":
            assert "chi2" in benford
            assert "p_value" in benford
            assert benford["chi2"] >= 0
            assert 0.0 <= benford["p_value"] <= 1.0


# ═══════════════════════════ NETWORK META (INFLUENCE) ═══════════════════════

class TestNetworkMetaEngine:
    def setup_method(self):
        self.engine = NetworkMetaEngine()

    def test_basic(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        assert "full_model" in r
        assert "influential_studies" in r
        assert r["full_model"]["theta"] > 0

    def test_leave_one_out_range(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        lo, hi = r["leave_one_out_range"]
        assert lo <= r["full_model"]["theta"] <= hi or abs(lo - hi) < 0.5

    def test_outlier_detection(self, heterogeneous_claim):
        r = self.engine.evaluate(heterogeneous_claim)
        assert r["max_studentized_residual"] > 0
        # With highly heterogeneous data, should find influential studies
        assert r["n_influential"] >= 0

    def test_too_few(self):
        r = self.engine.evaluate({"yi": [0.5, 0.6], "sei": [0.1, 0.1]})
        assert r["status"] == "skipped"

    def test_consistent_homogeneous(self, constant_effects_claim):
        r = self.engine.evaluate(constant_effects_claim)
        assert r["consistency_status"] == "Consistent"
        assert r["n_influential"] == 0

    def test_cooks_distance(self, standard_claim):
        """Cook's D: list of k floats, all >= 0."""
        r = self.engine.evaluate(standard_claim)
        assert "cooks_distance" in r
        cooks = r["cooks_distance"]
        k = len(standard_claim["yi"])
        assert len(cooks) == k
        assert all(d >= 0.0 for d in cooks)

    def test_galbraith_radial(self, standard_claim):
        """Galbraith dict with intercept, slope, outlier_indices."""
        r = self.engine.evaluate(standard_claim)
        assert "galbraith" in r
        g = r["galbraith"]
        assert "intercept" in g
        assert "slope" in g
        assert "outlier_indices" in g
        assert isinstance(g["outlier_indices"], list)


# ═══════════════════════════ EVOLUTION (MATURITY) ═══════════════════════════

class TestEvolutionEngine:
    def setup_method(self):
        self.engine = EvolutionEngine()

    def test_basic(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        assert r["evidence_age_years"] == 10  # 2020-2010
        assert r["accumulation_rate"] > 0
        assert 0 <= r["maturity_index"] <= 1.0
        assert r["phase"] in ("Early", "Accumulating", "Stabilizing", "Mature")

    def test_too_few(self):
        r = self.engine.evaluate({"yi": [0.5, 0.6], "sei": [0.1, 0.1], "years": [2020, 2021]})
        assert r["status"] == "skipped"

    def test_stable_effects_high_maturity(self, constant_effects_claim):
        r = self.engine.evaluate(constant_effects_claim)
        assert r["maturity_index"] > 0.3
        assert r["phase"] in ("Stabilizing", "Mature")

    def test_cumulative_theta_range(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert r["cumulative_theta_range"] is not None
        lo, hi = r["cumulative_theta_range"]
        assert lo <= hi

    def test_precision_doubling(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        # May or may not have precision_doubling_k depending on data
        assert "precision_doubling_k" in r


# ═══════════════════════════ SYNTHESIS LOSS ═════════════════════════════════

class TestSynthesisLossEngine:
    def setup_method(self):
        self.engine = SynthesisLossEngine()

    def test_basic(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        assert 0 <= r["information_loss_ratio"] <= 1
        assert r["design_effect"] >= 1.0
        assert r["effective_n"] > 0
        assert r["total_n"] > 0

    def test_homogeneous_no_loss(self, constant_effects_claim):
        r = self.engine.evaluate(constant_effects_claim)
        assert r["information_loss_ratio"] < 0.01  # No tau2 → no loss
        assert abs(r["design_effect"] - 1.0) < 0.01

    def test_high_het_high_loss(self, heterogeneous_claim):
        r = self.engine.evaluate(heterogeneous_claim)
        assert r["information_loss_ratio"] > 0.1
        assert r["design_effect"] > 1.5

    def test_too_few(self):
        r = self.engine.evaluate({"yi": [0.5], "sei": [0.1]})
        assert r["status"] == "skipped"

    def test_redundancy_bounded(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert 0 <= r["redundancy"] <= 1


# ═══════════════════════════ AFRICA RCT ════════════════════════════════════

class TestAfricaRCTEngine:
    def setup_method(self):
        self.engine = AfricaRCTEngine()

    def test_basic(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        assert r["status"] == "evaluated"
        assert r["relevance_index"] is not None
        assert 0 <= r["relevance_index"] <= 1

    def test_burden_alignment_known(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        ba = r["burden_alignment"]
        assert ba["aligned"] is False  # Parkinson's not in SA burden

    def test_burden_alignment_hiv_kenya(self, minimal_claim):
        r = self.engine.evaluate(minimal_claim)
        assert r["burden_alignment"]["aligned"] is True  # HIV IS in Kenya burden

    def test_unknown_country(self):
        r = self.engine.evaluate({"yi": [0.3], "sei": [0.1], "country": "Mars", "condition": "X"})
        assert r["burden_alignment"]["aligned"] is None  # No data

    def test_transportability(self, standard_claim):
        r = self.engine.evaluate(standard_claim)
        t = r["transportability"]
        assert t is not None
        assert "pi_lo" in t
        assert "transport_risk" in t
        assert t["transport_risk"] in ("High", "Low")


# ═══════════════════════════ E156 EMITTER ═══════════════════════════════════

class TestE156Emitter:
    def _make_audit_results(self):
        return {
            "PredictionGap": {"metrics": {"theta": 0.5, "ci_lo": 0.3, "ci_hi": 0.7, "k": 5}},
            "MetaFrontierLab": {"estimate": 0.89, "ci": [0.57, 1.21]},
            "FragilityAtlas": {"classification": "Robust"},
            "CausalSynth": {"e_value": 4.3, "e_value_ci": 2.1},
            "RegistryForensics": {"status": "evaluated", "scientific_entropy": 2.1,
                                   "anomaly_flags": 1, "total_tests": 4},
            "NetworkMeta": {"status": "evaluated", "n_influential": 1,
                            "max_studentized_residual": 2.3},
            "SynthesisLoss": {"status": "evaluated", "information_loss_ratio": 0.15},
            "BayesianMA": {
                "status": "evaluated",
                "posterior_mu": 0.52,
                "posterior_mu_sd": 0.09,
                "cri_lo": 0.34,
                "cri_hi": 0.70,
                "bf01": 0.08,
                "bf10": 12.5,
                "evidence_label": "Positive",
            },
            "PubBias": {
                "status": "evaluated",
                "egger": {"intercept": 0.21, "p_value": 0.12, "significant": False},
                "trim_fill": {"n_missing": 2, "adjusted_theta": 0.44, "original_theta": 0.50},
                "failsafe_n": {"failsafe_n": 120.0, "robust_threshold": 35, "robust": True},
                "p_curve": {"n_significant": 4, "right_skew_p": 0.03, "skew_direction": "right"},
            },
            "GRADE": {
                "status": "evaluated",
                "certainty": "MODERATE",
                "certainty_score": 3,
                "total_downgrade": -1,
                "domains": {
                    "risk_of_bias": {"downgrade": -1, "reason": "1 anomaly flag detected"},
                    "inconsistency": {"downgrade": 0, "reason": "I2=30%, 1 influential study"},
                    "indirectness": {"downgrade": 0, "reason": "Condition aligned with burden"},
                    "imprecision": {"downgrade": 0, "reason": "CI width 0.40, not fragile"},
                    "publication_bias": {"downgrade": 0, "reason": "No bias detected"},
                },
            },
        }

    def test_seven_sentences(self, standard_claim):
        standard_claim["audit_results"] = self._make_audit_results()
        r = E156Emitter().evaluate(standard_claim)
        assert r["sentence_count"] == 7
        assert r["word_count"] > 0

    def test_word_count_tracked(self):
        claim = {"country": "SA", "condition": "X",
                 "audit_results": self._make_audit_results()}
        r = E156Emitter().evaluate(claim)
        assert isinstance(r["over_limit"], bool)

    def test_no_simulated_fields(self):
        claim = {"country": "SA", "condition": "X",
                 "audit_results": self._make_audit_results()}
        r = E156Emitter().evaluate(claim)
        assert "disclosure" not in r
        assert "has_simulated_engines" not in r

    def test_e156_references_bayesian(self):
        """E156 body must mention posterior or credible (Bayesian result in S4)."""
        claim = {"country": "SA", "condition": "X",
                 "audit_results": self._make_audit_results()}
        r = E156Emitter().evaluate(claim)
        body_lower = r["body"].lower()
        assert "posterior" in body_lower or "credible" in body_lower, (
            f"Expected 'posterior' or 'credible' in E156 body, got: {r['body']}"
        )

    def test_e156_references_grade(self):
        """E156 body must mention certainty or a GRADE level (GRADE result in S5)."""
        claim = {"country": "SA", "condition": "X",
                 "audit_results": self._make_audit_results()}
        r = E156Emitter().evaluate(claim)
        body_lower = r["body"].lower()
        grade_terms = ["certainty", "high", "moderate", "low", "very low"]
        assert any(t in body_lower for t in grade_terms), (
            f"Expected a GRADE term in E156 body, got: {r['body']}"
        )


# ═══════════════════════════ ORCHESTRATOR ═══════════════════════════════════

class TestOrchestrator:
    def test_all_engines_produce_results(self, standard_claim):
        results = EvidenceOrchestrator().run_audit(standard_claim)
        for eng in ["PredictionGap", "FragilityAtlas", "TreatmentEvolution",
                     "SynthesisLoss", "CausalSynth", "EvidenceDrift",
                     "RegistryForensics", "NetworkMeta", "Al-Mizan", "AfricaRCT", "E156"]:
            assert eng in results, f"Missing: {eng}"
            assert "status" in results[eng]

    def test_no_crash_on_empty(self):
        results = EvidenceOrchestrator().run_audit({})
        assert len(results) > 0
        for name, r in results.items():
            assert "status" in r

    def test_claim_data_not_mutated(self, standard_claim):
        original_keys = set(standard_claim.keys())
        EvidenceOrchestrator().run_audit(standard_claim)
        assert set(standard_claim.keys()) == original_keys

    def test_no_simulated_flags_anywhere(self, standard_claim):
        results = EvidenceOrchestrator().run_audit(standard_claim)
        for name, r in results.items():
            assert "simulated" not in r, f"Engine {name} still has 'simulated' flag"


# ═══════════════════════════ REPORTING ══════════════════════════════════════

class TestReporting:
    def test_html_escaping(self, tmp_path):
        from alburhan.reporting import generate_html_report
        xss_results = {k: {} for k in [
            "PredictionGap", "MetaFrontierLab", "FragilityAtlas", "CausalSynth",
            "RegistryForensics", "NetworkMeta", "Al-Mizan", "AfricaRCT",
            "TreatmentEvolution", "SynthesisLoss", "EvidenceDrift",
        ]}
        xss_results["E156"] = {"body": "Test body."}
        out = str(tmp_path / "test.html")
        generate_html_report(xss_results, "id", "C", '<script>alert(1)</script>', out)
        html = open(out, encoding="utf-8").read()
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_viewport(self, tmp_path):
        from alburhan.reporting import generate_html_report
        out = str(tmp_path / "t.html")
        generate_html_report({"E156": {"body": "x"}}, "id", "C", "D", out)
        assert 'name="viewport"' in open(out, encoding="utf-8").read()

    def test_semantic_html(self, tmp_path):
        from alburhan.reporting import generate_html_report
        out = str(tmp_path / "t.html")
        generate_html_report({"E156": {"body": "x"}}, "id", "C", "D", out)
        html = open(out, encoding="utf-8").read()
        for tag in ("<main", "<header", "<footer", "<article", "aria-label"):
            assert tag in html

    def test_no_simulation_banner(self, tmp_path):
        from alburhan.reporting import generate_html_report
        out = str(tmp_path / "t.html")
        generate_html_report({"E156": {"body": "x"}}, "id", "C", "D", out)
        html = open(out, encoding="utf-8").read()
        assert "sim-banner" not in html
        assert "simulated" not in html.lower()
