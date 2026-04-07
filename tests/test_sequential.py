"""
Tests for SequentialTSAEngine (Task 6).

8 tests covering:
1. OBF boundary conservative early, liberal late
2. Pocock boundary roughly constant
3. Z-trajectory computed for each cumulative look
4. RIS > 0 and accounts for heterogeneity
5. Conditional power between 0 and 1
6. Futility boundary < efficacy boundary
7. Too-few-studies (k<2) returns skipped
8. Boundary crossing detected when Z exceeds OBF boundary
"""

import math
import numpy as np
import pytest
from scipy import stats

from alburhan.engines.sequential import SequentialTSAEngine


# ─────────────────────────── helpers ────────────────────────────────────────

def _engine():
    return SequentialTSAEngine()


def _claim(yi, sei, years, n_per_study=None, **kwargs):
    d = {"yi": yi, "sei": sei, "years": years}
    if n_per_study is not None:
        d["n_per_study"] = n_per_study
    d.update(kwargs)
    return d


# ─────────────────────────── fixtures ───────────────────────────────────────

@pytest.fixture
def standard_claim():
    """5 studies, moderate effect, 200 pts each."""
    return _claim(
        yi=[0.60, 0.50, 0.40, 0.45, 0.55],
        sei=[0.10, 0.15, 0.10, 0.12, 0.10],
        years=[2010, 2012, 2015, 2018, 2020],
        n_per_study=[200, 200, 200, 200, 200],
    )


@pytest.fixture
def large_homogeneous_claim():
    """8 studies with near-identical effects and large N."""
    return _claim(
        yi=[0.50] * 8,
        sei=[0.05] * 8,
        years=list(range(2010, 2018)),
        n_per_study=[500] * 8,
    )


@pytest.fixture
def high_heterogeneity_claim():
    """5 studies with large variance — high I2."""
    return _claim(
        yi=[0.10, 0.80, -0.20, 1.20, 0.05],
        sei=[0.10, 0.10, 0.10, 0.10, 0.10],
        years=[2010, 2012, 2014, 2016, 2018],
        n_per_study=[200, 200, 200, 200, 200],
    )


@pytest.fixture
def boundary_crossing_claim():
    """Strong, consistent effect that should cross OBF boundary."""
    return _claim(
        yi=[1.50, 1.60, 1.55, 1.70, 1.50, 1.65],
        sei=[0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        years=[2010, 2012, 2014, 2016, 2018, 2020],
        n_per_study=[400, 400, 400, 400, 400, 400],
    )


# ─────────────────────────── test 1 ─────────────────────────────────────────

def test_obf_boundary_conservative_early_liberal_late():
    """
    OBF boundary at t=0.5 should be > boundary at t=1.0
    (very conservative early → approaching z_alpha at final look).
    """
    engine = _engine()
    alpha = 0.05

    # Direct spending-function check
    alpha_spent_early = SequentialTSAEngine._obf_spending(0.5, alpha)
    alpha_spent_late = SequentialTSAEngine._obf_spending(1.0, alpha)

    z_early = SequentialTSAEngine._alpha_spent_to_boundary(alpha_spent_early)
    z_late = SequentialTSAEngine._alpha_spent_to_boundary(alpha_spent_late)

    assert z_early > z_late, (
        f"OBF boundary at t=0.5 ({z_early:.4f}) should exceed boundary at t=1.0 ({z_late:.4f})"
    )

    # Also verify the OBF property directly at a range of t values:
    # boundary should be strictly decreasing from t=0.1 to t=1.0
    z_half = stats.norm.ppf(1.0 - 0.05 / 2.0)
    prev_boundary = None
    for t in [0.1, 0.25, 0.5, 0.75, 1.0]:
        alpha_spent = SequentialTSAEngine._obf_spending(t, 0.05)
        z_b = SequentialTSAEngine._alpha_spent_to_boundary(alpha_spent)
        if prev_boundary is not None:
            assert z_b <= prev_boundary + 1e-9, (
                f"OBF boundary at t={t} ({z_b:.4f}) should not exceed "
                f"previous boundary ({prev_boundary:.4f})"
            )
        prev_boundary = z_b
    # At t=1 the OBF boundary equals z_alpha/2 (all alpha spent)
    assert abs(prev_boundary - z_half) < 0.01, (
        f"OBF boundary at t=1.0 ({prev_boundary:.4f}) should equal z_alpha/2 ({z_half:.4f})"
    )
    # At t=0.1 the OBF boundary must be much more conservative than z_alpha/2
    z_b_early = SequentialTSAEngine._alpha_spent_to_boundary(
        SequentialTSAEngine._obf_spending(0.1, 0.05)
    )
    assert z_b_early > z_half + 0.5, (
        f"OBF boundary at t=0.1 ({z_b_early:.4f}) should be well above z_alpha/2 ({z_half:.4f})"
    )


# ─────────────────────────── test 2 ─────────────────────────────────────────

def test_pocock_boundary_roughly_constant():
    """
    Pocock spending yields roughly constant boundaries across looks
    (less conservative early, more liberal late than OBF).
    """
    engine = _engine()
    alpha = 0.05
    # Compute Pocock boundaries at several information fractions
    ts = [0.2, 0.4, 0.6, 0.8, 1.0]
    boundaries = []
    for t in ts:
        alpha_spent = SequentialTSAEngine._pocock_spending(t, alpha)
        z = SequentialTSAEngine._alpha_spent_to_boundary(alpha_spent)
        boundaries.append(z)

    # Max deviation from mean should be small (< 0.5 z-units)
    mean_b = np.mean(boundaries)
    max_dev = max(abs(b - mean_b) for b in boundaries)
    assert max_dev < 0.5, (
        f"Pocock boundaries deviate too much (max_dev={max_dev:.4f}). "
        f"Boundaries: {[f'{b:.3f}' for b in boundaries]}"
    )

    # Via engine output
    claim = _claim(
        yi=[0.30] * 5,
        sei=[0.12] * 5,
        years=[2010, 2012, 2014, 2016, 2018],
        n_per_study=[200] * 5,
        spending="pocock",
    )
    result = engine.evaluate(claim)
    poc_b = result["pocock_alpha_boundaries"]
    range_b = max(poc_b) - min(poc_b)
    assert range_b < 1.0, (
        f"Pocock engine boundaries vary too widely: range={range_b:.4f}, values={poc_b}"
    )


# ─────────────────────────── test 3 ─────────────────────────────────────────

def test_z_trajectory_length(standard_claim):
    """Z-trajectory has one entry per cumulative look (looks = k-1 to k)."""
    engine = _engine()
    result = engine.evaluate(standard_claim)
    assert result["status"] == "evaluated"

    k = len(standard_claim["yi"])
    # Cumulative meta starts at k=2, so we get (k-1) looks
    expected_looks = k - 1
    assert result["n_looks"] == expected_looks, (
        f"Expected {expected_looks} looks for k={k} studies, got {result['n_looks']}"
    )
    assert len(result["z_trajectory"]) == expected_looks, (
        f"Z-trajectory length mismatch: {len(result['z_trajectory'])} vs {expected_looks}"
    )
    # Each z-value should be a finite float
    for i, z in enumerate(result["z_trajectory"]):
        assert math.isfinite(z), f"z_trajectory[{i}] = {z} is not finite"


# ─────────────────────────── test 4 ─────────────────────────────────────────

def test_ris_positive_and_heterogeneity_inflated():
    """RIS > 0 and high-I2 claim yields larger RIS than homogeneous claim."""
    engine = _engine()

    homo_claim = _claim(
        yi=[0.50] * 4,
        sei=[0.10] * 4,
        years=[2010, 2012, 2014, 2016],
        n_per_study=[200] * 4,
    )
    hetero_claim = _claim(
        yi=[0.10, 0.90, -0.30, 1.40],
        sei=[0.10] * 4,
        years=[2010, 2012, 2014, 2016],
        n_per_study=[200] * 4,
    )

    r_homo = engine.evaluate(homo_claim)
    r_hetero = engine.evaluate(hetero_claim)

    assert r_homo["ris"] > 0, "Homogeneous RIS must be positive"
    assert r_hetero["ris"] > 0, "Heterogeneous RIS must be positive"
    assert r_hetero["ris"] > r_homo["ris"], (
        f"High-heterogeneity RIS ({r_hetero['ris']:.1f}) should exceed "
        f"homogeneous RIS ({r_homo['ris']:.1f})"
    )


# ─────────────────────────── test 5 ─────────────────────────────────────────

def test_conditional_power_in_unit_interval(standard_claim):
    """All conditional powers must lie in [0, 1]."""
    engine = _engine()
    result = engine.evaluate(standard_claim)
    assert result["status"] == "evaluated"

    cps = result["conditional_powers"]
    assert len(cps) > 0, "No conditional powers returned"
    for i, cp in enumerate(cps):
        assert 0.0 <= cp <= 1.0, (
            f"conditional_powers[{i}] = {cp} is outside [0, 1]"
        )


# ─────────────────────────── test 6 ─────────────────────────────────────────

def test_futility_boundary_less_than_efficacy_boundary(standard_claim):
    """At every look, futility boundary <= efficacy boundary."""
    engine = _engine()
    result = engine.evaluate(standard_claim)
    assert result["status"] == "evaluated"

    eff = result["obf_alpha_boundaries"]
    fut = result["obf_futility_boundaries"]

    assert len(eff) == len(fut), "Boundary lists must have same length"
    for i, (e, f) in enumerate(zip(eff, fut)):
        assert f <= e, (
            f"Look {i}: futility boundary ({f:.4f}) exceeds efficacy boundary ({e:.4f})"
        )


# ─────────────────────────── test 7 ─────────────────────────────────────────

def test_too_few_studies_returns_skipped():
    """k=1 study → status='skipped'."""
    engine = _engine()

    single_study = _claim(
        yi=[0.50],
        sei=[0.10],
        years=[2020],
        n_per_study=[200],
    )
    result = engine.evaluate(single_study)
    assert result["status"] == "skipped", (
        f"Expected 'skipped' for k=1, got '{result['status']}'"
    )
    # Empty lists / zero studies also should skip
    empty = _claim(yi=[], sei=[], years=[])
    result2 = engine.evaluate(empty)
    assert result2["status"] == "skipped"


# ─────────────────────────── test 8 ─────────────────────────────────────────

def test_boundary_crossing_detected(boundary_crossing_claim):
    """
    Very strong effect → cumulative Z should eventually cross OBF boundary
    and boundary_crossed flag should be True.
    """
    engine = _engine()
    result = engine.evaluate(boundary_crossing_claim)
    assert result["status"] == "evaluated"

    # Verify the flag is consistent with the trajectory
    z_traj = result["z_trajectory"]
    obf_bounds = result["obf_alpha_boundaries"]

    manual_cross = any(
        abs(z) >= b for z, b in zip(z_traj, obf_bounds)
    )
    assert result["boundary_crossed"] == manual_cross, (
        "boundary_crossed flag inconsistent with z_trajectory vs obf_alpha_boundaries"
    )

    # For this strong-effect claim, we expect at least one crossing
    assert result["boundary_crossed"] is True, (
        f"Strong-effect claim should cross OBF boundary.\n"
        f"Z-trajectory: {[f'{z:.2f}' for z in z_traj]}\n"
        f"OBF boundaries: {[f'{b:.2f}' for b in obf_bounds]}"
    )
