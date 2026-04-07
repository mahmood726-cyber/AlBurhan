"""
Tests for alburhan.plots SVG generators.

All 8 required tests plus additional sanity checks.
"""

import pytest
from alburhan.plots import forest_plot, funnel_plot, cusum_plot, galbraith_plot


class TestPlots:
    # ── Test 1 ────────────────────────────────────────────────────────────────
    def test_forest_plot_valid_svg(self):
        svg = forest_plot(
            [0.5, 0.6, 0.4], [0.1, 0.15, 0.1],
            pooled_theta=0.5, pooled_ci=(0.3, 0.7)
        )
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "viewBox" in svg

    # ── Test 2 ────────────────────────────────────────────────────────────────
    def test_forest_plot_has_studies(self):
        svg = forest_plot([0.5, 0.6, 0.4], [0.1, 0.15, 0.1])
        # Should have 3 study rows (class="study-row") or at least 3 CI lines
        assert svg.count("study-row") >= 3 or svg.count("<line") >= 3

    # ── Test 3 ────────────────────────────────────────────────────────────────
    def test_funnel_plot_valid_svg(self):
        svg = funnel_plot(
            [0.5, 0.6, 0.4, 0.7], [0.1, 0.15, 0.1, 0.2],
            pooled_theta=0.55
        )
        assert "<svg" in svg
        assert "</svg>" in svg

    # ── Test 4 ────────────────────────────────────────────────────────────────
    def test_funnel_plot_with_trim_fill(self):
        svg = funnel_plot(
            [0.5, 0.6], [0.1, 0.15],
            pooled_theta=0.55,
            trim_fill_points=[(0.4, 0.12)]
        )
        # Trim-fill points use class="trim-fill", fill="none", or stroke-dasharray
        assert (
            "trim-fill" in svg
            or 'stroke-dasharray' in svg
            or 'fill="none"' in svg
        )

    # ── Test 5 ────────────────────────────────────────────────────────────────
    def test_cusum_plot_valid_svg(self):
        svg = cusum_plot(
            [1.5, 2.1, 2.8],
            efficacy_bounds=[3.0, 2.5, 2.2],
            info_fractions=[0.3, 0.6, 1.0]
        )
        assert "<svg" in svg
        assert "</svg>" in svg

    # ── Test 6 ────────────────────────────────────────────────────────────────
    def test_galbraith_plot_valid_svg(self):
        svg = galbraith_plot(
            [0.5, 0.6, 0.4, 0.7], [0.1, 0.15, 0.1, 0.2],
            pooled_theta=0.55
        )
        assert "<svg" in svg

    # ── Test 7 ────────────────────────────────────────────────────────────────
    def test_empty_data_returns_empty(self):
        """Empty arrays should return empty string, not crash."""
        assert forest_plot([], []) == ""
        assert funnel_plot([], []) == ""

    # ── Test 8 ────────────────────────────────────────────────────────────────
    def test_forest_plot_accessibility(self):
        svg = forest_plot([0.5, 0.6], [0.1, 0.15])
        assert "<title>" in svg
        assert "<desc>" in svg

    # ── Extra sanity checks ───────────────────────────────────────────────────

    def test_cusum_plot_empty_returns_empty(self):
        assert cusum_plot([]) == ""

    def test_galbraith_plot_empty_returns_empty(self):
        assert galbraith_plot([], []) == ""

    def test_forest_plot_with_labels(self):
        svg = forest_plot(
            [0.3, 0.5], [0.1, 0.12],
            labels=["Trial A", "Trial B"],
            pooled_theta=0.4, pooled_ci=(0.2, 0.6)
        )
        assert "Trial A" in svg
        assert "Trial B" in svg
        assert "Pooled" in svg

    def test_funnel_plot_accessibility(self):
        svg = funnel_plot([0.5, 0.6, 0.4], [0.1, 0.15, 0.1])
        assert "<title>" in svg
        assert "<desc>" in svg

    def test_cusum_plot_accessibility(self):
        svg = cusum_plot([1.5, 2.1])
        assert "<title>" in svg
        assert "<desc>" in svg

    def test_galbraith_plot_accessibility(self):
        svg = galbraith_plot([0.5, 0.6, 0.4], [0.1, 0.15, 0.1])
        assert "<title>" in svg
        assert "<desc>" in svg

    def test_forest_plot_diamond_in_svg(self):
        """Diamond (pooled) polygon should appear when pooled_theta provided."""
        svg = forest_plot(
            [0.5, 0.6, 0.4], [0.1, 0.15, 0.1],
            pooled_theta=0.5, pooled_ci=(0.3, 0.7)
        )
        assert "<polygon" in svg

    def test_cusum_plot_with_futility(self):
        svg = cusum_plot(
            [1.5, 2.1, 2.8],
            efficacy_bounds=[3.0, 2.5, 2.2],
            futility_bounds=[0.5, 0.8, 1.0],
            info_fractions=[0.3, 0.6, 1.0]
        )
        assert "<svg" in svg
        # Both boundary polylines should be present
        assert svg.count("<polyline") >= 2

    def test_galbraith_outlier_highlighted(self):
        """A study far from regression line should be coloured red."""
        # Study with yi=5 (z=50) will be an extreme outlier
        svg = galbraith_plot([0.5, 0.5, 0.5, 5.0], [0.1, 0.1, 0.1, 0.1])
        # Outlier fill colour (#f85149) should appear
        assert "#f85149" in svg
