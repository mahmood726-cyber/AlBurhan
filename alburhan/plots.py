"""
SVG plot generators for Al-Burhan evidence dashboard.

All plots are:
- Pure SVG (no JavaScript libraries, no external deps)
- Self-contained (work fully offline)
- Responsive (use viewBox)
- Accessible (include <title> and <desc> elements)
"""

import math


# ─── Colour palette (matches dark theme) ─────────────────────────────────────
_C_TEXT = "#c9d1d9"
_C_GRID = "#30363d"
_C_ACCENT = "#58a6ff"
_C_DANGER = "#f85149"
_C_SUCCESS = "#3fb950"
_C_OUTLIER = "rgba(248,81,73,0.5)"
_C_TRIM = "none"   # fill=none for hollow circles


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    """Minimal XML attribute/text escaping."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _linear_map(value, in_lo, in_hi, out_lo, out_hi):
    """Map value from [in_lo, in_hi] to [out_lo, out_hi]."""
    if in_hi == in_lo:
        return (out_lo + out_hi) / 2.0
    return out_lo + (value - in_lo) / (in_hi - in_lo) * (out_hi - out_lo)


def _fmt(v, decimals=3):
    """Format a number, returning 'N/A' for None/nan."""
    try:
        if v is None or math.isnan(float(v)):
            return "N/A"
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


# ─── Plot 1: Forest Plot ───────────────────────────────────────────────────────

def forest_plot(yi, sei, labels=None, pooled_theta=None, pooled_ci=None,
                width=600, height=None):
    """
    Generate SVG forest plot with study effects + CI bars + pooled diamond.

    Parameters
    ----------
    yi            : list of float  — study effect sizes
    sei           : list of float  — standard errors
    labels        : list of str    — study labels (optional)
    pooled_theta  : float          — pooled estimate (optional)
    pooled_ci     : (lo, hi) tuple — pooled CI (optional)
    width, height : int            — SVG dimensions

    Returns
    -------
    str : inline SVG element, or "" if no data.
    """
    if not yi or not sei or len(yi) == 0:
        return ""

    yi = list(yi)
    sei = list(sei)
    k = len(yi)

    if height is None:
        height = (k + 4) * 30

    if labels is None:
        labels = [f"Study {i+1}" for i in range(k)]

    # Layout constants
    PAD_TOP = 20
    ROW_H = 30
    LEFT_W = 150      # label column
    RIGHT_W = 100     # effect text column
    PLOT_X0 = LEFT_W + 10
    PLOT_X1 = width - RIGHT_W - 10
    PLOT_W = PLOT_X1 - PLOT_X0

    # Compute CI for each study
    ci_lo = [y - 1.96 * s for y, s in zip(yi, sei)]
    ci_hi = [y + 1.96 * s for y, s in zip(yi, sei)]

    # Data range for x-axis
    all_lo = ci_lo[:]
    all_hi = ci_hi[:]
    if pooled_ci:
        all_lo.append(pooled_ci[0])
        all_hi.append(pooled_ci[1])
    x_min = min(all_lo) - 0.1 * abs(min(all_lo) if min(all_lo) != 0 else 0.1)
    x_max = max(all_hi) + 0.1 * abs(max(all_hi) if max(all_hi) != 0 else 0.1)
    # Ensure null (0) is visible
    x_min = min(x_min, -0.1)
    x_max = max(x_max, 0.1)

    def xpx(v):
        return _linear_map(v, x_min, x_max, PLOT_X0, PLOT_X1)

    # Weights (proportional to 1/sei^2), normalised for square size
    weights = [1.0 / (s ** 2) for s in sei]
    max_w = max(weights) if weights else 1.0
    MAX_SQ = 9   # max half-side of weight square in px
    MIN_SQ = 3

    def sq_size(w):
        return MIN_SQ + (MAX_SQ - MIN_SQ) * (w / max_w) ** 0.5

    lines = []

    # Defs
    lines.append(
        f'<defs>'
        f'<marker id="fb-arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{_C_GRID}"/>'
        f'</marker>'
        f'</defs>'
    )

    # Background
    lines.append(f'<rect width="{width}" height="{height}" fill="transparent"/>')

    # Header row
    header_y = PAD_TOP + 14
    lines.append(
        f'<text x="10" y="{header_y}" fill="{_C_TEXT}" '
        f'font-size="11" font-weight="bold" opacity="0.7">Study</text>'
    )
    lines.append(
        f'<text x="{width - RIGHT_W + 5}" y="{header_y}" fill="{_C_TEXT}" '
        f'font-size="11" font-weight="bold" opacity="0.7">Effect [95% CI]</text>'
    )

    # Null line (x=0) — dashed
    null_x = xpx(0.0)
    lines.append(
        f'<line x1="{null_x:.1f}" y1="{PAD_TOP}" '
        f'x2="{null_x:.1f}" y2="{height - 30}" '
        f'stroke="{_C_GRID}" stroke-width="1.5" stroke-dasharray="4,3"/>'
    )

    # Study rows
    for i in range(k):
        row_y = PAD_TOP + 30 + i * ROW_H
        cy = row_y + ROW_H // 2

        lo_px = xpx(ci_lo[i])
        hi_px = xpx(ci_hi[i])
        est_px = xpx(yi[i])
        sq = sq_size(weights[i])

        # Clamp to plot area
        lo_px_c = max(PLOT_X0, min(PLOT_X1, lo_px))
        hi_px_c = max(PLOT_X0, min(PLOT_X1, hi_px))

        # CI line
        lines.append(
            f'<g class="study-row" role="img" aria-label="{_escape(labels[i])}: '
            f'{_fmt(yi[i])} [{_fmt(ci_lo[i])}, {_fmt(ci_hi[i])}]">'
        )
        lines.append(
            f'<line x1="{lo_px_c:.1f}" y1="{cy}" x2="{hi_px_c:.1f}" y2="{cy}" '
            f'stroke="{_C_ACCENT}" stroke-width="1.5"/>'
        )
        # Square (point estimate)
        lines.append(
            f'<rect x="{est_px - sq:.1f}" y="{cy - sq:.1f}" '
            f'width="{2*sq:.1f}" height="{2*sq:.1f}" '
            f'fill="{_C_ACCENT}" stroke="none"/>'
        )
        lines.append('</g>')

        # Label
        lines.append(
            f'<text x="{LEFT_W - 5}" y="{cy + 4}" '
            f'fill="{_C_TEXT}" font-size="11" text-anchor="end" '
            f'font-family="inherit">{_escape(labels[i])}</text>'
        )

        # Effect [CI] text
        lines.append(
            f'<text x="{width - RIGHT_W + 5}" y="{cy + 4}" '
            f'fill="{_C_TEXT}" font-size="10" font-family="inherit">'
            f'{_fmt(yi[i], 2)} [{_fmt(ci_lo[i], 2)}, {_fmt(ci_hi[i], 2)}]</text>'
        )

    # Pooled diamond
    if pooled_theta is not None:
        pool_y = PAD_TOP + 30 + k * ROW_H + 10
        pool_cy = pool_y + ROW_H // 2

        if pooled_ci is not None:
            p_lo, p_hi = pooled_ci
        else:
            p_lo = pooled_theta - 0.15
            p_hi = pooled_theta + 0.15

        p_lo_px = max(PLOT_X0, min(PLOT_X1, xpx(p_lo)))
        p_hi_px = max(PLOT_X0, min(PLOT_X1, xpx(p_hi)))
        p_est_px = xpx(pooled_theta)

        diam_half_h = 8
        # Diamond: left-mid-right-mid polygon
        pts = (
            f"{p_lo_px:.1f},{pool_cy:.1f} "
            f"{p_est_px:.1f},{pool_cy - diam_half_h:.1f} "
            f"{p_hi_px:.1f},{pool_cy:.1f} "
            f"{p_est_px:.1f},{pool_cy + diam_half_h:.1f}"
        )
        lines.append(
            f'<polygon points="{pts}" fill="{_C_ACCENT}" stroke="none" '
            f'role="img" aria-label="Pooled estimate: {_fmt(pooled_theta, 2)} '
            f'[{_fmt(p_lo, 2)}, {_fmt(p_hi, 2)}]"/>'
        )

        # Pooled line (dotted)
        lines.append(
            f'<line x1="{p_est_px:.1f}" y1="{PAD_TOP}" '
            f'x2="{p_est_px:.1f}" y2="{height - 20}" '
            f'stroke="{_C_ACCENT}" stroke-width="1" stroke-dasharray="2,3" opacity="0.6"/>'
        )

        # Label
        lines.append(
            f'<text x="{LEFT_W - 5}" y="{pool_cy + 4}" '
            f'fill="{_C_ACCENT}" font-size="11" text-anchor="end" '
            f'font-weight="bold" font-family="inherit">Pooled</text>'
        )
        lines.append(
            f'<text x="{width - RIGHT_W + 5}" y="{pool_cy + 4}" '
            f'fill="{_C_ACCENT}" font-size="10" font-weight="bold" font-family="inherit">'
            f'{_fmt(pooled_theta, 2)} [{_fmt(p_lo, 2)}, {_fmt(p_hi, 2)}]</text>'
        )

    # Axis label
    axis_y = height - 8
    lines.append(
        f'<text x="{(PLOT_X0 + PLOT_X1) / 2:.0f}" y="{axis_y}" '
        f'fill="{_C_TEXT}" font-size="10" text-anchor="middle" opacity="0.6">Effect Size</text>'
    )

    inner = "\n  ".join(lines)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" '
        f'font-family="inherit">\n'
        f'  <title>Forest Plot</title>\n'
        f'  <desc>Forest plot showing {k} study effect estimates with '
        f'95% confidence intervals{" and pooled diamond" if pooled_theta is not None else ""}.'
        f'</desc>\n'
        f'  {inner}\n'
        f'</svg>'
    )
    return svg


# ─── Plot 2: Funnel Plot ──────────────────────────────────────────────────────

def funnel_plot(yi, sei, pooled_theta=None, egger_intercept=None,
                trim_fill_points=None, width=500, height=400):
    """
    Generate SVG funnel plot: effect (x) vs SE (y, inverted).

    Parameters
    ----------
    yi                : list of float — study effect sizes
    sei               : list of float — standard errors
    pooled_theta      : float         — pooled estimate
    egger_intercept   : float         — Egger intercept (slope of regression line)
    trim_fill_points  : list of (y,se)— imputed trim-fill points (hollow circles)
    width, height     : int           — SVG dimensions

    Returns
    -------
    str : inline SVG element, or "" if no data.
    """
    if not yi or not sei or len(yi) == 0:
        return ""

    yi = list(yi)
    sei = list(sei)
    k = len(yi)

    PAD = {"top": 30, "right": 30, "bottom": 50, "left": 60}
    PLOT_X0 = PAD["left"]
    PLOT_X1 = width - PAD["right"]
    PLOT_Y0 = PAD["top"]
    PLOT_Y1 = height - PAD["bottom"]

    # x range: effect sizes
    x_vals = yi[:]
    if trim_fill_points:
        x_vals += [p[0] for p in trim_fill_points]
    if pooled_theta is not None:
        x_vals.append(pooled_theta)
    x_lo = min(x_vals) - 0.2
    x_hi = max(x_vals) + 0.2

    # y range: SE (inverted — 0 at top)
    all_se = sei[:]
    if trim_fill_points:
        all_se += [p[1] for p in trim_fill_points]
    max_se = max(all_se) * 1.1
    # y=0 maps to PLOT_Y0 (top), y=max_se maps to PLOT_Y1 (bottom)

    def xpx(v):
        return _linear_map(v, x_lo, x_hi, PLOT_X0, PLOT_X1)

    def ypx(se_val):
        # inverted: se=0 → top (PLOT_Y0), se=max_se → bottom (PLOT_Y1)
        return _linear_map(se_val, 0, max_se, PLOT_Y0, PLOT_Y1)

    lines = []
    lines.append(f'<rect width="{width}" height="{height}" fill="transparent"/>')

    # Grid lines (light horizontal)
    n_yticks = 5
    for i in range(n_yticks + 1):
        se_v = max_se * i / n_yticks
        yy = ypx(se_v)
        lines.append(
            f'<line x1="{PLOT_X0}" y1="{yy:.1f}" x2="{PLOT_X1}" y2="{yy:.1f}" '
            f'stroke="{_C_GRID}" stroke-width="0.5"/>'
        )
        lines.append(
            f'<text x="{PLOT_X0 - 5}" y="{yy + 4:.1f}" fill="{_C_TEXT}" '
            f'font-size="9" text-anchor="end" opacity="0.7">{_fmt(se_v, 2)}</text>'
        )

    # Funnel (pseudo-95% CI funnel around pooled theta)
    theta0 = pooled_theta if pooled_theta is not None else sum(yi) / len(yi)
    # Funnel lines: from (theta0, 0) to (theta0 ± 1.96*max_se, max_se)
    funnel_lo_x = theta0 - 1.96 * max_se
    funnel_hi_x = theta0 + 1.96 * max_se
    lines.append(
        f'<line x1="{xpx(theta0):.1f}" y1="{ypx(0):.1f}" '
        f'x2="{xpx(funnel_lo_x):.1f}" y2="{ypx(max_se):.1f}" '
        f'stroke="{_C_GRID}" stroke-width="1.5" stroke-dasharray="4,3"/>'
    )
    lines.append(
        f'<line x1="{xpx(theta0):.1f}" y1="{ypx(0):.1f}" '
        f'x2="{xpx(funnel_hi_x):.1f}" y2="{ypx(max_se):.1f}" '
        f'stroke="{_C_GRID}" stroke-width="1.5" stroke-dasharray="4,3"/>'
    )

    # Null line at pooled theta
    lines.append(
        f'<line x1="{xpx(theta0):.1f}" y1="{PLOT_Y0}" '
        f'x2="{xpx(theta0):.1f}" y2="{PLOT_Y1}" '
        f'stroke="{_C_ACCENT}" stroke-width="1" stroke-dasharray="3,3" opacity="0.5"/>'
    )

    # Egger regression line (if intercept provided)
    # Egger: precision (1/se) vs yi. Line: yi = intercept + slope*precision (slope≈theta)
    # In funnel space: yi = egger_intercept + theta * (1/se)  =>  yi is a function of se
    if egger_intercept is not None and pooled_theta is not None:
        # At se=max_se: yi = egger_intercept + pooled_theta / max_se
        # At se→0:  yi = egger_intercept (approaches intercept)
        # Draw as line from (yi@max_se, max_se) to (yi@0.001, ~0)
        se_vals_eg = [max_se, max_se * 0.1]
        for i_eg in range(len(se_vals_eg) - 1):
            se_a = se_vals_eg[i_eg]
            se_b = se_vals_eg[i_eg + 1]
            ya_x = egger_intercept + pooled_theta * (1.0 / se_a if se_a > 0 else 0)
            yb_x = egger_intercept + pooled_theta * (1.0 / se_b if se_b > 0 else 0)
            lines.append(
                f'<line x1="{xpx(ya_x):.1f}" y1="{ypx(se_a):.1f}" '
                f'x2="{xpx(yb_x):.1f}" y2="{ypx(se_b):.1f}" '
                f'stroke="{_C_DANGER}" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.7"/>'
            )

    # Trim-fill imputed points (hollow circles)
    if trim_fill_points:
        for pt in trim_fill_points:
            pt_y, pt_se = pt[0], pt[1]
            lines.append(
                f'<circle cx="{xpx(pt_y):.1f}" cy="{ypx(pt_se):.1f}" r="5" '
                f'fill="none" stroke="{_C_ACCENT}" stroke-width="1.5" '
                f'class="trim-fill" opacity="0.7" '
                f'aria-label="Imputed study: {_fmt(pt_y, 2)}, SE={_fmt(pt_se, 2)}"/>'
            )

    # Study circles (filled)
    weights = [1.0 / (s ** 2) for s in sei]
    max_w = max(weights) if weights else 1.0
    for i in range(k):
        r = 3 + 4 * (weights[i] / max_w) ** 0.5
        lines.append(
            f'<circle cx="{xpx(yi[i]):.1f}" cy="{ypx(sei[i]):.1f}" r="{r:.1f}" '
            f'fill="{_C_ACCENT}" opacity="0.75" '
            f'aria-label="Study {i+1}: effect={_fmt(yi[i], 2)}, SE={_fmt(sei[i], 2)}"/>'
        )

    # Axes
    lines.append(
        f'<line x1="{PLOT_X0}" y1="{PLOT_Y0}" x2="{PLOT_X0}" y2="{PLOT_Y1}" '
        f'stroke="{_C_TEXT}" stroke-width="1" opacity="0.4"/>'
    )
    lines.append(
        f'<line x1="{PLOT_X0}" y1="{PLOT_Y1}" x2="{PLOT_X1}" y2="{PLOT_Y1}" '
        f'stroke="{_C_TEXT}" stroke-width="1" opacity="0.4"/>'
    )

    # Axis labels
    lines.append(
        f'<text x="{(PLOT_X0 + PLOT_X1) / 2:.0f}" y="{height - 8}" '
        f'fill="{_C_TEXT}" font-size="11" text-anchor="middle">Effect Size</text>'
    )
    lines.append(
        f'<text x="12" y="{(PLOT_Y0 + PLOT_Y1) / 2:.0f}" '
        f'fill="{_C_TEXT}" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90,12,{(PLOT_Y0 + PLOT_Y1) / 2:.0f})">Standard Error</text>'
    )

    inner = "\n  ".join(lines)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" '
        f'font-family="inherit">\n'
        f'  <title>Funnel Plot</title>\n'
        f'  <desc>Funnel plot with inverted y-axis (SE), showing {k} studies. '
        f'Dashed lines indicate pseudo-95% CI funnel around pooled estimate.</desc>\n'
        f'  {inner}\n'
        f'</svg>'
    )
    return svg


# ─── Plot 3: CUSUM (Cumulative Z) Plot ────────────────────────────────────────

def cusum_plot(z_trajectory, efficacy_bounds=None, futility_bounds=None,
               info_fractions=None, width=500, height=350):
    """
    Generate SVG cumulative Z-statistic plot with TSA boundaries.

    Parameters
    ----------
    z_trajectory    : list of float — cumulative Z values
    efficacy_bounds : list of float — efficacy (upper) boundary at each look
    futility_bounds : list of float — futility (lower) boundary at each look
    info_fractions  : list of float — x positions (information fractions 0-1)
    width, height   : int           — SVG dimensions

    Returns
    -------
    str : inline SVG element, or "" if no data.
    """
    if not z_trajectory or len(z_trajectory) == 0:
        return ""

    z_trajectory = list(z_trajectory)
    n = len(z_trajectory)

    if info_fractions is None or len(info_fractions) == 0:
        info_fractions = [(i + 1) / n for i in range(n)]
    else:
        info_fractions = list(info_fractions)[:n]

    PAD = {"top": 30, "right": 30, "bottom": 50, "left": 60}
    PLOT_X0 = PAD["left"]
    PLOT_X1 = width - PAD["right"]
    PLOT_Y0 = PAD["top"]
    PLOT_Y1 = height - PAD["bottom"]

    # y range: cover z values + boundary values
    all_z = z_trajectory[:]
    if efficacy_bounds:
        all_z += [b for b in efficacy_bounds if b is not None]
    if futility_bounds:
        all_z += [b for b in futility_bounds if b is not None]
    all_z += [1.96, -1.96]
    y_lo = min(all_z) - 0.5
    y_hi = max(all_z) + 0.5

    def xpx(v):
        return _linear_map(v, 0.0, 1.0, PLOT_X0, PLOT_X1)

    def ypx(v):
        # y=y_hi → top (PLOT_Y0), y=y_lo → bottom (PLOT_Y1)
        return _linear_map(v, y_lo, y_hi, PLOT_Y1, PLOT_Y0)

    lines = []
    lines.append(f'<rect width="{width}" height="{height}" fill="transparent"/>')

    # Horizontal grid
    n_yticks = 6
    for i in range(n_yticks + 1):
        z_v = y_lo + (y_hi - y_lo) * i / n_yticks
        yy = ypx(z_v)
        lines.append(
            f'<line x1="{PLOT_X0}" y1="{yy:.1f}" x2="{PLOT_X1}" y2="{yy:.1f}" '
            f'stroke="{_C_GRID}" stroke-width="0.5"/>'
        )
        lines.append(
            f'<text x="{PLOT_X0 - 5}" y="{yy + 4:.1f}" fill="{_C_TEXT}" '
            f'font-size="9" text-anchor="end" opacity="0.7">{_fmt(z_v, 1)}</text>'
        )

    # Nominal significance lines at ±1.96
    for z_nom in [1.96, -1.96]:
        lines.append(
            f'<line x1="{PLOT_X0}" y1="{ypx(z_nom):.1f}" '
            f'x2="{PLOT_X1}" y2="{ypx(z_nom):.1f}" '
            f'stroke="{_C_GRID}" stroke-width="1" stroke-dasharray="3,3" opacity="0.7"/>'
        )
        lines.append(
            f'<text x="{PLOT_X1 + 3}" y="{ypx(z_nom) + 4:.1f}" '
            f'fill="{_C_TEXT}" font-size="9" opacity="0.7">±1.96</text>'
        )

    # Zero line
    lines.append(
        f'<line x1="{PLOT_X0}" y1="{ypx(0):.1f}" '
        f'x2="{PLOT_X1}" y2="{ypx(0):.1f}" '
        f'stroke="{_C_GRID}" stroke-width="1"/>'
    )

    # Efficacy boundary (red dashed)
    if efficacy_bounds and len(efficacy_bounds) >= 2:
        pts = " ".join(
            f"{xpx(info_fractions[i]):.1f},{ypx(efficacy_bounds[i]):.1f}"
            for i in range(min(n, len(efficacy_bounds)))
            if efficacy_bounds[i] is not None
        )
        if pts:
            lines.append(
                f'<polyline points="{pts}" fill="none" stroke="{_C_DANGER}" '
                f'stroke-width="2" stroke-dasharray="6,3"/>'
            )

    # Futility boundary (green dashed)
    if futility_bounds and len(futility_bounds) >= 2:
        pts = " ".join(
            f"{xpx(info_fractions[i]):.1f},{ypx(futility_bounds[i]):.1f}"
            for i in range(min(n, len(futility_bounds)))
            if futility_bounds[i] is not None
        )
        if pts:
            lines.append(
                f'<polyline points="{pts}" fill="none" stroke="{_C_SUCCESS}" '
                f'stroke-width="2" stroke-dasharray="6,3"/>'
            )

    # Z trajectory (blue solid)
    pts = " ".join(
        f"{xpx(info_fractions[i]):.1f},{ypx(z_trajectory[i]):.1f}"
        for i in range(n)
    )
    lines.append(
        f'<polyline points="{pts}" fill="none" stroke="{_C_ACCENT}" '
        f'stroke-width="2.5"/>'
    )

    # Circles at each look
    for i in range(n):
        lines.append(
            f'<circle cx="{xpx(info_fractions[i]):.1f}" cy="{ypx(z_trajectory[i]):.1f}" '
            f'r="3" fill="{_C_ACCENT}" '
            f'aria-label="Look {i+1}: Z={_fmt(z_trajectory[i], 2)}, '
            f'IF={_fmt(info_fractions[i], 2)}"/>'
        )

    # Axes
    lines.append(
        f'<line x1="{PLOT_X0}" y1="{PLOT_Y0}" x2="{PLOT_X0}" y2="{PLOT_Y1}" '
        f'stroke="{_C_TEXT}" stroke-width="1" opacity="0.4"/>'
    )
    lines.append(
        f'<line x1="{PLOT_X0}" y1="{PLOT_Y1}" x2="{PLOT_X1}" y2="{PLOT_Y1}" '
        f'stroke="{_C_TEXT}" stroke-width="1" opacity="0.4"/>'
    )

    # Axis labels
    lines.append(
        f'<text x="{(PLOT_X0 + PLOT_X1) / 2:.0f}" y="{height - 10}" '
        f'fill="{_C_TEXT}" font-size="11" text-anchor="middle">Information Fraction</text>'
    )
    lines.append(
        f'<text x="12" y="{(PLOT_Y0 + PLOT_Y1) / 2:.0f}" '
        f'fill="{_C_TEXT}" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90,12,{(PLOT_Y0 + PLOT_Y1) / 2:.0f})">Cumulative Z</text>'
    )

    # Legend
    leg_x = PLOT_X1 - 120
    leg_y = PLOT_Y0 + 10
    lines.append(
        f'<line x1="{leg_x}" y1="{leg_y}" x2="{leg_x + 20}" y2="{leg_y}" '
        f'stroke="{_C_ACCENT}" stroke-width="2.5"/>'
    )
    lines.append(
        f'<text x="{leg_x + 24}" y="{leg_y + 4}" fill="{_C_TEXT}" font-size="9">Z-stat</text>'
    )
    if efficacy_bounds:
        lines.append(
            f'<line x1="{leg_x}" y1="{leg_y + 14}" x2="{leg_x + 20}" y2="{leg_y + 14}" '
            f'stroke="{_C_DANGER}" stroke-width="2" stroke-dasharray="6,3"/>'
        )
        lines.append(
            f'<text x="{leg_x + 24}" y="{leg_y + 18}" fill="{_C_TEXT}" font-size="9">Efficacy</text>'
        )
    if futility_bounds:
        lines.append(
            f'<line x1="{leg_x}" y1="{leg_y + 28}" x2="{leg_x + 20}" y2="{leg_y + 28}" '
            f'stroke="{_C_SUCCESS}" stroke-width="2" stroke-dasharray="6,3"/>'
        )
        lines.append(
            f'<text x="{leg_x + 24}" y="{leg_y + 32}" fill="{_C_TEXT}" font-size="9">Futility</text>'
        )

    inner = "\n  ".join(lines)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" '
        f'font-family="inherit">\n'
        f'  <title>CUSUM Cumulative Z Plot</title>\n'
        f'  <desc>Cumulative Z-statistic trajectory across {n} interim looks '
        f'with alpha-spending boundaries.</desc>\n'
        f'  {inner}\n'
        f'</svg>'
    )
    return svg


# ─── Plot 4: Galbraith Radial Plot ────────────────────────────────────────────

def galbraith_plot(yi, sei, pooled_theta=None, width=500, height=400):
    """
    Generate SVG Galbraith radial plot: z/se (y) vs 1/se (x).

    Parameters
    ----------
    yi            : list of float — study effect sizes
    sei           : list of float — standard errors
    pooled_theta  : float         — pooled estimate (slope through origin)
    width, height : int           — SVG dimensions

    Returns
    -------
    str : inline SVG element, or "" if no data.
    """
    if not yi or not sei or len(yi) == 0:
        return ""

    yi = list(yi)
    sei = list(sei)
    k = len(yi)

    PAD = {"top": 30, "right": 40, "bottom": 50, "left": 60}
    PLOT_X0 = PAD["left"]
    PLOT_X1 = width - PAD["right"]
    PLOT_Y0 = PAD["top"]
    PLOT_Y1 = height - PAD["bottom"]

    # Radial coordinates
    x_vals = [1.0 / s for s in sei]   # precision (1/se)
    y_vals = [y / s for y, s in zip(yi, sei)]  # z-score

    x_lo = 0.0
    x_hi = max(x_vals) * 1.1
    y_range = max(abs(v) for v in y_vals) * 1.3
    y_lo = -y_range
    y_hi = y_range

    def xpx(v):
        return _linear_map(v, x_lo, x_hi, PLOT_X0, PLOT_X1)

    def ypx(v):
        return _linear_map(v, y_lo, y_hi, PLOT_Y1, PLOT_Y0)

    # Compute pooled theta if not provided (OLS slope through origin)
    if pooled_theta is None:
        sum_xx = sum(v ** 2 for v in x_vals)
        sum_xy = sum(x_vals[i] * y_vals[i] for i in range(k))
        pooled_theta = sum_xy / sum_xx if sum_xx > 0 else 0.0

    # Regression line (slope = pooled_theta, intercept=0 through origin)
    # Also compute intercept from OLS for ±2 bands
    x_bar = sum(x_vals) / k
    y_bar = sum(y_vals) / k
    ssxx = sum((v - x_bar) ** 2 for v in x_vals)
    ssxy = sum((x_vals[i] - x_bar) * (y_vals[i] - y_bar) for i in range(k))
    slope_full = ssxy / ssxx if ssxx > 0 else pooled_theta
    intercept_full = y_bar - slope_full * x_bar

    # Residuals from full regression line
    residuals = [y_vals[i] - (intercept_full + slope_full * x_vals[i]) for i in range(k)]
    outlier_mask = [abs(r) > 2.0 for r in residuals]

    lines = []
    lines.append(f'<rect width="{width}" height="{height}" fill="transparent"/>')

    # Grid
    n_xticks = 5
    for i in range(n_xticks + 1):
        xv = x_lo + (x_hi - x_lo) * i / n_xticks
        xx = xpx(xv)
        lines.append(
            f'<line x1="{xx:.1f}" y1="{PLOT_Y0}" x2="{xx:.1f}" y2="{PLOT_Y1}" '
            f'stroke="{_C_GRID}" stroke-width="0.5"/>'
        )
        lines.append(
            f'<text x="{xx:.1f}" y="{PLOT_Y1 + 15}" fill="{_C_TEXT}" '
            f'font-size="9" text-anchor="middle" opacity="0.7">{_fmt(xv, 1)}</text>'
        )
    n_yticks = 6
    for i in range(n_yticks + 1):
        yv = y_lo + (y_hi - y_lo) * i / n_yticks
        yy = ypx(yv)
        lines.append(
            f'<line x1="{PLOT_X0}" y1="{yy:.1f}" x2="{PLOT_X1}" y2="{yy:.1f}" '
            f'stroke="{_C_GRID}" stroke-width="0.5"/>'
        )
        lines.append(
            f'<text x="{PLOT_X0 - 5}" y="{yy + 4:.1f}" fill="{_C_TEXT}" '
            f'font-size="9" text-anchor="end" opacity="0.7">{_fmt(yv, 1)}</text>'
        )

    # Zero axes
    lines.append(
        f'<line x1="{PLOT_X0}" y1="{ypx(0):.1f}" x2="{PLOT_X1}" y2="{ypx(0):.1f}" '
        f'stroke="{_C_TEXT}" stroke-width="1" opacity="0.3"/>'
    )
    lines.append(
        f'<line x1="{xpx(0):.1f}" y1="{PLOT_Y0}" x2="{xpx(0):.1f}" y2="{PLOT_Y1}" '
        f'stroke="{_C_TEXT}" stroke-width="1" opacity="0.3"/>'
    )

    # Regression line (through origin, slope = pooled_theta)
    # Draw from x=0 to x_hi
    lines.append(
        f'<line x1="{xpx(x_lo):.1f}" y1="{ypx(pooled_theta * x_lo):.1f}" '
        f'x2="{xpx(x_hi):.1f}" y2="{ypx(pooled_theta * x_hi):.1f}" '
        f'stroke="{_C_ACCENT}" stroke-width="2"/>'
    )

    # ±2 confidence bands
    for band_sign in [+1, -1]:
        pts_lo_x = [xpx(x_lo), xpx(x_hi)]
        pts_lo_y = [
            ypx(pooled_theta * x_lo + band_sign * 2),
            ypx(pooled_theta * x_hi + band_sign * 2),
        ]
        lines.append(
            f'<line x1="{pts_lo_x[0]:.1f}" y1="{pts_lo_y[0]:.1f}" '
            f'x2="{pts_lo_x[1]:.1f}" y2="{pts_lo_y[1]:.1f}" '
            f'stroke="{_C_ACCENT}" stroke-width="1" stroke-dasharray="5,3" opacity="0.5"/>'
        )

    # Study points
    for i in range(k):
        cx = xpx(x_vals[i])
        cy = ypx(y_vals[i])
        fill = _C_DANGER if outlier_mask[i] else _C_ACCENT
        opacity = "0.8" if outlier_mask[i] else "0.75"
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{fill}" opacity="{opacity}" '
            f'aria-label="Study {i+1}: 1/SE={_fmt(x_vals[i], 2)}, '
            f'Z={_fmt(y_vals[i], 2)}{" (outlier)" if outlier_mask[i] else ""}"/>'
        )

    # Axes
    lines.append(
        f'<line x1="{PLOT_X0}" y1="{PLOT_Y0}" x2="{PLOT_X0}" y2="{PLOT_Y1}" '
        f'stroke="{_C_TEXT}" stroke-width="1" opacity="0.4"/>'
    )
    lines.append(
        f'<line x1="{PLOT_X0}" y1="{PLOT_Y1}" x2="{PLOT_X1}" y2="{PLOT_Y1}" '
        f'stroke="{_C_TEXT}" stroke-width="1" opacity="0.4"/>'
    )

    # Axis labels
    lines.append(
        f'<text x="{(PLOT_X0 + PLOT_X1) / 2:.0f}" y="{height - 10}" '
        f'fill="{_C_TEXT}" font-size="11" text-anchor="middle">Precision (1/SE)</text>'
    )
    lines.append(
        f'<text x="12" y="{(PLOT_Y0 + PLOT_Y1) / 2:.0f}" '
        f'fill="{_C_TEXT}" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90,12,{(PLOT_Y0 + PLOT_Y1) / 2:.0f})">Z-score (Effect/SE)</text>'
    )

    # Outlier count annotation
    n_out = sum(outlier_mask)
    if n_out > 0:
        lines.append(
            f'<text x="{PLOT_X1 - 5}" y="{PLOT_Y0 + 15}" '
            f'fill="{_C_DANGER}" font-size="10" text-anchor="end">'
            f'{n_out} outlier{"s" if n_out > 1 else ""} (|res|&gt;2)</text>'
        )

    inner = "\n  ".join(lines)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'role="img" '
        f'font-family="inherit">\n'
        f'  <title>Galbraith Radial Plot</title>\n'
        f'  <desc>Galbraith radial plot with {k} studies. '
        f'Points outside ±2 confidence bands are highlighted in red.</desc>\n'
        f'  {inner}\n'
        f'</svg>'
    )
    return svg
