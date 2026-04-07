"""
Effect size conversion from clinical trial outcome measures.

Supports: Hazard Ratio, Odds Ratio, Risk Ratio, Mean Difference,
Standardized Mean Difference.
"""
import math
from scipy import stats


def hr_to_yi_sei(hr, ci_lo, ci_hi, conf_level=0.95):
    """Hazard Ratio -> log(HR) with SE from CI."""
    yi = math.log(hr)
    z = stats.norm.ppf(1 - (1 - conf_level) / 2)
    sei = (math.log(ci_hi) - math.log(ci_lo)) / (2 * z)
    return yi, sei


def or_to_yi_sei(or_val, ci_lo, ci_hi, conf_level=0.95):
    """Odds Ratio -> log(OR) with SE from CI."""
    yi = math.log(or_val)
    z = stats.norm.ppf(1 - (1 - conf_level) / 2)
    sei = (math.log(ci_hi) - math.log(ci_lo)) / (2 * z)
    return yi, sei


def rr_to_yi_sei(rr, ci_lo, ci_hi, conf_level=0.95):
    """Risk Ratio -> log(RR) with SE from CI."""
    yi = math.log(rr)
    z = stats.norm.ppf(1 - (1 - conf_level) / 2)
    sei = (math.log(ci_hi) - math.log(ci_lo)) / (2 * z)
    return yi, sei


def md_to_yi_sei(md, ci_lo, ci_hi, conf_level=0.95):
    """Mean Difference -> yi=MD, SE from CI width."""
    yi = md
    z = stats.norm.ppf(1 - (1 - conf_level) / 2)
    sei = (ci_hi - ci_lo) / (2 * z)
    return yi, sei


def counts_to_yi_sei(events_t, total_t, events_c, total_c):
    """2x2 table -> log(OR) with SE via Woolf's method."""
    a, b = events_t, total_t - events_t
    c, d = events_c, total_c - events_c
    if any(x == 0 for x in [a, b, c, d]):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    yi = math.log(a * d / (b * c))
    sei = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return yi, sei


def parse_effect(measure_type, value, ci_lo, ci_hi, **kwargs):
    """Dispatch to correct converter based on measure type string."""
    measure_type = measure_type.upper().strip()
    if measure_type in ("HR", "HAZARD RATIO"):
        return hr_to_yi_sei(value, ci_lo, ci_hi)
    elif measure_type in ("OR", "ODDS RATIO"):
        return or_to_yi_sei(value, ci_lo, ci_hi)
    elif measure_type in ("RR", "RISK RATIO", "RELATIVE RISK"):
        return rr_to_yi_sei(value, ci_lo, ci_hi)
    elif measure_type in ("MD", "MEAN DIFFERENCE", "DIFFERENCE"):
        return md_to_yi_sei(value, ci_lo, ci_hi)
    else:
        return None, None  # Unknown type
