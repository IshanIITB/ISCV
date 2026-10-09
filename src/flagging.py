"""Transparent rule-based flags with explicit evidence and baseline checks."""
from __future__ import annotations
import pandas as pd

def bottleneck_flags(zone_metrics: pd.DataFrame, *, density_threshold: float = 1.0,
                     speed_threshold_mps: float = .25, persistence_seconds: float = 180,
                     observed_persistence: dict[str,float] | None = None,
                     max_low_quality_fraction: float = .25) -> list[dict]:
    """Flag zones only when density, speed, persistence and data quality pass."""
    flags = []
    if zone_metrics.empty:
        return flags
    for row in zone_metrics.to_dict("records"):
        zone = row["zone"]
        duration = float((observed_persistence or {}).get(zone, 0.0))
        density = float(row.get("density_people_per_m2", 0))
        speed = float(row.get("median_speed_mps", float("inf")))
        quality = float(row.get("low_quality_fraction", 1))
        if (density >= density_threshold and speed <= speed_threshold_mps and
                duration >= persistence_seconds and quality <= max_low_quality_fraction):
            flags.append({"zone": zone, "type": "bottleneck", "density": density,
                          "median_speed_mps": speed, "duration_seconds": duration,
                          "low_quality_fraction": quality, "status": "flagged"})
    return flags

def compare_zone_baseline(current: pd.DataFrame, baseline: pd.DataFrame,
                          metric: str = "track_count_approx", low_percentile: float = .2,
                          min_baseline_rows: int = 5) -> list[dict]:
    """Flag candidates only with enough like-for-like baseline samples."""
    flags = []
    if baseline.empty or len(baseline) < min_baseline_rows or metric not in baseline:
        return [{"status": "insufficient_data", "reason": "not enough baseline observations"}]
    cutoff = float(baseline[metric].quantile(low_percentile))
    for row in current.to_dict("records"):
        value = float(row.get(metric, 0))
        if value <= cutoff:
            flags.append({"zone": row.get("zone"), "type": "candidate dead zone: needs review",
                          "metric": metric, "value": value, "baseline_cutoff": cutoff,
                          "status": "needs_review"})
    return flags

def hot_zone_flags(current: pd.DataFrame, baseline: pd.DataFrame,
                   metric: str = "stationary_stop_seconds", high_percentile: float = .9,
                   min_baseline_rows: int = 5) -> list[dict]:
    """Flag high stop-time candidates; high presence alone is not interpreted as interest."""
    if baseline.empty or len(baseline) < min_baseline_rows or metric not in baseline:
        return [{"status": "insufficient_data", "reason": "not enough baseline observations"}]
    cutoff = float(baseline[metric].quantile(high_percentile))
    return [{"zone": r.get("zone"), "type": "hot-zone candidate",
             "metric": metric, "value": float(r.get(metric,0)), "baseline_cutoff": cutoff,
             "status": "needs_review"}
            for r in current.to_dict("records") if float(r.get(metric,0)) > cutoff]
