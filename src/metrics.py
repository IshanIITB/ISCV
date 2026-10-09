"""Compute separate presence, stop, queue, occupancy and heat-map metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

REQUIRED = {"timestamp","track_id","x_m","y_m","zone","speed_mps","low_quality","in_trust_mask"}

def compute_metrics(observations: pd.DataFrame, zone_areas: dict[str, float] | None = None,
                    observation_seconds: float | None = None, stop_speed_mps: float = .3,
                    min_stop_seconds: float = 3.0, grid_resolution_m: float = .1,
                    heat_sigma_cells: float = 2.0) -> dict:
    """Compute metrics from timestamped track observations. Durations are estimates."""
    missing = REQUIRED - set(observations.columns)
    if missing:
        raise ValueError(f"Missing observation columns: {sorted(missing)}")
    df = observations.copy()
    if df.empty:
        return {"zone_metrics": [], "heatmap": [], "summary": {"status": "insufficient_data"}}
    df = df.sort_values(["track_id","timestamp"])
    df["low_quality"] = df["low_quality"].fillna(True).astype(bool)
    df["in_trust_mask"] = df["in_trust_mask"].fillna(False).astype(bool)
    # Approximate each observation's contribution using the next sample interval,
    # capped to avoid counting long gaps as continuous presence.
    diffs = df.groupby("track_id")["timestamp"].diff().shift(-1)
    df["dt"] = diffs.clip(lower=0, upper=1.0).fillna(0)
    df.loc[~df["in_trust_mask"], "dt"] = 0
    df.loc[df["zone"].isin(["uncertain","unassigned"]), "dt"] = 0
    df["stop"] = df["speed_mps"].fillna(np.inf) < stop_speed_mps
    # Require a continuous-enough stop run to exceed min_stop_seconds.
    stop_seconds = {}
    for (tid, zone), g in df.groupby(["track_id","zone"], sort=False):
        run, total = 0.0, 0.0
        prev_t = None
        for r in g.itertuples():
            dt = float(r.dt)
            if bool(r.stop) and not bool(r.low_quality) and bool(r.in_trust_mask):
                run += dt
                if run > min_stop_seconds:
                    total += dt
            else:
                run = 0.0
            prev_t = r.timestamp
        stop_seconds[(tid, zone)] = total
    rows = []
    for zone, g in df.groupby("zone"):
        if zone in ("uncertain","unassigned"):
            continue
        presence = float(g["dt"].sum())
        stops = float(sum(v for (tid,z),v in stop_seconds.items() if z == zone))
        quality = float(g["low_quality"].mean()) if len(g) else 1.0
        uncertain = float((~g["in_trust_mask"]).mean()) if len(g) else 1.0
        area = max(float((zone_areas or {}).get(zone, 1.0)), 1e-6)
        window = max(float(observation_seconds or (df["timestamp"].max()-df["timestamp"].min())), 1.0)
        trusted = g[g["in_trust_mask"] & ~g["low_quality"]]
        rows.append({
            "zone": zone, "presence_seconds": presence, "stationary_stop_seconds": stops,
            "queue_wait_seconds": presence if zone.lower().find("queue") >= 0 else 0.0,
            "track_count_approx": int(g["track_id"].nunique()),
            "density_people_per_m2": float(trusted["track_id"].nunique()/area),
            "n_detections": int(len(g)), "low_quality_fraction": quality,
            "uncertain_fraction": uncertain, "area_m2": area,
            "presence_person_seconds_per_m2_per_hour": presence/area/window*3600,
        })
    # Presence heatmap in floor coordinates, normalized by duration; cells use 2D histogram.
    trusted = df[df["in_trust_mask"] & ~df["low_quality"] & ~df["zone"].isin(["uncertain","unassigned"])]
    if len(trusted):
        xmin, xmax = float(trusted.x_m.min()), float(trusted.x_m.max()) + grid_resolution_m
        ymin, ymax = float(trusted.y_m.min()), float(trusted.y_m.max()) + grid_resolution_m
        xedges = np.arange(xmin, xmax + grid_resolution_m, grid_resolution_m)
        yedges = np.arange(ymin, ymax + grid_resolution_m, grid_resolution_m)
        hist, _, _ = np.histogram2d(trusted.x_m, trusted.y_m, bins=[xedges,yedges], weights=trusted.get("dt", 0))
        smooth = gaussian_filter(hist, sigma=heat_sigma_cells)
        duration = max(float(observation_seconds or (df.timestamp.max()-df.timestamp.min())), 1.0)
        smooth = smooth / (duration * grid_resolution_m**2) * 3600
        heatmap = [{"x_m": float((xedges[i]+xedges[i+1])/2),
                    "y_m": float((yedges[j]+yedges[j+1])/2), "value": float(smooth[i,j])}
                   for i,j in zip(*np.nonzero(smooth > 0))]
    else:
        heatmap = []
    return {"zone_metrics": rows, "heatmap": heatmap,
            "summary": {"status": "ok", "n_observations": int(len(df)),
                        "n_tracks_approx": int(df.track_id.nunique()),
                        "low_quality_fraction": float(df.low_quality.mean()),
                        "observation_seconds": float(observation_seconds or (df.timestamp.max()-df.timestamp.min()))}}
