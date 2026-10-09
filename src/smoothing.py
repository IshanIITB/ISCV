"""Smooth track positions and estimate speed over a temporal window."""
from __future__ import annotations
import numpy as np
import pandas as pd

def smooth_and_speed(track: pd.DataFrame, window_seconds: float = 1.5) -> pd.DataFrame:
    """Add smoothed coordinates and speed_mps; expects timestamp, x_m, y_m, in_trust_mask."""
    if track.empty:
        out = track.copy()
        out["speed_mps"] = []
        return out
    out = track.sort_values("timestamp").copy()
    # Time-aware rolling median suppresses jitter while retaining stops.
    indexed = out.set_index(pd.to_timedelta(out["timestamp"], unit="s"))
    indexed["x_smooth"] = indexed["x_m"].rolling(f"{window_seconds}s", min_periods=1).median()
    indexed["y_smooth"] = indexed["y_m"].rolling(f"{window_seconds}s", min_periods=1).median()
    out["x_smooth"] = indexed["x_smooth"].to_numpy()
    out["y_smooth"] = indexed["y_smooth"].to_numpy()
    dt = out["timestamp"].diff()
    dx, dy = out["x_smooth"].diff(), out["y_smooth"].diff()
    speed = np.sqrt(dx**2+dy**2) / dt.replace(0, np.nan)
    # The first row and out-of-trust positions are deliberately unknown.
    out["speed_mps"] = speed.where(out["in_trust_mask"].astype(bool), np.nan)
    out.loc[dt > max(window_seconds*2, 3.0), "speed_mps"] = np.nan
    return out
