"""Generate deterministic synthetic zone-level observations for development."""
from __future__ import annotations
import numpy as np
import pandas as pd

def generate_synthetic(seed: int = 7, minutes: int = 30, fps: int = 2) -> tuple[pd.DataFrame, dict]:
    """Return synthetic observations and zone metadata; no video model required."""
    rng = np.random.default_rng(seed)
    zones = {
        "entrance": {"type": "entrance", "polygon": [(0,0),(4,0),(4,3),(0,3)]},
        "browse": {"type": "browse", "polygon": [(4,0),(8,0),(8,4),(4,4)]},
        "queue": {"type": "queue", "polygon": [(0,3),(4,3),(4,6),(0,6)]},
        "dead_zone": {"type": "browse", "polygon": [(8,0),(12,0),(12,4),(8,4)]},
    }
    rows, track = [], 0
    end = minutes * 60
    # Arrivals produce short journeys; a few linger in browse and queue.
    for start in np.arange(0, end, rng.uniform(3, 8)):
        track += 1
        path = rng.choice(["quick", "browse", "queue"], p=[.48,.32,.20])
        t = float(start)
        if path == "quick":
            sequence = [("entrance", rng.uniform(2,6)), ("browse", rng.uniform(2,5))]
        elif path == "browse":
            sequence = [("entrance", rng.uniform(2,5)), ("browse", rng.uniform(18,55))]
        else:
            sequence = [("entrance", rng.uniform(2,4)), ("queue", rng.uniform(25,100))]
        for zone, duration in sequence:
            stop = path == "browse" and zone == "browse" or path == "queue" and zone == "queue"
            n = max(1, int(duration * fps))
            base_x, base_y = {
                "entrance": (2,1.5), "browse": (6,2), "queue": (2,4.5)
            }[zone]
            for i in range(n):
                ts = t + i / fps
                jitter = .12 if stop else .55
                x, y = base_x + rng.normal(0,jitter), base_y + rng.normal(0,jitter)
                rows.append({
                    "timestamp": ts, "track_id": track, "camera_id": "synthetic_01",
                    "x_m": x, "y_m": y, "zone": zone, "speed_mps": abs(rng.normal(.08 if stop else .55,.05)),
                    "confidence": float(rng.uniform(.65,.99)),
                    "low_quality": bool(rng.random() < .05),
                    "in_trust_mask": True,
                })
            t += duration
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    # Inject gaps/quality issues and a few synthetic ID fragments.
    if len(df):
        bad = rng.choice(df.index, size=max(1, int(.025*len(df))), replace=False)
        df.loc[bad, "low_quality"] = True
        switch_rows = df.index[df.index % 211 == 0]
        df.loc[switch_rows, "track_id"] = df.loc[switch_rows, "track_id"] + 10000
    return df, zones
