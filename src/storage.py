"""Persistence helpers: aggregate metrics by default; raw video deletion is opt-in."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def save_outputs(observations: pd.DataFrame, metrics: dict, output_dir: str | Path,
                 delete_source_video: str | None = None) -> None:
    """Save observations and metrics. Track IDs are session-scoped and should not be reused."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    observations.to_csv(out / "observations.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False, default=float), encoding="utf-8")
    if delete_source_video:
        path = Path(delete_source_video)
        if path.exists():
            path.unlink()
