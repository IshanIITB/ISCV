"""Zone loading and uncertainty-aware point assignment."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from shapely.geometry import Point, Polygon

@dataclass
class Zone:
    name: str
    kind: str
    polygon: Polygon

def load_zones(path: str | Path) -> list[Zone]:
    """Load zone polygons from YAML, using coordinates in metres."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Zone(z["name"], z.get("type", "browse"), Polygon(z["polygon"])) for z in data["zones"]]

def zones_from_mapping(mapping: dict[str, dict[str, Any]]) -> list[Zone]:
    """Convert the synthetic zone mapping to Zone objects."""
    return [Zone(name, meta.get("type", "browse"), Polygon(meta["polygon"])) for name, meta in mapping.items()]

def assign_zone(x: float, y: float, zones: list[Zone], buffer_m: float = 0.15) -> str:
    """Return zone name, 'uncertain' near boundaries, or 'unassigned'."""
    point = Point(float(x), float(y))
    for zone in zones:
        if zone.polygon.boundary.distance(point) <= buffer_m and zone.polygon.buffer(buffer_m).covers(point):
            return "uncertain"
        if zone.polygon.covers(point):
            return zone.name
    return "unassigned"
