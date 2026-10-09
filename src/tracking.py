"""Single-camera ByteTrack wrapper with lightweight track quality statistics."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import supervision as sv

class PersonTracker:
    """Track detections within one camera; IDs are session-scoped."""
    def __init__(self, minimum_track_length: int = 8, maximum_gap: int = 15):
        self.tracker = sv.ByteTrack()
        self.minimum_track_length, self.maximum_gap = minimum_track_length, maximum_gap
        self.stats = defaultdict(lambda: {"length":0, "conf_sum":0.0, "low":0, "gaps":0, "last_frame":None})

    def update(self, detections: sv.Detections, frame_index: int) -> sv.Detections:
        """Update tracker and annotate per-track quality metadata in self.stats."""
        tracked = self.tracker.update_with_detections(detections)
        if tracked.tracker_id is None:
            return tracked
        confidences = tracked.confidence if tracked.confidence is not None else np.ones(len(tracked))
        for tid, conf in zip(tracked.tracker_id, confidences):
            s = self.stats[int(tid)]
            if s["last_frame"] is not None and frame_index - s["last_frame"] > 1:
                s["gaps"] += frame_index - s["last_frame"] - 1
            s["length"] += 1
            s["conf_sum"] += float(conf)
            s["low"] += int(float(conf) < .4)
            s["last_frame"] = frame_index
        return tracked

    def quality(self, track_id: int) -> dict:
        """Return quality summary for one ID."""
        s = self.stats[int(track_id)]
        return {"length": s["length"], "mean_confidence": s["conf_sum"]/max(s["length"],1),
                "low_quality_fraction": s["low"]/max(s["length"],1), "gaps": s["gaps"],
                "low_confidence_fragment": s["length"] < self.minimum_track_length or s["gaps"] > self.maximum_gap}
