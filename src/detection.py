"""Ultralytics YOLO person-only detector."""
from __future__ import annotations
import logging
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
log = logging.getLogger(__name__)

class PersonDetector:
    """Detect person class only and return supervision detections."""
    def __init__(self, weights: str = "yolo11n.pt", confidence: float = .35,
                 image_size: int = 640, device: str | None = None):
        self.model = YOLO(weights)
        self.confidence, self.image_size, self.device = confidence, image_size, device

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run inference on one BGR frame."""
        kwargs = {"conf": self.confidence, "imgsz": self.image_size, "classes": [0], "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        result = self.model.predict(frame, **kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return sv.Detections.empty()
        xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = result.boxes.conf.cpu().numpy().astype(np.float32)
        class_id = result.boxes.cls.cpu().numpy().astype(int)
        return sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)
