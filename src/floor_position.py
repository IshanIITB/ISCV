"""Map person bounding boxes to calibrated floor coordinates."""
from __future__ import annotations
import cv2
import numpy as np

def proxy_point(box: np.ndarray, proxy: str = "bottom_center") -> tuple[float,float]:
    """Estimate floor point from xyxy box; proxy is camera-specific."""
    x1,y1,x2,y2 = map(float, box)
    if proxy == "box_center":
        return ((x1+x2)/2, (y1+y2)/2)
    if proxy != "bottom_center":
        raise ValueError("proxy must be bottom_center or box_center")
    return ((x1+x2)/2, y2)

def map_to_floor(point: tuple[float,float], homography: np.ndarray) -> tuple[float,float]:
    """Transform an image point into floor-plan metres."""
    p = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, homography)[0,0]
    return float(q[0]), float(q[1])

def detection_quality(box: np.ndarray, confidence: float, frame_shape: tuple[int,...],
                      trust_mask_contains: bool, confidence_threshold: float = .35,
                      exclusion_overlap: bool = False) -> bool:
    """Return True when a detection should be marked low-quality."""
    x1,y1,x2,y2 = map(float, box)
    h,w = frame_shape[:2]
    bw,bh = max(x2-x1,1), max(y2-y1,1)
    aspect = bh/bw
    return bool(y2 >= h-2 or aspect < .7 or aspect > 6.5 or confidence < confidence_threshold
                or not trust_mask_contains or exclusion_overlap)
