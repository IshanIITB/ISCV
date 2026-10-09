"""CLI pipeline for recorded camera video.

Normal mode:
    python -m src.pipeline --video videos/test.mp4 --config config/camera_01.yaml --output outputs/

Detection-only mode (no calibration required):
    python -m src.pipeline --video videos/test.mp4 --config config/camera_01.yaml --output outputs/ --detection-only
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from .calibration import load_calibration, undistort_frame, acceptance_gate
from .detection import PersonDetector
from .tracking import PersonTracker
from .floor_position import proxy_point, map_to_floor, detection_quality
from .smoothing import smooth_and_speed
from .zones import assign_zone
from .metrics import compute_metrics
from .flagging import bottleneck_flags
from .storage import save_outputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return config


def run_detection_only(
    video: str,
    config: str,
    output: str,
    max_frames: int | None = None,
    stride_override: int | None = None,
) -> None:
    """Detect and track people, write an annotated preview, and skip calibration."""
    cfg = _load_config(config)
    video_path = Path(video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path.resolve()}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    stride = stride_override or int(cfg.get("detection", {}).get("stride", 2))
    if stride < 1:
        cap.release()
        raise ValueError("--stride must be at least 1")

    detector_cfg = dict(cfg.get("detection", {}).get("model", {}))
    # The detector wrapper accepts model constructor settings. Use YAML defaults
    # if values are present there; keep top-level confidence as a fallback.
    if "confidence" not in detector_cfg:
        detector_cfg["confidence"] = cfg.get("detection", {}).get("confidence", 0.35)
    detector = PersonDetector(**detector_cfg)
    tracker = PersonTracker(**cfg.get("tracking", {}))

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = out_dir / "detection_preview.mp4"

    writer = None
    frame_index = 0
    sampled_frames = 0
    boxes_seen = 0
    unique_ids: set[int] = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            current_index = frame_index
            frame_index += 1
            if current_index % stride != 0:
                continue

            detections = detector.detect(frame)
            tracked = tracker.update(detections, current_index)

            boxes = (
                tracked.xyxy
                if tracked.xyxy is not None
                else np.empty((0, 4), dtype=float)
            )
            ids = (
                tracked.tracker_id
                if tracked.tracker_id is not None
                else np.full(len(boxes), -1, dtype=int)
            )
            confidences = (
                tracked.confidence
                if tracked.confidence is not None
                else np.ones(len(boxes), dtype=float)
            )

            preview = frame.copy()
            for box, track_id, confidence in zip(boxes, ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                valid_id = int(track_id)
                label = f"ID {valid_id} ({float(confidence):.2f})" if valid_id >= 0 else f"Person ({float(confidence):.2f})"
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(
                    preview,
                    label,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
                if valid_id >= 0:
                    unique_ids.add(valid_id)

            if writer is None:
                height, width = preview.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(fps / stride, 1.0),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not create output video: {output_video.resolve()}"
                    )

            writer.write(preview)
            sampled_frames += 1
            boxes_seen += len(boxes)

            if sampled_frames % 50 == 0:
                log.info(
                    "Sampled frames: %d | boxes observed: %d",
                    sampled_frames,
                    boxes_seen,
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if sampled_frames == 0:
        raise RuntimeError("No frames were processed. Check the video and --max-frames.")

    log.info("Detection-only test completed.")
    log.info("Sampled frames: %d", sampled_frames)
    log.info("Person boxes observed across sampled frames: %d", boxes_seen)
    log.info("Distinct tracker IDs observed: %d", len(unique_ids))
    log.info("Annotated preview: %s", output_video.resolve())
    log.warning(
        "No calibration was used. This mode does not produce valid floor positions, "
        "dwell-time, zone-density, or bottleneck metrics."
    )


def run(
    video: str,
    config: str,
    output: str,
    max_frames: int | None = None,
    stride_override: int | None = None,
    debug_video: bool = False,
) -> None:
    """Run the calibrated video pipeline and save CSV/JSON outputs."""
    cfg = _load_config(config)

    calibration_path = Path(cfg["calibration"]["json_path"])
    if not calibration_path.is_absolute():
        calibration_path = Path.cwd() / calibration_path
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_path.resolve()}. "
            "Create a real camera calibration first, or run with --detection-only."
        )

    calib = load_calibration(str(calibration_path))
    ok, message = acceptance_gate(
        calib["heldout_rmse_cm"],
        cfg["calibration"].get("max_error_cm", 30),
    )
    log.info(message)
    if not ok:
        raise RuntimeError(message)

    video_path = Path(video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path.resolve()}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    stride = stride_override or int(cfg.get("detection", {}).get("stride", 2))
    if stride < 1:
        cap.release()
        raise ValueError("--stride must be at least 1")

    detector = PersonDetector(**cfg["detection"].get("model", {}))
    tracker = PersonTracker(**cfg.get("tracking", {}))

    from shapely.geometry import Polygon
    from .zones import Zone

    zones = [
        Zone(
            z["name"],
            z.get("type", "browse"),
            Polygon(z["polygon"]),
        )
        for z in cfg["zones"]
    ]
    homography = calib["homography"]
    rows: list[dict] = []
    frame_index = 0
    sampled_frames = 0
    writer = None
    out_dir = Path(output)
    if debug_video:
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            current_index = frame_index
            frame_index += 1
            if current_index % stride != 0:
                continue

            frame = undistort_frame(
                frame,
                calib.get("camera_matrix"),
                calib.get("distortion"),
            )
            detections = detector.detect(frame)
            tracked = tracker.update(detections, current_index)

            boxes = (
                tracked.xyxy
                if tracked.xyxy is not None
                else np.empty((0, 4), dtype=float)
            )
            ids = (
                tracked.tracker_id
                if tracked.tracker_id is not None
                else np.full(len(boxes), -1, dtype=int)
            )
            confidences = (
                tracked.confidence
                if tracked.confidence is not None
                else np.ones(len(boxes), dtype=float)
            )

            for box, track_id, confidence in zip(boxes, ids, confidences):
                tid = int(track_id)
                if tid < 0:
                    continue
                point = proxy_point(
                    box, cfg.get("camera", {}).get("foot_point_proxy", "bottom_center")
                )
                floor_x, floor_y = map_to_floor(point, homography)
                quality = detection_quality(
                    box,
                    float(confidence),
                    frame.shape,
                    True,
                    cfg.get("detection", {}).get("confidence", 0.35),
                )
                zone = assign_zone(
                    floor_x,
                    floor_y,
                    zones,
                    cfg.get("zone_settings", {}).get("buffer_m", 0.15),
                )
                rows.append(
                    {
                        "timestamp": current_index / fps,
                        "frame_index": current_index,
                        "track_id": tid,
                        "camera_id": cfg.get("camera", {}).get("id", "camera_01"),
                        "x_m": floor_x,
                        "y_m": floor_y,
                        "zone": zone,
                        "confidence": float(confidence),
                        "low_quality": bool(
                            quality
                            or tracker.quality(tid).get("low_confidence_fragment", False)
                        ),
                        # This prototype does not yet query the spatial trust mask.
                        "in_trust_mask": False,
                        "speed_mps": np.nan,
                    }
                )

            sampled_frames += 1
            if debug_video:
                preview = frame.copy()
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        preview,
                        str(int(track_id)),
                        (x1, max(y1 - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                if writer is None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(out_dir / "debug.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(fps / stride, 1.0),
                        (preview.shape[1], preview.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Could not create debug video.")
                writer.write(preview)

            if sampled_frames % 100 == 0:
                log.info("Processed %d sampled frames", sampled_frames)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No tracked person detections. Check the video, weights, confidence, and tracker."
        )

    smoothed = [
        smooth_and_speed(group, cfg.get("motion", {}).get("window_seconds", 1.5))
        for _, group in df.groupby("track_id")
    ]
    df = pd.concat(smoothed, ignore_index=True)

    area = {zone.name: float(zone.polygon.area) for zone in zones}
    duration = max(float(df["timestamp"].max() - df["timestamp"].min()), 1.0)
    metrics = compute_metrics(
        df,
        area,
        duration,
        cfg.get("motion", {}).get("stop_speed_mps", 0.3),
        cfg.get("motion", {}).get("min_stop_seconds", 3.0),
    )

    for row in metrics["zone_metrics"]:
        values = df.loc[
            (df["zone"] == row["zone"]) & df["speed_mps"].notna(),
            "speed_mps",
        ]
        row["median_speed_mps"] = (
            float(values.median()) if len(values) else float("inf")
        )

    metrics["bottleneck_flags"] = bottleneck_flags(
        pd.DataFrame(metrics["zone_metrics"]),
        density_threshold=cfg.get("flags", {}).get("density_threshold", 1.0),
        speed_threshold_mps=cfg.get("flags", {}).get("speed_threshold_mps", 0.25),
        persistence_seconds=cfg.get("flags", {}).get("persistence_seconds", 180),
    )
    save_outputs(df, metrics, output)
    log.info("Saved outputs to %s", output)
    log.warning(
        "Trust-mask integration is incomplete in this prototype. Treat calibrated "
        "zone metrics as experimental until that validation is implemented."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dwell-Time & Bottleneck Mapper")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--config", required=True, help="Camera YAML configuration")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--max-frames", type=int, help="Maximum source frames to read")
    parser.add_argument("--stride", type=int, help="Process every Nth frame")
    parser.add_argument("--debug-video", action="store_true", help="Save annotated video in calibrated mode")
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Test person detection/tracking without calibration",
    )
    args = parser.parse_args()

    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.stride is not None and args.stride < 1:
        parser.error("--stride must be positive")

    if args.detection_only:
        run_detection_only(
            args.video, args.config, args.output, args.max_frames, args.stride
        )
    else:
        run(
            args.video, args.config, args.output,
            args.max_frames, args.stride, args.debug_video
        )


if __name__ == "__main__":
    main()
