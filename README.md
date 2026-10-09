# Dwell-Time & Bottleneck Mapper (V1 prototype)

A prototype for estimating zone-level customer presence, stationary stops, heat maps and potential congestion from one recorded CCTV feed. It treats detections as noisy measurements and reports caveats.

## Scope and limitations
- Single camera only; no cross-camera journey stitching.
- Pretrained YOLO person detection and ByteTrack.
- Calibration and zones must be configured for the real camera and store.
- The included calibration JSON is intentionally not fabricated: create a valid one from measured image/floor correspondences before running real video.
- `in_trust_mask` is currently a conservative pipeline placeholder; connect the calibration trust grid to each projected point before using results operationally.
- Bottleneck persistence in this prototype is configured from observed duration supplied to the rule engine; a production version should maintain time-window state.
- Heat map and synthetic demo are illustrative, not business evidence.
- No face recognition. Keep track IDs session-scoped and delete raw footage when permitted.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Test the core logic
```bash
pytest -q
```

## Run the dashboard on synthetic data
From the project root:
```bash
streamlit run app/dashboard.py
```
Choose **Synthetic demo** in the sidebar.

## Run a recorded video
1. Create a calibration JSON using measured points. `src.calibration.fit_homography` takes image points, floor points in metres, fit indices and held-out validation indices.
2. Save the homography and validation error with `save_calibration`.
3. Edit `config/camera_01.yaml`: camera settings, real zone polygons and thresholds.
4. Run:
```bash
python -m src.pipeline --video path/to/video.mp4 --config config/camera_01.yaml --output outputs --debug-video
```
Use `--max-frames 1000` for a quick smoke test and `--stride 3` to process every third frame.

## Validation footage and labels
- Record short clips from the actual camera view during quiet and crowded periods, with carts, occlusions and varied lighting.
- Choose 8–12 visible floor points distributed across the image. Measure their corresponding floor-plan coordinates in metres. Fit the homography on some points and reserve others for validation.
- Hand-label the visible foot/floor point for about 100–200 frames per camera.
- Mark zone entry/exit and stationary stops on a short clip. Compare predictions to labels and report calibration error, detection precision/recall, tracking ID switches and stop-detection errors.
- Recalibrate after camera movement, zoom changes or significant layout changes.

## Important metric definitions
- Presence time is not customer interest.
- Stationary stops can include waiting or chatting.
- Track counts are approximate and can be inflated by ID switches.
- A high-presence zone may indicate congestion, not engagement.
- A candidate dead zone requires like-for-like historical data and manual context review.
