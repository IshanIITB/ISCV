"""Per-camera floor-plane calibration, held-out validation and JSON persistence."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

def undistort_frame(frame: np.ndarray, camera_matrix: np.ndarray | None,
                    distortion: np.ndarray | None) -> np.ndarray:
    """Undistort frame when intrinsics are configured."""
    if camera_matrix is None or distortion is None:
        return frame
    return cv2.undistort(frame, camera_matrix, distortion)

def fit_homography(image_points: np.ndarray, floor_points_m: np.ndarray,
                   fit_indices: list[int], validation_indices: list[int]) -> tuple[np.ndarray, float]:
    """Fit with RANSAC and return H plus held-out RMSE in centimetres."""
    src = np.asarray(image_points, np.float32).reshape(-1,2)
    dst = np.asarray(floor_points_m, np.float32).reshape(-1,2)
    if len(src) < 8 or len(src) != len(dst):
        raise ValueError("Provide at least 8 corresponding image/floor points.")
    if not fit_indices or not validation_indices:
        raise ValueError("Both fit and held-out validation points are required.")
    H, _ = cv2.findHomography(src[fit_indices], dst[fit_indices], cv2.RANSAC, 3.0)
    if H is None:
        raise ValueError("Homography fitting failed.")
    projected = cv2.perspectiveTransform(src[validation_indices].reshape(-1,1,2), H).reshape(-1,2)
    errors_cm = np.linalg.norm(projected-dst[validation_indices], axis=1)*100
    return H, float(np.sqrt(np.mean(errors_cm**2)))

def acceptance_gate(error_cm: float, tolerance_cm: float = 30.0) -> tuple[bool,str]:
    """Reject calibration whose held-out error exceeds tolerance."""
    ok = np.isfinite(error_cm) and error_cm <= tolerance_cm
    return ok, (f"Calibration accepted: held-out RMSE {error_cm:.1f} cm." if ok
                else f"Calibration rejected: held-out RMSE {error_cm:.1f} cm exceeds {tolerance_cm:.1f} cm.")

def build_trust_mask(H: np.ndarray, image_shape: tuple[int,int], floor_bounds: tuple[float,float,float,float],
                     validation_error_cm: float, max_error_cm: float = 30.0,
                     cell_m: float = .1) -> dict:
    """Build a floor grid trust mask using validation error plus local mapping scale.

    This is a conservative heuristic, not a full uncertainty model. Jacobian scale
    is approximated numerically and used to estimate the floor footprint of a pixel.
    """
    xmin,ymin,xmax,ymax = floor_bounds
    xs = np.arange(xmin, xmax+cell_m, cell_m)
    ys = np.arange(ymin, ymax+cell_m, cell_m)
    Hinv = np.linalg.inv(H)
    grid, errors = [], []
    base_px = 2.0
    for y in ys:
        row, erow = [], []
        for x in xs:
            q = cv2.perspectiveTransform(np.array([[[x,y]]],np.float32), Hinv)[0,0]
            qx = cv2.perspectiveTransform(np.array([[[x+cell_m,y]]],np.float32), Hinv)[0,0]
            qy = cv2.perspectiveTransform(np.array([[[x,y+cell_m]]],np.float32), Hinv)[0,0]
            px_per_cell = max(np.linalg.norm(qx-q), np.linalg.norm(qy-q), 1e-6)
            scale_error_cm = base_px / px_per_cell * cell_m * 100
            err = float(validation_error_cm + scale_error_cm)
            row.append(bool(err <= max_error_cm))
            erow.append(err)
        grid.append(row); errors.append(erow)
    return {"x_m": xs.tolist(), "y_m": ys.tolist(), "trusted": grid, "estimated_error_cm": errors}

def save_calibration(path: str | Path, H: np.ndarray, error_cm: float, trust_mask: dict,
                     camera_matrix: np.ndarray | None = None, distortion: np.ndarray | None = None) -> None:
    """Save calibration parameters to JSON."""
    data = {"homography": np.asarray(H).tolist(), "heldout_rmse_cm": float(error_cm),
            "trust_mask": trust_mask,
            "camera_matrix": None if camera_matrix is None else np.asarray(camera_matrix).tolist(),
            "distortion": None if distortion is None else np.asarray(distortion).tolist()}
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_calibration(path: str | Path) -> dict:
    """Load JSON calibration."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["homography"] = np.asarray(data["homography"], dtype=float)
    if data.get("camera_matrix") is not None:
        data["camera_matrix"] = np.asarray(data["camera_matrix"], dtype=float)
    if data.get("distortion") is not None:
        data["distortion"] = np.asarray(data["distortion"], dtype=float)
    return data

def checkerboard_calibration(images: list[str], pattern_size: tuple[int,int], square_size_m: float) -> tuple[np.ndarray,np.ndarray]:
    """Optional checkerboard calibration helper returning camera matrix and distortion."""
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)*square_size_m
    objpoints, imgpoints = [], []
    image_size = None
    for path in images:
        img = cv2.imread(path)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            objpoints.append(objp)
            imgpoints.append(cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.001)))
    if len(objpoints) < 5 or image_size is None:
        raise ValueError("Need at least five images with a detected checkerboard.")
    _, matrix, dist, _, _ = cv2.calibrateCamera(objpoints,imgpoints,image_size,None,None)
    return matrix, dist



def is_in_trust_mask(
    x_m: float,
    y_m: float,
    trust_mask: dict | None,
) -> bool:
    """Return whether a floor coordinate lies in a trusted calibration cell."""
    if not trust_mask:
        return False

    try:
        xs = np.asarray(trust_mask["x_m"], dtype=float)
        ys = np.asarray(trust_mask["y_m"], dtype=float)
        trusted = np.asarray(trust_mask["trusted"], dtype=bool)

        if (
            xs.size == 0
            or ys.size == 0
            or trusted.shape != (ys.size, xs.size)
            or not np.isfinite(x_m)
            or not np.isfinite(y_m)
        ):
            return False

        # Use the nearest grid coordinate, but reject points outside the grid.
        if x_m < xs.min() or x_m > xs.max():
            return False
        if y_m < ys.min() or y_m > ys.max():
            return False

        xi = int(np.argmin(np.abs(xs - x_m)))
        yi = int(np.argmin(np.abs(ys - y_m)))

        return bool(trusted[yi, xi])

    except (KeyError, TypeError, ValueError):
        return False