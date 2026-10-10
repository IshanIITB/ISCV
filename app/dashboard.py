import hashlib
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from scipy.ndimage import gaussian_filter, label as nd_label
from PIL import Image
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="StoreFlow | Dwell-Time Analytics", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
.stApp { background: #f7f3e9; color: #292720; }
[data-testid="stSidebar"] { background: #eee7d8; }
[data-testid="stMetric"] { background: #fffdf7; border: 1px solid #e5dece;
    padding: 14px; border-radius: 12px; }
.block-container { padding-top: 1.4rem; max-width: 1800px; }
h1, h2, h3 { color: #292720; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
STILL_SPEED = 0.03   # frame-widths/s below which a person counts as stationary
MIN_STOP_S = 3.0     # stationary seconds needed to count as a "stop"
BIN_S = 5            # seconds per occupancy bin
OTHER = "Open floor"
MAX_IMG_SIDE = 768   # images sent to the vision model are shrunk to this (keeps CPU runs feasible)

VLM_MODELS = {
    "SmolVLM-500M (fastest, weakest)": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "Qwen2-VL-2B (balanced)": "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2.5-VL-3B (best, slow on CPU)": "Qwen/Qwen2.5-VL-3B-Instruct",
}

STATUS_COLOR = {
    "Congestion candidate": "#c0392b",
    "Dead-zone candidate": "#7f8c8d",
    "Hot / high engagement": "#d68910",
    "High traffic, few stops": "#2471a3",
}

VLM_PROMPT = """You are a retail store-layout analyst. You are given:
IMAGE 1: a heat map of where shoppers were observed (red/orange = more presence, transparent = little or none). Coloured rectangles are store areas: red = congestion candidate, grey = dead-zone candidate, orange = high engagement, blue = high traffic but few stops.
{frame_line}
Below are the measured numbers from people tracking. Use them together with what you SEE in the images. Never invent numbers.

{facts}

Write a clear report in plain text with these sections:
1. WHAT THE HEAT MAP SHOWS: describe in simple words where customers actually go, which areas are busy, which are empty, and which routes are visible.
2. WHAT IS HAPPENING AND WHY: explain the patterns. Start each statement with "Observed:" (visible in the images or numbers) or "Hypothesis:" (a possible reason that must be checked in the store).
3. PROBLEMS FOUND: queues, crowding, ignored areas, wasted space, dead ends.
4. HOW TO IMPROVE THE STORE: five concrete actions. For each one give the metric that would show it worked.
5. WHAT TO CHECK BEFORE ACTING: data limits (tracking errors, no validation, synthetic data, camera coverage).
Be specific and brief. If the data is synthetic or not validated, say so at the start."""

# =========================================================
# SIDEBAR (two tabs)
# =========================================================
st.sidebar.title("🗺️ StoreFlow")
page = st.sidebar.radio("Mode", ["📹 Recorded video", "🧪 Synthetic demo"], label_visibility="collapsed")
synthetic_mode = page.startswith("🧪")

with st.sidebar.expander("Analysis settings", expanded=False):
    min_visit = st.slider("Min. seconds to count as a visit", 1, 10, 3)
    occ_thr = st.slider("Congestion: people at once", 2, 12, 3)
    persist_s = st.slider("Congestion: must last (s)", 10, 180, 30)
    show_paths = st.checkbox("Show movement paths on heat map", value=False)
    show_labels = st.checkbox("Show area labels", value=True)

with st.sidebar.expander("AI vision analyst", expanded=False):
    vlm_label = st.selectbox("Hugging Face vision model", list(VLM_MODELS), index=1)
    vlm_name = VLM_MODELS[vlm_label]

uploaded_video = None
if not synthetic_mode:
    uploaded_video = st.sidebar.file_uploader("Upload store video", type=["mp4", "avi", "mov", "mkv"])

st.title("StoreFlow")
st.caption("Dwell-Time & Bottleneck Mapper")

# =========================================================
# DRAWING HELPERS
# =========================================================
def make_density(x, y, width, height, bins=120):
    bins_y = max(30, round(bins * height / width))
    hist, xe, ye = np.histogram2d(x, y, bins=[bins, bins_y], range=[[0, width], [0, height]])
    return gaussian_filter(hist, sigma=2.5), xe, ye


def draw_density(ax, density, xe, ye, background=None):
    peak = float(density.max())
    if peak <= 0:
        return
    if background is not None:
        ax.imshow(background, extent=(xe[0], xe[-1], ye[-1], ye[0]), aspect="auto")
    image = np.ma.masked_where(density < peak * 0.08, density)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad((1, 1, 1, 0))
    mesh = ax.pcolormesh(xe, ye, image.T, cmap=cmap, shading="auto", vmin=0, vmax=peak,
                         alpha=0.6 if background is not None else 0.92)
    plt.colorbar(mesh, ax=ax, label="Relative presence (observations, not unique visitors)",
                 fraction=0.035, pad=0.02)

#original
# def draw_areas(ax, at, areas, labels=True):
#     status = dict(zip(at["Area"], at["Status"])) if len(at) else {}
#     for a in areas:
#         x1, y1, x2, y2 = a["bounds"]
#         col = STATUS_COLOR.get(status.get(a["name"]), "#655847")
#         ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, linewidth=2.2))
#         if labels:
#             ax.text((x1 + x2) / 2, (y1 + y2) / 2, a["name"], ha="center", va="center",
#                     fontsize=9, color="#382719",
#                     bbox=dict(facecolor="#fff4d8", alpha=0.92, edgecolor="#d8b477", boxstyle="round,pad=0.3"))

#verison1
def draw_areas(ax, at, areas, labels=True):
    status = dict(zip(at["Area"], at["Status"])) if len(at) else {}

    for a in areas:
        col = STATUS_COLOR.get(
            status.get(a["name"]), "#655847"
        )

        if "polygon" in a:
            points = np.asarray(
                a["polygon"], dtype=np.float32
            ).reshape((-1, 2))

            if len(points) < 3:
                continue

            ax.add_patch(
                Polygon(
                    points,
                    closed=True,
                    fill=False,
                    edgecolor=col,
                    linewidth=2.2
                )
            )

            # Use the polygon centroid for the label.
            contour = points.reshape((-1, 1, 2))
            moments = cv2.moments(contour)

            if moments["m00"] != 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            else:
                cx, cy = points.mean(axis=0)

        elif "bounds" in a:
            x1, y1, x2, y2 = a["bounds"]

            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=col,
                    linewidth=2.2
                )
            )

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

        else:
            continue

        if labels:
            ax.text(
                cx,
                cy,
                a["name"],
                ha="center",
                va="center",
                fontsize=9,
                color="#382719",
                bbox=dict(
                    facecolor="#fff4d8",
                    alpha=0.92,
                    edgecolor="#d8b477",
                    boxstyle="round,pad=0.3"
                )
            )



def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def shrink(img, max_side=MAX_IMG_SIDE):
    img = img.copy()
    img.thumbnail((max_side, max_side))
    return img

# =========================================================
# VIDEO PROCESSING
# =========================================================

#otgina;

# def process_video(video_bytes, filename, sample_fps=5):
#     from ultralytics import YOLO

#     video_path, cap = None, None
#     try:
#         with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".mp4", delete=False) as temp:
#             temp.write(video_bytes)
#             video_path = temp.name

#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise RuntimeError("Could not open the uploaded video.")

#         fps = cap.get(cv2.CAP_PROP_FPS)
#         total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         if not np.isfinite(fps) or fps <= 0 or total <= 0:
#             raise RuntimeError("Video metadata is invalid.")

#         duration = total / fps
#         model = YOLO("yolo11n.pt")
#         rows, unique_ids = [], set()
#         preview, clean, objects = None, None, []
#         frame_no = 0
#         stride = max(1, round(fps / sample_fps))
#         progress, status = st.progress(0), st.empty()

#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             if frame_no % stride == 0:
#                 result = model.track(frame, persist=True, tracker="bytetrack.yaml",
#                                      classes=[0], conf=0.35, verbose=False)[0]
#                 boxes = result.boxes
#                 if boxes is not None and len(boxes) > 0:
#                     coords = boxes.xyxy.cpu().numpy()
#                     ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(coords)
#                     for box, tid in zip(coords, ids):
#                         rows.append({"timestamp": frame_no / fps, "track_id": int(tid),
#                                      "x_px": float((box[0] + box[2]) / 2), "y_px": float(box[3])})
#                         if tid >= 0:
#                             unique_ids.add(tid)
#                 if preview is None:
#                     preview = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
#                     clean = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     fr = model.predict(frame, classes=[72], conf=0.25, verbose=False)[0]  # refrigerator
#                     objects = [{"name": "Fridge", "bounds": tuple(map(float, b))}
#                                for b in fr.boxes.xyxy.cpu().numpy()]
#             frame_no += 1
#             if frame_no % 10 == 0:
#                 progress.progress(min(frame_no / total, 1.0))
#                 status.write(f"Processing {frame_no:,}/{total:,} frames")

#         progress.progress(1.0)
#         cap.release()
#         cap = None
#         if not rows:
#             raise RuntimeError("No people were detected. Try a clearer video.")
#         return (pd.DataFrame(rows), width, height, fps, duration, len(unique_ids), preview, clean, objects)
#     finally:
#         if cap is not None:
#             cap.release()
#         if video_path and Path(video_path).exists():
#             Path(video_path).unlink(missing_ok=True)

#v1

def process_video(video_bytes, filename, sample_fps=5):
    from ultralytics import YOLO

    video_path, cap, writer = None, None, None
    output_path = OUTPUT_DIR / f"processed_{Path(filename).stem}.mp4"

    try:
        with tempfile.NamedTemporaryFile(
            suffix=Path(filename).suffix or ".mp4",
            delete=False
        ) as temp:
            temp.write(video_bytes)
            video_path = temp.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open the uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not np.isfinite(fps) or fps <= 0 or total <= 0:
            raise RuntimeError("Video metadata is invalid.")

        duration = total / fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Could not create the output MP4. "
                "Check OpenCV video codec support."
            )

        model = YOLO("yolo11n.pt")
        rows, unique_ids = [], set()
        preview, clean, objects = None, None, []
        frame_no = 0
        stride = max(1, round(fps / sample_fps))

        # Reuse the most recent detection result on intermediate frames.
        last_annotated = None

        progress, status = st.progress(0), st.empty()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_no % stride == 0:
                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=0.35,
                    verbose=False,
                )[0]

                last_annotated = result.plot()

                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    coords = boxes.xyxy.cpu().numpy()
                    ids = (
                        boxes.id.int().cpu().tolist()
                        if boxes.id is not None
                        else [-1] * len(coords)
                    )

                    for box, tid in zip(coords, ids):
                        rows.append({
                            "timestamp": frame_no / fps,
                            "track_id": int(tid),
                            "x_px": float((box[0] + box[2]) / 2),
                            "y_px": float(box[3]),
                        })
                        if tid >= 0:
                            unique_ids.add(tid)

                if preview is None:
                    preview = cv2.cvtColor(
                        last_annotated, cv2.COLOR_BGR2RGB
                    )
                    clean = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    fr = model.predict(
                        frame,
                        classes=[72],
                        conf=0.25,
                        verbose=False,
                    )[0]

                    objects = [
                        {
                            "name": "Fridge",
                            "bounds": tuple(map(float, b)),
                        }
                        for b in fr.boxes.xyxy.cpu().numpy()
                    ]

            # Write every frame, keeping the original FPS and duration.
            # Frames between detection steps have no fresh detections.
            writer.write(
                last_annotated if last_annotated is not None else frame
            )

            frame_no += 1

            if frame_no % 10 == 0:
                progress.progress(min(frame_no / total, 1.0))
                status.write(
                    f"Processing {frame_no:,}/{total:,} frames"
                )

        progress.progress(1.0)

        if not rows:
            raise RuntimeError(
                "No people were detected. Try a clearer video."
            )

        return (
            pd.DataFrame(rows),
            width,
            height,
            fps,
            duration,
            len(unique_ids),
            preview,
            clean,
            objects,
            str(output_path),
        )

    finally:
        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        if video_path and Path(video_path).exists():
            Path(video_path).unlink(missing_ok=True)



def auto_areas(tracks, width, height, objects=None):
    """Generate candidate retail zones from customer movement hotspots.

    Returns the existing {"name": ..., "bounds": ...} format.
    These are image-space activity regions, not calibrated floor boundaries.
    """
    required = {"track_id", "timestamp", "x_px", "y_px"}
    if tracks.empty or not required.issubset(tracks.columns):
        return [
            {"name": f"{o['name']} {i}", "bounds": o["bounds"]}
            for i, o in enumerate(objects or [], 1)
        ]

    t = tracks.loc[tracks["track_id"] >= 0].copy()
    t = t.dropna(subset=["track_id", "timestamp", "x_px", "y_px"])
    t = t.sort_values(["track_id", "timestamp"])

    if t.empty:
        return [
            {"name": f"{o['name']} {i}", "bounds": o["bounds"]}
            for i, o in enumerate(objects or [], 1)
        ]

    # Estimate per-track movement in normalized image coordinates.
    grouped = t.groupby("track_id", sort=False)
    dt = grouped["timestamp"].diff().to_numpy(dtype=float)
    dx = grouped["x_px"].diff().to_numpy(dtype=float)
    dy = grouped["y_px"].diff().to_numpy(dtype=float)

    valid_dt = np.isfinite(dt) & (dt > 0)
    speed = np.full(len(t), np.nan, dtype=float)
    speed[valid_dt] = (
        np.hypot(dx[valid_dt], dy[valid_dt])
        / dt[valid_dt]
        / max(float(width), float(height), 1.0)
    )
    t["speed_norm"] = speed

    # Smooth the spatial histogram to suppress isolated noisy detections.
    density, xe, ye = make_density(
        t["x_px"].to_numpy(),
        t["y_px"].to_numpy(),
        width,
        height,
        bins=120,
    )

    peak = float(density.max())
    if not np.isfinite(peak) or peak <= 0:
        zones = []
    else:
        # Try a relative threshold, with a small absolute floor relative
        # to the map's peak. Avoid treating the entire weak background as a zone.
        thresholds = [0.35, 0.25, 0.18]
        labels = np.zeros_like(density, dtype=int)
        n = 0

        for fraction in thresholds:
            candidate, count = nd_label(density >= peak * fraction)
            min_cells = max(12, int(density.size * 0.0008))
            keep = [
                label_id
                for label_id in range(1, count + 1)
                if np.count_nonzero(candidate == label_id) >= min_cells
            ]

            # Prefer a few substantial regions over many tiny fragments.
            if keep:
                labels = np.zeros_like(candidate)
                for new_id, old_id in enumerate(keep, start=1):
                    labels[candidate == old_id] = new_id
                n = len(keep)
                break

        zones, counts = [], {}

        # Include first/last observed positions as weak evidence of entry/exit.
        ends = pd.concat([
            t.groupby("track_id", sort=False).head(1),
            t.groupby("track_id", sort=False).tail(1),
        ]).drop_duplicates()

        for region_id in range(1, n + 1):
            cells_x, cells_y = np.where(labels == region_id)
            if len(cells_x) == 0:
                continue

            x1 = float(xe[cells_x.min()])
            x2 = float(xe[cells_x.max() + 1])
            y1 = float(ye[cells_y.min()])
            y2 = float(ye[cells_y.max() + 1])

            # Skip implausibly tiny or nearly full-frame regions.
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = (box_w * box_h) / max(float(width * height), 1.0)

            if box_w < 0.04 * width or box_h < 0.04 * height:
                continue
            if box_area > 0.70:
                continue

            inside = (
                t["x_px"].between(x1, x2)
                & t["y_px"].between(y1, y2)
            )
            sub = t.loc[inside]

            visitor_count = int(sub["track_id"].nunique())
            if visitor_count < 2:
                continue

            per_track = sub.groupby("track_id").agg(
                first_time=("timestamp", "min"),
                last_time=("timestamp", "max"),
                median_speed=("speed_norm", "median"),
                observations=("timestamp", "size"),
            )
            per_track["span_s"] = (
                per_track["last_time"] - per_track["first_time"]
            )

            median_dwell = float(per_track["span_s"].median())
            median_speed = float(per_track["median_speed"].median())
            stationary_share = float(
                (per_track["median_speed"].fillna(0) < 0.0005).mean()
            )

            edge_margin_x = 0.08 * width
            edge_margin_y = 0.08 * height
            near_edge = (
                x1 < edge_margin_x
                or x2 > width - edge_margin_x
                or y1 < edge_margin_y
                or y2 > height - edge_margin_y
            )

            end_inside = (
                ends["x_px"].between(x1, x2)
                & ends["y_px"].between(y1, y2)
            )
            end_fraction = float(end_inside.mean()) if len(ends) else 0.0

            aspect = max(
                box_w / max(box_h, 1.0),
                box_h / max(box_w, 1.0),
            )

            if near_edge and end_fraction >= 0.25:
                kind = "Entrance / Exit candidate"
            elif stationary_share >= 0.5 and median_dwell >= 5:
                kind = "Browsing / Queue candidate"
            elif aspect >= 1.8 and median_speed >= 0.0005:
                kind = "Transit / Aisle candidate"
            else:
                kind = "Customer activity zone"

            counts[kind] = counts.get(kind, 0) + 1
            zones.append({
                "name": f"{kind} {counts[kind]}",
                "bounds": (
                    max(0.0, x1),
                    max(0.0, y1),
                    min(float(width), x2),
                    min(float(height), y2),
                ),
            })

    # Preserve detected objects as contextual candidates.
    for i, obj in enumerate(objects or [], 1):
        zones.append({
            "name": f"{obj['name']} {i}",
            "bounds": tuple(map(float, obj["bounds"])),
        })

    return zones









# def auto_areas(tracks, width, height, objects=None):
#     """Find hotspots and name them from how people behave there."""
#     t = tracks[tracks["track_id"] >= 0].sort_values(["track_id", "timestamp"]).copy()
#     if t.empty:
#         return [{"name": f"{o['name']} {k}", "bounds": o["bounds"]} for k, o in enumerate(objects or [], 1)]

#     g = t.groupby("track_id")
#     dt = g["timestamp"].diff()
#     t["speed"] = np.hypot(g["x_px"].diff(), g["y_px"].diff()) / dt

#     density, xe, ye = make_density(t["x_px"].to_numpy(), t["y_px"].to_numpy(), width, height)
#     labels, n = nd_label(density >= density.max() * 0.30)
#     ends = pd.concat([g.head(1), g.tail(1)])
#     zones, counts = [], {}

#     for i in range(1, n + 1):
#         xs, ys = np.where(labels == i)
#         if len(xs) < 15:
#             continue
#         x1, x2 = xe[xs.min()], xe[xs.max() + 1]
#         y1, y2 = ye[ys.min()], ye[ys.max() + 1]

#         def inside(d, x1=x1, x2=x2, y1=y1, y2=y2):
#             return d["x_px"].between(x1, x2) & d["y_px"].between(y1, y2)

#         sub = t[inside(t)]
#         if sub["track_id"].nunique() < 2:
#             continue
#         speed = sub["speed"].median() / width
#         dwell = sub.groupby("track_id")["timestamp"].agg(lambda s: s.max() - s.min()).median()
#         w, h = x2 - x1, y2 - y1
#         elong = max(w / h, h / w)
#         end_frac = inside(ends).sum() / max(1, len(ends))
#         near_edge = (x1 < 0.08 * width or x2 > 0.92 * width or y1 < 0.08 * height or y2 > 0.92 * height)

#         if near_edge and end_frac > 0.25:
#             kind = "Entrance / Exit"
#         elif speed < 0.03 and dwell > 5:
#             kind = "Counter / Queue"
#         elif elong > 1.8 and speed >= 0.03:
#             kind = "Aisle"
#         else:
#             kind = "Shelf / Browsing area"
#         counts[kind] = counts.get(kind, 0) + 1
#         zones.append({"name": f"{kind} {counts[kind]}",
#                       "bounds": (max(0, x1), max(0, y1), min(width, x2), min(height, y2))})

#     for k, obj in enumerate(objects or [], 1):
#         zones.append({"name": f"{obj['name']} {k}", "bounds": obj["bounds"]})
#     return zones








# =========================================================
# SYNTHETIC DATA (clearly fake; demo only)
# =========================================================
SYN_W, SYN_H = 17.0, 15.0
SYN_AREAS = [
    {"name": "Entrance", "bounds": (1, 12, 4, 14.5)},
    {"name": "Produce", "bounds": (1, 2, 5, 5)},
    {"name": "Aisle 1", "bounds": (6, 2, 8, 9)},
    {"name": "Aisle 2", "bounds": (9, 2, 11, 9)},
    {"name": "Dairy", "bounds": (12, 2, 15.5, 5)},
    {"name": "Checkout", "bounds": (11, 11, 16, 14)},
]


@st.cache_data(show_spinner=False)
def simulate_store(seed=42, n=140):
    rng = np.random.default_rng(seed)
    bounds = {a["name"]: a["bounds"] for a in SYN_AREAS}
    browse = ["Produce", "Aisle 1", "Aisle 2", "Dairy"]
    weights = [0.30, 0.40, 0.05, 0.25]  # Aisle 2 is deliberately ignored
    rows = []
    for tid in range(1, n + 1):
        t = rng.uniform(0, 1200) if rng.random() < 0.5 else rng.normal(900, 80)  # rush near t=900
        b = bounds["Entrance"]
        pos = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        stops = list(rng.choice(browse, size=int(rng.integers(1, 4)), replace=False, p=weights)) + ["Checkout"]
        for name in stops:
            x1, y1, x2, y2 = bounds[name]
            target = np.array([rng.uniform(x1, x2), rng.uniform(y1, y2)])
            steps = max(1, int(np.linalg.norm(target - pos) / 1.3))
            for s in range(1, steps + 1):
                p = pos + (target - pos) * s / steps + rng.normal(0, 0.08, 2)
                rows.append((t, tid, p[0], p[1]))
                t += 1
            pos = target
            dwell = int(rng.gamma(5, 6)) if name == "Checkout" else int(rng.gamma(4, 2.5))
            for _ in range(dwell):
                p = pos + rng.normal(0, 0.1, 2)
                rows.append((t, tid, p[0], p[1]))
                t += 1
    df = pd.DataFrame(rows, columns=["timestamp", "track_id", "x_px", "y_px"])
    df["timestamp"] -= df["timestamp"].min()
    return df

# =========================================================
# ANALYTICS
# =========================================================

#original
# def prepare(tracks, areas, width):
#     t = tracks[tracks["track_id"] >= 0].sort_values(["track_id", "timestamp"]).copy()
#     if t.empty:
#         return t, 1.0
#     g = t.groupby("track_id")
#     for c in ("x_px", "y_px"):
#         t[c + "_s"] = g[c].transform(lambda s: s.rolling(3, center=True, min_periods=1).median())
#     g = t.groupby("track_id")
#     dt = g["timestamp"].diff()
#     dist = np.hypot(g["x_px_s"].diff(), g["y_px_s"].diff())
#     t["speed"] = (dist / dt / width).where(dt > 0)
#     t["speed"] = t.groupby("track_id")["speed"].transform(lambda s: s.rolling(3, center=True, min_periods=1).median())
#     interval = float(np.nanmedian(dt)) if dt.notna().any() else 1.0
#     interval = interval if interval > 0 else 1.0
#     nxt = t.groupby("track_id")["timestamp"].shift(-1) - t["timestamp"]
#     t["dur"] = nxt.clip(upper=interval * 2.5).fillna(0.0)

#     t["area"] = OTHER
#     size = lambda a: (a["bounds"][2] - a["bounds"][0]) * (a["bounds"][3] - a["bounds"][1])
#     for a in sorted(areas, key=size, reverse=True):  # smaller areas win overlaps
#         x1, y1, x2, y2 = a["bounds"]
#         m = t["x_px_s"].between(x1, x2) & t["y_px_s"].between(y1, y2)
#         t.loc[m, "area"] = a["name"]
#     return t, interval

#verison1


def prepare(tracks, areas, width):
    t = tracks[tracks["track_id"] >= 0].sort_values(
        ["track_id", "timestamp"]
    ).copy()

    if t.empty:
        return t, 1.0

    g = t.groupby("track_id")

    for c in ("x_px", "y_px"):
        t[c + "_s"] = g[c].transform(
            lambda s: s.rolling(
                3, center=True, min_periods=1
            ).median()
        )

    g = t.groupby("track_id")
    dt = g["timestamp"].diff()
    dist = np.hypot(
        g["x_px_s"].diff(),
        g["y_px_s"].diff()
    )

    t["speed"] = (dist / dt / width).where(dt > 0)

    t["speed"] = t.groupby("track_id")["speed"].transform(
        lambda s: s.rolling(
            3, center=True, min_periods=1
        ).median()
    )

    interval = float(np.nanmedian(dt)) if dt.notna().any() else 1.0
    interval = interval if interval > 0 else 1.0

    nxt = (
        t.groupby("track_id")["timestamp"].shift(-1)
        - t["timestamp"]
    )
    t["dur"] = nxt.clip(upper=interval * 2.5).fillna(0.0)

    # Default assignment for points outside every zone.
    t["area"] = OTHER

    def zone_size(a):
        if "polygon" in a:
            points = np.asarray(a["polygon"], dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 3:
                return 0.0
            return abs(cv2.contourArea(points))

        if "bounds" in a:
            x1, y1, x2, y2 = a["bounds"]
            return max(0, x2 - x1) * max(0, y2 - y1)

        return 0.0

    # Assign larger zones first; smaller zones overwrite them,
    # so the smallest zone wins when zones overlap.
    for a in sorted(areas, key=zone_size, reverse=True):
        name = a["name"]

        if "polygon" in a:
            points = np.asarray(
                a["polygon"], dtype=np.float32
            ).reshape((-1, 2))

            if len(points) < 3:
                continue

            inside = np.array([
                cv2.pointPolygonTest(
                    points,
                    (float(x), float(y)),
                    False
                ) >= 0
                for x, y in zip(t["x_px_s"], t["y_px_s"])
            ])

            mask = pd.Series(inside, index=t.index)

        elif "bounds" in a:
            x1, y1, x2, y2 = a["bounds"]

            mask = (
                t["x_px_s"].between(x1, x2)
                & t["y_px_s"].between(y1, y2)
            )

        else:
            continue

        t.loc[mask, "area"] = name

    return t, interval


def build_runs(t, min_visit):
    prev = t.groupby("track_id")["area"].shift()
    t["run"] = (t["area"] != prev).cumsum()
    t["still_dur"] = np.where(t["speed"] < STILL_SPEED, t["dur"], 0.0)
    runs = t.groupby("run").agg(track_id=("track_id", "first"), area=("area", "first"),
                                t0=("timestamp", "min"), dur=("dur", "sum"),
                                still=("still_dur", "sum"), n=("timestamp", "size")).reset_index(drop=True)
    return runs[(runs["n"] >= 2) & (runs["dur"] >= min_visit)]


def area_table(t, runs, areas, occ_thr, persist_s):
    valid = runs[runs["area"] != OTHER]
    total = max(1, valid["track_id"].nunique())
    rows = []
    for a in areas:
        name = a["name"]
        r = runs[runs["area"] == name]
        rt = t[t["area"] == name]
        visitors = r["track_id"].nunique()
        stoppers = r.loc[r["still"] >= MIN_STOP_S, "track_id"].nunique()
        if len(rt):
            occ = rt.groupby(rt["timestamp"] // BIN_S)["track_id"].nunique()
            peak, cong = int(occ.max()), int((occ >= occ_thr).sum()) * BIN_S
            med_speed = float(rt["speed"].median())
        else:
            peak, cong, med_speed = 0, 0, np.nan
        rows.append({
            "Area": name, "Visitors": visitors,
            "Share %": round(100 * visitors / total, 1),
            "Median visit s": round(float(r["dur"].median()), 1) if len(r) else np.nan,
            "Presence s": round(float(r["dur"].sum()), 1),
            "Stationary s": round(float(r["still"].sum()), 1),
            "Stop rate %": round(100 * stoppers / visitors, 1) if visitors else 0.0,
            "Peak occupancy": peak, "Congested s": cong,
            "Median speed": round(med_speed, 3) if np.isfinite(med_speed) else np.nan,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        df["Status"] = []
        return df

    mean_share = df["Share %"].mean()
    med_visit = df["Median visit s"].median()
    enough = total >= 10 and len(df) >= 3
    statuses = []
    for _, r in df.iterrows():
        is_gate = any(k in r["Area"].lower() for k in ("entr", "exit"))
        if not enough:
            s = "Not enough data"
        elif r["Congested s"] >= persist_s and (pd.isna(r["Median speed"]) or r["Median speed"] < 0.06):
            s = "Congestion candidate"
        elif is_gate:
            s = "Normal"
        elif r["Share %"] < 0.35 * mean_share:
            s = "Dead-zone candidate"
        elif r["Stop rate %"] >= 60 and r["Median visit s"] > med_visit:
            s = "Hot / high engagement"
        elif r["Share %"] >= mean_share and r["Stop rate %"] < 25:
            s = "High traffic, few stops"
        else:
            s = "Normal"
        statuses.append(s)
    df["Status"] = statuses
    return df


def flow_tables(runs):
    r = runs[runs["area"] != OTHER].sort_values(["track_id", "t0"]).copy()
    if r.empty:
        e = pd.Series(dtype=int)
        return pd.DataFrame(columns=["Move", "People"]), e, e, e
    r = r[r["area"] != r.groupby("track_id")["area"].shift()]
    r["next"] = r.groupby("track_id")["area"].shift(-1)
    flows = (r.dropna(subset=["next"]).groupby(["area", "next"]).size()
             .reset_index(name="People").sort_values("People", ascending=False))
    flows["Move"] = flows["area"] + " → " + flows["next"]
    paths = r.groupby("track_id")["area"].apply(lambda s: " → ".join(s)).value_counts().head(5)
    entries = r.groupby("track_id")["area"].first().value_counts()
    exits = r.groupby("track_id")["area"].last().value_counts()
    return flows[["Move", "People"]].head(10), paths, entries, exits


def quality_stats(tracks, t):
    life = t.groupby("track_id")["timestamp"].agg(lambda s: s.max() - s.min())
    return {
        "untracked_pct": round(100 * float((tracks["track_id"] < 0).mean()), 1),
        "short_track_pct": round(100 * float((life < 3).mean()), 1) if len(life) else 0.0,
        "track_ids": int(t["track_id"].nunique()),
    }


def rule_findings(at, entries, exits, paths):
    recs = at.to_dict("records")
    out = []
    if not recs:
        return out
    byv = sorted(recs, key=lambda r: r["Visitors"], reverse=True)
    out.append(f"**Where people go:** {byv[0]['Area']} gets the most visitors ({byv[0]['Share %']:.0f}% of "
               f"shoppers); {byv[-1]['Area']} gets the fewest ({byv[-1]['Share %']:.0f}%).")
    if len(entries) and len(exits):
        out.append(f"**Entry / exit:** most shoppers start in **{entries.index[0]}** and finish in **{exits.index[0]}**.")
    if len(paths):
        out.append(f"**Most common route:** {paths.index[0]} ({int(paths.iloc[0])} shoppers).")
    for r in recs:
        s = r["Status"]
        if s == "Congestion candidate":
            out.append(f"🔴 **{r['Area']}**: up to {r['Peak occupancy']} people at once, congested for "
                       f"{r['Congested s']:.0f}s. Consider more capacity, a wider approach, or spreading demand.")
        elif s == "Dead-zone candidate":
            out.append(f"⚪ **{r['Area']}**: only {r['Share %']:.0f}% of shoppers. Check for stockouts, blocked "
                       f"access, poor sightlines or camera coverage before acting.")
        elif s == "High traffic, few stops":
            out.append(f"🔵 **{r['Area']}**: {r['Share %']:.0f}% of shoppers pass but only {r['Stop rate %']:.0f}% "
                       f"stop. Good spot to test a promo display or better signage.")
        elif s == "Hot / high engagement":
            out.append(f"🟠 **{r['Area']}**: {r['Stop rate %']:.0f}% stop, median visit {r['Median visit s']}s. "
                       f"Protect this space and test adjacent complementary products.")
    return out


def facts_text(at, flows, paths, entries, exits, q, source, validation):
    val = "NOT validated against manual counts yet." if validation is None else (
        f"Validated: mean abs count error {validation['count_err']:.0f}%"
        + (f", dwell error {validation['dwell_err']:.0f}%" if validation.get("dwell_err") is not None else "") + ".")
    return (
        f"DATA SOURCE: {source}\nVALIDATION: {val}\n"
        f"TRACK QUALITY: {q['track_ids']} track IDs, {q['untracked_pct']}% detections untracked, "
        f"{q['short_track_pct']}% tracks shorter than 3s (possible ID switches).\n"
        "NOTES: speeds are frame-widths/second, not m/s. Visitors are approximate (ID switches inflate them). "
        "Stop = stationary >=3s. Status labels are rule-based candidates, not verdicts.\n\n"
        f"AREA TABLE (CSV):\n{at.to_csv(index=False)}\n"
        f"TOP AREA-TO-AREA MOVES:\n{flows.to_csv(index=False)}\n"
        f"COMMON ROUTES:\n{paths.to_string()}\n\n"
        f"FIRST AREA ENTERED:\n{entries.to_string()}\n\nLAST AREA VISITED:\n{exits.to_string()}\n"
    )

# =========================================================
# VISION-LANGUAGE MODEL (sees the heat map + video frame)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_vlm(name):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    cuda = torch.cuda.is_available()
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(
        name, torch_dtype=torch.float16 if cuda else torch.float32)
    model.to("cuda" if cuda else "cpu").eval()
    return processor, model


def run_vlm(name, images, prompt):
    import torch
    processor, model = load_vlm(name)
    images = [shrink(i) for i in images]
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    # SmolVLM expects one list of images per prompt; Qwen-VL takes a flat list.
    img_arg = [images] if "smol" in name.lower() else images
    inputs = processor(text=[text], images=img_arg, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=800, do_sample=False)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

# =========================================================
# DASHBOARD RENDERER (shared by both modes)
# =========================================================
def render(tracks, meta, areas, synthetic, background=None):
    width, height = meta["width"], meta["height"]
    source = ("SYNTHETIC simulated shoppers (NOT real measurements)" if synthetic
              else "Real video, YOLO + ByteTrack, unvalidated unless stated")

    if synthetic:
        st.error("🧪 SYNTHETIC DEMO: these shoppers are simulated. Nothing below is a real store measurement. "
                 "It only shows how the dashboard works. Use the Recorded video tab for real analysis.")

    t, interval = prepare(tracks, areas, width)
    if t.empty:
        st.warning("No tracked people (track IDs missing). Try a clearer video.")
        return
    runs = build_runs(t, min_visit)
    at = area_table(t, runs, areas, occ_thr, persist_s)
    flows, paths, entries, exits = flow_tables(runs)
    q = quality_stats(tracks, t)
    total_sys = int(runs.loc[runs["area"] != OTHER, "track_id"].nunique())

    tab_over, tab_flow, tab_val, tab_ai = st.tabs(["📊 Overview", "🔀 Flow", "✅ Validation", "🤖 AI insights"])
    validation = None
    heat_img = None

    # ---------- OVERVIEW ----------
    with tab_over:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{meta['duration']:.0f}s")
        c2.metric("Shoppers (approx.)", total_sys)
        busiest = at.sort_values("Visitors", ascending=False)["Area"].iloc[0] if len(at) else "-"
        c3.metric("Busiest area", busiest)
        flagged = int((~at["Status"].isin(["Normal", "Not enough data"])).sum()) if len(at) else 0
        c4.metric("Flagged areas", flagged)

        if q["short_track_pct"] > 30:
            st.warning(f"{q['short_track_pct']}% of tracks are shorter than 3s. Likely ID switches or flicker, "
                       "so visitor counts may be inflated.")
        if q["untracked_pct"] > 10:
            st.warning(f"{q['untracked_pct']}% of detections have no track ID and are ignored in area metrics.")
        if not synthetic:
            st.info("These numbers are **not validated yet** unless you completed the Validation tab.")

        density, xe, ye = make_density(tracks["x_px"].to_numpy(), tracks["y_px"].to_numpy(), width, height)
        fig, ax = plt.subplots(figsize=(11, 7))
        draw_density(ax, density, xe, ye, background=background)
        draw_areas(ax, at, areas, show_labels)
        if show_paths:
            for _, grp in tracks[tracks["track_id"] >= 0].groupby("track_id"):
                grp = grp.sort_values("timestamp")
                ax.plot(grp["x_px"], grp["y_px"], alpha=0.3, linewidth=0.8, color="#8b5e91")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal")
        ax.set_title("SYNTHETIC heat map (simulated)" if synthetic else "Occupancy by image position")
        ax.set_facecolor("#eee5d3")
        heat_img = fig_to_pil(fig)   # this exact image is what the AI model will look at
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Box colours: red = congestion, grey = dead-zone candidate, orange = hot, blue = high traffic/few stops. "
                   "Heat map shows observed presence, not unique visitors or interest.")

        st.subheader("Area analytics")
        st.dataframe(at, hide_index=True, use_container_width=True)
        st.caption("Presence = time inside area. Stationary = time below walking speed. "
                   "Queueing is not browsing. Speed is in frame-widths/s, not m/s.")
        st.download_button("Download area analytics CSV", at.to_csv(index=False).encode("utf-8"),
                           "storeflow_area_analytics.csv", "text/csv")

    # ---------- FLOW ----------
    with tab_flow:
        st.subheader("Where customers actually go")
        if flows.empty:
            st.info("No area-to-area movement found. Define more areas or use a longer video.")
        else:
            a, b = st.columns([3, 2])
            with a:
                st.markdown("**Top moves between areas**")
                st.bar_chart(flows.set_index("Move")["People"])
            with b:
                st.markdown("**Most common routes**")
                st.dataframe(paths.rename("Shoppers").reset_index().rename(columns={"area": "Route"}),
                             hide_index=True, use_container_width=True)
            c, d = st.columns(2)
            c.markdown("**First area entered**")
            c.dataframe(entries.rename("Shoppers"), use_container_width=True)
            d.markdown("**Last area visited**")
            d.dataframe(exits.rename("Shoppers"), use_container_width=True)
        st.caption("Routes assume one track ID per person. ID switches split journeys, so treat routes as indicative.")

    # ---------- VALIDATION ----------
    with tab_val:
        st.subheader("Validate against manually checked video")
        if synthetic:
            st.info("Validation only applies to real video. Synthetic data has no real-world ground truth.")
        else:
            st.write("Watch the video yourself. For each area, count how many different people entered it and "
                     "estimate their typical dwell time. Leave blanks for areas you skipped.")
            base = pd.DataFrame({"Area": at["Area"], "Manual customers": np.nan, "Manual typical dwell (s)": np.nan})
            ed = st.data_editor(
                base, hide_index=True, disabled=["Area"], key="val_" + "|".join(at["Area"]),
                column_config={"Manual customers": st.column_config.NumberColumn(min_value=0, step=1),
                               "Manual typical dwell (s)": st.column_config.NumberColumn(min_value=0)})
            manual_total = st.number_input("Manual total customers in the clip (optional)", min_value=0, value=0, step=1)

            m = at[["Area", "Visitors", "Median visit s"]].merge(ed, on="Area")
            m["Count error %"] = np.where(m["Manual customers"] > 0,
                                          (m["Visitors"] - m["Manual customers"]) / m["Manual customers"] * 100, np.nan)
            m["Dwell error %"] = np.where(m["Manual typical dwell (s)"] > 0,
                                          (m["Median visit s"] - m["Manual typical dwell (s)"]) / m["Manual typical dwell (s)"] * 100,
                                          np.nan)
            checked = m.dropna(subset=["Count error %"])
            tot_err = None
            if manual_total > 0:
                tot_err = (total_sys - manual_total) / manual_total * 100
                st.metric("Total customers: system vs manual", f"{total_sys} vs {manual_total}", f"{tot_err:+.0f}%")
            if checked.empty and manual_total == 0:
                st.info("Enter at least one manual count to see accuracy.")
            else:
                st.dataframe(m.round(1), hide_index=True, use_container_width=True)
                errs = [checked["Count error %"].abs().mean()] if not checked.empty else []
                if tot_err is not None:
                    errs.append(abs(tot_err))
                count_err = float(np.mean(errs))
                dw = m["Dwell error %"].abs().dropna()
                dwell_err = float(dw.mean()) if len(dw) else None
                validation = {"count_err": count_err, "dwell_err": dwell_err}
                st.session_state["sf_validation"] = validation
                worst = max(count_err, dwell_err or 0)
                if worst <= 15:
                    st.success(f"✅ Good: mean count error {count_err:.0f}%.")
                elif worst <= 30:
                    st.warning(f"⚠️ Fair: mean count error {count_err:.0f}%. Use for rough trends only.")
                else:
                    st.error(f"❌ Poor: mean count error {count_err:.0f}%. Do not base layout decisions on this yet.")

    # ---------- AI INSIGHTS ----------
    with tab_ai:
        validation = validation or st.session_state.get("sf_validation")
        st.subheader("What the data says and what to do")
        for line in rule_findings(at, entries, exits, paths):
            st.markdown("- " + line)

        st.divider()
        st.markdown(f"**AI vision analyst:** `{vlm_name}` looks at the heat map image"
                    + ("" if synthetic or background is None else " and the video frame")
                    + ", reads the measured numbers, and writes a report in plain text. "
                      "It sees the pictures, but its 'why' statements are still hypotheses to check in the store.")

        images, frame_line = [], ""
        if heat_img is not None:
            images.append(heat_img)
        if background is not None and not synthetic:
            images.append(Image.fromarray(background))
            frame_line = "IMAGE 2: a plain frame from the store video, so you can relate the heat map to real shelves, counters and doors."
        else:
            frame_line = "(No video frame available; this is a simulated floor plan.)" if synthetic else ""

        facts = facts_text(at, flows, paths, entries, exits, q, source, validation)
        prompt = VLM_PROMPT.format(frame_line=frame_line, facts=facts)

        cols = st.columns(len(images)) if images else []
        for col, img, cap in zip(cols, images, ["Heat map the AI sees", "Video frame the AI sees"]):
            col.image(img, caption=cap, use_container_width=True)
        with st.expander("Numbers sent to the model together with the images"):
            st.code(facts)

        key = hashlib.md5((vlm_name + facts + str(len(images))).encode()).hexdigest()
        if st.button("Analyse heat map with AI", type="primary"):
            try:
                with st.spinner("Loading vision model (first run downloads it) and analysing the images..."):
                    st.session_state["ai_" + key] = run_vlm(vlm_name, images, prompt)
            except Exception as exc:
                st.error(f"AI analysis failed: {exc}. The rule-based findings above still apply.")
        if "ai_" + key in st.session_state:
            st.markdown("### AI report")
            st.write(st.session_state["ai_" + key])
            st.caption("AI-generated reading of the images and numbers, not ground truth. "
                       "Small vision models can misread charts, so check it against the table and your own eyes "
                       "before changing the store layout.")

# =========================================================
def extract_preview(video_bytes, filename):
    """Extract a clean representative frame before running detection."""
    suffix = Path(filename).suffix or ".mp4"
    video_path = None
    cap = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(video_bytes)
            video_path = temp.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open the uploaded video.")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError("The uploaded video contains no readable frames.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = cap.read()

        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()

        if not ok:
            raise RuntimeError("Could not extract a preview frame.")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width, height

    finally:
        if cap is not None:
            cap.release()
        if video_path is not None:
            Path(video_path).unlink(missing_ok=True)


# =========================================================
# PAGES
# =========================================================
if synthetic_mode:
    tracks = simulate_store()
    meta = {
        "width": SYN_W,
        "height": SYN_H,
        "duration": float(tracks["timestamp"].max()),
        "fps": 1.0,
        "ids": int(tracks["track_id"].nunique()),
    }
    render(tracks, meta, SYN_AREAS, synthetic=True)

else:
    if uploaded_video is None:
        st.info("Upload a store video in the sidebar.")
        st.stop()

    from streamlit_drawable_canvas import st_canvas

    video_bytes = uploaded_video.getvalue()
    video_hash = hashlib.md5(video_bytes).hexdigest()
    video_key = (uploaded_video.name, video_hash)
    widget_suffix = video_hash[:10]

    # Reset zones and results whenever the selected video changes.
    if st.session_state.get("sf_upload_key") != video_key:
        for key in (
            "sf_tracks",
            "sf_meta",
            "sf_preview",
            "sf_clean",
            "sf_areas",
            "sf_validation",
            "sf_results_video_key",
        ):
            st.session_state.pop(key, None)

        st.session_state["sf_upload_key"] = video_key

    with st.expander("Preview uploaded video", expanded=False):
        st.video(video_bytes)

    # Extract a clean frame before running detection.
    if st.session_state.get("sf_preview_video_key") != video_key:
        try:
            with st.spinner("Preparing a frame for zone selection..."):
                clean, preview_width, preview_height = extract_preview(
                    video_bytes, uploaded_video.name
                )

            st.session_state["sf_clean"] = clean
            st.session_state["sf_preview_size"] = (
                preview_width, preview_height
            )
            st.session_state["sf_preview_video_key"] = video_key

        except Exception as exc:
            st.error(f"Could not prepare video preview: {exc}")
            st.stop()

    width, height = st.session_state["sf_preview_size"]
    clean = st.session_state["sf_clean"]

    st.subheader("Step 1 — Draw your store zones")
    st.caption(
        "Draw one rectangle per zone on the frame. Then enter one name "
        "per rectangle, in the same order you drew them. "
        "You can rename saved zones before running the analysis."
    )

    canvas_height = min(650, max(200, round(800 * height / width)))
    canvas_width = max(1, round(canvas_height * width / height))
    background = Image.fromarray(clean).resize(
        (canvas_width, canvas_height)
    )

    canvas = st_canvas(
        fill_color="rgba(245, 185, 65, 0.20)",
        stroke_width=2,
        stroke_color="#a65a16",
        background_image=background,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="rect",
        key=f"sf_area_canvas_{widget_suffix}",
    )
    #st.write("Canvas data:", canvas.json_data)

    
    canvas_objects = (
        canvas.json_data.get("objects", [])
        if canvas.json_data else []
    )
    #st.write("DEBUG — Canvas objects:", canvas_objects)

    polygons = [
        obj for obj in canvas_objects
        if obj.get("type", "").lower() in ("rect", "polygon")
        and (
            obj.get("type", "").lower() == "rect"
            or len(obj.get("points", [])) >= 3
        )
    ]

    st.subheader("Name your zones")
    names = []

    for i in range(len(polygons)):
        name = st.text_input(
            f"Zone {i + 1} name",
            key=f"sf_area_name_{widget_suffix}_{i}",
            placeholder=f"e.g. Browsing Area {i + 1}",
        )
        if name.strip():
            names.append(name.strip())
        else:
            names.append("")

    #names = [line.strip() for line in names_text.splitlines() if line.strip()]
    ##origina
    # if st.button("Save drawn zones", key=f"sf_save_areas_{widget_suffix}"):
    #     objects = (
    #         canvas.json_data.get("objects", [])
    #         if canvas.json_data else []
    #     )
    #     rectangles = [obj for obj in objects if obj.get("type") == "rect"]

    #     if not rectangles:
    #         st.warning("Draw at least one rectangle on the preview first.")
    #     elif len(names) != len(rectangles):
    #         st.warning(
    #             f"You drew {len(rectangles)} rectangles but entered "
    #             f"{len(names)} names. Enter exactly one name per rectangle."
    #         )
    #     else:
    #         scale_x = width / canvas_width
    #         scale_y = height / canvas_height
    #         saved = []

    #         for name, obj in zip(names, rectangles):
    #             left = float(obj.get("left", 0))
    #             top = float(obj.get("top", 0))
    #             rect_width = float(obj.get("width", 0)) * float(
    #                 obj.get("scaleX", 1)
    #             )
    #             rect_height = float(obj.get("height", 0)) * float(
    #                 obj.get("scaleY", 1)
    #             )

    #             x1 = max(0.0, min(float(width), left * scale_x))
    #             y1 = max(0.0, min(float(height), top * scale_y))
    #             x2 = max(
    #                 0.0, min(float(width), (left + rect_width) * scale_x)
    #             )
    #             y2 = max(
    #                 0.0, min(float(height), (top + rect_height) * scale_y)
    #             )

    #             if x2 <= x1 or y2 <= y1:
    #                 continue

    #             saved.append({
    #                 "name": name,
    #                 "bounds": (x1, y1, x2, y2),
    #             })

    #         if saved:
    #             st.session_state["sf_areas"] = saved
    #             st.success(f"Saved {len(saved)} zones.")
    #         else:
    #             st.warning("The rectangles were empty or invalid. Draw them again.")

    ##veeison1

    # if st.button("Save drawn zones", key=f"sf_save_areas_{widget_suffix}"):
    #     objects = (
    #         canvas.json_data.get("objects", [])
    #         if canvas.json_data else []
    #     )

    #     polygons = [
    #         obj for obj in objects
    #         if obj.get("type", "").lower() == "polygon"
    #         and len(obj.get("points", [])) >= 3
    #     ]

    #     if not polygons:
    #         st.warning(
    #             "No valid polygons found. Draw a rectangle and reshape "
    #             "its vertices before saving."
    #         )
    #     elif any(not name for name in names):
    #         st.warning("Please enter a name for every polygon.")
    #         st.warning(
    #             f"You drew {len(polygons)} polygons but entered "
    #             f"{len(names)} names. Enter exactly one name per polygon."
    #         )
    #     else:
    #         scale_x = width / canvas_width
    #         scale_y = height / canvas_height
    #         saved = []

    #         for name, obj in zip(names, polygons):
    #             points = obj["points"]

    #             # Fabric polygon points are relative to the object's
    #             # center. Convert them to canvas coordinates.
    #             left = float(obj.get("left", 0))
    #             top = float(obj.get("top", 0))
    #             origin_x = obj.get("originX", "center")
    #             origin_y = obj.get("originY", "center")

    #             if origin_x == "center":
    #                 center_x = left
    #             elif origin_x == "right":
    #                 center_x = left - float(obj.get("width", 0)) / 2
    #             else:
    #                 center_x = left + float(obj.get("width", 0)) / 2

    #             if origin_y == "center":
    #                 center_y = top
    #             elif origin_y == "bottom":
    #                 center_y = top - float(obj.get("height", 0)) / 2
    #             else:
    #                 center_y = top + float(obj.get("height", 0)) / 2

    #             polygon_points = []

    #             for point in points:
    #                 px = (
    #                     center_x
    #                     + float(point["x"]) * float(obj.get("scaleX", 1))
    #                 )
    #                 py = (
    #                     center_y
    #                     + float(point["y"]) * float(obj.get("scaleY", 1))
    #                 )

    #                 px = max(0.0, min(float(canvas_width), px))
    #                 py = max(0.0, min(float(canvas_height), py))

    #                 polygon_points.append((
    #                     px * scale_x,
    #                     py * scale_y,
    #                 ))

    #             if len(set(polygon_points)) < 3:
    #                 continue

    #             saved.append({
    #                 "name": name,
    #                 "polygon": polygon_points,
    #             })

    #         if saved:
    #             st.session_state["sf_areas"] = saved
    #             st.success(f"Saved {len(saved)} polygon zones.")
    #         else:
    #             st.warning("The polygons were invalid. Please reshape them.")

    ##version2

    
    # if st.button("Save drawn zones", key=f"sf_save_areas_{widget_suffix}"):
    #     objects = (
    #         canvas.json_data.get("objects", [])
    #         if canvas.json_data else []
    #     )

    #     shapes = [
    #         obj for obj in objects
    #         if obj.get("type", "").lower() in ("rect", "polygon")
    #     ]

    #     if not shapes:
    #         st.warning("Draw at least one rectangle or polygon first.")
    #     elif len(names) != len(shapes) or any(not name for name in names):
    #         st.warning(
    #             f"You drew {len(shapes)} shapes. "
    #             "Enter exactly one name for every shape."
    #         )
    #     else:
    #         scale_x = width / canvas_width
    #         scale_y = height / canvas_height
    #         saved = []

    #         for name, obj in zip(names, shapes):
    #             shape_type = obj.get("type", "").lower()

    #             if shape_type == "rect":
    #                 left = float(obj.get("left", 0))
    #                 top = float(obj.get("top", 0))
    #                 rect_width = float(obj.get("width", 0)) * float(
    #                     obj.get("scaleX", 1)
    #                 )
    #                 rect_height = float(obj.get("height", 0)) * float(
    #                     obj.get("scaleY", 1)
    #                 )

    #                 x1 = max(0.0, min(float(canvas_width), left))
    #                 y1 = max(0.0, min(float(canvas_height), top))
    #                 x2 = max(
    #                     0.0,
    #                     min(float(canvas_width), left + rect_width)
    #                 )
    #                 y2 = max(
    #                     0.0,
    #                     min(float(canvas_height), top + rect_height)
    #                 )

    #                 if x2 <= x1 or y2 <= y1:
    #                     continue

    #                 saved.append({
    #                     "name": name,
    #                     "polygon": [
    #                         (x1 * scale_x, y1 * scale_y),
    #                         (x2 * scale_x, y1 * scale_y),
    #                         (x2 * scale_x, y2 * scale_y),
    #                         (x1 * scale_x, y2 * scale_y),
    #                     ],
    #                 })

    #             elif shape_type == "polygon":
    #                 points = obj.get("points", [])
    #                 if len(points) < 3:
    #                     continue

    #                 left = float(obj.get("left", 0))
    #                 top = float(obj.get("top", 0))
    #                 origin_x = obj.get("originX", "center")
    #                 origin_y = obj.get("originY", "center")

    #                 center_x = (
    #                     left if origin_x == "center"
    #                     else left - float(obj.get("width", 0)) / 2
    #                     if origin_x == "right"
    #                     else left + float(obj.get("width", 0)) / 2
    #                 )
    #                 center_y = (
    #                     top if origin_y == "center"
    #                     else top - float(obj.get("height", 0)) / 2
    #                     if origin_y == "bottom"
    #                     else top + float(obj.get("height", 0)) / 2
    #                 )

    #                 polygon_points = []
    #                 for point in points:
    #                     px = center_x + float(point["x"]) * float(
    #                         obj.get("scaleX", 1)
    #                     )
    #                     py = center_y + float(point["y"]) * float(
    #                         obj.get("scaleY", 1)
    #                     )

    #                     px = max(0.0, min(float(canvas_width), px))
    #                     py = max(0.0, min(float(canvas_height), py))
    #                     polygon_points.append((px * scale_x, py * scale_y))

    #                 if len(set(polygon_points)) < 3:
    #                     continue

    #                 saved.append({
    #                     "name": name,
    #                     "polygon": polygon_points,
    #                 })

    #         if saved:
    #             st.session_state["sf_areas"] = saved
    #             st.success(f"Saved {len(saved)} zones.")
    #         else:
    #             st.warning("The drawn shapes were invalid. Please draw them again.")

    ##verison3

    if st.button("Save drawn zones", key=f"sf_save_areas_{widget_suffix}"):
        objects = (
            canvas.json_data.get("objects", [])
            if canvas.json_data else []
        )

        shapes = [
            obj for obj in objects
            if obj.get("type", "").lower() in ("rect", "polygon")
        ]

        if not shapes:
            st.warning("Draw at least one rectangle or polygon first.")
        elif len(names) != len(shapes) or any(not name for name in names):
            st.warning("Enter a name for every drawn shape.")
        else:
            scale_x = width / canvas_width
            scale_y = height / canvas_height
            saved = []

            for name, obj in zip(names, shapes):
                shape_type = obj.get("type", "").lower()

                if shape_type == "rect":
                    # Fabric Rect: left/top are its top-left corner.
                    x1 = float(obj.get("left", 0))
                    y1 = float(obj.get("top", 0))
                    x2 = x1 + float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
                    y2 = y1 + float(obj.get("height", 0)) * float(obj.get("scaleY", 1))

                    points = [
                        (x1, y1),
                        (x2, y1),
                        (x2, y2),
                        (x1, y2),
                    ]

                else:
                    # Fabric Polygon points are relative to its center.
                    left = float(obj.get("left", 0))
                    top = float(obj.get("top", 0))
                    origin_x = obj.get("originX", "center")
                    origin_y = obj.get("originY", "center")

                    center_x = (
                        left if origin_x == "center"
                        else left - float(obj.get("width", 0)) / 2
                        if origin_x == "right"
                        else left + float(obj.get("width", 0)) / 2
                    )
                    center_y = (
                        top if origin_y == "center"
                        else top - float(obj.get("height", 0)) / 2
                        if origin_y == "bottom"
                        else top + float(obj.get("height", 0)) / 2
                    )

                    points = [
                        (
                            center_x + float(p["x"]) * float(obj.get("scaleX", 1)),
                            center_y + float(p["y"]) * float(obj.get("scaleY", 1)),
                        )
                        for p in obj.get("points", [])
                    ]

                points = [
                    (
                        max(0.0, min(float(canvas_width), x)) * scale_x,
                        max(0.0, min(float(canvas_height), y)) * scale_y,
                    )
                    for x, y in points
                ]

                if len(set(points)) < 3:
                    continue

                saved.append({
                    "name": name,
                    "polygon": points,
                })

            if saved:
                st.session_state["sf_areas"] = saved
                st.success(f"Saved {len(saved)} zones.")
            else:
                st.warning("The drawn shapes were invalid. Please draw them again.")




    areas = st.session_state.get("sf_areas", [])

    if areas:
        st.subheader("Step 2 — Review your zones")

        edited = st.data_editor(
            pd.DataFrame({"Area": [area["name"] for area in areas]}),
            hide_index=True,
            key=f"sf_rename_{widget_suffix}_{len(areas)}",
        )

        renamed = []
        for area, name in zip(areas, edited["Area"]):
            clean_name = str(name).strip() or area["name"]
            renamed.append({**area, "name": clean_name})

        st.session_state["sf_areas"] = renamed
        areas = renamed

        st.caption(f"{len(areas)} zones ready for analysis.")
    else:
        st.info("Draw and save at least one zone before running analysis.")

    st.subheader("Step 3 — Run the analysis")

    if st.button(
        "Run Analysis",
        type="primary",
        disabled=not areas,
        key=f"sf_run_analysis_{widget_suffix}",
    ):
        try:
            with st.spinner("Detecting and tracking people..."):
                result = process_video(video_bytes, uploaded_video.name)

            (
                tracks, result_width, result_height, fps, duration,
                ids, annotated_preview, analysis_clean, objects,processed_video_path
            ) = result

            st.session_state["sf_processed_video"] = processed_video_path
            st.session_state["sf_tracks"] = tracks
            st.session_state["sf_meta"] = {
                "width": result_width,
                "height": result_height,
                "fps": fps,
                "duration": duration,
                "ids": ids,
            }
            st.session_state["sf_preview"] = annotated_preview
            st.session_state["sf_results_video_key"] = video_key
            st.session_state.pop("sf_validation", None)

            st.success("Analysis completed.")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    # Only show results belonging to the currently selected video.
    if st.session_state.get("sf_results_video_key") == video_key:
        render(
            st.session_state["sf_tracks"],
            st.session_state["sf_meta"],
            st.session_state.get("sf_areas", []),
            synthetic=False,
            background=st.session_state.get("sf_clean"),
        )



processed_video_path = st.session_state.get("sf_processed_video")

if processed_video_path and Path(processed_video_path).is_file():
    st.subheader("Processed video")

    video_bytes = Path(processed_video_path).read_bytes()

    st.video(video_bytes, format="video/mp4")

    st.download_button(
        "Download processed video",
        data=video_bytes,
        file_name=Path(processed_video_path).name,
        mime="video/mp4",
    )
