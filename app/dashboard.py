import hashlib, io, re, shutil, subprocess, tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter
from PIL import Image

OUTPUT_DIR = Path(tempfile.gettempdir())   # no new folder in your project

st.set_page_config(page_title="StoreFlow | Dwell-Time Analytics", page_icon="🗺️", layout="wide")
st.markdown("""<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"] { background:#111318; color:#e8eaf0; }
[data-testid="stHeader"] { background:rgba(17,19,24,.96); }
[data-testid="stSidebar"] { background:#191c23; border-right:1px solid #2b303b; }
[data-testid="stSidebar"] * { color:#e8eaf0; }
[data-testid="stMetric"] { background:#1b1f28; border:1px solid #303644; padding:14px; border-radius:12px; }
[data-testid="stMetricLabel"] { color:#aeb6c5; } [data-testid="stMetricValue"] { color:#f4f6fb; }
.block-container { padding-top:1.4rem; max-width:1800px; }
h1,h2,h3,h4 { color:#f4f6fb; } p,label,.stMarkdown,[data-testid="stCaptionContainer"] { color:#c4cad5; }
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div, [data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input { background:#202530; color:#f4f6fb; border-color:#3a4252; }
.stButton>button, [data-testid="stDownloadButton"] button { background:#252b37; color:#f4f6fb;
    border:1px solid #424b5d; border-radius:8px; }
.stButton>button:hover, [data-testid="stDownloadButton"] button:hover { background:#333c4d; border-color:#6d7b95; color:#fff; }
video { width:100%; max-height:560px; border-radius:12px; background:#000; }

/* ---- file uploader (dark, matches the app) ---- */
[data-testid="stFileUploader"] > div, [data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div, [data-testid="stFileUploaderDropzone"] {
    background:#202530 !important; border-radius:12px !important; }
[data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"] {
    border:1.5px dashed #3a4252 !important; }
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p { color:#c4cad5 !important; }
[data-testid="stFileUploaderDropzone"] button { background:#252b37 !important; border:1px solid #424b5d !important;
    border-radius:8px !important; }
[data-testid="stFileUploaderDropzone"] button * { color:#f4f6fb !important; }
[data-testid="stFileUploader"] ul, [data-testid="stFileUploader"] li { background:transparent !important; }
[data-testid="stFileUploaderFile"] { background:#2a3040 !important; border:1px solid #3a4252 !important;
    border-radius:10px !important; padding:6px 10px !important; margin-top:8px !important; }
[data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"] { color:#e8eaf0 !important; opacity:1 !important; }
[data-testid="stFileUploaderFile"] svg, [data-testid="stFileUploader"] svg { fill:#aeb6c5 !important; color:#aeb6c5 !important; }
</style>""", unsafe_allow_html=True)

# ---------------- constants ----------------
STILL_SPEED, MIN_STOP_S, BIN_S, OTHER, MAX_IMG_SIDE = 0.03, 3.0, 5, "Open floor", 768
VLM_MODELS = {
    "SmolVLM-500M (fastest, weakest)": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "Qwen2-VL-2B (balanced)": "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2.5-VL-3B (best, slow on CPU)": "Qwen/Qwen2.5-VL-3B-Instruct",
}
STATUS_COLOR = {"Congestion candidate": "#c0392b", "Dead-zone candidate": "#7f8c8d",
                "Hot / high engagement": "#d68910", "High traffic, few stops": "#2471a3"}
VLM_PROMPT = """You are a retail store-layout analyst. You are given:
IMAGE 1: a heat map of where shoppers were observed (red/orange = more presence). Coloured outlines are store areas: red = congestion candidate, grey = dead-zone candidate, orange = high engagement, blue = high traffic but few stops.
{frame_line}
Below are measured numbers from people tracking. Use them with what you SEE. Never invent numbers.

{facts}

Write a plain-text report with these sections:
1. WHAT THE HEAT MAP SHOWS: where customers go, which areas are busy or empty, visible routes.
2. WHAT IS HAPPENING AND WHY: start each statement with "Observed:" or "Hypothesis:" (a reason that must be checked in the store).
3. PROBLEMS FOUND: queues, crowding, ignored areas, wasted space, dead ends.
4. HOW TO IMPROVE THE STORE: five concrete actions, each with the metric that would show it worked.
5. WHAT TO CHECK BEFORE ACTING: data limits (tracking errors, no validation, synthetic data, camera coverage).
Be specific and brief. If the data is synthetic or not validated, say so at the start."""

# ---------------- sidebar ----------------
st.sidebar.title("🗺️ StoreFlow")
synthetic_mode = st.sidebar.radio("Mode", ["📹 Recorded video", "🧪 Synthetic demo"],
                                  label_visibility="collapsed").startswith("🧪")
with st.sidebar.expander("Analysis settings"):
    min_visit = st.slider("Min. seconds to count as a visit", 1, 10, 3)
    occ_thr = st.slider("Congestion: people at once", 2, 12, 3)
    persist_s = st.slider("Congestion: must last (s)", 10, 180, 30)
    show_paths = st.checkbox("Show movement paths on heat map", value=False)
    show_labels = st.checkbox("Show area labels", value=True)
with st.sidebar.expander("AI vision analyst"):
    vlm_name = VLM_MODELS[st.selectbox("Hugging Face vision model", list(VLM_MODELS), index=1)]
uploaded_video = None if synthetic_mode else st.sidebar.file_uploader(
    "Upload store video", type=["mp4", "avi", "mov", "mkv"])

st.title("StoreFlow")
st.caption("Dwell-Time & Bottleneck Mapper")

# ---------------- helpers ----------------
def poly(a):
    """Every area becomes an Nx2 polygon (rectangles included)."""
    if "polygon" in a:
        return np.asarray(a["polygon"], float).reshape(-1, 2)
    x1, y1, x2, y2 = a["bounds"]
    return np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], float)


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def make_density(x, y, w, h, bins=120):
    hist, xe, ye = np.histogram2d(x, y, bins=[bins, max(30, round(bins * h / w))], range=[[0, w], [0, h]])
    return gaussian_filter(hist, sigma=2.5), xe, ye


def draw_density(ax, density, xe, ye, background=None):
    peak = float(density.max())
    if peak <= 0:
        return
    if background is not None:
        ax.imshow(background, extent=(xe[0], xe[-1], ye[-1], ye[0]), aspect="auto")
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad((1, 1, 1, 0))
    mesh = ax.pcolormesh(xe, ye, np.ma.masked_where(density < peak * 0.08, density).T, cmap=cmap,
                         shading="auto", vmin=0, vmax=peak, alpha=0.6 if background is not None else 0.92)
    plt.colorbar(mesh, ax=ax, label="Relative presence (observations, not unique visitors)", fraction=0.035, pad=0.02)


def draw_areas(ax, at, areas, labels=True):
    status = dict(zip(at["Area"], at["Status"])) if len(at) else {}
    for a in areas:
        p = poly(a)
        if len(p) < 3:
            continue
        col = STATUS_COLOR.get(status.get(a["name"]), "#9aa5b8")
        ax.add_patch(Polygon(p, closed=True, fill=True, facecolor=(1, 1, 1, 0.07), edgecolor=col, linewidth=2.2))
        if labels:
            cx, cy = p.mean(axis=0)
            ax.text(cx, cy, a["name"], ha="center", va="center", fontsize=9, color="#382719",
                    bbox=dict(facecolor="#fff4d8", alpha=0.92, edgecolor="#d8b477", boxstyle="round,pad=0.3"))


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def shrink(img, side=MAX_IMG_SIDE):
    img = img.copy()
    img.thumbnail((side, side))
    return img


def find_ffmpeg():
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_browser_mp4(src, dst):
    """OpenCV's mp4v is not browser-playable; re-encode to H.264. Falls back to the raw file."""
    exe = find_ffmpeg()
    if exe:
        cmd = [exe, "-y", "-i", src, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
               "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(dst)]
        try:
            if subprocess.run(cmd, capture_output=True, timeout=900).returncode == 0 and Path(dst).stat().st_size > 0:
                return True
        except Exception:
            pass
    shutil.copy(src, dst)
    return False


def extract_preview(video_bytes, filename):
    """Clean middle frame, used for drawing zones before detection runs."""
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".mp4", delete=False) as f:
        f.write(video_bytes)
    cap = cv2.VideoCapture(f.name)
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open the uploaded video.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not extract a preview frame.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), w, h
    finally:
        cap.release()
        Path(f.name).unlink(missing_ok=True)


def canvas_shape_to_points(obj):
    """Fabric.js rect / path / polygon -> list of (x, y) in canvas pixels."""
    t = obj.get("type", "").lower()
    sx, sy = float(obj.get("scaleX", 1)), float(obj.get("scaleY", 1))
    left, top = float(obj.get("left", 0)), float(obj.get("top", 0))
    w, h = float(obj.get("width", 0)) * sx, float(obj.get("height", 0)) * sy
    if t == "rect":
        return [(left, top), (left + w, top), (left + w, top + h), (left, top + h)]
    if t == "path":   # the drawable-canvas polygon tool produces paths
        raw = [(float(c[-2]), float(c[-1])) for c in obj.get("path", []) if c and c[0] in ("M", "L") and len(c) >= 3]
    else:
        raw = [(float(p["x"]), float(p["y"])) for p in obj.get("points", [])]
    if not raw:
        return []
    off = obj.get("pathOffset") or {}
    ox = off.get("x", (min(p[0] for p in raw) + max(p[0] for p in raw)) / 2)
    oy = off.get("y", (min(p[1] for p in raw) + max(p[1] for p in raw)) / 2)
    ax, ay = obj.get("originX", "left"), obj.get("originY", "top")
    cx = left + w / 2 if ax == "left" else left - w / 2 if ax == "right" else left
    cy = top + h / 2 if ay == "top" else top - h / 2 if ay == "bottom" else top
    return [(cx + (x - ox) * sx, cy + (y - oy) * sy) for x, y in raw]

# ---------------- video processing ----------------
def process_video(video_bytes, filename, sample_fps=5):
    from ultralytics import YOLO
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_") or "video"
    out_path = OUTPUT_DIR / f"storeflow_processed_{stem}.mp4"
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".mp4", delete=False) as f:
        f.write(video_bytes)
    raw = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    cap, writer = cv2.VideoCapture(f.name), None
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open the uploaded video.")
        fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0 or total <= 0:
            raise RuntimeError("Video metadata is invalid.")
        writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Could not create the output video (OpenCV codec support).")

        model, rows, ids_seen, last = YOLO("yolo11n.pt"), [], set(), None
        stride, n = max(1, round(fps / sample_fps)), 0
        bar, status = st.progress(0), st.empty()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n % stride == 0:
                res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                                  conf=0.35, verbose=False)[0]
                last = res.plot()
                b = res.boxes
                if b is not None and len(b):
                    ids = b.id.int().cpu().tolist() if b.id is not None else [-1] * len(b)
                    for box, tid in zip(b.xyxy.cpu().numpy(), ids):
                        rows.append({"timestamp": n / fps, "track_id": int(tid),
                                     "x_px": float((box[0] + box[2]) / 2), "y_px": float(box[3])})
                        if tid >= 0:
                            ids_seen.add(tid)
            writer.write(last if last is not None else frame)
            n += 1
            if n % 10 == 0:
                bar.progress(min(n / total, 1.0))
                status.write(f"Processing {n:,}/{total:,} frames")
        writer.release()
        writer = None
        if not rows:
            raise RuntimeError("No people were detected. Try a clearer video.")
        status.write("Encoding processed video for the browser...")
        to_browser_mp4(raw, out_path)
        status.empty()
        bar.progress(1.0)
        meta = {"width": w, "height": h, "fps": fps, "duration": total / fps, "ids": len(ids_seen)}
        return pd.DataFrame(rows), meta, str(out_path)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        for p in (f.name, raw):
            Path(p).unlink(missing_ok=True)

# ---------------- synthetic data ----------------
SYN_W, SYN_H = 17.0, 15.0
SYN_AREAS = [{"name": n, "bounds": b} for n, b in [
    ("Entrance", (1, 12, 4, 14.5)), ("Produce", (1, 2, 5, 5)), ("Aisle 1", (6, 2, 8, 9)),
    ("Aisle 2", (9, 2, 11, 9)), ("Dairy", (12, 2, 15.5, 5)), ("Checkout", (11, 11, 16, 14))]]


@st.cache_data(show_spinner=False)
def simulate_store(seed=42, n=140):
    rng = np.random.default_rng(seed)
    bounds = {a["name"]: a["bounds"] for a in SYN_AREAS}
    browse, weights, rows = ["Produce", "Aisle 1", "Aisle 2", "Dairy"], [0.30, 0.40, 0.05, 0.25], []  # Aisle 2 ignored
    for tid in range(1, n + 1):
        t = rng.uniform(0, 1200) if rng.random() < 0.5 else rng.normal(900, 80)
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
            for _ in range(int(rng.gamma(5, 6)) if name == "Checkout" else int(rng.gamma(4, 2.5))):
                p = pos + rng.normal(0, 0.1, 2)
                rows.append((t, tid, p[0], p[1]))
                t += 1
    df = pd.DataFrame(rows, columns=["timestamp", "track_id", "x_px", "y_px"])
    df["timestamp"] -= df["timestamp"].min()
    return df

# ---------------- analytics ----------------
def prepare(tracks, areas, width):
    t = tracks[tracks["track_id"] >= 0].sort_values(["track_id", "timestamp"]).copy()
    if t.empty:
        return t, 1.0
    roll = lambda s: s.rolling(3, center=True, min_periods=1).median()
    g = t.groupby("track_id")
    for c in ("x_px", "y_px"):
        t[c + "_s"] = g[c].transform(roll)
    g = t.groupby("track_id")
    dt = g["timestamp"].diff()
    t["speed"] = (np.hypot(g["x_px_s"].diff(), g["y_px_s"].diff()) / dt / width).where(dt > 0)
    t["speed"] = t.groupby("track_id")["speed"].transform(roll)
    interval = float(np.nanmedian(dt)) if dt.notna().any() else 1.0
    interval = interval if interval > 0 else 1.0
    t["dur"] = (t.groupby("track_id")["timestamp"].shift(-1) - t["timestamp"]).clip(upper=interval * 2.5).fillna(0.0)

    t["area"] = OTHER
    xy = t[["x_px_s", "y_px_s"]].to_numpy()
    for a in sorted(areas, key=lambda a: poly_area(poly(a)), reverse=True):  # smallest zone wins overlaps
        p = poly(a)
        if len(p) >= 3:
            t.loc[MplPath(p).contains_points(xy), "area"] = a["name"]
    return t, interval


def build_runs(t, min_visit):
    t["run"] = (t["area"] != t.groupby("track_id")["area"].shift()).cumsum()
    t["still_dur"] = np.where(t["speed"] < STILL_SPEED, t["dur"], 0.0)
    runs = t.groupby("run").agg(track_id=("track_id", "first"), area=("area", "first"), t0=("timestamp", "min"),
                                dur=("dur", "sum"), still=("still_dur", "sum"), n=("timestamp", "size")).reset_index(drop=True)
    return runs[(runs["n"] >= 2) & (runs["dur"] >= min_visit)]


def area_table(t, runs, areas):
    total = max(1, runs[runs["area"] != OTHER]["track_id"].nunique())
    rows = []
    for a in areas:
        r, rt = runs[runs["area"] == a["name"]], t[t["area"] == a["name"]]
        vis = r["track_id"].nunique()
        stoppers = r.loc[r["still"] >= MIN_STOP_S, "track_id"].nunique()
        if len(rt):
            occ = rt.groupby(rt["timestamp"] // BIN_S)["track_id"].nunique()
            peak, cong, spd = int(occ.max()), int((occ >= occ_thr).sum()) * BIN_S, float(rt["speed"].median())
        else:
            peak, cong, spd = 0, 0, np.nan
        rows.append({"Area": a["name"], "Visitors": vis, "Share %": round(100 * vis / total, 1),
                     "Median visit s": round(float(r["dur"].median()), 1) if len(r) else np.nan,
                     "Presence s": round(float(r["dur"].sum()), 1), "Stationary s": round(float(r["still"].sum()), 1),
                     "Stop rate %": round(100 * stoppers / vis, 1) if vis else 0.0,
                     "Peak occupancy": peak, "Congested s": cong,
                     "Median speed": round(spd, 3) if np.isfinite(spd) else np.nan})
    df = pd.DataFrame(rows)
    if df.empty:
        df["Status"] = []
        return df
    mean_share, med_visit, enough = df["Share %"].mean(), df["Median visit s"].median(), total >= 10 and len(df) >= 3

    def status(r):
        if not enough: return "Not enough data"
        if r["Congested s"] >= persist_s and (pd.isna(r["Median speed"]) or r["Median speed"] < 0.06): return "Congestion candidate"
        if any(k in r["Area"].lower() for k in ("entr", "exit")): return "Normal"
        if r["Share %"] < 0.35 * mean_share: return "Dead-zone candidate"
        if r["Stop rate %"] >= 60 and r["Median visit s"] > med_visit: return "Hot / high engagement"
        if r["Share %"] >= mean_share and r["Stop rate %"] < 25: return "High traffic, few stops"
        return "Normal"
    df["Status"] = df.apply(status, axis=1)
    return df


def flow_tables(runs):
    r = runs[runs["area"] != OTHER].sort_values(["track_id", "t0"]).copy()
    if r.empty:
        e = pd.Series(dtype=int)
        return pd.DataFrame(columns=["Move", "People"]), e, e, e
    r = r[r["area"] != r.groupby("track_id")["area"].shift()]
    r["next"] = r.groupby("track_id")["area"].shift(-1)
    flows = r.dropna(subset=["next"]).groupby(["area", "next"]).size().reset_index(name="People").sort_values("People", ascending=False)
    flows["Move"] = flows["area"] + " → " + flows["next"]
    paths = r.groupby("track_id")["area"].apply(" → ".join).value_counts().head(5)
    return (flows[["Move", "People"]].head(10), paths,
            r.groupby("track_id")["area"].first().value_counts(), r.groupby("track_id")["area"].last().value_counts())


def quality_stats(tracks, t):
    life = t.groupby("track_id")["timestamp"].agg(lambda s: s.max() - s.min())
    return {"untracked_pct": round(100 * float((tracks["track_id"] < 0).mean()), 1),
            "short_track_pct": round(100 * float((life < 3).mean()), 1) if len(life) else 0.0,
            "track_ids": int(t["track_id"].nunique())}


def rule_findings(at, entries, exits, paths):
    recs = at.to_dict("records")
    if not recs:
        return []
    byv = sorted(recs, key=lambda r: r["Visitors"], reverse=True)
    out = [f"**Where people go:** {byv[0]['Area']} gets the most visitors ({byv[0]['Share %']:.0f}%); "
           f"{byv[-1]['Area']} the fewest ({byv[-1]['Share %']:.0f}%)."]
    if len(entries) and len(exits):
        out.append(f"**Entry / exit:** most shoppers start in **{entries.index[0]}** and finish in **{exits.index[0]}**.")
    if len(paths):
        out.append(f"**Most common route:** {paths.index[0]} ({int(paths.iloc[0])} shoppers).")
    msg = {"Congestion candidate": lambda r: f"🔴 **{r['Area']}**: up to {r['Peak occupancy']} people at once, congested {r['Congested s']:.0f}s. Consider more capacity or spreading demand.",
           "Dead-zone candidate": lambda r: f"⚪ **{r['Area']}**: only {r['Share %']:.0f}% of shoppers. Check stockouts, blocked access, sightlines, camera coverage first.",
           "High traffic, few stops": lambda r: f"🔵 **{r['Area']}**: {r['Share %']:.0f}% pass but only {r['Stop rate %']:.0f}% stop. Test a promo display or signage.",
           "Hot / high engagement": lambda r: f"🟠 **{r['Area']}**: {r['Stop rate %']:.0f}% stop, median visit {r['Median visit s']}s. Protect this space."}
    return out + [msg[r["Status"]](r) for r in recs if r["Status"] in msg]


def facts_text(at, flows, paths, entries, exits, q, source, validation):
    val = "NOT validated against manual counts yet." if validation is None else (
        f"Validated: mean abs count error {validation['count_err']:.0f}%"
        + (f", dwell error {validation['dwell_err']:.0f}%" if validation.get("dwell_err") is not None else "") + ".")
    return (f"DATA SOURCE: {source}\nVALIDATION: {val}\n"
            f"TRACK QUALITY: {q['track_ids']} track IDs, {q['untracked_pct']}% untracked, {q['short_track_pct']}% tracks <3s (possible ID switches).\n"
            "NOTES: speeds are frame-widths/second. Visitors are approximate. Stop = stationary >=3s. Status labels are candidates, not verdicts.\n\n"
            f"AREA TABLE (CSV):\n{at.to_csv(index=False)}\nTOP MOVES:\n{flows.to_csv(index=False)}\n"
            f"COMMON ROUTES:\n{paths.to_string()}\n\nFIRST AREA:\n{entries.to_string()}\n\nLAST AREA:\n{exits.to_string()}\n")

# ---------------- vision-language model ----------------
@st.cache_resource(show_spinner=False)
def load_vlm(name):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    cuda = torch.cuda.is_available()
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.float16 if cuda else torch.float32)
    return proc, model.to("cuda" if cuda else "cpu").eval()


def run_vlm(name, images, prompt):
    import torch
    proc, model = load_vlm(name)
    images = [shrink(i) for i in images]
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=[text], images=[images] if "smol" in name.lower() else images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=800, do_sample=False)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

# ---------------- dashboard ----------------
def render(tracks, meta, areas, synthetic, background=None):
    width, height = meta["width"], meta["height"]
    source = "SYNTHETIC simulated shoppers (NOT real measurements)" if synthetic else "Real video, YOLO + ByteTrack, unvalidated unless stated"
    if synthetic:
        st.error("🧪 SYNTHETIC DEMO: these shoppers are simulated. Nothing below is a real store measurement.")
    t, _ = prepare(tracks, areas, width)
    if t.empty:
        st.warning("No tracked people (track IDs missing). Try a clearer video.")
        return
    runs = build_runs(t, min_visit)
    at = area_table(t, runs, areas)
    flows, paths, entries, exits = flow_tables(runs)
    q = quality_stats(tracks, t)
    total_sys = int(runs.loc[runs["area"] != OTHER, "track_id"].nunique())
    tab_over, tab_flow, tab_val, tab_ai = st.tabs(["📊 Overview", "🔀 Flow", "✅ Validation", "🤖 AI insights"])
    validation, heat_img = None, None

    with tab_over:
        c = st.columns(4)
        c[0].metric("Duration", f"{meta['duration']:.0f}s")
        c[1].metric("Shoppers (approx.)", total_sys)
        c[2].metric("Busiest area", at.sort_values("Visitors", ascending=False)["Area"].iloc[0] if len(at) else "-")
        c[3].metric("Flagged areas", int((~at["Status"].isin(["Normal", "Not enough data"])).sum()) if len(at) else 0)
        if q["short_track_pct"] > 30:
            st.warning(f"{q['short_track_pct']}% of tracks are shorter than 3s: likely ID switches, counts may be inflated.")
        if q["untracked_pct"] > 10:
            st.warning(f"{q['untracked_pct']}% of detections have no track ID and are ignored in area metrics.")
        if not synthetic:
            st.info("These numbers are **not validated yet** unless you completed the Validation tab.")
        density, xe, ye = make_density(tracks["x_px"].to_numpy(), tracks["y_px"].to_numpy(), width, height)
        fig, ax = plt.subplots(figsize=(11, 7))
        draw_density(ax, density, xe, ye, background=background)
        draw_areas(ax, at, areas, show_labels)
        if show_paths:
            for _, g in tracks[tracks["track_id"] >= 0].groupby("track_id"):
                g = g.sort_values("timestamp")
                ax.plot(g["x_px"], g["y_px"], alpha=0.3, linewidth=0.8, color="#8b5e91")
        ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal",
               title="SYNTHETIC heat map (simulated)" if synthetic else "Occupancy by image position")
        ax.set_facecolor("#eee5d3")
        heat_img = fig_to_pil(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Outline colours: red = congestion, grey = dead-zone candidate, orange = hot, blue = high traffic/few stops. "
                   "The heat map shows observed presence, not unique visitors or interest.")
        st.subheader("Area analytics")
        st.dataframe(at, hide_index=True, use_container_width=True)
        st.caption("Presence = time inside area. Stationary = time below walking speed. Speed is in frame-widths/s, not m/s.")
        st.download_button("Download area analytics CSV", at.to_csv(index=False).encode(), "storeflow_area_analytics.csv", "text/csv")

    with tab_flow:
        st.subheader("Where customers actually go")
        if flows.empty:
            st.info("No area-to-area movement found. Define more areas or use a longer video.")
        else:
            a, b = st.columns([3, 2])
            a.markdown("**Top moves between areas**")
            a.bar_chart(flows.set_index("Move")["People"])
            b.markdown("**Most common routes**")
            b.dataframe(paths.rename("Shoppers").reset_index().rename(columns={"area": "Route"}), hide_index=True, use_container_width=True)
            c, d = st.columns(2)
            c.markdown("**First area entered**"); c.dataframe(entries.rename("Shoppers"), use_container_width=True)
            d.markdown("**Last area visited**"); d.dataframe(exits.rename("Shoppers"), use_container_width=True)
        st.caption("Routes assume one track ID per person; ID switches split journeys, so treat routes as indicative.")

    with tab_val:
        st.subheader("Validate against manually checked video")
        if synthetic:
            st.info("Validation only applies to real video.")
        else:
            st.write("Watch the video yourself. For each area, count the different people who entered and estimate typical dwell time.")
            base = pd.DataFrame({"Area": at["Area"], "Manual customers": np.nan, "Manual typical dwell (s)": np.nan})
            ed = st.data_editor(base, hide_index=True, disabled=["Area"], key="val_" + "|".join(at["Area"]),
                                column_config={"Manual customers": st.column_config.NumberColumn(min_value=0, step=1),
                                               "Manual typical dwell (s)": st.column_config.NumberColumn(min_value=0)})
            manual_total = st.number_input("Manual total customers in the clip (optional)", min_value=0, value=0, step=1)
            m = at[["Area", "Visitors", "Median visit s"]].merge(ed, on="Area")
            pct = lambda s, mnl: np.where(m[mnl] > 0, (m[s] - m[mnl]) / m[mnl] * 100, np.nan)
            m["Count error %"], m["Dwell error %"] = pct("Visitors", "Manual customers"), pct("Median visit s", "Manual typical dwell (s)")
            checked = m.dropna(subset=["Count error %"])
            tot_err = (total_sys - manual_total) / manual_total * 100 if manual_total > 0 else None
            if tot_err is not None:
                st.metric("Total customers: system vs manual", f"{total_sys} vs {manual_total}", f"{tot_err:+.0f}%")
            if checked.empty and tot_err is None:
                st.info("Enter at least one manual count to see accuracy.")
            else:
                st.dataframe(m.round(1), hide_index=True, use_container_width=True)
                errs = ([checked["Count error %"].abs().mean()] if not checked.empty else []) + ([abs(tot_err)] if tot_err is not None else [])
                count_err, dw = float(np.mean(errs)), m["Dwell error %"].abs().dropna()
                validation = {"count_err": count_err, "dwell_err": float(dw.mean()) if len(dw) else None}
                st.session_state["sf_validation"] = validation
                worst = max(count_err, validation["dwell_err"] or 0)
                if worst <= 15: st.success(f"✅ Good: mean count error {count_err:.0f}%.")
                elif worst <= 30: st.warning(f"⚠️ Fair: mean count error {count_err:.0f}%. Rough trends only.")
                else: st.error(f"❌ Poor: mean count error {count_err:.0f}%. Don't base layout decisions on this yet.")

    with tab_ai:
        validation = validation or st.session_state.get("sf_validation")
        st.subheader("What the data says and what to do")
        for line in rule_findings(at, entries, exits, paths):
            st.markdown("- " + line)
        st.divider()
        st.markdown(f"**AI vision analyst:** `{vlm_name}` reads the heat map"
                    + ("" if synthetic or background is None else ", a video frame")
                    + " and the measured numbers. Its 'why' statements are hypotheses to check in the store.")
        images = [heat_img] if heat_img is not None else []
        if background is not None and not synthetic:
            images.append(Image.fromarray(background))
            frame_line = "IMAGE 2: a plain frame from the store video, to relate the heat map to real shelves and doors."
        else:
            frame_line = "(No video frame; this is a simulated floor plan.)" if synthetic else ""
        facts = facts_text(at, flows, paths, entries, exits, q, source, validation)
        prompt = VLM_PROMPT.format(frame_line=frame_line, facts=facts)
        for col, img, cap in zip(st.columns(max(1, len(images))), images, ["Heat map the AI sees", "Video frame the AI sees"]):
            col.image(img, caption=cap, use_container_width=True)
        with st.expander("Numbers sent to the model together with the images"):
            st.code(facts)
        key = hashlib.md5((vlm_name + facts + str(len(images))).encode()).hexdigest()
        if st.button("Analyse heat map with AI", type="primary"):
            try:
                with st.spinner("Loading vision model (first run downloads it) and analysing..."):
                    st.session_state["ai_" + key] = run_vlm(vlm_name, images, prompt)
            except Exception as exc:
                st.error(f"AI analysis failed: {exc}. The rule-based findings above still apply.")
        if "ai_" + key in st.session_state:
            st.markdown("### AI report")
            st.write(st.session_state["ai_" + key])
            st.caption("AI-generated, not ground truth. Small vision models can misread charts; check against the table.")

# ---------------- pages ----------------
if synthetic_mode:
    tracks = simulate_store()
    render(tracks, {"width": SYN_W, "height": SYN_H, "duration": float(tracks["timestamp"].max()), "fps": 1.0,
                    "ids": int(tracks["track_id"].nunique())}, SYN_AREAS, synthetic=True)
    st.stop()

if uploaded_video is None:
    st.info("Upload a store video using the **Upload store video** control in the sidebar (MP4, AVI, MOV, MKV).")
    st.stop()
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Missing component. Run: python -m pip install streamlit-drawable-canvas")
    st.stop()

video_bytes = uploaded_video.getvalue()
vhash = hashlib.md5(video_bytes).hexdigest()
vkey, sfx = (uploaded_video.name, vhash), vhash[:10]

if st.session_state.get("sf_upload_key") != vkey:   # new video: reset everything
    for k in ("sf_tracks", "sf_meta", "sf_clean", "sf_areas", "sf_validation",
              "sf_results_video_key", "sf_processed_video", "sf_preview_size"):
        st.session_state.pop(k, None)
    st.session_state["sf_upload_key"] = vkey
    try:
        with st.spinner("Preparing a frame for zone selection..."):
            clean, pw, ph = extract_preview(video_bytes, uploaded_video.name)
        st.session_state["sf_clean"], st.session_state["sf_preview_size"] = clean, (pw, ph)
    except Exception as exc:
        st.error(f"Could not prepare video preview: {exc}")
        st.stop()

with st.expander("Preview uploaded video"):
    st.video(video_bytes)

width, height = st.session_state["sf_preview_size"]
clean = st.session_state["sf_clean"]

st.subheader("Step 1 — Draw your store zones")
st.caption("Draw one shape per zone, then name each below in the order you drew them. "
           "Polygon tool: click each corner, double-click to finish. Switching tool resets the canvas, so save first.")
mode = st.radio("Drawing tool", ["rect", "polygon"], horizontal=True, key=f"sf_mode_{sfx}",
                format_func=lambda m: "Rectangle" if m == "rect" else "Polygon")
cvh = min(650, max(200, round(800 * height / width)))
cvw = max(1, round(cvh * width / height))
canvas = st_canvas(fill_color="rgba(245, 185, 65, 0.20)", stroke_width=2, stroke_color="#a65a16",
                   background_image=Image.fromarray(clean).resize((cvw, cvh)), update_streamlit=True,
                   height=cvh, width=cvw, drawing_mode=mode, key=f"sf_area_canvas_{sfx}_{mode}")

shapes = [o for o in (canvas.json_data or {}).get("objects", [])
          if o.get("type", "").lower() in ("rect", "polygon", "path")
          and len(canvas_shape_to_points(o)) >= 3]
st.subheader("Name your zones")
names = [st.text_input(f"Zone {i + 1} name", key=f"sf_area_name_{sfx}_{i}", placeholder=f"e.g. Browsing Area {i + 1}").strip()
         for i in range(len(shapes))]

if st.button("Save drawn zones", key=f"sf_save_{sfx}"):
    if not shapes:
        st.warning("Draw at least one rectangle or polygon first.")
    elif any(not n for n in names):
        st.warning("Enter a name for every drawn shape.")
    else:
        sx, sy, saved = width / cvw, height / cvh, []
        for name, obj in zip(names, shapes):
            pts = [(max(0.0, min(float(cvw), x)) * sx, max(0.0, min(float(cvh), y)) * sy)
                   for x, y in canvas_shape_to_points(obj)]
            if len(set(pts)) >= 3:
                saved.append({"name": name, "polygon": pts})
        if saved:
            st.session_state["sf_areas"] = saved
            st.success(f"Saved {len(saved)} zones.")
        else:
            st.warning("The drawn shapes were invalid. Please draw them again.")

areas = st.session_state.get("sf_areas", [])
if areas:
    st.subheader("Step 2 — Review your zones")
    edited = st.data_editor(pd.DataFrame({"Area": [a["name"] for a in areas]}), hide_index=True,
                            key=f"sf_rename_{sfx}_{len(areas)}")
    areas = [{**a, "name": str(n).strip() or a["name"]} for a, n in zip(areas, edited["Area"])]
    st.session_state["sf_areas"] = areas
    st.caption(f"{len(areas)} zones ready for analysis.")
else:
    st.info("Draw and save at least one zone before running analysis.")

st.subheader("Step 3 — Run the analysis")
if st.button("Run Analysis", type="primary", disabled=not areas, key=f"sf_run_{sfx}"):
    try:
        with st.spinner("Detecting and tracking people..."):
            tracks, meta, out_path = process_video(video_bytes, uploaded_video.name)
        st.session_state.update(sf_tracks=tracks, sf_meta=meta, sf_processed_video=out_path, sf_results_video_key=vkey)
        st.session_state.pop("sf_validation", None)
        st.success("Analysis completed.")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

if st.session_state.get("sf_results_video_key") == vkey:
    render(st.session_state["sf_tracks"], st.session_state["sf_meta"], st.session_state.get("sf_areas", []),
           synthetic=False, background=clean)
    out = st.session_state.get("sf_processed_video")
    if out and Path(out).is_file():
        st.subheader("Processed video")
        data = Path(out).read_bytes()
        st.video(data, format="video/mp4")
        st.download_button("Download processed video", data, Path(out).name, "video/mp4")
