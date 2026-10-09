
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

st.set_page_config(
    page_title="StoreFlow | Store Analytics",
    page_icon="🗺️",
    layout="wide",
)

st.title("StoreFlow")
st.caption("Dwell-Time & Bottleneck Mapper")

# -------------------- SIDEBAR --------------------

mode = st.sidebar.selectbox(
    "Data source",
    ["Synthetic demo", "Recorded video"],
)

show_heatmap = st.sidebar.checkbox(
    "Show traffic heatmap", True
)
show_zones = st.sidebar.checkbox(
    "Show zone labels", True
)

uploaded_video = None

if mode == "Recorded video":
    uploaded_video = st.sidebar.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    st.sidebar.caption(
        "Video analytics use image coordinates until "
        "a real floor calibration is configured."
    )

# -------------------- STORE FLOOR PLAN --------------------

zones = [
    {"name": "Produce", "x": 1, "y": 7, "w": 4, "h": 3},
    {"name": "Aisle A", "x": 6, "y": 7, "w": 2, "h": 6},
    {"name": "Aisle B", "x": 9, "y": 7, "w": 2, "h": 6},
    {"name": "Dairy", "x": 12, "y": 7, "w": 3, "h": 3},
    {"name": "Checkout", "x": 11, "y": 1, "w": 5, "h": 3},
    {"name": "Entrance", "x": 1, "y": 1, "w": 3, "h": 2},
]

# -------------------- SYNTHETIC DEMO --------------------

if mode == "Synthetic demo":
    rng = np.random.default_rng(42)

    people = pd.DataFrame({
        "x": rng.uniform(0.5, 16, 128),
        "y": rng.uniform(0.5, 14, 128),
    })

    people.loc[:34, ["x", "y"]] = np.column_stack([
        rng.normal(13.2, 0.8, 35),
        rng.normal(2.4, 0.6, 35),
    ])

    people.loc[35:59, ["x", "y"]] = np.column_stack([
        rng.normal(7, 0.5, 25),
        rng.uniform(7, 13, 25),
    ])

    people["x"] = people["x"].clip(0.2, 16.8)
    people["y"] = people["y"].clip(0.2, 14.8)

    c1, c2, c3 = st.columns(3)
    c1.metric("Simulated people", len(people))
    c2.metric("Average dwell time", "2m 34s")
    c3.metric("Example bottlenecks", 2)

    st.info(
        "Synthetic demo: these positions and metrics are "
        "illustrative, not measurements from a real store."
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    if show_heatmap:
        ax.hist2d(
            people["x"],
            people["y"],
            bins=[34, 28],
            range=[[0, 17], [0, 15]],
            cmap="YlOrRd",
            alpha=0.8,
        )

    for z in zones:
        ax.add_patch(Rectangle(
            (z["x"], z["y"]),
            z["w"],
            z["h"],
            linewidth=1.5,
            edgecolor="#263238",
            facecolor="none",
        ))

        if show_zones:
            ax.text(
                z["x"] + z["w"] / 2,
                z["y"] + z["h"] / 2,
                z["name"],
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                },
            )

    # Illustrative shelf blocks.
    for x in [6.2, 9.2]:
        for y in [8, 10, 12]:
            ax.add_patch(Rectangle(
                (x, y), 1.6, 0.65,
                facecolor="#78909c",
                edgecolor="#455a64",
            ))

    ax.set_xlim(0, 17)
    ax.set_ylim(0, 15)
    ax.set_aspect("equal")
    ax.set_xlabel("Illustrative store width")
    ax.set_ylabel("Illustrative store depth")
    ax.set_title("Synthetic store traffic heatmap")
    ax.grid(alpha=0.15)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Example bottleneck alerts")
    st.dataframe(pd.DataFrame([
        {
            "Zone": "Checkout",
            "Condition": "High density / low movement",
            "Status": "Illustrative",
        },
        {
            "Zone": "Aisle A",
            "Condition": "Traffic concentration",
            "Status": "Illustrative",
        },
    ]), hide_index=True, use_container_width=True)

# -------------------- RECORDED VIDEO --------------------

else:
    st.subheader("Video analytics")

    if uploaded_video is None:
        st.info("Upload a video in the sidebar to begin.")
        st.stop()

    st.video(uploaded_video)

    st.caption(
        "The first run may download the YOLO model weights. "
        "Processing can take time for long or high-resolution videos."
    )

    if st.button("Run person detection and tracking",
                 type="primary"):
        video_path = None
        cap = None
        records = []

        try:
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_video.name).suffix or ".mp4",
                delete=False,
            ) as temp:
                temp.write(uploaded_video.getvalue())
                video_path = temp.name

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                st.error("Could not open this video.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if not np.isfinite(fps) or fps <= 0 or total_frames <= 0:
                st.error("Video metadata is invalid or unsupported.")
                st.stop()

            duration = total_frames / fps

            st.write(
                f"Duration: {duration:.1f} seconds · "
                f"Frame rate: {fps:.2f} FPS · "
                f"Frames: {total_frames}"
            )

            from ultralytics import YOLO

            model = YOLO("yolo11n.pt")

            preview = st.empty()
            progress = st.progress(0)
            status = st.empty()

            frame_index = 0
            seen_ids = set()
            max_dimension = 960

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                original_h, original_w = frame.shape[:2]

                # Resize very large frames to improve speed.
                scale = min(
                    1.0,
                    max_dimension / max(original_h, original_w),
                )

                if scale < 1:
                    frame = cv2.resize(
                        frame,
                        (
                            int(original_w * scale),
                            int(original_h * scale),
                        ),
                    )

                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=0.35,
                    verbose=False,
                )[0]

                if result.boxes is not None:
                    boxes = result.boxes

                    if boxes.id is not None:
                        ids = boxes.id.int().cpu().tolist()
                        xyxy = boxes.xyxy.cpu().numpy()

                        for track_id, box in zip(ids, xyxy):
                            seen_ids.add(track_id)

                            # Foot point: bottom centre of each box.
                            foot_x = (
                                (box[0] + box[2]) / 2 / scale
                            )
                            foot_y = box[3] / scale

                            records.append({
                                "timestamp": frame_index / fps,
                                "frame": frame_index,
                                "track_id": int(track_id),
                                "x_px": float(foot_x),
                                "y_px": float(foot_y),
                                "frame_width": original_w,
                                "frame_height": original_h,
                            })

                # Preview at intervals to keep the UI responsive.
                if frame_index % 3 == 0:
                    annotated = result.plot()
                    annotated = cv2.cvtColor(
                        annotated, cv2.COLOR_BGR2RGB
                    )

                    preview.image(
                        annotated,
                        channels="RGB",
                        use_container_width=True,
                    )

                    status.write(
                        f"Frame {frame_index + 1}/{total_frames} · "
                        f"Track IDs observed: {len(seen_ids)}"
                    )

                frame_index += 1
                progress.progress(
                    min(frame_index / total_frames, 1.0)
                )

            cap.release()
            cap = None

            st.success("Video processing finished.")

            if not records:
                st.warning(
                    "No tracked people were recorded. Try a "
                    "clearer video with visible people."
                )
                st.stop()

            tracks = pd.DataFrame(records)

            c1, c2, c3 = st.columns(3)
            c1.metric("Frames processed", frame_index)
            c2.metric("Track IDs observed", len(seen_ids))
            c3.metric("Video duration", f"{duration:.1f} s")

            st.subheader("Tracked positions")
            st.caption(
                "This is an image-coordinate map, not a real "
                "store floor plan. Calibration is required for "
                "physical distances and store zones."
            )

            # Image-coordinate occupancy heatmap.
            fig, ax = plt.subplots(figsize=(12, 7))

            if show_heatmap:
                ax.hist2d(
                    tracks["x_px"],
                    tracks["y_px"],
                    bins=45,
                    cmap="YlOrRd",
                )

            # Draw each track's path.
            for track_id, group in tracks.groupby("track_id"):
                group = group.sort_values("timestamp")
                ax.plot(
                    group["x_px"],
                    group["y_px"],
                    linewidth=0.8,
                    alpha=0.55,
                )

            width = int(tracks["frame_width"].iloc[0])
            height = int(tracks["frame_height"].iloc[0])

            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_aspect("equal")
            ax.set_xlabel("Image X (pixels)")
            ax.set_ylabel("Image Y (pixels)")
            ax.set_title("Person foot-point occupancy")
            ax.grid(alpha=0.15)

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.subheader("Track summary")

            summary = (
                tracks.groupby("track_id")
                .agg(
                    first_seen=("timestamp", "min"),
                    last_seen=("timestamp", "max"),
                    observations=("frame", "count"),
                )
                .reset_index()
            )

            summary["observed_seconds"] = (
                summary["last_seen"] - summary["first_seen"]
            ).round(2)

            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True,
            )

            st.download_button(
                "Download track observations (CSV)",
                data=tracks.to_csv(index=False).encode("utf-8"),
                file_name="storeflow_track_observations.csv",
                mime="text/csv",
            )

            st.warning(
                "Observed track duration is not necessarily "
                "customer dwell time. Zone-based dwell metrics, "
                "bottleneck alerts, and calibrated floor maps "
                "have not yet been connected to this video output."
            )

        except Exception as exc:
            st.error(f"Video processing failed: {exc}")

        finally:
            if cap is not None:
                cap.release()

            if video_path and Path(video_path).exists():
                Path(video_path).unlink(missing_ok=True)
