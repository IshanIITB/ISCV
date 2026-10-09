
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

st.set_page_config(
    page_title="StoreFlow | Store Analytics",
    page_icon="🗺️",
    layout="wide",
)

st.title("StoreFlow")
st.caption("Dwell-Time & Bottleneck Mapper")

mode = st.sidebar.selectbox(
    "Data source",
    ["Synthetic demo", "Recorded video"],
)

uploaded_video = None

if mode == "Recorded video":
    uploaded_video = st.sidebar.file_uploader(
        "Upload store video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_video is not None:
        st.video(uploaded_video)
        st.info(
            "Video loaded. Person detection and tracking "
            "will be connected in the next step."
        )

st.sidebar.subheader("Map settings")
show_heatmap = st.sidebar.checkbox("Show traffic heatmap", True)
show_zones = st.sidebar.checkbox("Show zone labels", True)


if mode == "Recorded video":
    if uploaded_video is None:
        st.info("Upload a video in the sidebar to begin.")
        st.stop()

    import cv2
    import tempfile
    from pathlib import Path
    from ultralytics import YOLO

    st.subheader("Person detection and tracking")
    st.caption(
        "YOLO + ByteTrack · Processing uploaded footage"
    )

    if st.button("Run detection", type="primary"):
        suffix = Path(uploaded_video.name).suffix or ".mp4"

        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp:
            temp.write(uploaded_video.getvalue())
            video_path = temp.name

        cap = None

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                st.error("Could not open this video file.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if fps <= 0 or total_frames <= 0:
                st.error("The video has invalid frame metadata.")
                st.stop()

            duration = total_frames / fps

            st.write(
                f"Duration: {duration:.1f} seconds · "
                f"Frame rate: {fps:.1f} FPS"
            )

            model = YOLO("yolo11n.pt")
            frame_view = st.empty()
            progress = st.progress(0)
            status = st.empty()

            seen_ids = set()
            frame_index = 0
            current_count = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=0.35,
                    verbose=False,
                )[0]

                if result.boxes is not None:
                    current_count = len(result.boxes)

                    if result.boxes.id is not None:
                        ids = (
                            result.boxes.id
                            .int()
                            .cpu()
                            .tolist()
                        )
                        seen_ids.update(ids)
                else:
                    current_count = 0

                # Display every third frame; process every frame.
                if frame_index % 3 == 0:
                    annotated = result.plot()
                    rgb = cv2.cvtColor(
                        annotated, cv2.COLOR_BGR2RGB
                    )

                    frame_view.image(
                        rgb,
                        channels="RGB",
                        use_container_width=True,
                    )

                    status.write(
                        f"Frame {frame_index + 1}/{total_frames} "
                        f"· People in current frame: {current_count} "
                        f"· Unique track IDs seen: {len(seen_ids)}"
                    )

                    progress.progress(
                        min(
                            (frame_index + 1) / total_frames,
                            1.0,
                        )
                    )

                frame_index += 1

            progress.progress(1.0)
            st.success("Video processing complete.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Frames processed", frame_index)
            col2.metric("Unique track IDs", len(seen_ids))
            col3.metric("Video duration", f"{duration:.1f} s")

            st.caption(
                "Track IDs are approximate counts of identities "
                "tracked during this clip, not guaranteed unique "
                "customer counts."
            )

        finally:
            if cap is not None:
                cap.release()

            Path(video_path).unlink(missing_ok=True)

    st.stop()


# Demo floor plan: coordinates are illustrative, in metres.
zones = [
    {"name": "Produce", "x": 1, "y": 7, "w": 4, "h": 3},
    {"name": "Aisle A", "x": 6, "y": 7, "w": 2, "h": 6},
    {"name": "Aisle B", "x": 9, "y": 7, "w": 2, "h": 6},
    {"name": "Dairy", "x": 12, "y": 7, "w": 3, "h": 3},
    {"name": "Checkout", "x": 11, "y": 1, "w": 5, "h": 3},
    {"name": "Entrance", "x": 1, "y": 1, "w": 3, "h": 2},
]

rng = np.random.default_rng(42)
people = pd.DataFrame({
    "x": rng.uniform(0.5, 16, 128),
    "y": rng.uniform(0.5, 14, 128),
})

# Simulate heavier traffic around checkout and Aisle A.
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

a, b, c = st.columns(3)
a.metric("People detected", "128", help="Synthetic demo value")
b.metric("Average dwell time", "2m 34s", help="Illustrative value")
c.metric("Potential bottlenecks", "2", help="Synthetic demo alerts")

st.subheader("Store floor heatmap")
st.caption("Synthetic data — not connected to a real camera yet.")

fig, ax = plt.subplots(figsize=(12, 7))

if show_heatmap:
    ax.hist2d(
        people["x"],
        people["y"],
        bins=[34, 28],
        range=[[0, 17], [0, 15]],
        cmap="YlOrRd",
        alpha=0.78,
    )

for z in zones:
    rect = Rectangle(
        (z["x"], z["y"]),
        z["w"],
        z["h"],
        linewidth=1.5,
        edgecolor="#263238",
        facecolor="none",
    )
    ax.add_patch(rect)

    if show_zones:
        ax.text(
            z["x"] + z["w"] / 2,
            z["y"] + z["h"] / 2,
            z["name"],
            ha="center",
            va="center",
            fontsize=9,
            color="#111111",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
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
ax.set_xlabel("Store width (metres)")
ax.set_ylabel("Store depth (metres)")
ax.set_facecolor("#fafafa")
ax.grid(alpha=0.15)

st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Bottleneck alerts")

alerts = pd.DataFrame([
    {
        "Zone": "Checkout",
        "Condition": "High density / low movement",
        "Status": "Demo alert",
    },
    {
        "Zone": "Aisle A",
        "Condition": "Sustained traffic concentration",
        "Status": "Demo alert",
    },
])

st.dataframe(alerts, hide_index=True, use_container_width=True)

st.info(
    "Next: replace synthetic positions with calibrated person tracks, "
    "calculate actual zone metrics, and trigger alerts from persistent "
    "density and speed conditions."
)
