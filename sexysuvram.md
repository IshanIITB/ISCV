# StoreFlow — Dwell-Time & Bottleneck Mapper

StoreFlow is a computer vision-based retail analytics dashboard that uses existing store video footage to analyze customer movement, understand how different areas of a store are used, and identify potential congestion and underutilized zones.

The project aims to turn ordinary in-store camera footage into actionable insights for improving store layouts, customer flow, and the placement of displays.

## How StoreFlow Addresses the Problem Statement

| Requirement | StoreFlow feature |
|---|---|
| Use standard in-store camera feeds | Processes ordinary CCTV-style video, either uploaded recordings (MP4, AVI, MOV, MKV) or live input, with no special hardware |
| Map customer journeys | Tracks each person across frames and draws their movement paths on the heat map |
| Real-time foot-traffic heat maps | Occupancy heat map overlaid on the camera view, with zones and area labels, updated as footage is processed in real-time mode |
| Section-by-section dwell times | Per-zone dwell time, stationary time, stop rate, peak occupancy, and median speed |
| Flag bottlenecks and dead zones | Congestion detection with severity classification, and underutilized-zone detection |

## Features

### Implemented

- **Real-Time Processing:** Analyzes live camera input as it arrives, updating the heat map, journey paths, and zone metrics continuously rather than only after a recording has been processed.
- **Person Detection:** Detects people in store video using YOLO.
- **Person Tracking:** Tracks people across video frames using ByteTrack.
- **Customer Journey Mapping:** Draws each tracked person's movement path over the store view, showing how customers actually move through the space. Paths can be toggled on or off on the heat map.
- **Traffic Heatmap:** Visualizes accumulated customer presence across the store, overlaid on the camera frame.
- **Zone-Based Analytics:** Analyzes manually defined rectangular and polygonal store zones, with optional on-screen area labels.
- **Visitor Statistics:** Reports detected visitors and their distribution across zones.
- **Dwell-Time Analysis:** Measures the time visitors spend in individual zones.
- **Stationary-Time Analysis:** Estimates how long visitors remain stationary.
- **Stop-Rate Analysis:** Measures the proportion of observed visits involving stationary behavior.
- **Occupancy and Movement Metrics:** Reports peak occupancy and median movement speed.
- **Bottleneck Detection:** Identifies congestion using occupancy, movement speed, congestion duration, and severity classification.
- **Dead-Zone Detection:** Identifies underutilized store zones based on observed visitor activity.
- **Adjustable Analysis Parameters:** Sidebar sliders let you tune the analysis without touching code (see below).
- **AI Vision Analyst Effort Control:** A sidebar setting that controls how much processing effort the analysis pipeline uses, so StoreFlow can be tuned to the available hardware and the speed needed.
- **Tracking-Quality Warnings:** The dashboard warns when a large share of tracks are very short, which suggests ID switches and possibly inflated visitor counts.
- **Validation Tab:** A dedicated tab for checking results. Until validation is completed, the dashboard clearly labels its figures as not yet validated.
- **Processed Video Preview:** Displays processed footage with detection boxes and tracking IDs.
- **Analytics Dashboard:** Presents zone-level statistics and congestion findings.
- **CSV Export:** Supports downloading zone analytics for further analysis.
- **Video Input:** Supports recorded video uploads, real-time processing, and a synthetic demo mode.

### Adjustable Parameters

The sidebar's **Analysis settings** panel lets you change how the analysis behaves and see the results update:

| Setting | What it controls |
|---|---|
| Min. seconds to count as a visit | Minimum time a person must spend in a zone for it to count as a visit, filtering out brief pass-throughs |
| Congestion: people at once | Number of people in a zone at the same time before it can be considered congested |
| Congestion: must last (s) | How long that crowding must persist before it is flagged as a bottleneck, separating real congestion from momentary crowding |
| AI vision analyst | Controls how much processing effort the pipeline uses, trading off speed against thoroughness depending on the hardware available |
| Show movement paths on heat map | Toggles the customer journey paths on or off |
| Show area labels | Toggles zone labels on the heat map |

Different stores and camera setups behave differently, so these thresholds are meant to be tuned to the footage being analyzed.

> **Note:** Heatmaps represent observed presence, not unique visitors, customer interest, or purchase intent. Bottleneck and dead-zone findings are based on observed video data and depend on footage quality, camera placement, zone definitions, and observation duration.

## Technology Stack

- **Python** — application and analytics logic
- **OpenCV** — video processing
- **YOLO** — person detection
- **ByteTrack** — multi-object tracking
- **Streamlit** — interactive dashboard
- **NumPy and Pandas** — numerical processing and tabular analytics
- **Matplotlib** — heatmap and visualization support

## Current Metrics

StoreFlow currently reports metrics including:

| Metric | Description |
|---|---|
| Visitors | Number of distinct tracked visitors observed in a zone, subject to tracking accuracy |
| Visitor Share | Percentage of detected visitors attributed to a zone |
| Median Visit Duration | Median time spent inside a zone per observed visit |
| Presence Time | Total observed time spent inside a zone |
| Stationary Time | Time spent below the configured walking-speed threshold |
| Stop Rate | Proportion of observed visits classified as involving a stop |
| Peak Occupancy | Maximum observed occupancy within a zone |
| Congested Duration | Duration classified as congested by the current detection logic |
| Median Speed | Median tracked movement speed, currently measured in frame-widths per second rather than metres per second |

Metric accuracy depends on video quality, camera perspective, zone placement, detection performance, and tracking continuity.

## Running the Dashboard

1. Clone the repository.
2. Set up a Python virtual environment.
3. Install the project's dependencies.
4. Launch the Streamlit application.

```bash
git clone <repository-url>
cd <repository-directory>

python -m venv .venv
```

Activate the virtual environment.

**Linux / macOS / WSL**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies and launch the dashboard:

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

These commands assume the repository contains `requirements.txt` and that `app/dashboard.py` is the dashboard entry point.

## Using the Dashboard

1. Choose **Recorded video** to analyze a file, use real-time mode to analyze live input, or choose **Synthetic demo** to try the dashboard without footage.
2. For recorded video, upload a store video from the sidebar (MP4, AVI, MOV, or MKV, up to 200 MB).
3. Adjust the analysis settings, including the AI vision analyst effort level, to suit the footage and your hardware.
4. Review the heat map with journey paths, the zone analytics table, and any congestion or dead-zone findings.
5. Export zone analytics as CSV if needed.

## Project Goal

The goal of StoreFlow is to develop a practical retail analytics tool that helps store operators:

- Understand where visitors spend their time and how they move through the store.
- Compare activity across store zones.
- Identify potential bottlenecks and underutilized areas.
- Use video-derived evidence to guide store-layout improvements.

## Privacy

StoreFlow does not perform face recognition. Track IDs are used only to follow movement within a processed video.

## Limitations

StoreFlow is currently a prototype. Congestion and dead-zone classifications should be treated as preliminary findings until validated against representative real-world footage. Visitor counts can be inflated by tracking ID switches, and real-time performance depends on the hardware and the processing effort selected.