# AutoCam Rover Portfolio Core

## Project Summary

AutoCam Rover is an autonomous rover pipeline that combines ROS 2 navigation, camera-based object/event detection, event snapshot capture, network upload, local dashboard streaming, YOLO analysis, and SQLite-backed event storage.

The system was organized for portfolio review around the minimum source files needed to understand the full data path:

1. The ROS 2 integrated rover node drives navigation, scans camera angles, detects risk events, and saves snapshots.
2. The ROS 2 snapshot uploader listens for snapshot events, attaches odometry-based location metadata, and uploads images to the server.
3. The Flask local server receives images, fans each frame out to GUI/DB/analysis queues, streams the latest frame, runs YOLO analysis, and stores event records.
4. The FastAPI DB API shows the database contract for saving and querying event/analysis results.
5. The ROS 2 launch file shows how the integrated node, uploader, camera, controller, and kinematics stack are started together.

## Included Files

| File | Role |
| --- | --- |
| `integrated_auto_rover_node.py` | Final ROS 2 integrated node. Handles RGB/depth input, autonomous obstacle-aware driving, YOLO event detection, camera/servo alignment, snapshot creation, and status publishing. |
| `snapshot_uploader_node.py` | ROS 2 bridge from detected snapshot events to HTTP upload. Reads odometry, packages image plus metadata, and posts to `/upload`. |
| `local_server_snapshot.py` | Flask receiving server. Accepts multipart image uploads, serves MJPEG/SSE dashboard updates, writes event records, and performs YOLO-based analysis. |
| `db_api_main.py` | FastAPI/SQLite API reference for event storage and querying. Documents the application-level database contract. |
| `integrated_auto_rover.launch.py` | Launch composition for camera, controller, kinematics, integrated detection node, and snapshot uploader. |

## End-To-End Flow

```text
RGB/depth camera + odometry
        |
        v
integrated_auto_rover_node.py
  - monitors RGB/depth frames
  - runs YOLO event detection
  - pauses navigation for risk events
  - aligns camera/servo toward target
  - saves event snapshot
  - publishes snapshot:{...} status JSON
        |
        v
snapshot_uploader_node.py
  - receives snapshot status message
  - reads current odometry x/y
  - posts image + captured_at + event_type + location_x/location_y
        |
        v
local_server_snapshot.py
  - POST /upload receives image
  - queues frame for GUI, DB, and analysis workers
  - stores image/index record
  - runs YOLO person analysis
  - streams latest frame and JSON updates to dashboard
        |
        v
autocam_rover.db
  - event_logs
  - analysis_results
```

## Database Structure

The portfolio database design is centered on two related SQLite tables.

```sql
CREATE TABLE event_logs (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    received_at TEXT,
    robot_location TEXT,
    event_type TEXT,
    image_path TEXT NOT NULL,
    analysis_status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE analysis_results (
    analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    person_detected INTEGER DEFAULT 0,
    fallen_object_detected INTEGER DEFAULT 0,
    obstacle_detected INTEGER DEFAULT 0,
    motion_detected INTEGER DEFAULT 0,
    risk_level TEXT,
    result_summary TEXT,
    analyzed_at TEXT,
    FOREIGN KEY (event_id) REFERENCES event_logs(event_id)
);
```

Important distinction: `event_logs.event_id` and `analysis_results.analysis_id` are auto-increment keys. `analysis_results.event_id` is not auto-generated; it links an analysis row back to the event row.

## Network Contract

The rover uploader sends images to the Flask server as `multipart/form-data`:

| Field | Type | Meaning |
| --- | --- | --- |
| `image` | file | JPEG snapshot |
| `captured_at` | string | robot-side capture timestamp |
| `event_type` | string | detected event label such as `motion`, `person`, or paired risk label |
| `location_x` | string/float | odometry x position |
| `location_y` | string/float | odometry y position |
| `metadata_json` | JSON string | optional scan/camera metadata |

The server responds with event metadata including `event_id`, `image_path`, `captured_at`, `received_at`, `event_type`, and `robot_location`.

## Key Technical Points

- ROS 2 package: `final_compilation`
- Core perception: YOLO model inference over camera frames
- Navigation safety: depth ROI checks pause/reverse/turn when obstacles are near
- Event handling: detection cooldown and target-centering refinement before snapshot capture
- Network architecture: HTTP upload plus Flask dashboard streaming through MJPEG and SSE
- Storage: SQLite event/analysis schema with event-to-analysis foreign key
- Deployment shape: JetRover ROS 2 stack on robot, Flask/SQLite receiver on local server PC

## Excluded From This Portfolio Folder

The following were intentionally excluded to keep the package reviewable:

- `.venv`, `__pycache__`, `.pyc`
- model weights such as `yolov8n.pt` or `yolov5n.pt`
- SQLite DB data files
- captured images and runtime logs
- large ZIP archives
- unrelated ROS example/backup packages

## Suggested Review Order

1. Read this `README.md`.
2. Open `integrated_auto_rover_node.py` to understand robot-side autonomous behavior.
3. Open `snapshot_uploader_node.py` to see how events leave ROS 2.
4. Open `local_server_snapshot.py` to see server-side ingestion, dashboard streaming, analysis, and DB writes.
5. Open `db_api_main.py` for the database API/query contract.
