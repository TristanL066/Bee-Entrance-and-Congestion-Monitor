import argparse
import csv
import time
from collections import deque

import cv2
from ultralytics import YOLO

# =========================================================================
# EASY-TO-EDIT SETTINGS
# Change these directly instead of passing command-line flags every time.
# Any of these can still be overridden with a --flag when running the
# script, but editing here is the quickest way to tune things.
# =========================================================================

CONF_THRESHOLD = 0.35          # Detection confidence (0-1). Lower catches more bees but risks false positives.
LOG_INTERVAL_SECONDS = 5.0     # Only write a row to the CSV this often, even though detection runs every frame.
CONGEST_ON = 5                # Smoothed bee count (inside the entrance box) at/above which flagged CONGESTED.
CONGEST_OFF = 2               # Smoothed bee count (inside the entrance box) at/below which the flag clears.
SMOOTHING_WINDOW_SECONDS = 2.0 # How many seconds of recent frames to average for the smoothed count.

OUTPUT_VIDEO = "bee_videos_results/annotated_output.mp4"
OUTPUT_CSV = "bee_videos_results/detection_log.csv"

# =========================================================================


def select_entrance_box(video_path):
    """
    Show the first frame of the video and let the user drag a box around
    the hive entrance opening. Returns (x, y, w, h) in pixel coordinates.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise IOError(f"Could not read a frame from: {video_path}")

    print("\nDrag a box around the hive entrance opening, then press ENTER or SPACE to confirm.")
    print("(Press 'c' to cancel and use the full frame instead.)")

    roi = cv2.selectROI("Select entrance area - press ENTER when done", frame, showCrosshair=True)
    cv2.destroyWindow("Select entrance area - press ENTER when done")

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("No box selected -- congestion will be counted across the whole frame instead.")
        frame_h, frame_w = frame.shape[:2]
        return (0, 0, frame_w, frame_h)

    print(f"Entrance box set to: x={x}, y={y}, width={w}, height={h}")
    return (x, y, w, h)


def box_center_inside_roi(box_xyxy, roi):
    """Check whether the center point of a detection box falls inside the ROI."""
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    rx, ry, rw, rh = roi
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def run_video_inference(weights_path, video_path, conf_threshold, log_interval_seconds,
                         congest_on, congest_off, smoothing_window_seconds, output_video, output_csv):
    model = YOLO(weights_path)

    entrance_roi = select_entrance_box(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log_interval_frames = max(1, round(source_fps * log_interval_seconds))
    smoothing_window_frames = max(1, round(source_fps * smoothing_window_seconds))
    target_frame_time = 1.0 / source_fps

    print(f"\nSource video: {source_fps:.1f} fps, {frame_width}x{frame_height}")
    print(f"Detecting on every frame, logging to CSV every {log_interval_seconds}s "
          f"(~every {log_interval_frames} frames)")
    print(f"Smoothing window: {smoothing_window_seconds}s (~{smoothing_window_frames} frames)")
    print(f"Congestion thresholds (entrance box only): ON at {congest_on}, OFF at {congest_off}")
    print("Playback is paced to match the source video's real-time speed as closely as processing allows.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, source_fps, (frame_width, frame_height))

    csv_file = open(output_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "video_time_sec",
        "total_bees_in_frame", "bees_in_entrance", "smoothed_entrance_count", "congested",
    ])

    counts_window = deque(maxlen=smoothing_window_frames)
    is_congested = False

    rx, ry, rw, rh = entrance_roi

    frame_idx = 0

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Detection runs every frame, continuously -- this is the "real-time" part
        results = model.predict(source=frame, conf=conf_threshold, verbose=False)
        result = results[0]

        total_in_frame = len(result.boxes)
        entrance_count = 0

        annotated = frame.copy()

        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            inside = box_center_inside_roi(xyxy, entrance_roi)
            if inside:
                entrance_count += 1

            # Bees inside the entrance box drawn in yellow, bees elsewhere in green --
            # both are still detected and drawn, only the count differs
            color = (0, 255, 255) if inside else (0, 200, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{conf:.2f}", (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        counts_window.append(entrance_count)
        smoothed_count = sum(counts_window) / len(counts_window)

        # Hysteresis: only flip state when crossing the relevant threshold,
        # so the flag doesn't rapidly toggle right at one boundary value.
        # This still updates every frame, using the rolling smoothed count.
        if not is_congested and smoothed_count >= congest_on:
            is_congested = True
        elif is_congested and smoothed_count <= congest_off:
            is_congested = False

        video_time_sec = frame_idx / source_fps

        # Only write a row to the CSV every N frames (~ every log_interval_seconds),
        # even though detection and the on-screen count update every frame
        if frame_idx % log_interval_frames == 0:
            csv_writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                round(video_time_sec, 2),
                total_in_frame,
                entrance_count,
                round(smoothed_count, 2),
                is_congested,
            ])

        # Draw the entrance box itself, so it's clear what area is being counted
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
        cv2.putText(annotated, "Entrance", (rx, max(ry - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        status_text = "CONGESTED" if is_congested else "Clear"
        status_color = (0, 0, 255) if is_congested else (0, 200, 0)  # BGR: red / green

        cv2.putText(annotated,
                    f"Entrance: {entrance_count} (avg: {smoothed_count:.1f})  Total in frame: {total_in_frame}  t={video_time_sec:.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, status_text,
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        writer.write(annotated)
        cv2.imshow("Bee Congestion Monitor", annotated)

        # Pace playback to match the source video's real-time speed: figure out
        # how long this frame's processing took, then wait only the remaining
        # time needed to hit the frame's natural duration. If processing is
        # slower than real-time (e.g. underpowered hardware), this just shows
        # frames as fast as it can instead of falling further behind.
        elapsed = time.time() - loop_start
        remaining = target_frame_time - elapsed
        wait_ms = max(1, int(remaining * 1000))

        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            print("\nStopped early by user.")
            break

        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print(f"\nDone. Processed {frame_idx} frames.")
    print(f"Annotated video saved to: {output_video}")
    print(f"Log data saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bee counting + congestion detection on video")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights file")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Detection confidence threshold")
    parser.add_argument("--log-interval-seconds", type=float, default=LOG_INTERVAL_SECONDS,
                         help="Write a CSV row this often, even though detection runs every frame")
    parser.add_argument("--congest-on", type=float, default=CONGEST_ON,
                         help="Smoothed entrance-box count at/above which flagged congested")
    parser.add_argument("--congest-off", type=float, default=CONGEST_OFF,
                         help="Smoothed entrance-box count at/below which congestion flag clears")
    parser.add_argument("--smoothing-window-seconds", type=float, default=SMOOTHING_WINDOW_SECONDS,
                         help="How many seconds of recent frames to average for the smoothed count")
    parser.add_argument("--output-video", default=OUTPUT_VIDEO,
                         help="Path to save the annotated output video")
    parser.add_argument("--output-csv", default=OUTPUT_CSV,
                         help="Path to save the CSV log")
    args = parser.parse_args()

    run_video_inference(
        args.weights, args.video, args.conf, args.log_interval_seconds,
        args.congest_on, args.congest_off, args.smoothing_window_seconds,
        args.output_video, args.output_csv,
    )