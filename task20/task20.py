"""Lab 8 - Task 3: video analytics - detect, count and track moving objects.

Per frame: MOG2 background subtraction -> threshold (drop shadows) -> morphology
(open, close, dilate) -> external contours above MIN_AREA -> boxes + centroids ->
nearest-neighbour centroid tracker (IDs + trails) -> count/FPS overlay -> VideoWriter.

Video options (pick one):
    python task20/task20.py                     # Intel sample street video (downloaded once, ~6 MB)
    python task20/task20.py --video clip.mp4    # your own video file
    python task20/task20.py --synthetic         # generated moving shapes (offline, used by the tests)
"""
import argparse
import shutil
import subprocess
import time
import urllib.request
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent
SAMPLE_VIDEO_URL = (
    "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/"
    "person-bicycle-car-detection.mp4"
)

MIN_AREA = 250  # minimum contour area (px^2)
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


class CentroidTracker:
    """Keeps an ID per object by matching each new centroid to the nearest old one."""

    def __init__(self, max_disappeared=15, max_distance=60, trail_len=30):
        self.next_id = 0
        self.objects = OrderedDict()  # id -> centroid
        self.trails = OrderedDict()  # id -> recent centroids
        self.missing = OrderedDict()  # id -> frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trail_len = trail_len

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.trails[self.next_id] = deque([centroid], maxlen=self.trail_len)
        self.missing[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, object_id):
        for table in (self.objects, self.trails, self.missing):
            del table[object_id]

    def _mark_missing(self, object_id):
        self.missing[object_id] += 1
        if self.missing[object_id] > self.max_disappeared:
            self._deregister(object_id)

    def update(self, centroids):
        if not centroids:
            for object_id in list(self.missing):
                self._mark_missing(object_id)
            return self.objects
        if not self.objects:
            for centroid in centroids:
                self._register(centroid)
            return self.objects

        ids = list(self.objects)
        old = np.array(list(self.objects.values()), dtype=float)
        new = np.array(centroids, dtype=float)
        distance = np.linalg.norm(old[:, None, :] - new[None, :, :], axis=2)

        # Greedy matching: closest pairs first, each old/new point used once
        rows = distance.min(1).argsort()
        cols = distance.argmin(1)[rows]
        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols or distance[row, col] > self.max_distance:
                continue
            object_id = ids[row]
            self.objects[object_id] = centroids[col]
            self.trails[object_id].append(centroids[col])
            self.missing[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)
        for row in set(range(len(ids))) - used_rows:
            self._mark_missing(ids[row])
        for col in set(range(len(centroids))) - used_cols:
            self._register(centroids[col])
        return self.objects


def colour_for(object_id):
    rng = np.random.RandomState(object_id * 7 + 3)
    return tuple(int(v) for v in rng.randint(60, 255, 3))


def make_synthetic_video(path, frames=150, size=(320, 240), fps=15):
    """Static textured background with three shapes moving across it."""
    rng = np.random.default_rng(42)
    width, height = size
    background = cv2.GaussianBlur(rng.integers(60, 140, (height, width, 3), dtype=np.uint8), (7, 7), 0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    movers = [  # start x, y, velocity x, y, radius, colour
        (-30, 60, 3, 0, 18, (0, 0, 255)),
        (width + 30, 150, -4, 0, 22, (0, 255, 0)),
        (100, -30, 1, 2, 15, (255, 0, 0)),
    ]
    for t in range(frames):
        frame = background.copy()
        for x0, y0, vx, vy, radius, colour in movers:
            cv2.circle(frame, (int(x0 + vx * t), int(y0 + vy * t)), radius, colour, -1)
        noise = rng.normal(0, 3, frame.shape)
        writer.write(np.clip(frame + noise, 0, 255).astype(np.uint8))
    writer.release()
    return path


def get_video(args):
    if args.video:
        return args.video
    DATA_DIR.mkdir(exist_ok=True)
    if args.synthetic:
        return make_synthetic_video(DATA_DIR / "synthetic_motion.mp4")
    path = DATA_DIR / "parking_lot.mp4"
    if not path.exists():
        print(f"Downloading sample video from {SAMPLE_VIDEO_URL} ...")
        urllib.request.urlretrieve(SAMPLE_VIDEO_URL, path)
    return path


def show(axis, image, title=""):
    if image.ndim == 2:
        axis.imshow(image, cmap="gray")
    else:
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axis.set_title(title)
    axis.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="path to a video file")
    source.add_argument("--synthetic", action="store_true", help="use a generated video (offline)")
    args = parser.parse_args()

    print("--- Task 20 (Lab 8 Task 3): Moving-object detection, counting and tracking ---")
    video_path = get_video(args)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps_in = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} | {fps_in:.1f} fps | {total} frames | {total / fps_in:.1f} s")

    # MOG2 learns the background over time; varThreshold 32 ignores small flicker,
    # detectShadows marks shadows as 127 so they can be thresholded away
    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=True)
    tracker = CentroidTracker(max_disappeared=15, max_distance=60, trail_len=30)
    output_path = OUTPUT_DIR / "task20_tracking.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    snapshot_at = min(560, max(total - 1, 1))
    snapshot, counts, fps_history = {}, [], []
    frame_index, previous_time = 0, time.time()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1

        foreground = subtractor.apply(frame)
        _, binary = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)  # keep 255, drop shadows (127)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=2)  # remove specks
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=2)  # fill holes
        clean = cv2.dilate(clean, OPEN_KERNEL, iterations=2)  # join broken parts

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, centroids = [], []
        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"]:
                centroid = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            else:
                centroid = (x + w // 2, y + h // 2)
            boxes.append((x, y, w, h))
            centroids.append(centroid)

        objects = tracker.update(centroids)
        annotated = frame.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for object_id, (cx, cy) in objects.items():
            if tracker.missing[object_id] > 0:
                continue
            colour = colour_for(object_id)
            trail = list(tracker.trails[object_id])
            for j in range(1, len(trail)):
                cv2.line(annotated, trail[j - 1], trail[j], colour, int(1 + 3 * j / len(trail)))
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(annotated, f"ID {object_id}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

        now = time.time()
        fps_history.append(1.0 / max(now - previous_time, 1e-6))
        previous_time = now
        counts.append(len(boxes))
        cv2.rectangle(annotated, (0, 0), (290, 75), (0, 0, 0), -1)
        cv2.putText(annotated, f"Moving objects: {len(boxes)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {np.mean(fps_history[-15:]):.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(annotated, f"Frame: {frame_index}/{total}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(annotated)

        if frame_index == snapshot_at:
            snapshot = dict(frame=frame, fg=foreground, binary=binary, clean=clean, out=annotated)
    capture.release()
    writer.release()
    if frame_index == 0:
        raise RuntimeError("The video contains no readable frames.")
    if not snapshot:
        snapshot = dict(frame=frame, fg=foreground, binary=binary, clean=clean, out=annotated)
        snapshot_at = frame_index

    print(f"Processed {frame_index} frames | mean processing FPS = {np.mean(fps_history[5:] or fps_history):.1f}")
    print(f"Average moving objects per frame = {np.mean(counts):.2f} | max = {max(counts)}")
    print(f"Total unique track IDs assigned = {tracker.next_id}")
    print(f"Saved {output_path.name}")

    # Optional H.264 copy (plays in browsers / Colab); needs ffmpeg
    if shutil.which("ffmpeg"):
        h264_path = OUTPUT_DIR / "task20_tracking_h264.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(output_path), "-vcodec", "libx264",
             "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", str(h264_path)],
            check=False,
        )
        if h264_path.exists():
            print(f"Saved {h264_path.name} ({h264_path.stat().st_size / 1e6:.1f} MB)")

    # Pipeline stages for one frame + count plot
    figure, axes = plt.subplots(2, 3, figsize=(18, 9))
    show(axes[0, 0], snapshot["frame"], f"1. Input frame #{snapshot_at}")
    show(axes[0, 1], snapshot["fg"], "2. MOG2 mask (white = fg, gray = shadow)")
    show(axes[0, 2], snapshot["binary"], "3. Thresholded (shadows removed)")
    show(axes[1, 0], snapshot["clean"], "4. After opening + closing + dilation")
    show(axes[1, 1], snapshot["out"], "5. Boxes, centroids, IDs, trails")
    axes[1, 2].plot(counts, lw=1)
    axes[1, 2].set_title("Moving-object count per frame")
    axes[1, 2].set_xlabel("frame")
    axes[1, 2].set_ylabel("count")
    axes[1, 2].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "task20_pipeline.png", dpi=100)
    plt.close(figure)
    print("Saved task20_pipeline.png")

    # Six annotated frames spread across the video
    capture = cv2.VideoCapture(str(output_path))
    picks = np.linspace(min(40, frame_index - 1), frame_index - 1, 6).astype(int)
    figure, axes = plt.subplots(2, 3, figsize=(18, 9))
    for axis, pick in zip(axes.ravel(), picks):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(pick))
        ok, annotated = capture.read()
        if ok:
            show(axis, annotated, f"frame {pick}")
        else:
            axis.axis("off")
    capture.release()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "task20_frames.png", dpi=100)
    plt.close(figure)
    print("Saved task20_frames.png")


if __name__ == "__main__":
    main()
