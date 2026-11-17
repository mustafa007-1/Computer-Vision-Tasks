"""Lab 8 - Task 2: camera calibration, distortion correction and perspective transformation.

Uses the 13 checkerboard views in images/calibration/ (OpenCV's right01-right14 sample
set; right10 does not exist). The board has 9 x 6 internal corners.

Steps: detect corners + sub-pixel refinement -> calibrateCamera (K, D) ->
re-projection error -> distortion field -> getOptimalNewCameraMatrix + undistort ->
homography of the tilted board to a top-down view.
"""
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images" / "calibration"
OUTPUT_DIR = Path(__file__).resolve().parent

PATTERN = (9, 6)  # internal corners (columns, rows)
SQUARE_SIZE = 1.0  # one board square = 1 unit (use mm for metric results)
TEST_IMAGE = "right12.jpg"
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def show(axis, image, title=""):
    if image.ndim == 2:
        axis.imshow(image, cmap="gray")
    else:
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axis.set_title(title)
    axis.axis("off")


def save(figure, name):
    path = OUTPUT_DIR / name
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    print(f"Saved {path.name}")


def order_points(points):
    """Return four points ordered TL, TR, BR, BL."""
    points = np.asarray(points, np.float32)
    sums, diffs = points.sum(1), np.diff(points, axis=1).ravel()
    return np.array(
        [points[np.argmin(sums)], points[np.argmin(diffs)], points[np.argmax(sums)], points[np.argmax(diffs)]],
        np.float32,
    )


def draw_reference_lines(image, corners):
    """Straight red lines between the outer corners make lens bending visible."""
    output = image.copy()
    grid = corners.reshape(PATTERN[1], PATTERN[0], 2)
    for row in (0, PATTERN[1] - 1):
        cv2.line(output, tuple(map(int, grid[row, 0])), tuple(map(int, grid[row, -1])), (0, 0, 255), 2)
    for col in (0, PATTERN[0] - 1):
        cv2.line(output, tuple(map(int, grid[0, col])), tuple(map(int, grid[-1, col])), (0, 0, 255), 2)
    return output


def main():
    print("--- Task 19 (Lab 8 Task 2): Camera calibration, undistortion and homography ---")
    files = sorted(IMAGE_DIR.glob("right*.jpg"))
    if not files:
        raise FileNotFoundError(f"No calibration images found in {IMAGE_DIR}")

    # 3-D board points (0,0,0), (1,0,0), ..., (8,5,0): the board is the plane Z = 0
    board = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    board[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2) * SQUARE_SIZE

    object_points, image_points, used_files, previews = [], [], [], []
    image_size = None
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not load {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (width, height)
        found, corners = cv2.findChessboardCorners(gray, PATTERN, None)
        if not found:
            print(f"{path.name}: corners NOT found")
            continue
        corners = corners.reshape(-1, 1, 2).astype(np.float32)
        # cornerSubPix refines in place, so pass a copy to keep the raw corners for comparison
        refined = cv2.cornerSubPix(gray, corners.copy(), (11, 11), (-1, -1), CRITERIA).reshape(-1, 1, 2)
        shift = np.linalg.norm(refined - corners, axis=2).mean()
        print(f"{path.name}: corners found, mean sub-pixel shift = {shift:.3f} px")
        object_points.append(board)
        image_points.append(refined)
        used_files.append(path)
        previews.append((cv2.drawChessboardCorners(image.copy(), PATTERN, refined, found), path.name))
    print(f"\nValid views: {len(object_points)} / {len(files)} | image size = {image_size}")
    if len(object_points) < 3:
        raise RuntimeError("Need at least 3 views with detected corners to calibrate.")

    figure, axes = plt.subplots(1, 4, figsize=(18, 4))
    for axis, (preview, name) in zip(axes, previews[:4]):
        show(axis, preview, f"{name} - detected corners")
    save(figure, "task19_corners.png")

    # Raw vs. refined corner, zoomed in
    first_gray = cv2.cvtColor(cv2.imread(str(used_files[0])), cv2.COLOR_BGR2GRAY)
    _, raw = cv2.findChessboardCorners(first_gray, PATTERN, None)
    raw = raw.reshape(-1, 1, 2)
    refined = image_points[0]
    cx, cy = refined[20, 0]
    radius = 12
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(first_gray, cmap="gray")
    axis.set_xlim(cx - radius, cx + radius)
    axis.set_ylim(cy + radius, cy - radius)
    axis.scatter(raw[:, 0, 0], raw[:, 0, 1], c="red", s=80, marker="x", label="findChessboardCorners")
    axis.scatter(refined[:, 0, 0], refined[:, 0, 1], c="lime", s=80, marker="+", label="cornerSubPix")
    axis.legend(loc="lower left")
    axis.set_title("Sub-pixel corner refinement (zoomed)")
    save(figure, "task19_subpixel_zoom.png")

    # Calibration
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    np.set_printoptions(precision=4, suppress=True)
    print("\nIntrinsic camera matrix K:\n", camera_matrix)
    print(
        f"fx = {camera_matrix[0, 0]:.2f}, fy = {camera_matrix[1, 1]:.2f}, "
        f"cx = {camera_matrix[0, 2]:.2f}, cy = {camera_matrix[1, 2]:.2f}"
    )
    print("Distortion coefficients D = [k1, k2, p1, p2, k3]:", distortion.ravel())
    print(f"k1 {'< 0 -> barrel' if distortion.ravel()[0] < 0 else '> 0 -> pincushion'} distortion")
    print(f"RMS re-projection error (calibrateCamera): {rms:.4f} px")

    per_view = []
    for points3d, points2d, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(points3d, rvec, tvec, camera_matrix, distortion)
        per_view.append(np.linalg.norm(points2d - projected, axis=2).mean())
    mean_error = float(np.mean(per_view))
    print(f"Mean re-projection error: {mean_error:.4f} px (< 1 px means an accurate calibration)")
    np.savez(OUTPUT_DIR / "task19_calibration.npz", K=camera_matrix, D=distortion, rms=rms, mean_error=mean_error)
    print("Saved task19_calibration.npz")

    figure, axis = plt.subplots(figsize=(10, 3.5))
    axis.bar([path.stem for path in used_files], per_view, color="steelblue")
    axis.axhline(mean_error, color="red", ls="--", label=f"mean = {mean_error:.4f} px")
    axis.set_ylabel("error (px)")
    axis.set_title("Re-projection error per calibration image")
    axis.legend()
    save(figure, "task19_reprojection_error.png")

    # Distortion displacement field
    width, height = image_size
    grid_x, grid_y = np.meshgrid(np.linspace(0, width - 1, 20), np.linspace(0, height - 1, 15))
    distorted = np.stack([grid_x.ravel(), grid_y.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)
    corrected = cv2.undistortPoints(distorted, camera_matrix, distortion, P=camera_matrix).reshape(-1, 2)
    distorted = distorted.reshape(-1, 2)
    displacement = corrected - distorted
    figure, axis = plt.subplots(figsize=(8, 6))
    quiver = axis.quiver(
        distorted[:, 0], distorted[:, 1], displacement[:, 0], displacement[:, 1],
        np.linalg.norm(displacement, axis=1), angles="xy", scale_units="xy", scale=1, cmap="viridis",
    )
    figure.colorbar(quiver, label="pixel displacement")
    axis.plot(camera_matrix[0, 2], camera_matrix[1, 2], "r*", ms=15, label="principal point")
    axis.invert_yaxis()
    axis.set_aspect("equal")
    axis.legend()
    axis.set_title("Lens distortion field (grows toward the edges)")
    save(figure, "task19_distortion_field.png")

    # Undistortion: alpha = 1 keeps every source pixel (black borders), ROI crops them
    test_image = cv2.imread(str(IMAGE_DIR / TEST_IMAGE))
    if test_image is None:
        raise FileNotFoundError(f"Could not load {IMAGE_DIR / TEST_IMAGE}")
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, image_size, 1, image_size)
    undistorted = cv2.undistort(test_image, camera_matrix, distortion, None, new_matrix)
    x, y, roi_w, roi_h = roi
    cropped = undistorted[y : y + roi_h, x : x + roi_w]
    print("\nOptimal new camera matrix (alpha = 1):\n", new_matrix)
    print("Valid ROI (x, y, w, h):", roi)

    found_before, corners_before = cv2.findChessboardCorners(cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY), PATTERN)
    found_after, corners_after = cv2.findChessboardCorners(cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY), PATTERN)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    show(axes[0], draw_reference_lines(test_image, corners_before) if found_before else test_image,
         "Original (distorted)")
    show(axes[1], draw_reference_lines(undistorted, corners_after) if found_after else undistorted,
         "Undistorted (alpha = 1)")
    show(axes[2], cropped, "Undistorted, cropped to ROI")
    save(figure, "task19_undistortion.png")

    # Homography on the undistorted image (a homography is linear: it cannot model lens bending)
    square, margin = 50, 40
    if found_after:
        grid = corners_after.reshape(PATTERN[1], PATTERN[0], 2)
        source = order_points([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]])
        columns_direction, rows_direction = grid[0, -1] - grid[0, 0], grid[-1, 0] - grid[0, 0]
        horizontal = abs(columns_direction[0]) > abs(rows_direction[0])
    else:
        # manual fallback: four corners picked by hand on the undistorted image (TL, TR, BR, BL)
        source = order_points(np.float32([[237, 97], [415, 87], [440, 394], [210, 396]]))
        horizontal = False
    nx, ny = (PATTERN[0] - 1, PATTERN[1] - 1) if horizontal else (PATTERN[1] - 1, PATTERN[0] - 1)
    out_w, out_h = nx * square, ny * square  # keeps the board's true square aspect ratio
    destination = np.float32(
        [[margin, margin], [margin + out_w, margin], [margin + out_w, margin + out_h], [margin, margin + out_h]]
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(undistorted, homography, (out_w + 2 * margin, out_h + 2 * margin))
    print("\nSource points (TL, TR, BR, BL):\n", source)
    print("Destination points:\n", destination)
    print("Homography H:\n", homography)

    marked = undistorted.copy()
    cv2.polylines(marked, [source.astype(int)], True, (0, 255, 0), 2)
    for label, (px, py) in zip(("TL", "TR", "BR", "BL"), source.astype(int)):
        cv2.circle(marked, (int(px), int(py)), 7, (0, 0, 255), -1)
        cv2.putText(marked, label, (int(px) + 8, int(py) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    show(axes[0], marked, "Undistorted - selected planar region")
    show(axes[1], warped, "Top-down view (warpPerspective)")
    save(figure, "task19_homography.png")


if __name__ == "__main__":
    main()
