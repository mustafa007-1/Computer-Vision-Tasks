# Computer Vision Labs

This repository contains the Computer Vision Lab 6, Lab 7 and Lab 8 task implementations, numbered as one continuous series of tasks (`task1` – `task20`).

| Tasks | Lab | Topic |
|---|---|---|
| `task1` – `task7` | Lab 6 (journal) | Feature detection, description, RANSAC matching, panorama |
| `task8` – `task12` | Lab 7 (journal) | Feature matching, homography and RANSAC |
| `task13` – `task17` | Lab 7 (manual) | Staged SIFT alignment walkthrough |
| `task18` – `task20` | Lab 8 | Image classification, camera calibration, video analytics |

## Setup Instructions

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate the necessary images to test the code. Since we don't have the original lab images, this script generates different synthetic test images as required by the tasks:
   ```bash
   python create_synthetic_images.py
   ```
   This creates the test images in `images/`. Alternatively, run `python download_images.py` to fetch the sample images used by the tasks. Choose one source; both scripts write to the same filenames.

   The Lab 8 calibration images (`images/calibration/right01–right14.jpg`, OpenCV's stereo sample set; `right10` does not exist) are committed to the repository.

3. Run any task from the project root or another working directory. For example:
   ```bash
   python task1/task1.py
   python task2/task2.py
   # ... similarly for task3 through task20
   ```
   Each task reads inputs from the project `images/` directory and saves generated visualizations alongside its script.

4. Run the integration tests:
   ```bash
   python -m unittest discover -s tests -v
   ```
   The tests verify that every task runs from outside the project directory, that synthetic fixtures are repeatable and that the camera calibration reproduces the reference values. GitHub Actions runs the same test suite on pushes and pull requests.

## Task Implementations

Each task is implemented in its own file.

### Lab 6
- **`task1.py`:** Compares keypoints from Harris and ORB across 3 distinct images.
- **`task2.py`:** Identifies keypoints under scale/rotation transformations.
- **`task3.py`:** Extracts and reports ORB descriptor counts and dimensions.
- **`task4.py`:** Visualizes the strongest ORB keypoints comparing `nfeatures=50` and `nfeatures=500`.
- **`task5.py`:** Tests feature detection on a low-texture image and high-texture image to show practical differences.
- **`task6.py`:** Performs Robust Feature Matching with Lowe's Ratio test and RANSAC outlier rejection.
- **`task7.py`:** Multi-Image Panorama Stitching of a sequence of 3 images (left, center, right) using homographies and overlap blending.

### Lab 7 — journal
- **`task8.py`:** Feature matching under rotation — ORB descriptors and Hamming-distance matching (top 100 matches).
- **`task9.py`:** Homography with and without RANSAC — compares transforms when seeded synthetic outliers are added.
- **`task10.py`:** Planar object localization — locates the box image in the scene and draws its projected boundary.
- **`task11.py`:** BF vs. FLANN matching — compares SIFT matching strategies on a transformed image.
- **`task12.py`:** Image alignment pipeline — estimates a RANSAC homography and aligns a shifted image.

### Lab 7 — manual (staged SIFT alignment)
- **`task13.py`:** Detect SIFT keypoints and find initial FLANN descriptor matches.
- **`task14.py`:** Filter descriptor matches with Lowe's ratio test.
- **`task15.py`:** Estimate the homography with RANSAC (saved as `task15_homography.npy`).
- **`task16.py`:** Visualize RANSAC inliers and outliers.
- **`task17.py`:** Warp the source image into alignment with the target.

### Lab 8 — Image Classification, Camera Calibration & Video Analytics
- **`task18.py`:** Classical image classification. Images resized to 128×128, 80/20 stratified split, HSV colour histogram (512-D) and HOG (1764-D) features, SVM (RBF) and Random Forest, accuracy chart, confusion matrix, classification report and the most-confused class pair.
  ```bash
  python task18/task18.py                    # Imagenette subset: church, garbage truck, golf ball, parachute (~94 MB download)
  python task18/task18.py --dataset my_data  # your own dataset, one sub-folder per class
  python task18/task18.py --synthetic        # generated shapes dataset, no download
  ```
- **`task19.py`:** Camera calibration on 13 checkerboard views (9×6 inner corners) with sub-pixel corner refinement, `calibrateCamera` (K, distortion D), re-projection error, distortion field, `getOptimalNewCameraMatrix` + undistortion, and a homography that turns the tilted board into a top-down view. Reference result: fx ≈ 542.35, fy ≈ 541.62, k1 ≈ −0.28 (barrel), mean re-projection error ≈ 0.26 px.
- **`task20.py`:** Video analytics. MOG2 background subtraction, shadow removal, morphology, contour filtering, centroid tracking with IDs and trails, a count/FPS overlay and an annotated output video (plus an H.264 copy when `ffmpeg` is installed).
  ```bash
  python task20/task20.py                    # Intel sample street video (~6 MB download)
  python task20/task20.py --video clip.mp4   # your own video
  python task20/task20.py --synthetic        # generated moving shapes, no download
  ```
  Reference result on the sample video: 647 frames, 0.69 moving objects per frame on average (max 5), 19 track IDs.

Downloaded datasets go to `data/`. Task outputs are generated when the scripts run. Both are excluded from version control.
