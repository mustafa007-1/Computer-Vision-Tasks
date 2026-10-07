# CV Lab 6 - Feature Detection & Description

This workspace contains the implementation for the "Computer Vision Lab Journal — Lab 6" tasks.

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

3. Run any task from the project root or another working directory. For example:
   ```bash
   python task1/task1.py
   python task2/task2.py
   # ... similarly for task3 through task7
   ```
   Each task reads inputs from the project `images/` directory and saves generated visualizations alongside its script.

## Task Implementations

Each task is implemented in its own file:
- **`task1.py`:** Compares keypoints from Harris and ORB across 3 distinct images.
- **`task2.py`:** Identifies keypoints under scale/rotation transformations.
- **`task3.py`:** Extracts and reports ORB descriptor counts and dimensions.
- **`task4.py`:** Visualizes the strongest ORB keypoints comparing `nfeatures=50` and `nfeatures=500`.
- **`task5.py`:** Tests feature detection on a low-texture image and high-texture image to show practical differences.
- **`task6.py`:** Performs Robust Feature Matching with Lowe's Ratio test and RANSAC outlier rejection.
- **`task7.py`:** Multi-Image Panorama Stitching of a sequence of 3 images (left, center, right) using homographies and overlap blending.

Task output images are generated when the scripts run and are excluded from version control.
