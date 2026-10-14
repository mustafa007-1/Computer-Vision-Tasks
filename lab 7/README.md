# Computer Vision Lab 7

Lab 7 is split into two sections from `CV_LAB_07.ipynb`. Every task is a standalone Python script and can be run from the repository root or another working directory.

## Setup

Install the repository dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

The scripts use the sample images in the repository's `images/` directory. To recreate those fixtures:

```bash
python create_synthetic_images.py
```

## Lab Journal

Run `python "lab 7/task N/task.py"` for a journal exercise:

1. **Feature matching under rotation** — ORB descriptors and Hamming-distance matching.
2. **Homography with and without RANSAC** — compares transforms with seeded synthetic outliers.
3. **Planar object localization** — locates the box image in the scene and draws its projected boundary.
4. **BF vs. FLANN matching** — compares SIFT matching strategies on a transformed image.
5. **Image alignment pipeline** — estimates a RANSAC homography and aligns a shifted image.

## Lab Manual

Run `python "lab 7/lab manual/task N/task.py"` for the staged SIFT alignment walkthrough:

1. Detect SIFT keypoints and find initial FLANN descriptor matches.
2. Filter descriptor matches with Lowe's ratio test.
3. Estimate the homography with RANSAC.
4. Visualize RANSAC inliers and outliers.
5. Warp the source image into alignment with the target.

Each task saves its visualization in its own task folder. Generated Lab 7 images are excluded from version control.
