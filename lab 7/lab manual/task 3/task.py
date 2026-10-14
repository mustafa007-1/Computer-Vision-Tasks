from pathlib import Path

import cv2
import numpy as np


IMAGE_DIR = Path(__file__).resolve().parents[3] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    image_path = IMAGE_DIR / "image3.png"
    source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"Could not load input image: {image_path}")
    height, width = source.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), 15, 1.0)
    transform[:, 2] += (20, -10)
    target = cv2.warpAffine(source, transform, (width, height))
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    source_keypoints, source_descriptors = sift.detectAndCompute(source_gray, None)
    target_keypoints, target_descriptors = sift.detectAndCompute(target_gray, None)
    if source_descriptors is None or target_descriptors is None:
        raise RuntimeError("SIFT could not find descriptors in both images.")
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50)
    )
    pairs = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
    good_matches = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
    ]
    if len(good_matches) < 4:
        raise RuntimeError(
            f"At least four matches are required to estimate a homography; found {len(good_matches)}."
        )

    source_points = np.float32(
        [source_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    target_points = np.float32(
        [target_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(
        source_points, target_points, cv2.RANSAC, 5.0
    )
    if homography is None or mask is None:
        raise RuntimeError("RANSAC could not estimate a homography.")

    print("Homography matrix:")
    print(homography)
    print(f"RANSAC inliers: {int(mask.sum())}")
    print(f"RANSAC outliers: {len(good_matches) - int(mask.sum())}")

    output_path = OUTPUT_DIR / "task7_manual_homography.npy"
    np.save(output_path, homography)
    if not output_path.is_file():
        raise OSError(f"Could not save homography matrix: {output_path}")
    print(f"Saved homography matrix to {output_path}")


if __name__ == "__main__":
    main()
