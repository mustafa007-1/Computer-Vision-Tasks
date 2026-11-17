from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    image_path = IMAGE_DIR / "image3.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load input image: {image_path}")

    rows, cols = image.shape
    rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1.0)
    rotated = cv2.warpAffine(image, rotation, (cols, rows))
    orb = cv2.ORB_create(nfeatures=1500)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    rotated_keypoints, rotated_descriptors = orb.detectAndCompute(rotated, None)
    if descriptors is None or rotated_descriptors is None:
        raise RuntimeError("ORB could not find descriptors in both images.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(descriptors, rotated_descriptors, k=2)
    good_matches = [
        pair[0] for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good_matches) < 4:
        raise RuntimeError(
            f"At least four ratio-test matches are required; found {len(good_matches)}."
        )

    source_points = np.float32(
        [keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    target_points = np.float32(
        [rotated_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    rng = np.random.default_rng(42)
    noisy_source = np.vstack(
        (source_points, rng.uniform([0, 0], [cols, rows], (20, 1, 2)).astype(np.float32))
    )
    noisy_target = np.vstack(
        (target_points, rng.uniform([0, 0], [cols, rows], (20, 1, 2)).astype(np.float32))
    )

    homography_plain, _ = cv2.findHomography(noisy_source, noisy_target, 0)
    homography_ransac, inlier_mask = cv2.findHomography(
        noisy_source, noisy_target, cv2.RANSAC, 5.0
    )
    if homography_plain is None or homography_ransac is None or inlier_mask is None:
        raise RuntimeError("Could not estimate both homographies.")

    aligned_plain = cv2.warpPerspective(image, homography_plain, (cols, rows))
    aligned_ransac = cv2.warpPerspective(image, homography_ransac, (cols, rows))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, result, title in zip(
        axes,
        (rotated, aligned_plain, aligned_ransac),
        ("Target rotated image", "Homography without RANSAC", "Homography with RANSAC"),
    ):
        axis.imshow(result, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "task9_homography_comparison.jpg"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Ratio-test matches: {len(good_matches)}")
    print(f"RANSAC inliers: {int(inlier_mask.sum())}")
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
