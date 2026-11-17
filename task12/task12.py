from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent


def align_images(source, reference):
    sift = cv2.SIFT_create()
    source_keypoints, source_descriptors = sift.detectAndCompute(source, None)
    reference_keypoints, reference_descriptors = sift.detectAndCompute(reference, None)
    if source_descriptors is None or reference_descriptors is None:
        raise RuntimeError("SIFT could not find descriptors in both images.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(source_descriptors, reference_descriptors, k=2)
    good_matches = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good_matches) < 4:
        raise RuntimeError(
            f"At least four matches are required for alignment; found {len(good_matches)}."
        )

    source_points = np.float32(
        [source_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    reference_points = np.float32(
        [reference_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(
        source_points, reference_points, cv2.RANSAC, 5.0
    )
    if homography is None or mask is None:
        raise RuntimeError("RANSAC could not estimate the alignment homography.")

    height, width = reference.shape
    aligned = cv2.warpPerspective(source, homography, (width, height))
    return aligned, int(mask.sum())


def main():
    image_path = IMAGE_DIR / "image3.png"
    reference = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise FileNotFoundError(f"Could not load input image: {image_path}")

    height, width = reference.shape
    shift = int(width * 0.2)
    shifted = np.zeros_like(reference)
    shifted[:, : width - shift] = reference[:, shift:]
    aligned, inlier_count = align_images(shifted, reference)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, image, title in zip(
        axes,
        (reference, shifted, aligned),
        ("Reference image", "Shifted image", f"Aligned image ({inlier_count} inliers)"),
    ):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "task12_image_alignment.jpg"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"RANSAC inliers: {inlier_count}")
    print(f"Saved aligned image comparison to {output_path}")


if __name__ == "__main__":
    main()
