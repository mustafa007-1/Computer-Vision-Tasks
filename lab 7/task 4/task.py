from pathlib import Path
from time import perf_counter

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_DIR = Path(__file__).resolve().parents[2] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    image_path = IMAGE_DIR / "image3.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load input image: {image_path}")

    rows, cols = image.shape
    transform = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1.0)
    transformed = cv2.warpAffine(image, transform, (cols, rows))
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    transformed_keypoints, transformed_descriptors = sift.detectAndCompute(
        transformed, None
    )
    if descriptors is None or transformed_descriptors is None:
        raise RuntimeError("SIFT could not find descriptors in both images.")

    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
    start = perf_counter()
    bf_pairs = bf_matcher.knnMatch(descriptors, transformed_descriptors, k=2)
    bf_seconds = perf_counter() - start
    good_bf = [
        pair for pair in bf_pairs
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
    ]

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50)
    )
    start = perf_counter()
    flann_pairs = flann.knnMatch(descriptors, transformed_descriptors, k=2)
    flann_seconds = perf_counter() - start
    good_flann = [
        pair for pair in flann_pairs
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
    ]
    if not good_bf or not good_flann:
        raise RuntimeError("At least one matcher returned no good matches.")

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    for axis, matches, title in (
        (axes[0], good_bf, f"Brute Force ({len(good_bf)} matches, {bf_seconds:.4f}s)"),
        (axes[1], good_flann, f"FLANN ({len(good_flann)} matches, {flann_seconds:.4f}s)"),
    ):
        visualization = cv2.drawMatchesKnn(
            image,
            keypoints,
            transformed,
            transformed_keypoints,
            matches[:50],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        axis.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "task7_matcher_comparison.jpg"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Brute Force: {len(good_bf)} good matches in {bf_seconds:.5f}s")
    print(f"FLANN: {len(good_flann)} good matches in {flann_seconds:.5f}s")
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
