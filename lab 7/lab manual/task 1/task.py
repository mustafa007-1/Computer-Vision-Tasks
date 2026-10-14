from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_DIR = Path(__file__).resolve().parents[3] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    image_path = IMAGE_DIR / "image3.png"
    source = cv2.imread(str(image_path))
    if source is None:
        raise FileNotFoundError(f"Could not load input image: {image_path}")

    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    height, width = source_gray.shape
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), 15, 1.0)
    transform[:, 2] += (20, -10)
    target = cv2.warpAffine(source, transform, (width, height))
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    source_keypoints, source_descriptors = sift.detectAndCompute(source_gray, None)
    target_keypoints, target_descriptors = sift.detectAndCompute(target_gray, None)
    if source_descriptors is None or target_descriptors is None:
        raise RuntimeError("SIFT could not find descriptors in both images.")

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50)
    )
    pairs = flann.knnMatch(source_descriptors, target_descriptors, k=2)
    raw_matches = [pair[0] for pair in pairs if pair]
    if not raw_matches:
        raise RuntimeError("FLANN did not find any descriptor matches.")

    source_keypoint_image = cv2.drawKeypoints(
        source, source_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    target_keypoint_image = cv2.drawKeypoints(
        target, target_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    match_image = cv2.drawMatches(
        source,
        source_keypoints,
        target,
        target_keypoints,
        raw_matches[:100],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    figure, axes = plt.subplots(3, 1, figsize=(14, 18))
    for axis, image, title in zip(
        axes,
        (source_keypoint_image, target_keypoint_image, match_image),
        ("Source SIFT keypoints", "Transformed SIFT keypoints", "Initial FLANN matches"),
    ):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "task7_manual_initial_matches.jpg"
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    print(f"Source keypoints: {len(source_keypoints)}")
    print(f"Target keypoints: {len(target_keypoints)}")
    print(f"Initial FLANN matches: {len(raw_matches)}")
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
