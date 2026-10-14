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
    if not good_matches:
        raise RuntimeError("Lowe's ratio test rejected every descriptor match.")

    visualization = cv2.drawMatches(
        source,
        source_keypoints,
        target,
        target_keypoints,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    figure, axis = plt.subplots(figsize=(16, 8))
    axis.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
    axis.set_title(f"Lowe's ratio-test matches ({len(good_matches)})")
    axis.axis("off")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "task7_manual_good_matches.jpg"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Good matches after Lowe's ratio test: {len(good_matches)}")
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
