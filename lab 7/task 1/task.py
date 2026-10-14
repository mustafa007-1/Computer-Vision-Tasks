from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")


IMAGE_DIR = Path(__file__).resolve().parents[2] / "images"
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

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(descriptors, rotated_descriptors), key=lambda m: m.distance)
    selected_matches = matches[:100]
    if not selected_matches:
        raise RuntimeError("ORB did not find any matches between the images.")

    visualization = cv2.drawMatches(
        image,
        keypoints,
        rotated,
        rotated_keypoints,
        selected_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    output_path = OUTPUT_DIR / "task7_feature_matches.jpg"
    if not cv2.imwrite(str(output_path), visualization):
        raise OSError(f"Could not write visualization: {output_path}")

    print(f"Total matches: {len(matches)}")
    print(f"Selected matches: {len(selected_matches)}")
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
