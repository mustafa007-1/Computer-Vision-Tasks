import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent


def estimate_homography(source, target, sift, matcher):
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    source_keypoints, source_descriptors = sift.detectAndCompute(source_gray, None)
    target_keypoints, target_descriptors = sift.detectAndCompute(target_gray, None)
    if source_descriptors is None or target_descriptors is None:
        raise RuntimeError("Could not find features in a panorama image.")

    candidates = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
    good_matches = [
        pair[0]
        for pair in candidates
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good_matches) < 4:
        raise RuntimeError(
            f"Not enough matches to estimate a panorama transform: {len(good_matches)}/4."
        )

    source_points = np.float32(
        [source_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    target_points = np.float32(
        [target_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(
        source_points, target_points, cv2.RANSAC, 5.0
    )
    if homography is None:
        raise RuntimeError("RANSAC could not estimate a panorama transform.")
    return homography


def main():
    print("--- Task 7: Multi-Image Panorama Stitching ---")
    paths = [IMAGE_DIR / f"{name}.jpg" for name in ("left", "center", "right")]
    images = [cv2.imread(str(path)) for path in paths]
    if any(image is None for image in images):
        missing = [str(path) for path, image in zip(paths, images) if image is None]
        raise FileNotFoundError(f"Could not load panorama image(s): {', '.join(missing)}")

    img_left, img_center, img_right = images
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    transforms = [
        estimate_homography(img_left, img_center, sift, matcher),
        np.eye(3, dtype=np.float64),
        estimate_homography(img_right, img_center, sift, matcher),
    ]

    corners = []
    for image, transform in zip(images, transforms):
        height, width = image.shape[:2]
        image_corners = np.float32(
            [[0, 0], [width, 0], [width, height], [0, height]]
        ).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(image_corners, transform))

    all_corners = np.concatenate(corners, axis=0).reshape(-1, 2)
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
    translation = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64
    )
    canvas_size = (max_x - min_x, max_y - min_y)

    accumulated = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)
    weights = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)
    for image, transform in zip(images, transforms):
        warped = cv2.warpPerspective(image, translation @ transform, canvas_size)
        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(
            source_mask, translation @ transform, canvas_size
        )
        valid = warped_mask > 0
        accumulated[valid] += warped[valid].astype(np.float32)
        weights[valid] += 1

    panorama = np.zeros_like(accumulated, dtype=np.uint8)
    valid = weights > 0
    panorama[valid] = (accumulated[valid] / weights[valid, None]).astype(np.uint8)
    y_coords, x_coords = np.where(valid)
    panorama = panorama[
        y_coords.min() : y_coords.max() + 1, x_coords.min() : x_coords.max() + 1
    ]

    output_path = OUTPUT_DIR / "task7_panorama.jpg"
    if not cv2.imwrite(str(output_path), panorama):
        raise OSError(f"Could not write panorama image: {output_path}")
    print(f"Panorama saved successfully to {output_path.name}.")

if __name__ == "__main__":
    main()
