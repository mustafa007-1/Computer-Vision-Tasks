from pathlib import Path

import cv2
import numpy as np


IMAGE_DIR = Path(__file__).resolve().parents[2] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent


def main():
    template_path = IMAGE_DIR / "box.png"
    scene_path = IMAGE_DIR / "box_in_scene.png"
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    scene = cv2.imread(str(scene_path), cv2.IMREAD_GRAYSCALE)
    if template is None or scene is None:
        missing = [
            str(path)
            for path, image in ((template_path, template), (scene_path, scene))
            if image is None
        ]
        raise FileNotFoundError(f"Could not load image(s): {', '.join(missing)}")

    sift = cv2.SIFT_create()
    template_keypoints, template_descriptors = sift.detectAndCompute(template, None)
    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene, None)
    if template_descriptors is None or scene_descriptors is None:
        raise RuntimeError("SIFT could not find features in the template and scene.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(template_descriptors, scene_descriptors, k=2)
    good_matches = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
    ]
    if len(good_matches) < 4:
        raise RuntimeError(
            f"At least four matches are required to locate the object; found {len(good_matches)}."
        )

    template_points = np.float32(
        [template_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    scene_points = np.float32(
        [scene_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(
        template_points, scene_points, cv2.RANSAC, 5.0
    )
    if homography is None or mask is None:
        raise RuntimeError("RANSAC could not estimate the object's location.")

    height, width = template.shape
    corners = np.float32(
        [[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]
    ).reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(corners, homography)
    scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    result = cv2.polylines(
        scene_color,
        [np.int32(projected_corners)],
        True,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    output_path = OUTPUT_DIR / "task7_object_localization.jpg"
    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Could not write visualization: {output_path}")
    print(f"Ratio-test matches: {len(good_matches)}")
    print(f"RANSAC inliers: {int(mask.sum())}")
    print(f"Saved localized object to {output_path}")


if __name__ == "__main__":
    main()
