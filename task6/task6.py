import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent

def main():
    print("--- Task 6: Robust Feature Matching with RANSAC Outlier Rejection ---")
    image1_path = IMAGE_DIR / "box.png"
    image2_path = IMAGE_DIR / "box_in_scene.png"
    img1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load box.png and box_in_scene.png from the images directory.")
    
    # SIFT is better for Lowe's ratio test
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        raise RuntimeError("SIFT could not find descriptors in both input images.")
    
    # k-NN match
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good_matches.append(pair[0])
            
    print(f"Total initial matches (after Lowe's): {len(good_matches)}")
    
    # RANSAC
    if len(good_matches) < 4:
        raise RuntimeError(f"Not enough matches to estimate a homography: {len(good_matches)}/4.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        raise RuntimeError("RANSAC could not estimate a homography for the matched features.")

    matches_mask = mask.ravel().tolist()
    print(f"Total surviving inliers (after RANSAC): {sum(matches_mask)}")
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
    output_path = OUTPUT_DIR / "task6_ransac_matches.jpg"
    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Could not write output image: {output_path}")
    print(f"Saved RANSAC inlier matches to {output_path.name}.")

if __name__ == "__main__":
    main()
