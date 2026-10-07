import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent

def main():
    print("--- Task 1: Harris vs ORB on three different images ---")
    images = ["image1.jpg", "image2.jpg", "image3.png"]
    for img_name in images:
        image_path = IMAGE_DIR / img_name
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imread(str(image_path))
        if img is None or img_color is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Harris
        dst = cv2.cornerHarris(img, 2, 3, 0.04)
        harris_keypoints = np.argwhere(dst > 0.01 * dst.max())
        
        # ORB
        orb = cv2.ORB_create()
        keypoints, _ = orb.detectAndCompute(img, None)
        
        print(f"{img_name}:")
        print(f"  Harris keypoints: {len(harris_keypoints)}")
        print(f"  ORB keypoints: {len(keypoints)}")
        
        # Visualization
        img_harris = np.copy(img_color)
        img_harris[dst > 0.01 * dst.max()] = [0, 0, 255]
        
        img_orb = cv2.drawKeypoints(img_color, keypoints, None, color=(0, 255, 0), flags=0)
        
        cv2.imwrite(str(OUTPUT_DIR / f"task1_{Path(img_name).stem}_harris.jpg"), img_harris)
        cv2.imwrite(str(OUTPUT_DIR / f"task1_{Path(img_name).stem}_orb.jpg"), img_orb)
        print(f"  Saved visualizations for {img_name}")

if __name__ == "__main__":
    main()
