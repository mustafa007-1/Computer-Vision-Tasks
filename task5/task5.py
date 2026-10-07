import cv2
from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent

def main():
    print("--- Task 5: Low-texture vs Highly textured image ---")
    low_path = IMAGE_DIR / "low_texture.jpg"
    high_path = IMAGE_DIR / "high_texture.jpg"
    img_low = cv2.imread(str(low_path), cv2.IMREAD_GRAYSCALE)
    img_high = cv2.imread(str(high_path), cv2.IMREAD_GRAYSCALE)
    if img_low is None or img_high is None:
        raise FileNotFoundError("Could not load low- and high-texture test images.")
    
    orb = cv2.ORB_create()
    kp_low, _ = orb.detectAndCompute(img_low, None)
    kp_high, _ = orb.detectAndCompute(img_high, None)
    
    print(f"Low-texture image keypoints: {len(kp_low)}")
    print(f"Highly textured image keypoints: {len(kp_high)}")
    print("Practical difference: Highly textured images have more intensity variations (corners/edges), leading to more detected features.")
    
    img_low_color = cv2.imread(str(low_path))
    img_high_color = cv2.imread(str(high_path))
    
    img_low_kp = cv2.drawKeypoints(img_low_color, kp_low, None, color=(255,0,0))
    img_high_kp = cv2.drawKeypoints(img_high_color, kp_high, None, color=(255,0,0))
    
    cv2.imwrite(str(OUTPUT_DIR / "task5_low_texture_kp.jpg"), img_low_kp)
    cv2.imwrite(str(OUTPUT_DIR / "task5_high_texture_kp.jpg"), img_high_kp)
    print("Saved visualizations for low and high texture image keypoints.")

if __name__ == "__main__":
    main()
