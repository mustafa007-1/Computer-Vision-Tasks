import cv2
from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent

def main():
    print("--- Task 2: Rotation/Scale Changes ---")
    image_path = IMAGE_DIR / "image1.jpg"
    img1 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Create rotated/scaled version
    rows, cols = img1.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 0.5) # 45 degrees, 0.5 scale
    img2 = cv2.warpAffine(img1, M, (cols, rows))
    
    cv2.imwrite(str(OUTPUT_DIR / "task2_original.jpg"), img1)
    cv2.imwrite(str(OUTPUT_DIR / "task2_transformed.jpg"), img2)
    
    orb = cv2.ORB_create()
    kp1, _ = orb.detectAndCompute(img1, None)
    kp2, _ = orb.detectAndCompute(img2, None)
    
    print(f"Original image keypoints (ORB): {len(kp1)}")
    print(f"Transformed image keypoints (ORB): {len(kp2)}")
    
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0))
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0))
    cv2.imwrite(str(OUTPUT_DIR / "task2_orig_kp.jpg"), img1_kp)
    cv2.imwrite(str(OUTPUT_DIR / "task2_trans_kp.jpg"), img2_kp)
    print("Saved original and transformed feature visualizations.")

if __name__ == "__main__":
    main()
