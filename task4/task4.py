import cv2
from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
OUTPUT_DIR = Path(__file__).resolve().parent

def main():
    print("--- Task 4: Visualize strongest keypoints & detector parameters ---")
    image_path = IMAGE_DIR / "building.jpg"
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # ORB with different parameters
    orb1 = cv2.ORB_create(nfeatures=50) # Strongest 50
    kp1, _ = orb1.detectAndCompute(img, None)
    
    orb2 = cv2.ORB_create(nfeatures=500) # Strongest 500
    kp2, _ = orb2.detectAndCompute(img, None)
    
    print(f"ORB (nfeatures=50): {len(kp1)} keypoints")
    print(f"ORB (nfeatures=500): {len(kp2)} keypoints")
    
    img_color = cv2.imread(str(image_path))
    img_kp1 = cv2.drawKeypoints(img_color, kp1, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_kp2 = cv2.drawKeypoints(img_color, kp2, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imwrite(str(OUTPUT_DIR / "task4_orb_50.jpg"), img_kp1)
    cv2.imwrite(str(OUTPUT_DIR / "task4_orb_500.jpg"), img_kp2)
    print("Saved visualizations for strongest 50 and 500 keypoints.")

if __name__ == "__main__":
    main()
