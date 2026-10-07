import cv2
from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"

def main():
    print("--- Task 3: Extract descriptors ---")
    paths = (IMAGE_DIR / "box.png", IMAGE_DIR / "box_in_scene.png")
    img1, img2 = (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths)
    if img1 is None or img2 is None:
        missing = [str(path) for path, image in zip(paths, (img1, img2)) if image is None]
        raise FileNotFoundError(f"Could not load image(s): {', '.join(missing)}")
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    print(f"Image 1 (box.png): {len(kp1)} features, Descriptor dimensions: {des1.shape if des1 is not None else 'None'}")
    print(f"Image 2 (box_in_scene.png): {len(kp2)} features, Descriptor dimensions: {des2.shape if des2 is not None else 'None'}")

if __name__ == "__main__":
    main()
