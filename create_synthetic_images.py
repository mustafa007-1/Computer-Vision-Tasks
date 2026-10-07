import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR = Path(__file__).resolve().parent / "images"

def create_synthetic_images():
    IMAGE_DIR.mkdir(exist_ok=True)

    # Image 1: Shapes
    img1 = np.ones((500, 500, 3), dtype=np.uint8) * 255
    cv2.rectangle(img1, (100, 100), (200, 200), (0, 0, 0), -1)
    cv2.circle(img1, (350, 350), 50, (0, 0, 0), -1)
    cv2.imwrite(str(IMAGE_DIR / "image1.jpg"), img1)

    # Image 2: Chessboard pattern
    img2 = np.zeros((500, 500, 3), dtype=np.uint8)
    for i in range(10):
        for j in range(10):
            if (i+j) % 2 == 0:
                cv2.rectangle(img2, (i*50, j*50), ((i+1)*50, (j+1)*50), (255, 255, 255), -1)
    cv2.imwrite(str(IMAGE_DIR / "image2.jpg"), img2)

    # Image 3: Random noise and some text
    img3 = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    cv2.putText(img3, "Test Image 3", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imwrite(str(IMAGE_DIR / "image3.png"), img3)
    
    # Box and Box in scene
    box = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(box, (50, 50), (150, 150), (0, 255, 0), 5)
    cv2.putText(box, "BOX", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(str(IMAGE_DIR / "box.png"), box)
    
    scene = np.ones((600, 600, 3), dtype=np.uint8) * 100
    # Place box rotated in scene
    M = cv2.getRotationMatrix2D((100, 100), 30, 1)
    box_rot = cv2.warpAffine(box, M, (200, 200))
    scene[200:400, 200:400] = box_rot
    # Add some other shapes to scene
    cv2.circle(scene, (100, 100), 40, (255, 0, 0), -1)
    cv2.imwrite(str(IMAGE_DIR / "box_in_scene.png"), scene)
    
    # Building (high texture)
    building = np.random.randint(50, 200, (500, 500, 3), dtype=np.uint8)
    for i in range(10):
        cv2.line(building, (0, i*50), (500, i*50), (0,0,0), 2)
        cv2.line(building, (i*50, 0), (i*50, 500), (0,0,0), 2)
    cv2.imwrite(str(IMAGE_DIR / "building.jpg"), building)
    cv2.imwrite(str(IMAGE_DIR / "high_texture.jpg"), building)
    
    # Low-texture image
    apple = np.ones((500, 500, 3), dtype=np.uint8) * 200
    cv2.circle(apple, (250, 250), 100, (0, 0, 255), -1)
    cv2.imwrite(str(IMAGE_DIR / "low_texture.jpg"), apple)
    
    # Panorama images (Left, Center, Right)
    pano_scene = np.ones((500, 1200, 3), dtype=np.uint8) * 255
    cv2.rectangle(pano_scene, (100, 200), (300, 400), (255, 0, 0), -1) # Blue rect
    cv2.circle(pano_scene, (600, 300), 100, (0, 255, 0), -1) # Green circle
    cv2.fillPoly(pano_scene, [np.array([[900, 400], [1000, 200], [1100, 400]])], (0, 0, 255)) # Red triangle
    # Add noise to make features trackable
    noise = np.random.randint(0, 50, (500, 1200, 3), dtype=np.uint8)
    pano_scene = cv2.add(pano_scene, noise)
    
    # Extract left, center, right with overlap
    left = pano_scene[:, 0:500]
    center = pano_scene[:, 350:850]
    right = pano_scene[:, 700:1200]
    
    cv2.imwrite(str(IMAGE_DIR / "left.jpg"), left)
    cv2.imwrite(str(IMAGE_DIR / "center.jpg"), center)
    cv2.imwrite(str(IMAGE_DIR / "right.jpg"), right)
    print("Synthetic images created successfully in 'images' folder.")

if __name__ == '__main__':
    create_synthetic_images()
