import urllib.request
import urllib.error
from pathlib import Path

images = {
    "image1.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "image2.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
    "image3.png": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png",
    "box.png": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box.png",
    "box_in_scene.png": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box_in_scene.png",
    "building.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
    "left.jpg": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/stitching/boat1.jpg",
    "center.jpg": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/stitching/boat2.jpg",
    "right.jpg": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/stitching/boat3.jpg",
    "low_texture.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/apple.jpg",
    "high_texture.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
}

image_dir = Path(__file__).resolve().parent / "images"
image_dir.mkdir(exist_ok=True)

for filename, url in images.items():
    filepath = image_dir / filename
    if not filepath.exists():
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename}")
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError(f"Failed to download {filename} from {url}") from error
    else:
        print(f"{filename} already exists.")
