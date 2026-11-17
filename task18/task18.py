"""Lab 8 - Task 1: image dataset pipeline and classical classification.

Pipeline: load images -> resize to 128x128 -> 80/20 stratified split ->
HSV colour histogram and HOG features -> SVM (RBF) and Random Forest ->
accuracy table, confusion matrix and classification report.

Dataset options (pick one):
    python task18/task18.py                      # Imagenette subset (downloaded once, ~94 MB)
    python task18/task18.py --dataset my_data    # one sub-folder per class
    python task18/task18.py --synthetic          # generated shapes dataset (offline, used by the tests)
"""
import argparse
import random
import tarfile
import urllib.request
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent

SEED = 42
IMG_SIZE = (128, 128)
IMAGES_PER_CLASS = 100
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
# Imagenette folders are ImageNet WordNet IDs -> readable class names
IMAGENETTE_CLASSES = {
    "church": "n03028079",
    "garbage_truck": "n03417042",
    "golf_ball": "n03445777",
    "parachute": "n03888257",
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------------------------------------------------------- dataset
def load_folder_dataset(root, class_folders):
    images, labels = [], []
    for label, (name, folder) in enumerate(class_folders.items()):
        files = sorted(
            path for path in (Path(root) / folder).iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        random.Random(SEED).shuffle(files)
        loaded = 0
        for path in files:
            image = cv2.imread(str(path))
            if image is None:
                continue
            images.append(cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA))
            labels.append(label)
            loaded += 1
            if loaded == IMAGES_PER_CLASS:
                break
        print(f"{name:14s}: {loaded} images")
        if loaded == 0:
            raise RuntimeError(f"No readable images for class '{name}' in {Path(root) / folder}")
    return np.array(images), np.array(labels), list(class_folders)


def load_imagenette():
    DATA_DIR.mkdir(exist_ok=True)
    archive = DATA_DIR / "imagenette2-160.tgz"
    extracted = DATA_DIR / "imagenette2-160"
    if not extracted.is_dir():
        if not archive.exists():
            print(f"Downloading Imagenette (~94 MB) from {IMAGENETTE_URL} ...")
            urllib.request.urlretrieve(IMAGENETTE_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(DATA_DIR)
    return load_folder_dataset(extracted / "train", IMAGENETTE_CLASSES)


def make_synthetic_dataset():
    """Four shape classes with random colour, size, position and noise."""
    rng = np.random.default_rng(SEED)
    classes = ["circle", "square", "triangle", "stripes"]
    images, labels = [], []
    size = IMG_SIZE[0]
    for label, name in enumerate(classes):
        for _ in range(IMAGES_PER_CLASS):
            background = rng.integers(0, 256, 3)
            image = np.full((size, size, 3), background, dtype=np.uint8)
            colour = tuple(int(v) for v in rng.integers(0, 256, 3))
            cx, cy = rng.integers(40, size - 40, 2)
            r = int(rng.integers(18, 36))
            if name == "circle":
                cv2.circle(image, (int(cx), int(cy)), r, colour, -1)
            elif name == "square":
                cv2.rectangle(image, (int(cx - r), int(cy - r)), (int(cx + r), int(cy + r)), colour, -1)
            elif name == "triangle":
                points = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
                cv2.fillPoly(image, [points], colour)
            else:
                step = int(rng.integers(8, 16))
                for x in range(0, size, 2 * step):
                    cv2.rectangle(image, (x, 0), (x + step - 1, size - 1), colour, -1)
            noise = rng.normal(0, 12, image.shape)
            images.append(np.clip(image + noise, 0, 255).astype(np.uint8))
            labels.append(label)
        print(f"{name:14s}: {IMAGES_PER_CLASS} images (synthetic)")
    return np.array(images), np.array(labels), classes


# --------------------------------------------------------------- features
def hsv_histogram(image_bgr, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]).flatten()
    return hist / (hist.sum() + 1e-7)  # L1 normalisation -> independent of image size


def hog_features(image_bgr, visualize=False):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
        visualize=visualize,
    )


def extract(images, kind):
    features = []
    for image in images:
        if kind == "HSV":
            vector = hsv_histogram(image)
        elif kind == "HOG":
            vector = hog_features(image)
        else:
            vector = np.hstack([hsv_histogram(image), hog_features(image)])
        features.append(vector)
    return np.array(features, dtype=np.float32)


def show(axis, image_bgr, title=""):
    axis.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axis.set_title(title)
    axis.axis("off")


def save(figure, name):
    path = OUTPUT_DIR / name
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    print(f"Saved {path.name}")


# ------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=Path, help="folder with one sub-folder of images per class")
    source.add_argument("--synthetic", action="store_true", help="use a generated shapes dataset (offline)")
    args = parser.parse_args()

    print("--- Task 18 (Lab 8 Task 1): Image classification with HSV / HOG + SVM / Random Forest ---")
    random.seed(SEED)
    np.random.seed(SEED)
    if args.synthetic:
        images, labels, classes = make_synthetic_dataset()
    elif args.dataset:
        folders = sorted(path.name for path in args.dataset.iterdir() if path.is_dir())
        images, labels, classes = load_folder_dataset(args.dataset, {name: name for name in folders})
    else:
        images, labels, classes = load_imagenette()
    print(f"Dataset: {images.shape} | labels: {labels.shape}")

    # Sample grid
    figure, axes = plt.subplots(len(classes), 6, figsize=(12, 2.2 * len(classes)), squeeze=False)
    for row, name in enumerate(classes):
        for col, index in enumerate(np.where(labels == row)[0][:6]):
            show(axes[row, col], images[index], name if col == 0 else "")
    save(figure, "task18_samples.png")

    # 80/20 stratified split: every class keeps the same share in train and test
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.20, stratify=labels, random_state=SEED
    )
    print(f"Train: {len(x_train)}  Test: {len(x_test)}")

    features = {}
    for kind in ("HSV", "HOG", "HSV+HOG"):
        features[kind] = (extract(x_train, kind), extract(x_test, kind))
        print(f"{kind:8s} feature length = {features[kind][0].shape[1]}")

    # Feature visualisation for one image
    _, hog_image = hog_features(x_train[0], visualize=True)
    hsv = cv2.cvtColor(x_train[0], cv2.COLOR_BGR2HSV)
    figure, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    show(axes[0], x_train[0], f"Original ({classes[y_train[0]]})")
    axes[1].imshow(hog_image, cmap="inferno")
    axes[1].set_title("HOG visualisation")
    axes[1].axis("off")
    for channel, (name, upper, colour) in enumerate((("Hue", 180, "tab:orange"), ("Saturation", 256, "tab:green"))):
        histogram = cv2.calcHist([hsv], [channel], None, [32], [0, upper]).ravel()
        axes[2 + channel].bar(np.arange(32), histogram, color=colour)
        axes[2 + channel].set_title(f"{name} histogram (32 bins)")
    save(figure, "task18_features.png")

    # Train and evaluate (SVM needs scaled features; trees do not)
    models = {
        "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    results, predictions = [], {}
    for feature_name, (train_features, test_features) in features.items():
        for model_name, make_model in models.items():
            prediction = make_model().fit(train_features, y_train).predict(test_features)
            accuracy = accuracy_score(y_test, prediction)
            results.append((feature_name, model_name, accuracy))
            predictions[(feature_name, model_name)] = prediction

    print(f"\n{'Features':10s} | {'Classifier':14s} | Test accuracy")
    print("-" * 44)
    for feature_name, model_name, accuracy in results:
        print(f"{feature_name:10s} | {model_name:14s} | {accuracy * 100:6.2f} %")
    best_feature, best_model, best_accuracy = max(results, key=lambda result: result[2])
    print(f"\nBest: {best_feature} + {best_model} -> accuracy = {best_accuracy * 100:.2f} %")

    # Accuracy chart
    feature_names, model_names = list(features), list(models)
    figure, axis = plt.subplots(figsize=(8, 4))
    width = 0.38
    for i, model_name in enumerate(model_names):
        values = [100 * a for f, m, a in results if m == model_name]
        bars = axis.bar(np.arange(len(feature_names)) + (i - 0.5) * width, values, width, label=model_name)
        axis.bar_label(bars, fmt="%.1f")
    axis.set_xticks(np.arange(len(feature_names)), feature_names)
    axis.set_ylabel("Test accuracy (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Classifier accuracy by feature type")
    axis.legend()
    save(figure, "task18_accuracy.png")

    # Confusion matrix + report for the best model
    prediction = predictions[(best_feature, best_model)]
    matrix = confusion_matrix(y_test, prediction)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"Confusion matrix (counts)\n{best_feature} + {best_model}")
    ConfusionMatrixDisplay(matrix / matrix.sum(1, keepdims=True), display_labels=classes).plot(
        ax=axes[1], cmap="Greens", colorbar=False, values_format=".2f"
    )
    axes[1].set_title("Row-normalised (per-class recall)")
    save(figure, "task18_confusion_matrix.png")

    print(classification_report(y_test, prediction, target_names=classes, digits=3))
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    true_index, predicted_index = np.unravel_index(off_diagonal.argmax(), off_diagonal.shape)
    if off_diagonal.max() > 0:
        print(
            f"Most confused pair: {classes[true_index]} predicted as "
            f"{classes[predicted_index]} ({off_diagonal.max()} test images)"
        )
    else:
        print("No confusions: every test image was classified correctly.")

    # Example predictions (green = correct, red = wrong)
    picks = np.random.RandomState(1).choice(len(x_test), min(10, len(x_test)), replace=False)
    figure, axes = plt.subplots(2, 5, figsize=(14, 6))
    for axis, index in zip(axes.ravel(), picks):
        show(axis, x_test[index], f"T: {classes[y_test[index]]}\nP: {classes[prediction[index]]}")
        axis.title.set_color("green" if prediction[index] == y_test[index] else "red")
    save(figure, "task18_predictions.png")


if __name__ == "__main__":
    main()
