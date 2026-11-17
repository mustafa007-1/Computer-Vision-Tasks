import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class ProjectIntegrationTests(unittest.TestCase):
    def test_all_tasks_run_from_outside_project_directory(self):
        # task -> (extra command-line arguments, expected output files)
        expected_outputs = {
            "task1": ([], [
                f"task1_{image}_{kind}.jpg"
                for image in ("image1", "image2", "image3")
                for kind in ("harris", "orb")
            ]),
            "task2": ([], [
                "task2_original.jpg",
                "task2_transformed.jpg",
                "task2_orig_kp.jpg",
                "task2_trans_kp.jpg",
            ]),
            "task3": ([], []),
            "task4": ([], ["task4_orb_50.jpg", "task4_orb_500.jpg"]),
            "task5": ([], ["task5_low_texture_kp.jpg", "task5_high_texture_kp.jpg"]),
            "task6": ([], ["task6_ransac_matches.jpg"]),
            "task7": ([], ["task7_panorama.jpg"]),
            # Lab 7 journal
            "task8": ([], ["task8_feature_matches.jpg"]),
            "task9": ([], ["task9_homography_comparison.jpg"]),
            "task10": ([], ["task10_object_localization.jpg"]),
            "task11": ([], ["task11_matcher_comparison.jpg"]),
            "task12": ([], ["task12_image_alignment.jpg"]),
            # Lab 7 manual
            "task13": ([], ["task13_initial_matches.jpg"]),
            "task14": ([], ["task14_good_matches.jpg"]),
            "task15": ([], ["task15_homography.npy"]),
            "task16": ([], ["task16_ransac_inliers_outliers.jpg"]),
            "task17": ([], ["task17_aligned.jpg"]),
            # Lab 8 (offline datasets so the tests need no downloads)
            "task18": (["--synthetic"], [
                "task18_samples.png",
                "task18_features.png",
                "task18_accuracy.png",
                "task18_confusion_matrix.png",
                "task18_predictions.png",
            ]),
            "task19": ([], [
                "task19_corners.png",
                "task19_subpixel_zoom.png",
                "task19_reprojection_error.png",
                "task19_distortion_field.png",
                "task19_undistortion.png",
                "task19_homography.png",
                "task19_calibration.npz",
            ]),
            "task20": (["--synthetic"], [
                "task20_tracking.mp4",
                "task20_pipeline.png",
                "task20_frames.png",
            ]),
        }

        with tempfile.TemporaryDirectory() as working_directory:
            for task, (arguments, outputs) in expected_outputs.items():
                with self.subTest(task=task):
                    script = PROJECT_ROOT / task / f"{task}.py"
                    for filename in outputs:
                        (PROJECT_ROOT / task / filename).unlink(missing_ok=True)
                    result = subprocess.run(
                        [PYTHON, str(script), *arguments],
                        cwd=working_directory,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    self.assertEqual(
                        result.returncode,
                        0,
                        msg=f"{result.stdout}\n{result.stderr}",
                    )
                    for filename in outputs:
                        self.assertTrue(
                            (PROJECT_ROOT / task / filename).is_file(),
                            msg=f"{task} did not create {filename}",
                        )

    def test_calibration_matches_reference_values(self):
        result = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / "task19" / "task19.py")],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Valid views: 13 / 13", result.stdout)
        calibration = np.load(PROJECT_ROOT / "task19" / "task19_calibration.npz")
        self.assertAlmostEqual(calibration["K"][0, 0], 542.35, delta=1.0)
        self.assertLess(float(calibration["mean_error"]), 1.0)
        self.assertLess(calibration["D"].ravel()[0], 0)  # barrel distortion

    def test_synthetic_image_generation_is_deterministic(self):
        source_script = PROJECT_ROOT / "create_synthetic_images.py"
        generated_paths = [
            "image1.jpg",
            "image2.jpg",
            "image3.png",
            "box.png",
            "box_in_scene.png",
            "building.jpg",
            "high_texture.jpg",
            "low_texture.jpg",
            "left.jpg",
            "center.jpg",
            "right.jpg",
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            script = root / "create_synthetic_images.py"
            shutil.copy2(source_script, script)
            first_run_hashes = None

            for _ in range(2):
                result = subprocess.run(
                    [PYTHON, str(script)],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(
                    result.returncode, 0, msg=f"{result.stdout}\n{result.stderr}"
                )
                hashes = {
                    filename: hashlib.sha256(
                        (root / "images" / filename).read_bytes()
                    ).hexdigest()
                    for filename in generated_paths
                }
                if first_run_hashes is None:
                    first_run_hashes = hashes
                else:
                    self.assertEqual(hashes, first_run_hashes)


if __name__ == "__main__":
    unittest.main()
