import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class ProjectIntegrationTests(unittest.TestCase):
    def test_all_tasks_run_from_outside_project_directory(self):
        expected_outputs = {
            "task1": [
                f"task1_{image}_{kind}.jpg"
                for image in ("image1", "image2", "image3")
                for kind in ("harris", "orb")
            ],
            "task2": [
                "task2_original.jpg",
                "task2_transformed.jpg",
                "task2_orig_kp.jpg",
                "task2_trans_kp.jpg",
            ],
            "task3": [],
            "task4": ["task4_orb_50.jpg", "task4_orb_500.jpg"],
            "task5": ["task5_low_texture_kp.jpg", "task5_high_texture_kp.jpg"],
            "task6": ["task6_ransac_matches.jpg"],
            "task7": ["task7_panorama.jpg"],
        }

        with tempfile.TemporaryDirectory() as working_directory:
            for task, outputs in expected_outputs.items():
                with self.subTest(task=task):
                    script = PROJECT_ROOT / task / f"{task}.py"
                    result = subprocess.run(
                        [PYTHON, str(script)],
                        cwd=working_directory,
                        capture_output=True,
                        text=True,
                        timeout=120,
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

    def test_lab7_tasks_run_from_outside_project_directory(self):
        tasks = [
            (
                PROJECT_ROOT / "lab 7" / f"task {number}" / "task.py",
                PROJECT_ROOT / "lab 7" / f"task {number}",
                output,
            )
            for number, output in (
                (1, "task7_feature_matches.jpg"),
                (2, "task7_homography_comparison.jpg"),
                (3, "task7_object_localization.jpg"),
                (4, "task7_matcher_comparison.jpg"),
                (5, "task7_image_alignment.jpg"),
            )
        ]
        tasks.extend(
            (
                PROJECT_ROOT / "lab 7" / "lab manual" / f"task {number}" / "task.py",
                PROJECT_ROOT / "lab 7" / "lab manual" / f"task {number}",
                output,
            )
            for number, output in (
                (1, "task7_manual_initial_matches.jpg"),
                (2, "task7_manual_good_matches.jpg"),
                (3, "task7_manual_homography.npy"),
                (4, "task7_manual_ransac_inliers_outliers.jpg"),
                (5, "task7_manual_aligned.jpg"),
            )
        )

        with tempfile.TemporaryDirectory() as working_directory:
            for script, output_directory, output in tasks:
                with self.subTest(script=str(script.relative_to(PROJECT_ROOT))):
                    result = subprocess.run(
                        [PYTHON, str(script)],
                        cwd=working_directory,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    self.assertEqual(
                        result.returncode,
                        0,
                        msg=f"{result.stdout}\n{result.stderr}",
                    )
                    self.assertTrue(
                        (output_directory / output).is_file(),
                        msg=f"{script} did not create {output}",
                    )

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
