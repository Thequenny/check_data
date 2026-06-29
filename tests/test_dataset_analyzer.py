from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_analyzer import analyze_dataset, save_dataset_analysis  # noqa: E402


def write_nifti(
    path: Path,
    data: np.ndarray,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    for axis, value in enumerate(spacing[:3]):
        affine[axis, axis] = value

    image = nib.Nifti1Image(data, affine)
    image.header.set_zooms(spacing[: data.ndim])
    nib.save(image, str(path))


class DatasetAnalyzerTests(unittest.TestCase):
    def test_train_split_dataset_analysis_contains_report_ready_information(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_nifti(
                root / "imagesTr" / "patient_a.nii.gz",
                np.arange(8, dtype=np.float32).reshape((2, 2, 2)),
                spacing=(1.0, 1.0, 2.0),
            )
            write_nifti(
                root / "labelsTr" / "patient_a.nii.gz",
                np.ones((2, 2, 2), dtype=np.uint8),
                spacing=(1.0, 1.0, 2.0),
            )
            write_nifti(
                root / "imagesTr" / "patient_b.nii.gz",
                np.arange(12, dtype=np.float32).reshape((2, 2, 3)),
                spacing=(1.0, 1.0, 3.0),
            )
            write_nifti(
                root / "labelsTr" / "patient_b.nii.gz",
                np.ones((2, 2, 3), dtype=np.uint8),
                spacing=(1.0, 1.0, 3.0),
            )

            analysis = analyze_dataset(root, split="train")
            categories = {
                recommendation.category
                for recommendation in analysis.evaluation.preprocessing_recommendations
            }

            self.assertEqual(analysis.split_filter, "train")
            self.assertEqual(analysis.counts.patients_detected, 2)
            self.assertEqual(analysis.counts.patients_analyzed, 2)
            self.assertEqual(analysis.counts.patients_failed, 0)
            self.assertEqual(analysis.counts.annotations_present, 2)
            self.assertEqual(analysis.evaluation.consistency.percentage_same_resolution, 100.0)
            self.assertEqual(analysis.evaluation.consistency.percentage_same_thickness, 50.0)
            self.assertEqual(analysis.evaluation.consistency.percentage_same_voxel_spacing, 50.0)
            self.assertFalse(analysis.evaluation.consistency.dimensions_are_consistent)
            self.assertEqual(
                analysis.evaluation.memory.minimum_required_memory_readable,
                "48 B",
            )
            self.assertEqual(analysis.report_preparation.overview.split_filter, "train")
            self.assertEqual(
                analysis.report_preparation.overview.analysis_success_percentage,
                100.0,
            )
            self.assertIn("spatial_resampling", categories)
            self.assertIn("slice_thickness", categories)
            self.assertIn("shape_standardization", categories)
            self.assertIn("intensity_statistics", analysis.report_preparation.sections_ready)

    def test_dataset_analysis_can_be_saved_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "analysis.json"
            write_nifti(
                root / "imagesTr" / "patient_a.nii.gz",
                np.zeros((2, 2, 2), dtype=np.float32),
            )
            write_nifti(
                root / "labelsTr" / "patient_a.nii.gz",
                np.ones((2, 2, 2), dtype=np.uint8),
            )

            analysis = analyze_dataset(root, split="train")
            save_dataset_analysis(analysis, output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(saved["counts"]["patients_analyzed"], 1)
            self.assertEqual(saved["report_preparation"]["overview"]["split_filter"], "train")
            self.assertIn("evaluation", saved)


if __name__ == "__main__":
    unittest.main()
