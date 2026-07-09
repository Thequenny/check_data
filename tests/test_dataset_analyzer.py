from __future__ import annotations

import json
import sys
import tempfile
import unittest
from decimal import Decimal
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
    def test_default_dataset_analysis_uses_train_split_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_nifti(
                root / "imagesTr" / "patient_train.nii.gz",
                np.zeros((2, 2, 2), dtype=np.float32),
            )
            write_nifti(
                root / "labelsTr" / "patient_train.nii.gz",
                np.ones((2, 2, 2), dtype=np.uint8),
            )
            write_nifti(
                root / "imagesTs" / "patient_test.nii.gz",
                np.zeros((2, 2, 2), dtype=np.float32),
            )

            analysis = analyze_dataset(root)

            self.assertEqual(analysis.split_filter, "train")
            self.assertEqual(analysis.counts.patients_detected, 1)
            self.assertEqual(analysis.counts.patients_analyzed, 1)
            self.assertEqual(analysis.counts.image_files_detected, 1)
            self.assertEqual(analysis.counts.label_files_detected, 1)
            self.assertEqual(analysis.counts.annotations_missing, 0)
            self.assertEqual(analysis.report_preparation.overview.split_filter, "train")

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
            self.assertEqual(
                analysis.evaluation.consistency.slice_thickness_frequencies[0].thickness_mm,
                2.0,
            )
            self.assertFalse(
                hasattr(
                    analysis.evaluation.consistency.slice_thickness_frequencies[0],
                    "display_value",
                )
            )
            self.assertFalse(analysis.evaluation.consistency.dimensions_are_consistent)
            self.assertEqual(
                analysis.evaluation.memory.minimum_required_memory_readable,
                "48 B",
            )
            self.assertEqual(analysis.report_preparation.overview.split_filter, "train")
            self.assertEqual(
                analysis.report_preparation.task_detection.detected_task,
                "segmentation",
            )
            self.assertTrue(analysis.report_preparation.task_detection.labels_found)
            self.assertEqual(
                analysis.report_preparation.overview.analysis_success_percentage,
                100.0,
            )
            self.assertEqual(analysis.evaluation.missing_data, [])
            self.assertIn(
                "p50",
                analysis.evaluation.intensity.patient_percentile_summaries,
            )
            self.assertEqual(
                analysis.evaluation.intensity.patient_percentile_table[0].percentile,
                "P0.5",
            )
            intensity_statistics = {
                item.statistic
                for item in analysis.evaluation.intensity.patient_intensity_statistics_table
            }
            self.assertIn("Mean intensity", intensity_statistics)
            self.assertIn("Standard deviation", intensity_statistics)
            self.assertEqual(
                analysis.evaluation.intensity.voxel_validity.valid_voxels_readable,
                "20 voxels",
            )
            self.assertTrue(
                analysis.evaluation.consistency.physical_size_frequencies
            )
            self.assertIn("spatial_resampling", categories)
            self.assertIn("slice_thickness", categories)
            self.assertIn("shape_standardization", categories)
            self.assertIn("intensity_statistics", analysis.report_preparation.sections_ready)
            self.assertIn("missing_data", analysis.report_preparation.sections_ready)
            self.assertIn("image_label_alignment", analysis.report_preparation.sections_ready)
            self.assertIn("intensity_scale", analysis.report_preparation.sections_ready)

    def test_frequency_percentages_sum_to_100_after_rounding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for index, spacing in enumerate([1.0, 2.0, 3.0], start=1):
                patient_id = f"patient_{index}"
                write_nifti(
                    root / "imagesTr" / f"{patient_id}.nii.gz",
                    np.zeros((2, 2, 2), dtype=np.float32),
                    spacing=(1.0, 1.0, spacing),
                )
                write_nifti(
                    root / "labelsTr" / f"{patient_id}.nii.gz",
                    np.ones((2, 2, 2), dtype=np.uint8),
                    spacing=(1.0, 1.0, spacing),
                )

            analysis = analyze_dataset(root, split="train")
            percentage_sum = sum(
                Decimal(str(item.percentage))
                for item in analysis.evaluation.consistency.voxel_spacing_frequencies
            )
            decimal_lengths = [
                -Decimal(str(item.percentage)).as_tuple().exponent
                for item in analysis.evaluation.consistency.voxel_spacing_frequencies
            ]

            self.assertEqual(percentage_sum, Decimal("100.0"))
            self.assertTrue(all(length <= 3 for length in decimal_lengths))
            self.assertEqual(
                [
                    item.percentage
                    for item in analysis.evaluation.consistency.voxel_spacing_frequencies
                ],
                [33.334, 33.333, 33.333],
            )

    def test_alignment_and_normalized_intensity_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_nifti(
                root / "imagesTr" / "patient_aligned.nii.gz",
                np.array(
                    [-1024, -500, 0, 200, 500, 1000, 1500, 2000],
                    dtype=np.float32,
                ).reshape((2, 2, 2)),
            )
            write_nifti(
                root / "labelsTr" / "patient_aligned.nii.gz",
                np.ones((2, 2, 2), dtype=np.uint8),
            )
            write_nifti(
                root / "imagesTr" / "patient_misaligned.nii.gz",
                np.array(
                    [-1024, -500, 0, 200, 500, 1000, 1500, 2000],
                    dtype=np.float32,
                ).reshape((2, 2, 2)),
            )
            write_nifti(
                root / "labelsTr" / "patient_misaligned.nii.gz",
                np.ones((2, 2, 3), dtype=np.uint8),
            )
            write_nifti(
                root / "imagesTr" / "patient_normalized.nii.gz",
                np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape((2, 2, 2)),
            )
            write_nifti(
                root / "labelsTr" / "patient_normalized.nii.gz",
                np.ones((2, 2, 2), dtype=np.uint8),
            )

            analysis = analyze_dataset(root, split="train")

            self.assertEqual(analysis.evaluation.alignment.checked_pairs, 3)
            self.assertEqual(analysis.evaluation.alignment.misaligned_pairs, 1)
            self.assertEqual(
                analysis.evaluation.alignment.misaligned_patients[0].patient_id,
                "patient_misaligned",
            )
            self.assertIn(
                "dimensions",
                analysis.evaluation.alignment.misaligned_patients[0].issues,
            )
            self.assertEqual(
                analysis.evaluation.intensity.intensity_scale.normalized_patient_ids,
                ["patient_normalized"],
            )
            categories = {
                recommendation.category
                for recommendation in analysis.evaluation.preprocessing_recommendations
            }
            self.assertIn("label_alignment", categories)
            self.assertIn("intensity_scale", categories)

    def test_missing_annotation_is_reported_with_patient_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_nifti(
                root / "imagesTs" / "patient_unlabeled.nii.gz",
                np.zeros((2, 2, 2), dtype=np.float32),
            )

            analysis = analyze_dataset(root, split="test")

            self.assertEqual(len(analysis.evaluation.missing_data), 1)
            self.assertEqual(
                analysis.evaluation.missing_data[0].patient_id,
                "patient_unlabeled",
            )
            self.assertEqual(
                analysis.evaluation.missing_data[0].missing_fields,
                ["annotation_label"],
            )
            self.assertEqual(
                analysis.report_preparation.missing_data[0].patient_id,
                "patient_unlabeled",
            )

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
