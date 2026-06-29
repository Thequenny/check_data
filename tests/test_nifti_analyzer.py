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

from nifti_analyzer import (  # noqa: E402
    analyze_nifti_volume,
    analyze_patient,
    read_nifti_metadata,
    save_patient_analysis,
)


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


class NiftiAnalyzerTests(unittest.TestCase):
    def test_metadata_and_intensity_are_read_from_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "patient_01.nii.gz"
            data = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
            write_nifti(path, data, spacing=(1.0, 2.0, 3.0))

            metadata = read_nifti_metadata(path)
            analysis = analyze_nifti_volume(path)

            self.assertEqual(metadata.dimensions, [2, 3, 4])
            self.assertEqual(metadata.voxel_spacing, [1.0, 2.0, 3.0])
            self.assertEqual(metadata.orientation, ["R", "A", "S"])
            self.assertEqual(metadata.datatype, "float32")
            self.assertEqual(metadata.voxel_count, 24)
            self.assertEqual(metadata.physical_voxel_size_mm3, 6.0)
            self.assertEqual(metadata.memory_estimate.native_array_bytes, 96)
            self.assertEqual(analysis.intensity.min, 0.0)
            self.assertEqual(analysis.intensity.max, 23.0)
            self.assertEqual(analysis.intensity.mean, 11.5)
            self.assertAlmostEqual(
                analysis.intensity.std,
                float(np.std(data, dtype=np.float64)),
            )

    def test_patient_analysis_handles_present_and_missing_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "patient_02.nii.gz"
            label_path = root / "patient_02_label.nii.gz"
            write_nifti(image_path, np.ones((2, 2, 2), dtype=np.float32))
            write_nifti(label_path, np.ones((2, 2, 2), dtype=np.uint8))

            with_label = analyze_patient(image_path, label_path)
            without_label = analyze_patient(image_path)

            self.assertEqual(with_label.patient_id, "patient_02")
            self.assertTrue(with_label.annotation.present)
            self.assertEqual(with_label.annotation.metadata.dimensions, [2, 2, 2])
            self.assertEqual(with_label.annotation.metadata.datatype, "uint8")
            self.assertFalse(without_label.annotation.present)
            self.assertIsNone(without_label.annotation.metadata)

    def test_patient_analysis_can_be_saved_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "patient_03.nii.gz"
            output_path = root / "patient_03.json"
            write_nifti(image_path, np.zeros((2, 2, 2), dtype=np.float32))

            analysis = analyze_patient(image_path)
            save_patient_analysis(analysis, output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(saved["patient_id"], "patient_03")
            self.assertFalse(saved["annotation"]["present"])
            self.assertEqual(saved["image"]["metadata"]["dimensions"], [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
