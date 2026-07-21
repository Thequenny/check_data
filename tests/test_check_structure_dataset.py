from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from check_structure_dataset import identify_dataset_structure  # noqa: E402


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


class DatasetStructureDiscoveryTests(unittest.TestCase):
    def test_decathlon_layout_is_segmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            touch(root / "imagesTr" / "spleen_001.nii.gz")
            touch(root / "labelsTr" / "spleen_001.nii.gz")
            touch(root / "imagesTs" / "spleen_002.nii.gz")
            (root / "dataset.json").write_text(
                json.dumps(
                    {
                        "training": [
                            {
                                "image": "./imagesTr/spleen_001.nii.gz",
                                "label": "./labelsTr/spleen_001.nii.gz",
                            }
                        ],
                        "test": ["./imagesTs/spleen_002.nii.gz"],
                    }
                ),
                encoding="utf-8",
            )

            structure = identify_dataset_structure(root)

            self.assertEqual(structure.task_type, "segmentation")
            self.assertEqual(structure.training_folders, ["imagesTr", "labelsTr"])
            self.assertEqual(structure.test_folders, ["imagesTs"])
            self.assertEqual(structure.image_folders, ["imagesTr", "imagesTs"])
            self.assertEqual(structure.label_folders, ["labelsTr"])
            self.assertEqual(structure.classification_classes, [])
            self.assertEqual(len(structure.image_label_pairs), 1)
            self.assertEqual(structure.image_label_pairs[0].source, "dataset.json")
            self.assertEqual(structure.unmatched_images, [])

    def test_image_and_mask_folders_are_paired_by_subject_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            touch(root / "images" / "patient_01_ct.nii.gz")
            touch(root / "masks" / "patient_01_mask.nii.gz")

            structure = identify_dataset_structure(root)

            self.assertEqual(structure.task_type, "segmentation")
            self.assertEqual(structure.image_folders, ["images"])
            self.assertEqual(structure.label_folders, ["masks"])
            self.assertEqual(len(structure.image_label_pairs), 1)
            self.assertEqual(structure.image_label_pairs[0].subject_id, "patient_01")

    def test_folder_names_containing_image_and_label_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            touch(root / "raw_image_volumes" / "case_a.nii.gz")
            touch(root / "manual_label_maps" / "case_a.nii.gz")

            structure = identify_dataset_structure(root)

            self.assertEqual(structure.task_type, "segmentation")
            self.assertEqual(structure.image_folders, ["raw_image_volumes"])
            self.assertEqual(structure.label_folders, ["manual_label_maps"])
            self.assertEqual(len(structure.image_label_pairs), 1)

    def test_converted_series_folders_are_detected_as_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            touch(root / "serie_003" / "0003_serie_003.nii.gz")

            structure = identify_dataset_structure(root)

            self.assertEqual(structure.task_type, "classification")
            self.assertEqual(structure.classification_classes, [])
            self.assertEqual(structure.image_folders, ["serie_003"])
            self.assertEqual(len(structure.image_files), 1)
            self.assertEqual(structure.unknown_files, [])

    def test_class_folders_are_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            touch(root / "benign" / "case_001.nii.gz")
            touch(root / "malignant" / "case_002.nii.gz")

            structure = identify_dataset_structure(root)

            self.assertEqual(structure.task_type, "classification")
            self.assertEqual(len(structure.image_files), 2)
            self.assertEqual(structure.classification_classes, ["benign", "malignant"])
            self.assertIn("class-like folders", structure.task_reason)


if __name__ == "__main__":
    unittest.main()
