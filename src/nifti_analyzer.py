"""NIfTI volume analysis for one CT image or one patient.

This module implements Step 1 of the roadmap:
- load .nii and .nii.gz files with nibabel;
- read NIfTI metadata;
- compute basic CT intensity statistics for one image;
- return patient-level information as a dictionary/JSON structure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - depends on environment setup
    nib = None
    _NIBABEL_IMPORT_ERROR = exc
else:
    _NIBABEL_IMPORT_ERROR = None


NIFTI_EXTENSIONS = (".nii.gz", ".nii")


@dataclass
class MemoryEstimate:
    """Estimated storage and in-memory size for a NIfTI volume."""

    file_size_bytes: int
    native_array_bytes: int
    float32_array_bytes: int
    float64_array_bytes: int


@dataclass
class NiftiMetadata:
    """Metadata read from a NIfTI file without computing intensity stats."""

    path: str
    dimensions: list[int]
    voxel_spacing: list[float]
    orientation: list[str]
    datatype: str
    affine: list[list[float]]
    voxel_count: int
    physical_voxel_size_mm3: float | None
    physical_size_mm: list[float]
    memory_estimate: MemoryEstimate


@dataclass
class IntensityInformation:
    """Basic intensity distribution for one CT volume."""

    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    percentiles: dict[str, float | None]
    finite_voxel_count: int
    non_finite_voxel_count: int


@dataclass
class NiftiVolumeAnalysis:
    """Full analysis of one NIfTI image volume."""

    metadata: NiftiMetadata
    intensity: IntensityInformation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnnotationInformation:
    """Information about an optional patient annotation/label."""

    present: bool
    path: str | None
    metadata: NiftiMetadata | None
    message: str


@dataclass
class PatientAnalysis:
    """Patient-level structure expected by later dataset analysis steps."""

    patient_id: str
    image: NiftiVolumeAnalysis
    annotation: AnnotationInformation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_nifti(path: str | Path) -> Any:
    """Load a .nii or .nii.gz file with nibabel."""

    _require_nibabel()
    nifti_path = _validate_nifti_path(path)
    return nib.load(str(nifti_path))


def read_nifti_metadata(path: str | Path) -> NiftiMetadata:
    """Read dimensions, spacing, orientation, datatype, and affine matrix."""

    nifti_path = _validate_nifti_path(path)
    image = load_nifti(nifti_path)
    shape = tuple(int(value) for value in image.shape)
    zooms = tuple(float(value) for value in image.header.get_zooms()[: len(shape)])
    datatype = str(image.get_data_dtype())

    voxel_count = _compute_voxel_count(shape)
    physical_voxel_size_mm3 = _compute_physical_voxel_size(zooms)
    physical_size_mm = [
        float(dimension * spacing)
        for dimension, spacing in zip(shape[:3], zooms[:3], strict=False)
    ]

    metadata = NiftiMetadata(
        path=str(nifti_path),
        dimensions=list(shape),
        voxel_spacing=list(zooms),
        orientation=list(nib.orientations.aff2axcodes(image.affine)),
        datatype=datatype,
        affine=_matrix_to_list(image.affine),
        voxel_count=voxel_count,
        physical_voxel_size_mm3=physical_voxel_size_mm3,
        physical_size_mm=physical_size_mm,
        memory_estimate=_estimate_memory(nifti_path, shape, datatype),
    )
    image.uncache()
    return metadata


def analyze_nifti_volume(path: str | Path) -> NiftiVolumeAnalysis:
    """Analyze one CT NIfTI volume and compute intensity information."""

    nifti_path = _validate_nifti_path(path)
    metadata = read_nifti_metadata(nifti_path)
    image = load_nifti(nifti_path)
    data = image.get_fdata(dtype=np.float32)
    intensity = _compute_intensity_information(data)
    image.uncache()
    return NiftiVolumeAnalysis(metadata=metadata, intensity=intensity)


def analyze_patient(
    image_path: str | Path,
    label_path: str | Path | None = None,
) -> PatientAnalysis:
    """Analyze one patient image and optional annotation label."""

    nifti_image_path = _validate_nifti_path(image_path)
    image_analysis = analyze_nifti_volume(nifti_image_path)

    if label_path is None:
        annotation = AnnotationInformation(
            present=False,
            path=None,
            metadata=None,
            message="No annotation label was provided for this patient.",
        )
    else:
        nifti_label_path = _validate_nifti_path(label_path)
        # Labels are read as metadata only here. Dataset-level checks need their
        # dimensions, spacing, orientation, and affine, not their voxel values.
        annotation = AnnotationInformation(
            present=True,
            path=str(nifti_label_path),
            metadata=read_nifti_metadata(nifti_label_path),
            message="Annotation label is present.",
        )

    return PatientAnalysis(
        patient_id=_patient_id_from_path(nifti_image_path),
        image=image_analysis,
        annotation=annotation,
    )


def save_patient_analysis(
    analysis: PatientAnalysis,
    output_path: str | Path,
) -> None:
    """Save patient analysis as formatted JSON."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(analysis.to_dict(), file, indent=2)


def _compute_intensity_information(data: np.ndarray) -> IntensityInformation:
    # CT statistics should ignore NaN and infinite values, while still counting
    # them so the dataset report can warn about invalid voxels.
    finite_mask = np.isfinite(data)
    finite_voxel_count = int(np.count_nonzero(finite_mask))
    non_finite_voxel_count = int(data.size - finite_voxel_count)

    if finite_voxel_count == 0:
        return IntensityInformation(
            min=None,
            max=None,
            mean=None,
            std=None,
            percentiles=_empty_percentiles(),
            finite_voxel_count=0,
            non_finite_voxel_count=non_finite_voxel_count,
        )

    finite_data = data[finite_mask]
    return IntensityInformation(
        min=float(np.min(finite_data)),
        max=float(np.max(finite_data)),
        mean=float(np.mean(finite_data, dtype=np.float64)),
        std=float(np.std(finite_data, dtype=np.float64)),
        percentiles=_compute_percentiles(finite_data),
        finite_voxel_count=finite_voxel_count,
        non_finite_voxel_count=non_finite_voxel_count,
    )


def _compute_percentiles(data: np.ndarray) -> dict[str, float | None]:
    percentiles = [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]
    values = np.percentile(data, percentiles)
    return {
        _percentile_key(percentile): float(value)
        for percentile, value in zip(percentiles, values, strict=True)
    }


def _empty_percentiles() -> dict[str, float | None]:
    return {
        _percentile_key(percentile): None
        for percentile in [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]
    }


def _percentile_key(percentile: float) -> str:
    label = str(percentile).replace(".", "_")
    return f"p{label}"


def _estimate_memory(
    path: Path,
    shape: tuple[int, ...],
    datatype: str,
) -> MemoryEstimate:
    voxel_count = _compute_voxel_count(shape)
    try:
        native_dtype_size = np.dtype(datatype).itemsize
    except TypeError:
        native_dtype_size = 0

    return MemoryEstimate(
        file_size_bytes=int(path.stat().st_size),
        native_array_bytes=int(voxel_count * native_dtype_size),
        float32_array_bytes=int(voxel_count * np.dtype(np.float32).itemsize),
        float64_array_bytes=int(voxel_count * np.dtype(np.float64).itemsize),
    )


def _compute_voxel_count(shape: tuple[int, ...]) -> int:
    if not shape:
        return 0
    return int(np.prod(shape, dtype=np.int64))


def _compute_physical_voxel_size(zooms: tuple[float, ...]) -> float | None:
    spatial_zooms = zooms[:3]
    if len(spatial_zooms) < 3:
        return None
    return float(np.prod(spatial_zooms, dtype=np.float64))


def _matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.tolist()]


def _patient_id_from_path(path: Path) -> str:
    name = path.name
    for extension in NIFTI_EXTENSIONS:
        if name.lower().endswith(extension):
            return name[: -len(extension)]
    return path.stem


def _validate_nifti_path(path: str | Path) -> Path:
    nifti_path = Path(path).expanduser().resolve()
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file does not exist: {nifti_path}")
    if not nifti_path.is_file():
        raise FileNotFoundError(f"NIfTI path is not a file: {nifti_path}")
    if not _is_nifti_file(nifti_path):
        raise ValueError(f"Expected a .nii or .nii.gz file: {nifti_path}")
    return nifti_path


def _is_nifti_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(extension) for extension in NIFTI_EXTENSIONS)


def _require_nibabel() -> None:
    if nib is None:
        raise ImportError(
            "nibabel is required to load NIfTI files. Install it with "
            "`pip install nibabel`."
        ) from _NIBABEL_IMPORT_ERROR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze one NIfTI CT volume.")
    parser.add_argument("image", help="Path to the patient CT image (.nii or .nii.gz).")
    parser.add_argument(
        "--label",
        help="Optional path to the patient annotation/label (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. If omitted, JSON is printed to stdout.",
    )
    args = parser.parse_args(argv)

    analysis = analyze_patient(args.image, args.label)
    if args.output:
        save_patient_analysis(analysis, args.output)
    else:
        print(json.dumps(analysis.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
