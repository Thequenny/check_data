# check_data

`check_data` analyzes NIfTI CT datasets before AI training. It summarizes the
dataset structure, patient metadata, consistency issues and preprocessing considerations.

The project was tested with datasets from the Medical Segmentation Decathlon,
including the Spleen and Heart tasks.

## Features

- Detect image and label folders, including non-standard folder names that
  contain `image` or `label`.
- Identify image/label relationships and missing annotations.
- Infer whether the dataset is likely segmentation, classification, or unknown.
- Read `.nii` and `.nii.gz` files with `nibabel`.
- Extract dimensions, voxel spacing, orientation, datatype, affine matrix,
  physical size, storage size, and memory estimates.
- Compute patient-level CT intensity statistics.
- Analyze the training split by default to avoid mixing test data into the
  training data quality report.
- Compare voxel spacing, slice thickness, resolution, dimensions, and
  orientation across patients.
- Check whether image and label volumes are aligned for each patient.
- Detect CT images that look already normalized instead of raw Hounsfield units.
- Generate JSON outputs and an HTML report.

## Requirements

The code uses Python and requires:

- `numpy`
- `nibabel`

Install the dependencies with:

```powershell
pip install numpy nibabel
```

## Usage

### Step 0: Check Dataset Structure

```powershell
python src\check_structure_dataset.py dataset\Task09_Spleen
```

To print the full detected structure as JSON:

```powershell
python src\check_structure_dataset.py dataset\Task09_Spleen --json
```

### Step 1: Analyze One Patient

```powershell
python src\nifti_analyzer.py dataset\Task09_Spleen\imagesTr\spleen_10.nii.gz --label dataset\Task09_Spleen\labelsTr\spleen_10.nii.gz --output data\CT_syntesis.json
```

This stores patient-level metadata and intensity statistics in:

```text
data\CT_syntesis.json
```

### Step 2: Analyze a Full Dataset

```powershell
python src\dataset_analyzer.py dataset\Task09_Spleen --output data\analyse_dataset.json
```

By default, the dataset analyzer uses the `train` split only. This avoids
counting unlabeled test cases as missing training labels.

To choose a specific split:

```powershell
python src\dataset_analyzer.py dataset\Task09_Spleen --split train --output data\analyse_dataset.json
```

To analyze every detected split:

```powershell
python src\dataset_analyzer.py dataset\Task09_Spleen --split all --output data\analyse_dataset.json
```

### Step 3: Generate the HTML Report

Generate the report from the default dataset analysis JSON:

```powershell
python data\report.py --input data\analyse_dataset.json --html data\report.html
```

Open the generated report:

```powershell
start data\report.html
```

The report contains:

- dataset overview
- patients information
- task detection
- missing data
- image/label alignment
- slice count and slice thickness
- voxel size
- patient intensity summary
- CT intensity scale check
- voxel validity
- consistency statistics
- warnings
- preprocessing recommendations

## Example Workflow

```powershell
python src\check_structure_dataset.py dataset\Task02_Heart
python src\dataset_analyzer.py dataset\Task02_Heart --output data\analyse_dataset.json
python data\report.py --input data\analyse_dataset.json --html data\report.html
start data\report.html
```

## Tests

Run all tests with:

```powershell
python -m unittest discover -s tests -v
```

The tests generate small temporary NIfTI files, so they do not require a full
medical imaging dataset.
