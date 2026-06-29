# check_data

This program analyzes NIfTI CT datasets before AI training. It is designed to
summarize dataset structure, patient metadata, consistency issues and preprocessing considerations.
For testing, I used the Spleen dataset from the `Medical Segmentation Decathlon`.


## Features

- Detect image and label folders, including non-standard names containing
  `image` or `label`.
- Identify image/label relationships.
- Infer whether a dataset is likely segmentation or classification.
- Read `.nii` and `.nii.gz` files with `nibabel`.
- Extract dimensions, voxel spacing, orientation, datatype, affine matrix, and
  memory estimates.
- Compute patient-level CT intensity statistics.
- Analyze all detected patients in a dataset.
- Compare voxel spacing, slice thickness, resolution, dimensions, and
  orientation across patients.
- Generate JSON outputs for later report generation.


## Requirements

The code uses Python and requires:

- `numpy`
- `nibabel`

Install them with:

```powershell
pip install numpy nibabel
```

## Usage

### Step 0: Check Dataset Structure

```powershell
python src\check_structure_dataset.py dataset\Task09_Spleen
```

To print the full structure as JSON:

```powershell
python src\check_structure_dataset.py dataset\Task09_Spleen --json
```

### Step 1: Analyze One Patient

```powershell
python src\nifti_analyzer.py dataset\Task09_Spleen\imagesTr\spleen_10.nii.gz --label dataset\Task09_Spleen\labelsTr\spleen_10.nii.gz --output data\CT_syntesis.json
```

This stores patient-level metadata and intensity statistics in JSON format.

### Step 2: Analyze a Full Dataset

```powershell
python src\dataset_analyzer.py dataset\Task09_Spleen --output data\analyse_dataset.json
```

To analyze only one split:

```powershell
python src\dataset_analyzer.py dataset\Task09_Spleen --split train
```

The dataset analysis includes:

- number of detected, analyzed, and failed patients
- annotation coverage
- storage size in readable units
- minimum memory estimate
- slice count and slice thickness distributions
- voxel spacing and resolution distributions
- dimension consistency checks
- intensity statistics
- warnings and preprocessing recommendations

## Tests

Run all tests with:

```powershell
python -m unittest discover -s tests -v
```

The tests generate small temporary NIfTI files, so they do not require the full
example dataset.

