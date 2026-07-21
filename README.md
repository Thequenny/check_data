# check_data

`check_data` is a Python tool for checking CT medical imaging datasets before
AI training. It inspects NIfTI datasets, detects image/label relationships,
computes patient-level and dataset-level statistics, and generates a readable
HTML report.

The project was tested with Medical Segmentation Decathlon datasets, including
`Task02_Heart` and `Task09_Spleen`.

## What It Does

- Detects dataset structure, training/test folders, image folders, label
  folders, metadata files, and image/label pairs.
- Supports standard Medical Segmentation Decathlon layouts such as
  `imagesTr`, `labelsTr`, and `imagesTs`.
- Supports non-standard folder names when they contain image/label hints, such
  as `raw_image_volumes` or `manual_label_maps`.
- Infers whether the dataset is likely a segmentation, classification, or
  unknown task.
- Reads `.nii` and `.nii.gz` files with `nibabel`.
- Extracts dimensions, number of slices, voxel spacing, slice thickness,
  orientation, datatype, affine matrix, physical size, file size, and memory
  estimates.
- Computes CT intensity statistics: minimum, maximum, mean, standard deviation,
  percentiles, finite voxels, and non-finite voxels.
- Compares patients for consistency of dimensions, resolution, voxel spacing,
  slice thickness, slice count, physical size, and orientation.
- Checks whether image and label volumes are aligned.
- Detects CT images that look normalized instead of raw Hounsfield units.
- Writes JSON outputs and an HTML report.
- Can prepare datasets containing DICOM files by converting valid CT series to
  NIfTI before analysis.

## Requirements

Use Python 3.10 or newer.

Required for NIfTI analysis:

```powershell
pip install numpy nibabel
```

Required only when the input dataset contains DICOM files:

```powershell
pip install pydicom dicom2nifti
```

The HTML and PDF report generator does not require extra Python packages.

## Project Structure

```text
check_data/
  src/
    check_structure_dataset.py
    nifti_analyzer.py
    dataset_analyzer.py
    general_conversion.py
  data/
    report.py
    analyse_dataset.json
    report.html
    Warning.txt
  dataset/
    Task02_Heart/
    Task09_Spleen/
  tests/
```

## Supported Inputs

The analysis itself only uses CT files in NIfTI format:

- `.nii`
- `.nii.gz`

If a dataset contains DICOM files (`.dcm`), no extra command is required. The program detects the file type automatically, converts valid CT DICOM series to NIfTI first, then continues the analysis on the converted files.

```text
converted_nifti/
```

If a conversion problem occurs, the program stops and generates:

```text
data/Warning.txt
```

In that case, the HTML report should not be generated because the dataset
analysis is incomplete.

## Usage

Run commands from the repository root.

### Step 0: Check Dataset Structure

This step does not read voxel data. It only detects folders, files,splits, image/label pairs, task type, and structure warnings.

```powershell
python src\check_structure_dataset.py dataset\Task09_Spleen
```

Print the full detected structure as JSON:

```powershell
python src\check_structure_dataset.py dataset\Task09_Spleen --json
```


### Step 1: Analyze One Patient

For the analyze one CT image without a label:

```powershell
python src\nifti_analyzer.py dataset\Task09_Spleen\imagesTr\spleen_10.nii.gz
```

Analyze one CT image with its label:

```powershell
python src\nifti_analyzer.py dataset\Task09_Spleen\imagesTr\spleen_10.nii.gz --label dataset\Task09_Spleen\labelsTr\spleen_10.nii.gz
```

By default, the output is written to:

```text
data/CT_data.json
```

Choose a custom output path:

```powershell
python src\nifti_analyzer.py dataset\Task09_Spleen\imagesTr\spleen_10.nii.gz --label dataset\Task09_Spleen\labelsTr\spleen_10.nii.gz --output data\CT_data.json
```

### Step 2: Analyze a Full Dataset

```powershell
python src\dataset_analyzer.py dataset\Task09_Spleen
```

By default, the output is written to:

```text
data/analyse_dataset.json
```

The default split is `train`, so test cases are not counted as missing training
labels.

Choose a split:

```powershell
python src\dataset_analyzer.py dataset\Task09_Spleen --split train --output data\analyse_dataset.json
```

Analyze all detected splits:

```powershell
python src\dataset_analyzer.py dataset\Task09_Spleen --split all --output data\analyse_dataset.json
```

Print the full dataset analysis JSON to the terminal:

```powershell
python src\dataset_analyzer.py dataset\Task09_Spleen --json
```

### Step 3: Generate the Report

Generate the HTML report from the default dataset analysis JSON:

```powershell
python data\report.py --input data\analyse_dataset.json --html data\report.html
```

Open the report on Windows:

```powershell
start data\report.html
```

The report includes:

- Dataset overview
- Patient counts and missing data
- Task detection
- Slice count and slice thickness
- Voxel size
- Intensity statistics
- Consistency statistics
- preprocessing recommendations

## Example Workflow

```powershell
python src\check_structure_dataset.py dataset\Task02_Heart
python src\dataset_analyzer.py dataset\Task02_Heart --output data\analyse_dataset.json
python data\report.py --input data\analyse_dataset.json --html data\report.html
start data\report.html
```

## Tests

Run all tests:

```powershell
python -m unittest discover -s tests -v
```

The tests create small temporary NIfTI files automatically, so no real medical dataset is required to run them.
