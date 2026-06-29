# Roadmap of Medical Imaging Dataset Analyzer 

# Objective

Create a tool that analyzes NIfTI CT datasets
and generates a report about dataset characteristics, potential issues and preprocessing considerations before AI training

--------------------------------------
# Step 0:

- Identify dataset structure
 -> Detect image folders
 -> Detect label folders
 -> Handle different naming conventions
- Identify image/label relationship
- Understand if task is classification or segmentation


# Step 1: creation of a first file `nifti_analyzer.py`

# Phase 1.1: Reading of NIfTI files

Goals:
- Load a NIfTI content
- Read metadata

To do:
- Load .nii/.nii.gz files with nibabel
- Extract:
  -> volume dimensions
  -> voxel spacing
  -> orientation
  -> datatype
- affine matrix

# Phase 1.2: Analysis of one patient

Goal:
Analyze one CT and determinate important parameters.

To do:
Compute:
- Compute physical voxel size
- Estimate memory usage
- min, max and mean intensity
- Standard deviation 

# Phase 1.3: Storage of information

Goal:
Storage and output of the informations.

To do:
Store patient information in a dictionary or JSON structure 
- dimensions
- The annotation (if present and say when no)
- Intensities informations

# Step 2: A second file `dataset_analyzer.py`

# Phase 2.1: Data recovery
Goal:
Analyse all patients

To do:
-  Count number of patients
- Iterate through dataset folder
- Run analyze_patient.py for each .nii.gz file
- Store all informations in an adequate structure

# Phase 2.2: Data evaluation
Goal:
Compare the information, parameters by parameters and evaluate the potential issues of data

To do:
- Compute:
    -> Estimate storage size
    -> Estimate memory requirement for loading one volume 
    -> The percentages of CT with the same resolution
    -> The percentage of CT with the same thickness 
    -> Percentage of different voxel spacing
    -> Dimension consistency check
- Store all informations in an adequate structure


# Step 3: Report generation in a file `report.py`

Output:

HTML and PDF report:
- Dataset overview
- Statistics
- Warnings
- Possible preprocessing recommendations
