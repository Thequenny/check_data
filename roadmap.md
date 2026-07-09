# Roadmap of Medical Imaging Dataset Analyzer 

## Objective

Create a tool that analyzes NIfTI CT datasets
and generates a report about dataset characteristics, potential issues and preprocessing considerations before AI training.

This program must always take in input, CT format and not DICOM format. 

--------------------------------------
### Step 0:

- Identify dataset structure
 -> Identify trainning and test folders
 -> Detect image folders
 -> Detect label folders
 -> Handle different naming conventions
- Identify image/label relationship
- Understand if task is classification or segmentation
 -> if classification, specify the classes

IMPORTANT: Keep in mind that the names of the dataset folders won't always be the same, as long as they contain the strings 'image' and 'label'.

### Step 1: creation of a first file `nifti_analyzer.py`

Goals:
- Load a NIfTI content
- Read metadata and store the extracted informations in a `.json` file named `CT_data` that you always save in `data/`.

#### Phase 1.1: Reading of NIfTI files

To do:
- Load `.nii/.nii.gz` files with always nibabel
- Extract:
 -> volume dimensions
 -> voxel spacing
 -> orientation
 -> datatype
- affine matrix

Use dataclass for more practicality.

#### Phase 1.2: Analysis of one patient

Goal:
Analyze one CT and compute important parameters which are going to be used in the step 2.

To do:
Compute:
- Compute physical voxel size
- Estimate memory usage
- min, max and mean intensity
- Standard deviation.

#### Phase 1.3: Storage of information

Goal:
Storage and output of the informations. 
Patient information must be saved in the JSON structure as announced the the step 1 goals.

To do:

Save: 
- Dimensions computed
- Number of slices and the thickness
- Intensities computed
- The annotation (if present and say when no).

### Step 2: creation of a second file `dataset_analyzer.py`

This function take in input a folder of `.nii.gz` or `.nii` files and out a `.json` file which contains all variables with values computed. This file will be named `analyse_dataset` and will be saved in `data/`. 
Use `nifti_analyzer.py` for not having overlapping functions.

#### Phase 2.1: Data recovery
Goal:
Analyse all patients.

To do:
- Count number of patients
- Iterate through dataset folder
- Run `nifti_analyzer.py` for each `.nii.gz` or `.nii` file
- Store all informations in a JSON structure.

#### Phase 2.2: Data evaluation
Goal:
Compare the information, parameters by parameters and evaluate the potential issues of the data.

To do:
- Browse the JSON structure and compute:
 -> Estimate storage size
 -> Estimate memory requirement for loading one volume 
 -> The percentages of CT with the same resolution
 -> The percentage of CT with the same thickness 
 -> Percentage of different voxel spacing
 -> Dimension consistency check
- Save all informations in the .json output file.

### Step 3: Report generation named `report.py` in the folder `data/`

Input: `data\analyse_dataset.json`

Output: 
HTML report:
- Number of patients
- Number of analysed patients
- Missing data (specified the patient ID and the data missing)
- Task detection:
  -> Segmentation (if labels found)
  -> Classification (if labels absent)
  -> Unknown
- Number of failed analyses
- Minimum memory needed to run the program (in MB or KB)
- Number and thickness of slices 2D/3D
 -> if slices different, write number of patient by slices and corresponding percentages
- The voxel size (resolution) in the form: XxYxZ
 -> if differents sizes, write number of patiens (and percantages) having each voxel
- Intensity:
 -> Minimum
 -> Maximum
 -> Mean
 -> Standard deviation
- Statistics 
- Warnings
- Possible preprocessing recommendations

Format:
- Medical format
- Use table if necessary (you must makes them understandable for a human)
- For the big values, always use "." and not ","
- Not write words like "x_y" form except dataset name or patients ID
- Specify the units if necessary 
- If they are percentages, you must make sure that the total percentage for each dada is 100% and take up to 3 digits after the decimal point for each percentage


