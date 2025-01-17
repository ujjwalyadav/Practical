
---

# Medical Imaging Pipelines for TNM Stage Classification

This repository provides multiple **Python** scripts for processing medical images (2D and 3D) and classifying them into TNM stages using different machine-learning and deep-learning approaches. You will find scripts employing:

1. **Segment Anything Model (SAM) with XGBoost** for feature extraction and classification.
2. **2D Medical Image Classification with SVM**  
3. **3D Volume Processing with Random Forest**  
4. **2D Slice Classification with XGBoost**  
5. **3D CNN Pipeline**  

Each approach demonstrates a distinct workflow for medical image preprocessing, feature extraction, dimensionality reduction, and classification (or deep-learning training). Most scripts also implement **patient-level cross-validation** to avoid data leakage and to ensure realistic performance estimates.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Requirements](#requirements)
3. [Data Preparation](#data-preparation)
4. [Usage](#usage)
   - [Segment Anything Model (SAM) + XGBoost Pipeline](#segment-anything-model-sam--xgboost-pipeline)
   - [2D Medical Image Classification with SVM](#2d-medical-image-classification-with-svm)
   - [3D Volume Processing with Random Forest](#3d-volume-processing-with-random-forest)
   - [2D Slice Classification with XGBoost](#2d-slice-classification-with-xgboost)
   - [3D CNN Pipeline](#3d-cnn-pipeline)
5. [License](#license)
6. [Acknowledgments](#acknowledgments)

---

## Directory Structure

A typical directory structure might look like this (you can adapt it to your own project layout):

```
.
├── data/
│   ├── TNM.xlsx          # Excel file containing patient IDs and TNM stages
│   ├── here_we_come.csv  # Containing metadata image paths, TNM labels, and patient IDs.  
│   └── niig_unprocessed/ # Example folder containing patient subdirectories
│       ├── patientA/
│       │   └── MR/CT subfolders ...
│       ├── patientB/
│       │   └── ...
│       └── ...
├── segment_anything_final.py        # Script demonstrating SAM feature extraction + XGBoost
├── svm_classifier_final.py             # 2D SVM classification pipeline
├── random_forest_final.py   # 3D volume processing + Random Forest classification
├── gradient_boost_final.py         # 2D slice classification with XGBoost
├── 3d_cnn_final.py             # 3D CNN pipeline for TNM stage classification
├── requirements.txt      # List of Python dependencies
└── README.md
```

**Note**: The directory names and script names may differ based on how you organize your repository.

---

## Requirements

All scripts assume **Python 3.7+** (or higher). Below is a minimal set of dependencies used across the pipelines:

- **NumPy** (for array operations)
- **Pandas** (for dataframes and CSV/Excel handling)
- **scikit-learn** (for machine-learning utilities, splitting, feature selection)
- **scikit-image** (for resizing and basic image operations)
- **nibabel** (for reading NIfTI medical image files)
- **tqdm** (optional, for progress bars in some scripts)
- **xgboost** (XGBoost classification in two of the scripts)
- **PyTorch** + **segment-anything** (only for the SAM-based pipeline; PyTorch needed for GPU usage of SAM)
- **TensorFlow/Keras** (for the 3D CNN pipeline)
- **OpenCV** (optional, if you incorporate additional image I/O, not strictly required by all scripts)

GPU usage is highly recommended for:
- **Segment Anything Model (SAM)** extraction (PyTorch + CUDA-enabled device).
- **3D CNN training** (TensorFlow/Keras with GPU support).

You can manage these dependencies via **conda**, **pip** + **virtualenv**, or any other environment manager. For example:

```bash
pip install -r requirements.txt
```

or : 

```bash
pip install numpy pandas scikit-learn scikit-image nibabel xgboost torch torchvision torchaudio tensorflow
```

(Adjust package versions as needed for your environment.)

---

## Data Preparation

1. **TNM Stage File**:  
   Ensure you have an **Excel file** named `TNM.xlsx` (or a CSV) containing columns like:
   - `PatientID`
   - `TNMStage`

   Each row represents a patient and their stage.

2. **Images**:  
   Place your 2D or 3D **NIfTI** (`.nii` / `.nii.gz`) files under subfolders for each patient. The scripts typically scan subdirectories named **MR** or **CT**, looking for `.nii.gz`.

3. **CSV Metadata** (SAM-based script only):  
   A sample CSV file (`here_we_come.csv`) with columns such as:
   - `Image` (full path to the .nii file)
   - `TNM` (the stage label)
   - `Patient ID` (unique patient identifier)

4. **Verify Dimensions**:  
   - 2D pipelines expect 2D slices or single slices extracted from 3D volumes.
   - 3D pipelines expect 3D volumes (e.g., `(height, width, depth)`).

---

## Usage

Below is a high-level guide for each script. **Important**: Adjust file paths (Excel/CSV, base directory) and parameter settings (like output image sizes, PCA components, number of folds, etc.) inside each script before running.

### Segment Anything Model (SAM) + XGBoost Pipeline

- **Script Name**: `segment_anything_final.py`
- **Purpose**:  
  1. Reads metadata from a CSV (e.g., `here_we_come.csv`) containing image paths, TNM labels, and patient IDs.  
  2. Loads the Segment Anything Model (SAM) for feature extraction.  
  3. Optionally applies PCA for dimensionality reduction.  
  4. Performs multi-task feature selection (mutual information + XGBoost feature importance).  
  5. Trains and evaluates XGBoost with patient-level cross-validation.

- **Run**:
  ```bash
  python segment_anything_final.py
  ```
  Make sure your GPU is available and that you have the correct **segment-anything** and **torch** versions installed.

---

### 2D Medical Image Classification with SVM

- **Script Name**: `svm_classifier_final.py`
- **Purpose**:  
  1. Loads 2D slices (or extracts the middle slice from 3D images).  
  2. Resizes/normalizes each slice.  
  3. Performs patient-level cross-validation with an SVM classifier.

- **Run**:
  ```bash
  python svm_classifier_final.py
  ```

---

### 3D Volume Processing with Random Forest

- **Script Name**: `random_forest_final.py`
- **Purpose**:  
  1. Collects 3D volumes from patient subdirectories.  
  2. Resamples them to a target shape (e.g., `[Depth, Height, Width]`).  
  3. Optionally applies PCA.  
  4. Performs patient-level cross-validation using a **Random Forest** classifier.

- **Run**:
  ```bash
  python random_forest_final.py
  ```

---

### 2D Slice Classification with XGBoost

- **Script Name**: `gradient_boost_final.py`
- **Purpose**:  
  1. Extracts 2D slices (middle slice if 3D, or directly 2D).  
  2. Optionally applies PCA for dimensionality reduction.  
  3. Uses **patient-level cross-validation** to train an XGBoost classifier.

- **Run**:
  ```bash
  python gradient_boost_final.py
  ```

---

### 3D CNN Pipeline

- **Script Name**: `3d_cnn_final.py`
- **Purpose**:  
  1. Loads 3D volumes from patient directories and resizes them to a uniform shape (64×64×64 by default).  
  2. Implements a **3D CNN** in TensorFlow/Keras.  
  3. Uses patient-level cross-validation to train and evaluate the 3D CNN.

- **Run**:
  ```bash
  python 3d_cnn_final.py
  ```
  **Note**: This script requires a working **GPU** with compatible CUDA drivers for timely training.

---

## License

> (c) 2025 Ujjwal Yadav

---

## Acknowledgments

- **scikit-learn**, **Nibabel**, **scikit-image**, **xgboost**, **PyTorch**, **segment-anything**, **TensorFlow**, etc. for making these pipelines possible.

