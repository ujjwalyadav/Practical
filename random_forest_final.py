import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm  # For progress visualization
from collections import Counter

# ------------------------------- Configuration ------------------------------- #

# Paths
TNM_STAGE_FILE = "TNM.xlsx"
BASE_DIR = "TEST"  # Directory with preprocessed images

# Preprocessing Parameters
TARGET_SPACING = (1.0, 1.0, 1.0)  # Assuming images are already at this spacing
NORMALIZATION_METHOD = "z-score"  # Assuming images are already normalized

# Model Parameters
N_SPLITS = 5  # Number of folds for patient-level cross-validation
RANDOM_STATE = 42
PCA_COMPONENTS = 50  # Set to None if PCA is not desired
N_ESTIMATORS = 100  # Number of trees in the Random Forest

# ------------------------------- Helper Functions ------------------------------- #

def load_tnm_stage(tnm_df: pd.DataFrame, patient_id: str) -> float:
    """
    Retrieve the TNM stage for a given patient ID from a TNM DataFrame.

    Args:
        tnm_df (pd.DataFrame): DataFrame containing columns 'PatientID' and 'TNMStage'.
        patient_id (str): Unique patient identifier used to look up the TNM stage.

    Returns:
        float: The TNM stage for the specified patient. Returns None if not found.
    """
    tnm_stage_row = tnm_df.loc[tnm_df['PatientID'] == patient_id, 'TNMStage']
    if tnm_stage_row.empty:
        return None
    return tnm_stage_row.values[0]


def resample_image(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resample the input 3D image to the specified target shape using linear interpolation.

    Args:
        image (np.ndarray): A 3D NumPy array representing the image volume.
        target_shape (tuple): The desired output shape (depth, height, width).

    Returns:
        np.ndarray: The resampled 3D image volume as a NumPy array.
    """
    current_shape = image.shape
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
    resampled_image = zoom(image, zoom=zoom_factors, order=1)  # Linear interpolation
    return resampled_image


def collect_image_paths(base_dir: str) -> tuple:
    """
    Collect all .nii.gz file paths from the base directory and track corresponding patient IDs.

    Args:
        base_dir (str): Path to the base directory containing subdirectories of patient data.

    Returns:
        tuple:
            image_paths (list of str): List of complete file paths for each .nii.gz file.
            patient_ids (list of str): Corresponding patient IDs for each file path.
    """
    image_paths = []
    patient_ids = []
    for patient_id in os.listdir(base_dir):
        patient_dir = os.path.join(base_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue
        for root, _, files in os.walk(patient_dir):
            for file in files:
                if file.endswith('.nii.gz'):
                    image_paths.append(os.path.join(root, file))
                    patient_ids.append(patient_id)
    return image_paths, patient_ids


def determine_target_shape(image_paths: list) -> tuple:
    """
    Determine the maximum shape across a collection of 3D images, 
    which will serve as the target shape for resampling.

    Args:
        image_paths (list of str): A list of file paths to .nii.gz images.

    Returns:
        tuple: The maximum shape (depth, height, width) found among all images.
    """
    shapes = []
    for path in tqdm(image_paths, desc="Determining target shape"):
        img = nib.load(path).get_fdata()
        shapes.append(img.shape)
    max_shape = np.max(np.array(shapes), axis=0)
    return tuple(map(int, max_shape))


def load_and_process_images(
    image_paths: list,
    patient_ids: list,
    tnm_df: pd.DataFrame,
    target_shape: tuple
) -> tuple:
    """
    Load, resample, and flatten a set of 3D images, associating each with its TNM stage.

    Args:
        image_paths (list of str): A list of file paths to .nii.gz images.
        patient_ids (list of str): Patient IDs corresponding to the image_paths list.
        tnm_df (pd.DataFrame): DataFrame containing columns 'PatientID' and 'TNMStage'.
        target_shape (tuple): Desired (depth, height, width) for resampling each image.

    Returns:
        tuple:
            images (np.ndarray): Array of flattened images, shape (num_samples, depth*height*width).
            labels (np.ndarray): Array of TNM stage labels, shape (num_samples,).
            valid_patient_ids (list of str): List of patient IDs for which valid data was found.
    """
    images = []
    labels = []
    valid_patient_ids = []

    for path, pid in tqdm(
        zip(image_paths, patient_ids),
        total=len(image_paths),
        desc="Loading and processing images"
    ):
        tnm_stage = load_tnm_stage(tnm_df, pid)
        if tnm_stage is None:
            print(f"Skipping patient {pid}: TNM stage not found.")
            continue

        img = nib.load(path).get_fdata()
        resampled_img = resample_image(img, target_shape)
        images.append(resampled_img.flatten())
        labels.append(tnm_stage)
        valid_patient_ids.append(pid)

    return np.array(images), np.array(labels), valid_patient_ids


def patient_stratified_split(X, y, patient_ids, test_size=0.4, seed=None):
    """
    Perform a patient-level stratified train-test split.

    Ensures:
      1. All scans from a single patient are in either train or test set.
      2. Class distribution is maintained across splits.
      3. Representative sampling of each class.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features). 
            (Not used in splitting, but kept for interface consistency.)
        y (np.ndarray): Labels, shape (n_samples,).
        patient_ids (list[str] or np.ndarray): 
            Identifiers (IDs) for each sample. 
        test_size (float, optional): Fraction of data to use for the test set. 
            Defaults to 0.4.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple:
            train_indices (np.ndarray): Indices of training samples.
            test_indices (np.ndarray): Indices of test samples.
    """
    patient_ids = np.array(patient_ids)
    y = np.array(y)
    
    unique_patients = np.unique(patient_ids)

    # Map each patient to its majority class label (for stratification)
    patient_class_dict = {}
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        majority_label = Counter(y[patient_mask]).most_common(1)[0][0]
        if majority_label not in patient_class_dict:
            patient_class_dict[majority_label] = []
        patient_class_dict[majority_label].append(patient)
    
    train_patients, test_patients = [], []
    for class_label, class_patients in patient_class_dict.items():
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(class_patients)
        num_test_patients = max(1, int(len(class_patients) * test_size))
        test_patients.extend(class_patients[:num_test_patients])
        train_patients.extend(class_patients[num_test_patients:])
    
    train_indices = np.where(np.isin(patient_ids, train_patients))[0]
    test_indices = np.where(np.isin(patient_ids, test_patients))[0]
    
    return train_indices, test_indices


def train_and_evaluate_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: list,
    n_splits: int = 5
) -> dict:
    """
    Train and evaluate a Random Forest classifier using patient-level cross-validation.

    For each fold:
      - Perform patient-level stratified split.
      - Standardize the training data and transform the test data accordingly.
      - (Optionally) apply PCA on the training data, then transform the test data.
      - Train a Random Forest model and evaluate on the test set.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Labels, shape (n_samples,).
        patient_ids (list[str]): Patient IDs (one per sample).
        n_splits (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        dict: {
            'accuracies': List of accuracy scores for each fold,
            'mean_accuracy': Mean accuracy across folds
        }
    """
    accuracies = []

    for fold in range(n_splits):
        fold_seed = RANDOM_STATE + fold
        train_idx, test_idx = patient_stratified_split(
            X, y, patient_ids, test_size=0.4, seed=fold_seed
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Check if the test set has classes not in the train set
        train_classes = set(np.unique(y_train))
        test_classes = set(np.unique(y_test))
        if not test_classes.issubset(train_classes):
            print(f"Skipping fold {fold+1}: Test set has unseen classes.")
            continue

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Optional PCA
        if PCA_COMPONENTS and PCA_COMPONENTS < X_train_scaled.shape[1]:
            pca = PCA(n_components=PCA_COMPONENTS, random_state=fold_seed)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)

        # Train Random Forest
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=fold_seed)
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    if accuracies:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print("\nCross-Validation Results:")
        print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        mean_acc = None
        print("No valid folds were evaluated.")

    return {
        'accuracies': accuracies,
        'mean_accuracy': mean_acc
    }

# ------------------------------- Main Execution ------------------------------- #

def main() -> None:
    """
    Main function to orchestrate the pipeline:
      1. Load TNM stage data from an Excel file.
      2. Collect all .nii.gz file paths from the specified base directory.
      3. Determine a target shape from the largest volume found.
      4. Load and resample all images.
      5. Perform patient-level cross-validation with Random Forest.
    """
    # Load TNM stage data
    print("Loading TNM stage data...")
    tnm_df = pd.read_excel(TNM_STAGE_FILE)
    print(f"Total patients in TNM data: {tnm_df['PatientID'].nunique()}")

    # Collect image paths
    print("Collecting image paths...")
    image_paths, patient_ids = collect_image_paths(BASE_DIR)
    print(f"Total images found: {len(image_paths)}")

    # Determine target shape dynamically
    print("Determining target shape for resampling...")
    target_shape = determine_target_shape(image_paths)
    print(f"Target shape set to: {target_shape}")

    # Load and process images
    print("Loading and processing images...")
    image_data, labels, valid_patient_ids = load_and_process_images(
        image_paths, patient_ids, tnm_df, target_shape
    )
    print(f"Total valid samples after processing: {len(image_data)}")

    # Check if we have sufficient data
    if len(image_data) == 0:
        print("No valid data available for training.")
        return

    # Perform patient-level cross-validation
    print("Starting patient-level cross-validation...")
    results = train_and_evaluate_random_forest(
        image_data, 
        labels, 
        valid_patient_ids, 
        n_splits=N_SPLITS
    )

    print("\nFinal Cross-Validation Results:")
    print(results)


if __name__ == "__main__":
    main()
