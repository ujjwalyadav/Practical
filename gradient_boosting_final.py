import os
import numpy as np
import pandas as pd
import nibabel as nib
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from collections import Counter

def load_tnm_stage_labels(excel_path: str) -> pd.DataFrame:
    """
    Load TNM stage labels from an Excel file.

    Args:
        excel_path (str): Path to the Excel file containing TNM data. 
            The file should have 'PatientID' and 'TNMStage' columns.

    Returns:
        pd.DataFrame: DataFrame with the columns 'PatientID' and 'TNMStage'.
    """
    return pd.read_excel(excel_path)


def gather_middle_slice_data(
    base_dir: str, 
    tnm_df: pd.DataFrame, 
    output_size_2d: tuple = (128, 128)
) -> tuple:
    """
    Traverse patient directories and extract the middle slice of 3D images 
    (or the entire 2D image if it's already 2D). Resizes to output_size_2d 
    and normalizes each slice before flattening.

    Args:
        base_dir (str): Path to the base directory containing patient subfolders.
        tnm_df (pd.DataFrame): DataFrame containing 'PatientID' and 'TNMStage' columns.
        output_size_2d (tuple, optional): Desired (height, width) for the 
            2D images. Defaults to (128, 128).

    Returns:
        tuple: (image_data, labels, patient_ids)
            image_data (np.ndarray): Flattened 2D slices, shape (num_samples, height*width).
            labels (np.ndarray): Array of TNM stage labels, shape (num_samples,).
            patient_ids (list of str): Patient IDs corresponding to the slices.
    """
    patient_ids = []
    image_data = []
    labels = []

    for patient_id in os.listdir(base_dir):
        patient_dir = os.path.join(base_dir, patient_id)

        # Skip if not a directory
        if not os.path.isdir(patient_dir):
            continue

        # Find row matching patient_id
        tnm_stage_row = tnm_df.loc[tnm_df['PatientID'] == patient_id]
        if tnm_stage_row.empty:
            print(f"Skipping patient {patient_id}: No corresponding TNM stage found.")
            continue

        tnm_stage = tnm_stage_row['TNMStage'].values[0]

        # Traverse subdirectories to find .nii.gz files
        for subfolder in os.listdir(patient_dir):
            if 'MR' in subfolder or 'CT' in subfolder:  # We only look in MR/CT subfolders
                modality_dir = os.path.join(patient_dir, subfolder)
                if not os.path.isdir(modality_dir):
                    continue

                for scan_type in os.listdir(modality_dir):
                    scan_dir = os.path.join(modality_dir, scan_type)
                    if os.path.isdir(scan_dir):
                        for file in os.listdir(scan_dir):
                            if file.endswith('.nii.gz'):
                                img_path = os.path.join(scan_dir, file)
                                # Load the NIfTI image
                                img = nib.load(img_path).get_fdata()

                                if img.ndim == 3:
                                    # Take the middle slice for 3D volumes
                                    middle_slice_idx = img.shape[2] // 2
                                    img_slice = img[:, :, middle_slice_idx]
                                elif img.ndim == 2:
                                    # Already a 2D image
                                    img_slice = img
                                else:
                                    print(f"Skipping image with unexpected dimensions: {img.shape}")
                                    continue

                                # Check if the slice can be resized (avoid upsampling tiny images)
                                if (img_slice.shape[0] < output_size_2d[0] or
                                        img_slice.shape[1] < output_size_2d[1]):
                                    print(f"Skipping resizing for image with shape {img_slice.shape} "
                                          f"because it's smaller than target size {output_size_2d}.")
                                    continue

                                # Resize and normalize the image
                                img_resized = resize(img_slice, output_size_2d, anti_aliasing=True)
                                img_normalized = (
                                    (img_resized - np.mean(img_resized)) /
                                    (np.std(img_resized) + 1e-8)
                                )

                                image_data.append(img_normalized.flatten())  # Flatten to 1D
                                labels.append(tnm_stage)
                                patient_ids.append(patient_id)

    return np.array(image_data), np.array(labels), patient_ids


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
        patient_ids (list[str]): Identifiers (IDs) for each sample.
        test_size (float, optional): Fraction of data to use for the test set. Defaults to 0.4.
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


def train_and_evaluate_xgboost_patient_cv(
    X: np.ndarray, 
    y: np.ndarray, 
    patient_ids: list, 
    n_splits: int = 5,
    n_components: int = 90
):
    """
    Train and evaluate an XGBoost classifier using patient-level cross-validation.

    For each fold:
      1. Patient-level stratified train/test split.
      2. Standardization on X_train, then transform X_test.
      3. Optional PCA (controlled by n_components).
      4. Train an XGBoost model, predict, and evaluate.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Labels, shape (n_samples,).
        patient_ids (list[str]): Patient IDs for each sample.
        n_splits (int): Number of cross-validation folds. Defaults to 5.
        n_components (int): Number of principal components to keep. Defaults to 90.
            If n_components >= the number of features, PCA will be skipped.

    Returns:
        None: Prints fold-specific and aggregated results.
    """
    accuracies = []
    # We can keep the random seed for reproducibility
    base_seed = 42

    for fold in range(n_splits):
        fold_seed = base_seed + fold
        train_idx, test_idx = patient_stratified_split(X, y, patient_ids, 
                                                       test_size=0.4, seed=fold_seed)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Check if test set has classes not seen in train set
        train_classes = set(np.unique(y_train))
        test_classes = set(np.unique(y_test))
        if not test_classes.issubset(train_classes):
            print(f"Skipping fold {fold+1}: Test set has unseen classes: {test_classes - train_classes}")
            continue

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # (Optional) PCA
        if n_components and n_components < X_train_scaled.shape[1]:
            pca = PCA(n_components=n_components, random_state=fold_seed)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)

        # Train XGBoost
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=fold_seed
        )
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"\nFold {fold+1} Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    if accuracies:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print("\nCross-Validation Summary:")
        print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        print("No valid folds were evaluated.")


def main():
    """
    Main function to:
    1. Load TNM labels from Excel.
    2. Gather 2D slices (middle slice if 3D).
    3. Remove samples with NaN labels.
    4. Perform patient-level cross-validation with XGBoost (including standardization and PCA).
    """
    # 1. Load TNM stage labels
    tnm_df = load_tnm_stage_labels("TNM.xlsx")

    # 2. Collect 2D slice data
    base_dir = "niig_unprocessed"
    image_data, labels, patient_ids = gather_middle_slice_data(
        base_dir, 
        tnm_df, 
        output_size_2d=(128, 128)
    )

    # 3. Remove samples with NaN labels
    valid_indices = ~np.isnan(labels)
    image_data = image_data[valid_indices]
    labels = labels[valid_indices]
    patient_ids = np.array(patient_ids)[valid_indices]

    # 4. Patient-level cross-validation with XGBoost
    print("Starting patient-level cross-validation for XGBoost...")
    train_and_evaluate_xgboost_patient_cv(
        X=image_data, 
        y=labels, 
        patient_ids=patient_ids, 
        n_splits=5, 
        n_components=90  # Adjust or set to None if you want to skip PCA
    )


if __name__ == "__main__":
    main()
