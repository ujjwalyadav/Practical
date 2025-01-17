import os
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

def load_tnm_stages(excel_path: str) -> pd.DataFrame:
    """
    Load the TNM stage labels from an Excel file.

    Args:
        excel_path (str): Path to the Excel file containing TNM data with 
            'PatientID' and 'TNMStage' columns.

    Returns:
        pd.DataFrame: A DataFrame with the loaded TNM stage data.
    """
    return pd.read_excel(excel_path)

def collect_and_preprocess_images(
    base_dir: str,
    tnm_df: pd.DataFrame,
    output_size_2d: tuple = (128, 128)
) -> tuple:
    """
    Collect medical images from a directory structure, extract middle slices for 3D volumes,
    resize them to output_size_2d, normalize, and flatten them into feature vectors.

    Args:
        base_dir (str): Directory path containing patient subfolders with image data.
        tnm_df (pd.DataFrame): DataFrame containing columns 'PatientID' and 'TNMStage'.
        output_size_2d (tuple, optional): The target (height, width) for 
            2D slices. Defaults to (128, 128).

    Returns:
        tuple: A 3-tuple:
            - image_data (np.ndarray): Array of shape (num_samples, height*width).
            - labels (np.ndarray): TNM stage labels, shape (num_samples,).
            - patient_ids (list): Patient IDs corresponding to each sample.
    """
    patient_ids = []
    image_data = []
    labels = []

    # Iterate over patient directories
    for patient_id in os.listdir(base_dir):
        patient_dir = os.path.join(base_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        # Get the corresponding TNM stage from the DataFrame
        tnm_stage_row = tnm_df.loc[tnm_df['PatientID'] == patient_id]
        if tnm_stage_row.empty:
            print(f"Skipping patient {patient_id}: No corresponding TNM stage found.")
            continue
        
        tnm_stage = tnm_stage_row['TNMStage'].values[0]
        
        # Traverse the directory to find .nii.gz files
        for subfolder in os.listdir(patient_dir):
            if 'MR' in subfolder or 'CT' in subfolder:
                modality_dir = os.path.join(patient_dir, subfolder)
                
                if not os.path.isdir(modality_dir):
                    continue
                
                for scan_type in os.listdir(modality_dir):
                    scan_dir = os.path.join(modality_dir, scan_type)
                    if os.path.isdir(scan_dir):
                        for file in os.listdir(scan_dir):
                            if file.endswith('.nii.gz'):
                                # Load the NIfTI image
                                img_path = os.path.join(scan_dir, file)
                                img = nib.load(img_path).get_fdata()
                                
                                # Check dimensionality (2D or 3D)
                                if img.ndim == 3:
                                    middle_slice = img.shape[2] // 2
                                    img_slice = img[:, :, middle_slice]
                                elif img.ndim == 2:
                                    img_slice = img
                                else:
                                    print(f"Skipping image with unexpected dimensions: {img.shape}")
                                    continue
                                
                                # Check shape before resizing
                                if (img_slice.shape[0] < output_size_2d[0] or 
                                        img_slice.shape[1] < output_size_2d[1]):
                                    print(f"Skipping resizing for image with shape {img_slice.shape}: dimensions smaller than target size.")
                                    continue
                                
                                # Resize and normalize
                                img_resized = resize(img_slice, output_size_2d, anti_aliasing=True)
                                img_normalized = (img_resized - np.mean(img_resized)) / (np.std(img_resized) + 1e-8)
                                
                                # Flatten to 1D feature vector
                                image_data.append(img_normalized.flatten())
                                labels.append(tnm_stage)
                                patient_ids.append(patient_id)
    
    return np.array(image_data), np.array(labels), patient_ids

def remove_nan_labels(image_data: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Remove samples whose labels are NaN.

    Args:
        image_data (np.ndarray): Feature matrix of shape (num_samples, features).
        labels (np.ndarray): Label array of shape (num_samples,).

    Returns:
        tuple: (cleaned_image_data, cleaned_labels)
    """
    valid_indices = ~np.isnan(labels)
    cleaned_image_data = image_data[valid_indices]
    cleaned_labels = labels[valid_indices]
    return cleaned_image_data, cleaned_labels

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
    # Convert to numpy arrays
    patient_ids = np.array(patient_ids)
    y = np.array(y)
    
    # Get unique patients
    unique_patients = np.unique(patient_ids)
    
    # Map each patient to its majority class label (for stratification)
    from collections import Counter
    patient_class_dict = {}
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_label = Counter(y[patient_mask]).most_common(1)[0][0]
        if patient_label not in patient_class_dict:
            patient_class_dict[patient_label] = []
        patient_class_dict[patient_label].append(patient)
    
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

def train_svm_classifier(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    C: float = 1.0, 
    gamma: str = 'auto', 
    kernel: str = 'rbf',
    random_state: int = 42
) -> SVC:
    """
    Train an SVM classifier with specified parameters.

    Args:
        X_train (np.ndarray): Training data, shape (n_samples, n_features).
        y_train (np.ndarray): Training labels, shape (n_samples,).
        C (float, optional): Regularization parameter. Defaults to 1.0.
        gamma (str, optional): Kernel coefficient for certain kernels. Defaults to 'auto'.
        kernel (str, optional): Kernel type ('linear', 'poly', 'rbf', etc.). Defaults to 'rbf'.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        SVC: Trained SVM model.
    """
    clf = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

def train_and_evaluate_svm(X, y, patient_ids, n_splits=5):
    """
    Train and evaluate an SVM model using patient-level cross-validation.

    For each fold:
      - Perform patient-level stratified split.
      - Standardize the training data and transform the test data accordingly.
      - (Optional) Apply PCA on the training data and transform the test data.
      - Train an SVM on the training fold.
      - Evaluate on the test fold.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Labels, shape (n_samples,).
        patient_ids (np.ndarray or list[str]): Patient IDs (one per sample). 
        n_splits (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        dict: Contains per-fold and average accuracy scores.
    """
    accuracies = []

    for fold in range(n_splits):
        fold_seed = 42 + fold
        # Patient-level train-test split
        train_idx, test_idx = patient_stratified_split(
            X, y, patient_ids, test_size=0.4, seed=fold_seed
        )
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check if test set has classes not in train set
        train_classes = set(np.unique(y_train))
        test_classes = set(np.unique(y_test))
        if not test_classes.issubset(train_classes):
            print(f"Skipping fold {fold+1}: Test set has unknown classes.")
            continue
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # (Optional) PCA can be done here if desired:
        # pca = PCA(n_components=40)
        # X_train_scaled = pca.fit_transform(X_train_scaled)
        # X_test_scaled = pca.transform(X_test_scaled)

        # Train SVM
        clf = train_svm_classifier(
            X_train_scaled, y_train, 
            C=1.0, gamma='auto', kernel='rbf', random_state=fold_seed
        )
        
        # Predict on test set
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
        print("No valid folds were evaluated.")
    
    return {
        'accuracies': accuracies,
        'mean_accuracy': np.mean(accuracies) if accuracies else None
    }

def main():
    """
    Main execution function:
    1. Load TNM stage data.
    2. Collect, resize, and preprocess 2D image slices.
    3. Remove samples with NaN labels.
    4. Perform patient-level cross-validation to train and evaluate an SVM classifier.
    """
    # 1. Load TNM stage labels
    tnm_df = load_tnm_stages("TNM.xlsx")

    # 2. Collect and preprocess images
    base_dir = "niig_unprocessed"
    image_data, labels, patient_ids = collect_and_preprocess_images(
        base_dir, tnm_df, output_size_2d=(128, 128)
    )

    # 3. Remove samples with NaN labels
    image_data, labels = remove_nan_labels(image_data, labels)

    y_numeric = labels

    # 4. Perform patient-level cross-validation with an SVM classifier
    results = train_and_evaluate_svm(image_data, y_numeric, patient_ids, n_splits=5)

    # Final summary
    print("\nFinal Cross-Validation Results:")
    print(results)

if __name__ == "__main__":
    main()
