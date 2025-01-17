"""
Multi-task Feature Selection and Model Training with Segment Anything Model (SAM)

This script demonstrates a pipeline that:
1. Loads image metadata from a CSV file.
2. Uses a Segment Anything Model (SAM) to extract deep features from medical images.
3. Applies optional PCA for dimensionality reduction.
4. Employs a MultiTaskFeatureSelector to select features (using mutual information and XGBoost).
5. Performs patient-level stratified splitting for cross-validation.
6. Trains and evaluates an XGBoost classifier.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from collections import Counter

from segment_anything import sam_model_registry
import nibabel as nib
from sklearn.decomposition import PCA


class MultiTaskFeatureSelector:
    """
    MultiTaskFeatureSelector

    This class implements an ensemble-based feature selection approach 
    that combines mutual information with XGBoost feature importances.

    Attributes:
        n_features_to_select (int): Number of top features to select.
        n_estimators (int): Number of estimators for the XGBoost classifier.
    """

    def __init__(self, n_features_to_select=50, n_estimators=100):
        """
        Initialize the MultiTaskFeatureSelector with given parameters.

        Args:
            n_features_to_select (int, optional): 
                Number of top features to select. Defaults to 50.
            n_estimators (int, optional): 
                Number of XGBoost estimators. Defaults to 100.
        """
        self.n_features_to_select = n_features_to_select
        self.n_estimators = n_estimators
        
    def mutual_information_selection(self, X, y, fold_seed=42):
        """
        Select features based on mutual information, with a fold-specific random seed.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Integer-encoded labels, shape (n_samples,).
            fold_seed (int, optional): Random seed for mutual_info_classif. 
                Defaults to 42.

        Returns:
            np.ndarray: Indices of the top `n_features_to_select` features 
                based on mutual information.
        """
        mi_scores = mutual_info_classif(X, y, random_state=fold_seed)
        top_feature_indices = np.argsort(mi_scores)[-self.n_features_to_select:]
        return top_feature_indices
    
    def xgboost_importance(self, X, y):
        """
        Select features using XGBoost importance for multi-class data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Integer-encoded labels, shape (n_samples,).

        Returns:
            np.ndarray: Indices of the top `n_features_to_select` features 
                based on XGBoost feature importances.
        """
        # Ensure y is converted to integer type 
        y = np.array(y).astype(int)
        
        # Ensure consecutive integer labels starting from 0
        unique_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_mapped = np.array([label_map[label] for label in y])
        
        clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators, 
            learning_rate=0.1, 
            objective='multi:softprob',
            num_class=len(unique_labels)
        )
        clf.fit(X, y_mapped)
        feature_importance = clf.feature_importances_
        top_feature_indices = np.argsort(feature_importance)[-self.n_features_to_select:]
        return top_feature_indices
    
    def ensemble_selection(self, X, y, fold_seed=42):
        """
        Combine mutual information and XGBoost importance for feature selection.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Integer-encoded labels, shape (n_samples,).
            fold_seed (int, optional): Random seed used for consistent 
                feature selection. Defaults to 42.

        Returns:
            np.ndarray: Indices of the top `n_features_to_select` features 
                chosen by the ensemble of mutual information and XGBoost.
        """
        # Ensure y is converted to integer type 
        y = np.array(y).astype(int)
        
        # Ensure consecutive integer labels starting from 0
        unique_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_mapped = np.array([label_map[label] for label in y])
        
        # Combine multiple feature selection techniques
        mi_features = self.mutual_information_selection(X, y_mapped, fold_seed=fold_seed)
        xgb_features = self.xgboost_importance(X, y_mapped)
        
        # Combine and get unique top features
        combined_features = np.unique(np.concatenate([mi_features, xgb_features]))
        
        # If more features than needed, re-rank them using mutual information
        if len(combined_features) > self.n_features_to_select:
            mi_scores = mutual_info_classif(X[:, combined_features], y_mapped, random_state=fold_seed)
            top_feature_indices = combined_features[np.argsort(mi_scores)[-self.n_features_to_select:]]
        else:
            top_feature_indices = combined_features
        
        return top_feature_indices


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
        y (np.ndarray): Integer-encoded labels, shape (n_samples,).
        patient_ids (list[str] or np.ndarray): 
            Identifiers (IDs) for each sample. 
        test_size (float, optional): Fraction of data to use for the test set. 
            Defaults to 0.4 (i.e., 60% train, 40% test).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple:
            train_indices (np.ndarray): Indices of training samples.
            test_indices (np.ndarray): Indices of test samples.
    """
    # Convert to numpy arrays if not already
    patient_ids = np.array(patient_ids)
    y = np.array(y)
    
    # Get unique patients
    unique_patients = np.unique(patient_ids)
    
    # Build a mapping from patient -> majority class (simplistic approach)
    from collections import Counter
    patient_class_dict = {}
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_class = Counter(y[patient_mask]).most_common(1)[0][0]
        if patient_class not in patient_class_dict:
            patient_class_dict[patient_class] = []
        patient_class_dict[patient_class].append(patient)
    
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


def extract_sam_features(image_path, sam_model, device):
    """
    Extract feature embeddings from a medical image using the 
    Segment Anything Model (SAM).

    Args:
        image_path (str): Path to a NIfTI (.nii, .nii.gz) image.
        sam_model (nn.Module): Preloaded SAM model.
        device (torch.device): Device to run computations on 
            (e.g., 'cuda' or 'cpu').

    Returns:
        np.ndarray or None:
            - Mean-pooled SAM embeddings of shape (C,) if successful.
            - None if an error occurs or no valid slices are processed.
    """
    try:
        img_nifti = nib.load(image_path)
        image_array = img_nifti.get_fdata()
        
        # Normalize the image to zero-mean, unit-variance
        image_array = (image_array - image_array.mean()) / (image_array.std() + 1e-7)
        
        num_slices = image_array.shape[2]
        slice_features = []
        
        for i in range(num_slices):
            slice_2d = image_array[:, :, i]
            
            # Skip problematic slices (NaN/Inf)
            if np.any(np.isnan(slice_2d)) or np.any(np.isinf(slice_2d)):
                continue
            
            # Convert slice to a 4D tensor (batch=1, channels=1, height, width)
            slice_2d_tensor = torch.tensor(slice_2d).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # Resize slice to (1024, 1024) using bilinear interpolation
            resized_slice = torch.nn.functional.interpolate(
                slice_2d_tensor, 
                size=(1024, 1024), 
                mode="bilinear",
                align_corners=True
            )
            
            # Duplicate 1-channel to 3-channels for the SAM image encoder
            with torch.no_grad():
                features = sam_model.image_encoder(resized_slice.repeat(1, 3, 1, 1))
            
            # Move back to CPU, detach from computational graph
            slice_features.append(features.cpu().numpy().squeeze())
        
        # Return the mean embedding across slices if there are any valid slices
        return np.mean(slice_features, axis=0) if slice_features else None
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def train_and_evaluate(X, y, patient_ids, n_splits=5):
    """
    Train and evaluate an XGBoost model using patient-level cross-validation.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Integer-encoded class labels, shape (n_samples,).
        patient_ids (np.ndarray or list[str]): 
            Patient IDs (one per sample). 
        n_splits (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        dict: A dictionary containing:
            'accuracies' (list[float]): Accuracy scores for each fold.
            'balanced_accuracies' (list[float]): Balanced accuracy scores for each fold.
            'aucs' (list[float]): Area under ROC curve (one-vs-rest) for each fold.
    """
    # Ensure consecutive integer labels starting from 0
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_mapped = np.array([label_map[label] for label in y])

    accuracies = []
    balanced_accuracies = []
    aucs = []
    
    for fold in range(n_splits):
        fold_seed = 42 + fold
        np.random.seed(fold_seed)

        # Patient-level train-test split
        train_index, val_index = patient_stratified_split(
            X, y_mapped, patient_ids, test_size=0.4, seed=fold_seed
        )
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y_mapped[train_index], y_mapped[val_index]
        
        # Ensure validation classes exist in the training set
        train_classes = set(np.unique(y_train))
        val_classes = set(np.unique(y_val))
        if not val_classes.issubset(train_classes):
            print(f"Skipping fold {fold+1}: Validation set contains classes not in training set.")
            continue
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Multi-task feature selection
        feature_selector = MultiTaskFeatureSelector(n_features_to_select=50)
        selected_feature_indices = feature_selector.ensemble_selection(
            X_train_scaled, y_train, fold_seed=fold_seed
        )
        
        X_train_selected = X_train_scaled[:, selected_feature_indices]
        X_val_selected = X_val_scaled[:, selected_feature_indices]
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(unique_labels),
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=fold_seed
        )
        
        model.fit(X_train_selected, y_train)
        
        # Predictions
        y_pred = model.predict(X_val_selected)
        y_pred_proba = model.predict_proba(X_val_selected)
        
        # Metrics
        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        
        # One-hot encode for multi-class AUC
        y_val_one_hot = pd.get_dummies(y_val).values
        
        auc_score = roc_auc_score(y_val_one_hot, y_pred_proba, multi_class='ovr')
        
        accuracies.append(acc)
        balanced_accuracies.append(bal_acc)
        aucs.append(auc_score)
        
        print(f"Fold {fold+1}: Accuracy = {acc:.4f}, Balanced Accuracy = {bal_acc:.4f}, AUC = {auc_score:.4f}")
    
    # Aggregate results
    print("\nCross-Validation Results:")
    if accuracies:
        print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Mean Balanced Accuracy: {np.mean(balanced_accuracies):.4f} ± {np.std(balanced_accuracies):.4f}")
        print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    else:
        print("No valid folds to evaluate.")
    
    return {
        'accuracies': accuracies,
        'balanced_accuracies': balanced_accuracies,
        'aucs': aucs
    }


def main():
    """
    Main entry point for loading data, extracting SAM features, applying PCA, 
    and training/evaluating a multi-class XGBoost model.

    Steps:
        1. Load CSV metadata (image paths, labels, patient IDs).
        2. Load and prepare the Segment Anything Model (SAM).
        3. Extract features for each valid image slice.
        4. Align features with labels and patient IDs.
        5. Optionally apply PCA for dimensionality reduction.
        6. Perform patient-level cross-validation with XGBoost.
    """
    # Load CSV file
    csv_path = "here_we_come.csv"
    df = pd.read_csv(csv_path)

    images = df["Image"].tolist()
    labels = df["TNM"].tolist()
    patient_ids = df["Patient ID"].tolist()

    # Map labels to numeric
    label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    mapped_labels = np.array([label_mapping[label] for label in labels])

    # Print label mapping
    print("Label Mapping:", label_mapping)
    print("Class Distribution:", Counter(mapped_labels))

    # Choose computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Optimize GPU usage if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0, device=0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Load SAM model
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to(device)

    # Extract features
    sam_features = []
    valid_indices = []
    for idx, img_path in enumerate(images):
        try:
            features = extract_sam_features(img_path, sam, device)
            if features is not None:
                sam_features.append(features.flatten())
                valid_indices.append(idx)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    # Align features and labels
    sam_features = np.array(sam_features)
    labels_aligned = np.array([mapped_labels[idx] for idx in valid_indices])
    patient_ids_aligned = np.array(patient_ids)[valid_indices]

    # Report shapes
    print(f"Total features extracted: {sam_features.shape}")
    print(f"Aligned labels shape: {labels_aligned.shape}")
    print(f"Aligned patient IDs shape: {patient_ids_aligned.shape}")

    # Dimensionality Reduction (PCA)
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=300)  # Adjust as needed
    sam_features_reduced = pca.fit_transform(sam_features)
    print(f"Reduced feature dimensions: {sam_features_reduced.shape}")

    # Train and evaluate
    results = train_and_evaluate(sam_features_reduced, labels_aligned, patient_ids_aligned)


if __name__ == "__main__":
    main()
