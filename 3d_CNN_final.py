import os
import nibabel as nib
import numpy as np
import pandas as pd
from collections import Counter

from skimage.transform import resize
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ------------------------------- Helper Functions ------------------------------- #

def load_tnm_stage_labels(excel_path: str) -> pd.DataFrame:
    """
    Load TNM stage labels from an Excel file.

    Args:
        excel_path (str): Path to the Excel file containing TNM stage data.
          The file is expected to have a 'PatientID' column and a 'TNMStage' column.

    Returns:
        pd.DataFrame: DataFrame containing patient IDs and their corresponding TNM stages.
    """
    return pd.read_excel(excel_path)


def collect_image_data_and_labels(
    base_dir: str,
    tnm_df: pd.DataFrame,
    output_size_3d: tuple = (64, 64, 64)
) -> tuple:
    """
    Traverse a directory of patient folders, load 3D medical images, resize them,
    normalize them, and collect their TNM labels.

    This function expects each patient to have a corresponding row in the TNM
    DataFrame. Folders are skipped if they have no matching entry or if their images
    are not in NIfTI (.nii.gz) format with the expected 3D shape.

    Args:
        base_dir (str): Path to the base directory containing patient subfolders.
        tnm_df (pd.DataFrame): DataFrame containing 'PatientID' and 'TNMStage' columns.
        output_size_3d (tuple, optional): The desired 3D shape for resizing. 
          Defaults to (64, 64, 64).

    Returns:
        tuple: A tuple of (image_data, labels, patient_ids).
          - image_data (np.ndarray): 3D volumes of shape (num_samples, *output_size_3d).
          - labels (np.ndarray): Array of TNM stage labels (one per sample).
          - patient_ids (list[str]): List of patient IDs corresponding to the volumes.
    """
    patient_ids = []
    image_data = []
    labels = []

    for patient_id in os.listdir(base_dir):
        patient_dir = os.path.join(base_dir, patient_id)

        # Skip if it's not a directory
        if not os.path.isdir(patient_dir):
            continue

        # Check if the patient has a corresponding TNM stage
        tnm_stage_row = tnm_df.loc[tnm_df['PatientID'] == patient_id]
        if tnm_stage_row.empty:
            print(f"Skipping patient {patient_id}: No corresponding TNM stage found.")
            continue

        tnm_stage = tnm_stage_row['TNMStage'].values[0]

        # Traverse subdirectories to locate .nii.gz files
        for subfolder in os.listdir(patient_dir):
            if 'MR' in subfolder or 'CT' in subfolder:  # restrict to MR or CT folders
                modality_dir = os.path.join(patient_dir, subfolder)
                if not os.path.isdir(modality_dir):
                    continue

                for scan_type in os.listdir(modality_dir):
                    scan_dir = os.path.join(modality_dir, scan_type)
                    if os.path.isdir(scan_dir):
                        for file in os.listdir(scan_dir):
                            if file.endswith('.nii.gz'):
                                # Load NIfTI image
                                img_path = os.path.join(scan_dir, file)
                                img = nib.load(img_path).get_fdata()

                                # Check if it's 3D
                                if img.ndim != 3:
                                    print(f"Skipping image with unexpected dimensions: {img.shape}")
                                    continue

                                # Resize
                                img_resized = resize(img, output_size_3d, anti_aliasing=True)

                                # Volume-wise standardization
                                img_mean = np.mean(img_resized)
                                img_std = np.std(img_resized) + 1e-8
                                img_normalized = (img_resized - img_mean) / img_std

                                image_data.append(img_normalized)
                                labels.append(tnm_stage)
                                patient_ids.append(patient_id)

    image_data = np.array(image_data)
    labels = np.array(labels)
    return image_data, labels, patient_ids


def build_3d_cnn(input_shape=(64, 64, 64, 1), num_classes=5):
    """
    Build a 3D CNN model for multiclass classification of TNM stages.

    Args:
        input_shape (tuple, optional): Shape of the 3D input data, including channels.
          Defaults to (64, 64, 64, 1).
        num_classes (int, optional): Number of output classes (e.g., TNM stages). 
          Defaults to 5.

    Returns:
        tensorflow.keras.models.Sequential: A compiled Keras Sequential model.
    """
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Conv3D(128, kernel_size=(3, 3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    return model


def patient_stratified_split(X, y, patient_ids, test_size=0.4, seed=None):
    """
    Perform a patient-level stratified train-test split.

    Ensures:
      1. All volumes from a single patient are in either train or test set.
      2. Class distribution is maintained across splits.
      3. Representative sampling of each class.

    Args:
        X (np.ndarray): Feature array of shape (n_samples, ...). 
            (Not directly used in splitting, but kept for interface consistency.)
        y (np.ndarray): Label array of shape (n_samples,).
        patient_ids (list[str]): Patient IDs (one per sample).
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
        p_mask = (patient_ids == patient)
        majority_label = Counter(y[p_mask]).most_common(1)[0][0]
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


def train_and_evaluate_3d_cnn_patient_cv(
    image_data: np.ndarray,
    labels: np.ndarray,
    patient_ids: list,
    n_classes: int = 5,
    n_splits: int = 5,
    epochs: int = 20,
    batch_size: int = 16
):
    """
    Train and evaluate a 3D CNN using patient-level cross-validation.

    For each fold:
      - Perform patient-level stratified split.
      - Prepare data (channel dimension, one-hot encoding).
      - Build and compile a fresh 3D CNN model.
      - Fit the model, evaluate on the test set, and record accuracy.

    Args:
        image_data (np.ndarray): Array of shape (n_samples, D, H, W).
        labels (np.ndarray): Array of shape (n_samples,).
        patient_ids (list[str]): Patient IDs corresponding to each sample.
        n_classes (int, optional): Number of output classes. Defaults to 5.
        n_splits (int, optional): Number of cross-validation folds. Defaults to 5.
        epochs (int, optional): Training epochs per fold. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 16.

    Returns:
        None: Prints fold-by-fold metrics and final average accuracy.
    """
    accuracies = []
    base_seed = 42

    for fold in range(n_splits):
        fold_seed = base_seed + fold
        
        # Patient-level split
        train_idx, test_idx = patient_stratified_split(
            image_data, labels, patient_ids, 
            test_size=0.4, 
            seed=fold_seed
        )
        X_train, X_test = image_data[train_idx], image_data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Check for unseen classes in the test set
        train_classes = set(np.unique(y_train))
        test_classes = set(np.unique(y_test))
        if not test_classes.issubset(train_classes):
            print(f"Skipping fold {fold+1}: Test set has unseen classes.")
            continue

        # Add channel dimension
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # One-hot encode based on actual classes present
        # If your data always has labels from 0 to n_classes-1, you can do direct to_categorical
        # Otherwise, you might map them to consecutive integers first.
        y_train_ohe = to_categorical(y_train, num_classes=n_classes)
        y_test_ohe = to_categorical(y_test, num_classes=n_classes)

        # Build the 3D CNN
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
        model = build_3d_cnn(input_shape=input_shape, num_classes=n_classes)

        # Train
        model.fit(
            X_train, y_train_ohe,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test_ohe),
            verbose=1
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test_ohe, verbose=0)
        accuracies.append(test_acc)

        # Predictions and classification report
        y_pred = np.argmax(model.predict(X_test), axis=-1)
        print(f"\nFold {fold+1} Test Accuracy: {test_acc:.4f}")
        
        y_test_classes = y_test  # already integer-labeled
        print("Classification Report:")
        print(classification_report(y_test_classes, y_pred))

    if accuracies:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print("\nCross-Validation Summary:")
        print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        print("No valid folds were evaluated.")


# ------------------------------- Main Execution ------------------------------- #

def main():
    """
    Main function to orchestrate the data loading, patient-level cross-validation,
    and reporting for a 3D CNN-based TNM stage classification system.
    """
    # 1. Load TNM stage labels
    tnm_df = load_tnm_stage_labels("TNM.xlsx")

    # 2. Collect images and labels
    base_dir = "niig_unprocessed"
    image_data, labels, patient_ids = collect_image_data_and_labels(
        base_dir, tnm_df, output_size_3d=(64, 64, 64)
    )

    # 3. Remove NaN labels
    valid_indices = ~np.isnan(labels)
    image_data = image_data[valid_indices]
    labels = labels[valid_indices]
    patient_ids = np.array(patient_ids)[valid_indices]

    # Determine number of unique classes
    n_classes = len(np.unique(labels))

    # 4. Perform patient-level cross-validation with the 3D CNN
    train_and_evaluate_3d_cnn_patient_cv(
        image_data=image_data,
        labels=labels,
        patient_ids=patient_ids,
        n_classes=n_classes,
        n_splits=5,       # Number of folds
        epochs=20,        # Training epochs
        batch_size=16     # Batch size
    )


if __name__ == "__main__":
    main()
