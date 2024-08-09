import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.signal import convolve2d as conv2
from skimage.filters import median
from skimage.morphology import square
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def func1(F, B, c, f, b):
    name = 'MEAN'
    measure = np.nanmean(f)
    return name, measure

def func2(F, B, c, f, b):
    name = 'RNG'
    measure = np.ptp(f) if len(f) > 0 else np.nan
    return name, measure

def func3(F, B, c, f, b):
    name = 'VAR'
    measure = np.nanvar(f)
    return name, measure

def func4(F, B, c, f, b):
    name = 'CV'
    mean_f = np.nanmean(f)
    measure = (np.nanstd(f) / mean_f) * 100 if mean_f != 0 else np.nan
    return name, measure

def func5(F, B, c, f, b):
    name = 'CPP'
    filt = np.array([[ -1/8, -1/8, -1/8],[-1/8, 1, -1/8],[ -1/8, -1/8,  -1/8]])
    I_hat = conv2(F, filt, mode='same')
    measure = np.nanmean(I_hat)
    return name, measure

def func6(F, B, c, f, b):
    name = 'PSNR'
    epsilon = 1e-10
    I_hat = median(F / (np.max(F) + epsilon), square(5)) if np.max(F) != 0 else np.zeros_like(F)

    measure = psnr(F, I_hat)
    return name, measure

def func7(F, B, c, f, b):
    name = 'SNR1'
    bg_std = np.nanstd(b)
    measure = np.nanstd(f) / (bg_std + 1e-9)
    return name, measure

def func8(F, B, c, f, b):
    name = 'SNR2'
    bg_std = np.nanstd(b)
    measure = np.nanmean(patch(F, 5)) / (bg_std + 1e-9)
    return name, measure

def func9(F, B, c, f, b):
    name = 'SNR3'
    fore_patch = patch(F, 5)
    std_diff = np.nanstd(fore_patch - np.nanmean(fore_patch))
    std_diff = std_diff if std_diff != 0 else 1e-9
    measure = np.nanmean(fore_patch) / std_diff
    return name, measure

def func10(F, B, c, f, b):
    name = 'SNR4'
    fore_patch = patch(F, 5)
    back_patch = patch(B, 5)
    bg_std = np.nanstd(back_patch)
    bg_std = bg_std if bg_std != 0 else 1e-9
    measure = np.nanmean(fore_patch) / bg_std
    return name, measure

from scipy.ndimage import convolve

def func11(F, B, c, f, b):
    name = 'SNR5'
    window_size = 5  # Example window size

    # Check if the input is 2D or 3D and adjust the kernel accordingly
    if F.ndim == 2:
        kernel = np.ones((window_size, window_size))
        normalization_factor = window_size**2
    elif F.ndim == 3:
        kernel = np.ones((window_size, window_size, window_size))
        normalization_factor = window_size**3
    else:
        raise ValueError("Input data F must be either 2D or 3D")

    # Calculate the local mean of the squared image (F**2)
    local_mean_squared = convolve(F**2, kernel) / normalization_factor

    # Calculate the square of the local mean
    local_mean = convolve(F, kernel) / normalization_factor
    local_mean_squared_of_F = local_mean ** 2

    # Compute the local variance
    local_variance = local_mean_squared - local_mean_squared_of_F

    # Estimate noise as the square root of the average local variance
    noise_estimate = np.sqrt(np.nanmean(local_variance))

    # Estimate signal as the mean of the flattened 2D/3D image (foreground)
    signal_estimate = np.nanmean(f)

    # Calculate the SNR
    measure = signal_estimate / (noise_estimate + 1e-9)  # Adding a small number to avoid division by zero

    return name, measure


def func12(F, B, c, f, b):
    name = 'SNR6'
    noise_estimate = np.median(np.abs(f - np.median(f))) / 0.6745  # 0.6745 is the consistency constant for normally distributed data
    signal_estimate = np.mean(f)
    measure = signal_estimate / (noise_estimate + 1e-9)  # Adding a small number to avoid division by zero
    return name, measure

def func13(F, B, c, f, b):
    name = 'SNR7'
    edges = sobel(F)
    edge_pixels = F[(edges > np.percentile(edges, 95))]  # Consider top 5% of edges by magnitude
    noise_estimate = np.std(edge_pixels)
    signal_estimate = np.mean(f)
    measure = signal_estimate / (noise_estimate + 1e-9)
    return name, measure

def func14(F, B, c, f, b):
    name = 'SNR8'
    F_freq = np.fft.fft2(F)
    F_shifted = np.fft.fftshift(F_freq)
    rows, cols = F.shape
    crow, ccol = rows // 2 , cols // 2
    mask_size = 5  # Exclude the center
    mask = np.ones(F.shape, np.uint8)
    mask[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
    noise_freq = F_shifted * mask
    noise_time = np.fft.ifft2(np.fft.ifftshift(noise_freq)).real
    noise_estimate = np.std(noise_time)
    signal_estimate = np.mean(f)
    measure = signal_estimate / (noise_estimate + 1e-9)
    return name, measure


def func15(F, B, c, f, b):
    name = 'SNR9'
    LBP_texture = local_binary_pattern(F, P=8, R=1)
    texture_regions = LBP_texture[(LBP_texture > np.percentile(LBP_texture, 95))]

    if len(texture_regions) == 0:
        noise_estimate = 1e-9
    else:
        noise_estimate = np.std(texture_regions)

    signal_estimate = np.mean(f)
    measure = signal_estimate / (noise_estimate + 1e-9)
    return name, measure
    


def psnr(img1, img2):
    mse = np.square(np.subtract(img1, img2)).mean()
    if mse == 0:
        return np.inf
    return 20 * np.log10(np.nanmax(img1) / np.sqrt(mse))

def patch(img, patch_size):
    h = int(np.floor(patch_size / 2))
    U = np.pad(img, pad_width=5, mode='constant')
    a, b = np.where(img == np.max(img))
    a, b = a[0], b[0]
    return U[a:a+2*h+1, b:b+2*h+1]


def extract_metadata(nifti_file):
    img = nib.load(nifti_file)
    header = img.header
    metadata = {
        'MFR': header.get('manufacturer', 'Unknown'),
        'MFS': header.get('magnetic_field_strength', 'Unknown'),
        'VRX': header.get_zooms()[0] if len(header.get_zooms()) > 0 else 'Unknown',
        'VRY': header.get_zooms()[1] if len(header.get_zooms()) > 1 else 'Unknown',
        'VRZ': header.get_zooms()[2] if len(header.get_zooms()) > 2 else 'Unknown',
        'ROWS': header.get_data_shape()[0] if len(header.get_data_shape()) > 0 else 'Unknown',
        'COLS': header.get_data_shape()[1] if len(header.get_data_shape()) > 1 else 'Unknown',
        'TR': header.get('repetition_time', 'Unknown'),
        'TE': header.get('echo_time', 'Unknown'),
        'NUM': header.get_data_shape()[2] if len(header.get_data_shape()) > 2 else 'Unknown'
    }
    return metadata

def analyze_nifti(nifti_file):
    img = nib.load(nifti_file)
    img_data = img.get_fdata()

    if len(img_data.shape) == 3:  # Handle 3D images
        slices = [img_data[:, :, i] for i in range(img_data.shape[2])]
    elif len(img_data.shape) == 4:  # Handle 4D images (e.g., fMRI)
        slices = [img_data[:, :, :, i] for i in range(img_data.shape[3])]
    else:
        slices = [img_data]

    all_measures = []

    for slice_ in slices:
        foreground = slice_
        background = np.zeros_like(slice_)  # Adjust as needed
        context = {}  # Adjust as needed
        flat_foreground = foreground.flatten()
        flat_background = background.flatten()

        functions = [func1, func2, func3, func4, func5, func6, func7, func8, func9, func10, 
                     func11, func12, func13, func14, func15]

        measures = {}
        for func in functions:
            name, measure = func(foreground, background, context, flat_foreground, flat_background)
            measures[name] = measure

        all_measures.append(measures)

    avg_measures = {k: np.nanmean([m[k] for m in all_measures]) for k in all_measures[0]}

    return avg_measures

def process_nifti_files(input_folder, output_csv):
    results = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nifti_file = os.path.join(root, file)
                measures = analyze_nifti(nifti_file)
                metadata = extract_metadata(nifti_file)
                measures.update(metadata)
                measures['File'] = nifti_file  # Store full path
                results.append(measures)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    input_folder = 'output_of_pt4'
    output_csv = 'output_measures_pt4_niigz_snr5.csv'
    process_nifti_files(input_folder, output_csv)
