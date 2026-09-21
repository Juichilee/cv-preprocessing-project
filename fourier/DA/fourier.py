import cv2
import numpy as np
import math
from functools import lru_cache


@lru_cache(maxsize=8)
def _dft_matrix(n):
    k = np.arange(n)
    return np.exp(-2j*np.pi*np.outer(k, k)/n)

@lru_cache(maxsize=8)
def _idft_matrix(n):
    k = np.arange(n)
    return np.exp(+2j*np.pi*np.outer(k, k)/n) / n

def dft(x):
    x = np.asarray(x, complex)
    if x.ndim != 1:
        raise ValueError("dft: input must be 1D")
    return _dft_matrix(len(x)) @ x

def idft(X):
    X = np.asarray(X, complex)
    if X.ndim != 1:
        raise ValueError("idft: input must be 1D")
    return _idft_matrix(len(X)) @ X

def dft2(img):
    img = np.asarray(img, complex)
    if img.ndim != 2:
        raise ValueError("dft2: input must be 2D")
    Wm = _dft_matrix(img.shape[0])
    Wn = _dft_matrix(img.shape[1])
    return Wm @ img @ Wn

def idft2(F):
    F = np.asarray(F, complex)
    if F.ndim != 2:
        raise ValueError("idft2: input must be 2D")
    Wm_inv = _idft_matrix(F.shape[0])
    Wn_inv = _idft_matrix(F.shape[1])
    return Wm_inv @ F @ Wn_inv


def compress_image_fft(img_bgr, threshold_percentage):
    """Return compressed image by converting to fourier domain, thresholding based on threshold percentage, and converting back to spatial domain
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        threshold_percentage (float): between 0 and 1 representing what percentage of Fourier image to keep
    Returns:
        img_compressed (np.array): numpy array of shape (n,m,3) representing compressed image. (Make sure the data type of the np array is float64)
        compressed_frequency_img (np.array): numpy array of shape (n,m,3) representing the compressed image in the frequency domain
    """
    # Input validation
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be a 3D NumPy array with shape (height, width, 3).")
    if img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0:
        raise ValueError("img_bgr must have non-zero height and width.")
    if not np.issubdtype(img_bgr.dtype, np.number) or not np.all(np.isfinite(img_bgr)):
        raise ValueError("img_bgr must contain finite numeric values.")
    if not isinstance(threshold_percentage, (int, float)) or not (0 <= threshold_percentage <= 1):
        raise ValueError("threshold_percentage must be a number between 0 and 1 (inclusive).")
    
    # Ensure float64 precision
    img_bgr = img_bgr.astype(np.float64)
    rows, cols, _ = img_bgr.shape
    total_pixels = rows * cols
    thresh_idx = math.floor(total_pixels * threshold_percentage)
    
    compressed_frequency_img = np.zeros((rows, cols, 3), dtype=np.complex128)
    img_compressed = np.zeros((rows, cols, 3), dtype=np.float64)
    
    for ch in range(3):
        curr_channel = img_bgr[:, :, ch]
        freq = np.fft.fft2(curr_channel)  # Faster replacement for dft2
        
        freq_flat = freq.ravel()
        magnitudes = np.abs(freq_flat)
        
        # Use np.partition for efficient threshold finding (avoids full sort)
        if thresh_idx > 0:
            # Get the thresh_idx-th largest magnitude as the cutoff
            magnitude_threshold = np.partition(magnitudes, -thresh_idx)[-thresh_idx]
            mask = magnitudes >= magnitude_threshold
        else:
            mask = np.zeros_like(magnitudes, dtype=bool)  # Edge case: keep nothing
        
        freq_masked_flat = np.zeros_like(freq_flat, dtype=np.complex128)
        freq_masked_flat[mask] = freq_flat[mask]
        freq_masked = freq_masked_flat.reshape((rows, cols))
        
        compressed_frequency_img[:, :, ch] = freq_masked
        channel_compressed = np.fft.ifft2(freq_masked)  # Faster replacement for idft2
        img_compressed[:, :, ch] = np.real(channel_compressed)
    
    # Compute frequency visualization (shifted and log-scaled)
    compressed_frequency_img = 20 * np.log(np.abs(np.fft.fftshift(compressed_frequency_img)) + 1).astype(np.float64)
    
    return img_compressed, compressed_frequency_img


def low_pass_filter(img_bgr, r):
    """Return low pass filtered image by keeping a circle of radius r centered on the frequency domain image
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        r (float): radius of low pass circle
    Returns:
        img_low_pass (np.array): numpy array of shape (n,m,3) representing low pass filtered image. (Make sure the data type of the np array is float64)
        low_pass_frequency_img (np.array): numpy array of shape (n,m,3) representing the low pass filtered image in the frequency domain
    """
    # Input validation (shared with compress_image_fft where applicable)
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be a 3D NumPy array with shape (height, width, 3).")
    if img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0:
        raise ValueError("img_bgr must have non-zero height and width.")
    if not np.issubdtype(img_bgr.dtype, np.number) or not np.all(np.isfinite(img_bgr)):
        raise ValueError("img_bgr must contain finite numeric values.")
    if not isinstance(r, (int, float)) or r < 0:
        raise ValueError("r must be a non-negative number (int or float).")
    
    # Ensure float64 precision
    img_bgr = img_bgr.astype(np.float64)
    rows, cols, _ = img_bgr.shape
    
    # Precompute circular mask
    center_row, center_col = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - center_row)**2 + (X - center_col)**2)
    mask = distance <= r
    
    # Prepare output arrays
    freq_masked_all = np.zeros((rows, cols, 3), dtype=np.complex128)  # Renamed for clarity
    img_low_pass = np.zeros((rows, cols, 3), dtype=np.float64)
    
    for ch in range(3):
        freq = np.fft.fft2(img_bgr[:, :, ch])  # Faster replacement for dft2
        freq_shifted = np.fft.fftshift(freq)
        freq_shifted_masked = freq_shifted * mask
        freq_masked = np.fft.ifftshift(freq_shifted_masked)
        freq_masked_all[:, :, ch] = freq_masked
        channel_filtered = np.fft.ifft2(freq_masked)  # Faster replacement for idft2
        img_low_pass[:, :, ch] = np.real(channel_filtered)
    
    # Compute frequency visualization correctly (log-scaled, no overwrite)
    low_pass_frequency_img = 20 * np.log(np.abs(np.fft.fftshift(freq_masked_all)) + 1).astype(np.float64)
    
    return img_low_pass, low_pass_frequency_img