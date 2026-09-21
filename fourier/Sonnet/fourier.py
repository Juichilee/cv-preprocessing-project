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
        raise ValueError("img_bgr must be a 3-channel BGR image (numpy array with shape (n,m,3))")
    
    if not 0 <= threshold_percentage <= 1:
        raise ValueError("threshold_percentage must be between 0 and 1")
    
    # Ensure working in float64 precision
    img_bgr = img_bgr.astype(np.float64)
    rows, cols, _ = img_bgr.shape
    
    # Use NumPy's FFT which is much faster than our custom DFT implementation
    compressed_frequency_img = np.zeros((rows, cols, 3), dtype=np.complex128)
    img_compressed = np.zeros((rows, cols, 3), dtype=np.float64)
    
    # Calculate threshold index based on percentage
    total_pixels = rows * cols
    thresh_idx = math.floor(total_pixels * threshold_percentage)
    
    for ch in range(3):
        # Use NumPy's FFT which is much faster
        freq = np.fft.fft2(img_bgr[:, :, ch])
        
        # Create a mask based on magnitude thresholding
        magnitude = np.abs(freq)
        # Flatten, find threshold value, and create binary mask
        flat_magnitude = magnitude.flatten()
        sorted_indices = np.argsort(flat_magnitude)[::-1]  # Descending order
        threshold_value = flat_magnitude[sorted_indices[min(thresh_idx, len(sorted_indices)-1)]]
        
        # Create mask directly without flattening/reshaping operations
        mask = magnitude >= threshold_value
        
        # Apply mask to frequency domain
        freq_masked = freq * mask
        compressed_frequency_img[:, :, ch] = freq_masked
        
        # Convert back to spatial domain
        img_compressed[:, :, ch] = np.real(np.fft.ifft2(freq_masked))
    
    # Prepare frequency domain visualization
    compressed_frequency_img = 20 * np.log(np.abs(np.fft.fftshift(compressed_frequency_img)) + 1)
    
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
    # Input validation
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be a 3-channel BGR image (numpy array with shape (n,m,3))")
    
    if r <= 0:
        raise ValueError("Radius must be positive")
    
    # Ensure working in float64 precision
    img_bgr = img_bgr.astype(np.float64)
    rows, cols, _ = img_bgr.shape

    # Precompute a circular mask (in the frequency domain, after fftshift)
    center_row, center_col = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - center_row)**2 + (X - center_col)**2)
    mask = distance <= r

    # Prepare output arrays
    low_pass_frequency_img = np.zeros((rows, cols, 3), dtype=np.complex128)
    img_low_pass = np.zeros((rows, cols, 3), dtype=np.float64)

    # Process each channel independently
    for ch in range(3):
        # Use NumPy's FFT which is much faster
        freq = np.fft.fft2(img_bgr[:, :, ch])
        # Shift the zero-frequency component to the center
        freq_shifted = np.fft.fftshift(freq)
        # Apply the low pass mask
        freq_shifted_masked = freq_shifted * mask
        # Inverse shift to return to the original frequency ordering
        freq_masked = np.fft.ifftshift(freq_shifted_masked)
        
        low_pass_frequency_img[:, :, ch] = freq_masked
        # Convert back to spatial domain
        img_low_pass[:, :, ch] = np.real(np.fft.ifft2(freq_masked))

    # Prepare frequency domain visualization
    low_pass_frequency_img = 20 * np.log(np.abs(np.fft.fftshift(low_pass_frequency_img)) + 1)

    return img_low_pass, low_pass_frequency_img