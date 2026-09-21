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
    img_bgr = img_bgr.astype(np.float64)

    rows, cols, _ = img_bgr.shape
    total_pixels = rows * cols
    thresh_idx = math.floor(total_pixels * threshold_percentage)

    compressed_frequency_img = np.zeros((rows, cols, 3), dtype=np.complex128)
    img_compressed = np.zeros((rows, cols, 3), dtype=np.float64)

    for ch in range(3):
        curr_channel = img_bgr[:, :, ch]
        freq = dft2(curr_channel)   # shape: (rows, cols)

        freq_flat = freq.flatten()  # shape: (rows*cols,)
        magnitudes = np.abs(freq_flat)
        sort_indices = np.argsort(magnitudes)[::-1]  # largest first

        freq_masked_flat = np.zeros_like(freq_flat, dtype=np.complex128)
        freq_masked_flat[sort_indices[:thresh_idx]] = freq_flat[sort_indices[:thresh_idx]]

        freq_masked = freq_masked_flat.reshape((rows, cols))

        compressed_frequency_img[:, :, ch] = freq_masked

        channel_compressed = idft2(freq_masked)  # shape: (rows, cols)
        img_compressed[:, :, ch] = np.real(channel_compressed)

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
    # Ensure working in float64 precision.
    img_bgr = img_bgr.astype(np.float64)
    rows, cols, _ = img_bgr.shape

    # Precompute a circular mask (in the frequency domain, after fftshift).
    center_row, center_col = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - center_row)**2 + (X - center_col)**2)
    mask = distance <= r

    # Prepare output arrays.
    low_pass_frequency_img = np.zeros((rows, cols, 3), dtype=np.complex128)
    img_low_pass = np.zeros((rows, cols, 3), dtype=np.float64)

    # Process each channel independently.
    for ch in range(3):
        # Compute the 2D Fourier transform for the current channel.
        freq = dft2(img_bgr[:, :, ch])
        # Shift the zero-frequency component to the center.
        freq_shifted = np.fft.fftshift(freq)
        # Apply the low pass mask.
        freq_shifted_masked = freq_shifted * mask
        # Inverse shift to return to the original frequency ordering.
        freq_masked = np.fft.ifftshift(freq_shifted_masked)
        low_pass_frequency_img[:, :, ch] = freq_masked
        channel_filtered = idft2(freq_masked)
        img_low_pass[:, :, ch] = np.real(channel_filtered)

    low_pass_frequency_img = np.zeros((rows, cols, 3), dtype=np.float64)
        
    low_pass_frequency_img = 20 * np.log(np.abs(np.fft.fftshift(low_pass_frequency_img)) + 1)

    return img_low_pass, low_pass_frequency_img