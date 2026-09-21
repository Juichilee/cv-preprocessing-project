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
    """Return compressed image by converting to frequency domain,
    thresholding based on threshold_percentage, and converting back to spatial domain.

    Args:
        img_bgr (np.ndarray): H×W×3 BGR image
        threshold_percentage (float): fraction in [0, 1] of Fourier coefficients to keep

    Returns:
        img_compressed (np.ndarray, float64): H×W×3 compressed spatial-domain image
        compressed_frequency_img (np.ndarray, float64): H×W×3 log-magnitude spectrum of compressed frequencies
    """
    # Validate inputs
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("img_bgr must be a numpy array")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must have shape (H, W, 3)")
    if not isinstance(threshold_percentage, (int, float)):
        raise TypeError("threshold_percentage must be a number")
    if not (0 <= threshold_percentage <= 1):
        raise ValueError("threshold_percentage must be between 0 and 1")

    img = img_bgr.astype(np.float64)
    H, W, _ = img.shape
    total = H * W
    k = math.floor(total * threshold_percentage)

    freq_comp = np.zeros((H, W, 3), dtype=np.complex128)
    img_comp = np.zeros((H, W, 3), dtype=np.float64)

    for ch in range(3):
        channel = img[:, :, ch]
        F = np.fft.fft2(channel)

        if k <= 0:
            Fm = np.zeros_like(F)
        elif k >= total:
            Fm = F
        else:
            flat = F.ravel()
            mags = np.abs(flat)
            # pick top-k magnitudes
            idx_top = np.argpartition(-mags, k)[:k]
            mask = np.zeros_like(flat, dtype=complex)
            mask[idx_top] = flat[idx_top]
            Fm = mask.reshape(H, W)

        freq_comp[:, :, ch] = Fm
        img_comp[:, :, ch] = np.real(np.fft.ifft2(Fm))

    compressed_frequency_img = 20 * np.log(np.abs(np.fft.fftshift(freq_comp)) + 1)
    return img_comp, compressed_frequency_img


def low_pass_filter(img_bgr, radius):
    """Return low-pass filtered image by keeping a circle of radius
    `radius` in the frequency domain, then converting back to spatial domain.

    Args:
        img_bgr (np.ndarray): H×W×3 BGR image
        radius (float): non-negative radius in pixels

    Returns:
        img_low_pass (np.ndarray, float64): H×W×3 low-pass filtered spatial image
        low_pass_frequency_img (np.ndarray, float64): H×W×3 log-magnitude spectrum of low-pass frequencies
    """
    # Validate inputs
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("img_bgr must be a numpy array")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must have shape (H, W, 3)")
    if not isinstance(radius, (int, float)):
        raise TypeError("radius must be a number")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    img = img_bgr.astype(np.float64)
    H, W, _ = img.shape

    # build circular mask (no sqrt)
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist2 = (Y - cy) ** 2 + (X - cx) ** 2
    mask = dist2 <= (radius ** 2)

    freq_low = np.zeros((H, W, 3), dtype=np.complex128)
    img_low = np.zeros((H, W, 3), dtype=np.float64)

    for ch in range(3):
        F = np.fft.fft2(img[:, :, ch])
        Fs = np.fft.fftshift(F)
        Fs_masked = Fs * mask
        Fm = np.fft.ifftshift(Fs_masked)

        freq_low[:, :, ch] = Fm
        img_low[:, :, ch] = np.real(np.fft.ifft2(Fm))

    low_pass_frequency_img = 20 * np.log(np.abs(np.fft.fftshift(freq_low)) + 1)
    return img_low, low_pass_frequency_img