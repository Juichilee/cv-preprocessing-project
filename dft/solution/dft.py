import cv2
import numpy as np
import math
from functools import lru_cache

@lru_cache(maxsize=8)
def _dft_matrix(n):
    """
    Build an n×n DFT matrix.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("_dft_matrix: n must be a positive integer")
    k = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(k, k) / n)


@lru_cache(maxsize=8)
def _idft_matrix(n):
    """
    Build an n×n IDFT matrix (with 1/n scaling).
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("_idft_matrix: n must be a positive integer")
    k = np.arange(n)
    return np.exp(+2j * np.pi * np.outer(k, k) / n) / n


def dft(x):
    """
    Compute a 1D DFT using a cached matrix.

    Args:
        x (array_like, complex): 1D input.

    Returns:
        complex128 ndarray: DFT of x.

    Raises:
        ValueError: if x is not 1D.
    """
    x = np.asarray(x, dtype=complex)
    if x.ndim != 1:
        raise ValueError("dft: input must be 1D")
    return _dft_matrix(len(x)) @ x


def idft(X):
    """
    Compute a 1D inverse DFT using a cached matrix.

    Args:
        X (array_like, complex): 1D frequency data.

    Returns:
        complex128 ndarray: inverse DFT of X.

    Raises:
        ValueError: if X is not 1D.
    """
    X = np.asarray(X, dtype=complex)
    if X.ndim != 1:
        raise ValueError("idft: input must be 1D")
    return _idft_matrix(len(X)) @ X


def dft2(img):
    """
    Compute a 2D DFT of a 2D array where zero‐frequency bin is moved to the middle of the array.

    Args:
        img (array_like): 2D input array.

    Returns:
        complex128 ndarray: centered 2D DFT.

    Raises:
        ValueError: if img is not 2D.
    """
    img = np.asarray(img, dtype=complex)
    if img.ndim != 2:
        raise ValueError("dft2: input must be 2D")
    Wm = _dft_matrix(img.shape[0])
    Wn = _dft_matrix(img.shape[1])
    raw = Wm @ img @ Wn
    return np.fft.fftshift(raw)


def idft2(F):
    """
    Compute a 2D inverse DFT from a centered 2D spectrum.

    Args:
        F (array_like, complex): 2D centered DFT array.

    Returns:
        complex128 ndarray: Reconstructed 2D spatial array.

    Raises:
        ValueError: If F is not 2D.
    """
    F = np.asarray(F, dtype=complex)
    if F.ndim != 2:
        raise ValueError("idft2: input must be 2D")
    F_unshift = np.fft.ifftshift(F)
    Wm_inv = _idft_matrix(F_unshift.shape[0])
    Wn_inv = _idft_matrix(F_unshift.shape[1])
    return Wm_inv @ F_unshift @ Wn_inv


def compress_image_fft(img_bgr, threshold_percentage):
    """Compress image by retaining top-magnitude Fourier coefficients per channel and return spectrum.

    Args:
        img_bgr (np.ndarray): Input BGR image of shape (H, W, 3).
        threshold_percentage (float): Fraction of coefficients to retain (0.0 to 1.0).

    Returns:
        img_compressed (np.ndarray): Compressed image as uint8.
        spectrum (np.ndarray): 3-channel centered log-magnitude spectrum, shape (H, W, 3).

    Raises:
        TypeError: If img_bgr is not a NumPy array or threshold_percentage is not a scalar.
        ValueError: If img_bgr shape is invalid or threshold_percentage not in [0.0, 1.0].
    """
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("compress_image_fft: img_bgr must be a NumPy array")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("compress_image_fft: img_bgr must have shape (H, W, 3)")
    if not np.isscalar(threshold_percentage):
        raise TypeError("compress_image_fft: threshold_percentage must be a scalar")
    if not (0.0 <= threshold_percentage <= 1.0):
        raise ValueError("compress_image_fft: threshold_percentage must be in [0.0, 1.0]")

    img = img_bgr.astype(np.float32)
    H, W, _ = img.shape
    total_coeffs = H * W

    if threshold_percentage <= 0.0:
        k = 0
    elif threshold_percentage >= 1.0:
        k = total_coeffs
    else:
        k = int(math.floor(total_coeffs * threshold_percentage))
        k = max(k, 1)

    freq_comp = np.zeros((H, W, 3), dtype=np.complex128)
    img_comp = np.zeros((H, W, 3), dtype=np.float32)

    for ch in range(3):
        channel = img[:, :, ch]
        try:
            F_centered = dft2(channel)
        except Exception as e:
            raise ValueError(f"compress_image_fft: dft2 failed on channel {ch}: {e}")

        F_unshift = np.fft.ifftshift(F_centered)
        if k == 0:
            Fm_unshift = np.zeros_like(F_unshift)
        elif k >= total_coeffs:
            Fm_unshift = F_unshift
        else:
            flat = F_unshift.ravel()
            mags = np.abs(flat)
            idx_top = np.argpartition(-mags, k - 1)[:k]
            mask_bool = np.zeros_like(mags, dtype=bool)
            mask_bool[idx_top] = True
            Fm_unshift = (flat * mask_bool).reshape(H, W)

        Fm_centered = np.fft.fftshift(Fm_unshift)
        freq_comp[:, :, ch] = Fm_centered

        try:
            recon = idft2(Fm_centered)
        except Exception as e:
            raise ValueError(f"compress_image_fft: idft2 failed on channel {ch}: {e}")
        img_comp[:, :, ch] = np.real(recon)

    img_compressed = np.clip(img_comp, 0, 255).astype(np.uint8)

    spectrum = 20.0 * np.log1p(np.abs(freq_comp)).astype(np.float32)
    return img_compressed, spectrum


def low_pass_filter(img_bgr, radius):
    """Apply low-pass filter in frequency domain and return image and spectrum.

    Args:
        img_bgr (np.ndarray): Input BGR image of shape (H, W, 3).
        radius (float or int): Non-negative radius in pixels.

    Returns:
        img_low_pass (np.ndarray): Low-pass filtered image as uint8.
        spectrum (np.ndarray): 3-channel centered log-magnitude spectrum, shape (H, W, 3).

    Raises:
        TypeError: If img_bgr is not a NumPy array or radius not a scalar.
        ValueError: If img_bgr shape is invalid or radius is negative/not finite.
    """
    if not isinstance(img_bgr, np.ndarray):
        raise TypeError("low_pass_filter: img_bgr must be a NumPy array")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("low_pass_filter: img_bgr must have shape (H, W, 3)")
    if not np.isscalar(radius):
        raise TypeError("low_pass_filter: radius must be a scalar")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("low_pass_filter: radius must be a non-negative finite number")

    img = img_bgr.astype(np.float32)
    H, W, _ = img.shape

    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist2 = (Y - cy).astype(np.float32)**2 + (X - cx).astype(np.float32)**2
    mask = dist2 <= (float(radius) ** 2)

    freq_low = np.zeros((H, W, 3), dtype=np.complex128)
    img_low = np.zeros((H, W, 3), dtype=np.float32)

    for ch in range(3):
        channel = img[:, :, ch]
        try:
            F_centered = dft2(channel)
        except Exception as e:
            raise ValueError(f"low_pass_filter: dft2 failed on channel {ch}: {e}")

        Fm_centered = F_centered * mask
        freq_low[:, :, ch] = Fm_centered

        try:
            recon = idft2(Fm_centered)
        except Exception as e:
            raise ValueError(f"low_pass_filter: idft2 failed on channel {ch}: {e}")
        img_low[:, :, ch] = np.real(recon)

    img_low_pass = np.clip(img_low, 0, 255).astype(np.uint8)

    spectrum = 20.0 * np.log1p(np.abs(freq_low)).astype(np.float32)
    return img_low_pass, spectrum
