import numpy as np
from functools import lru_cache

@lru_cache(maxsize=8)
def _dft_matrix(n):
    """
    Build an n×n DFT matrix.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("dft_matrix: n must be a positive integer")
    
    omega = np.exp(-2j * np.pi / n)
    j, k = np.meshgrid(np.arange(n), np.arange(n))
    return np.power(omega, j * k) / np.sqrt(n)

@lru_cache(maxsize=8)
def _idft_matrix(n):
    """
    Build an n×n IDFT matrix as the conjugate of the DFT matrix for unitary consistency.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("idft_matrix: n must be a positive integer")
    
    dft_mat = _dft_matrix(n)
    return np.conj(dft_mat)  # Updated: Removed incorrect scaling (/ n * np.sqrt(n)) to ensure unitary inverse. This fixes reconstruction errors by making IDFT the true inverse of DFT.

def dft(x):
    """
    Compute a 1D DFT using a cached matrix.
    Should reference _dft_matrix().

    Args:
        x (array_like, complex): 1D input.

    Returns:
        complex128 ndarray: DFT of x.

    Raises:
        ValueError: if x is not 1D.
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError("dft: input must be 1D")
    n = len(x)
    return np.dot(_dft_matrix(n), x)

def idft(X):
    """
    Compute a 1D inverse DFT using a cached matrix.
    Should reference _idft_matrix().

    Args:
        X (array_like, complex): 1D frequency data.

    Returns:
        complex128 ndarray: inverse DFT of X.

    Raises:
        ValueError: if X is not 1D.
    """
    X = np.asarray(X, dtype=np.complex128)
    if X.ndim != 1:
        raise ValueError("idft: input must be 1D")
    n = len(X)
    return np.dot(_idft_matrix(n), X)

def dft2(img):
    """
    Compute a 2D DFT of a 2D array where zero‐frequency bin is moved to the middle of the array.
    Should reference _dft_matrix().

    Args:
        img (array_like): 2D input array.

    Returns:
        complex128 ndarray: centered 2D DFT.

    Raises:
        ValueError: if img is not 2D.
    """
    img = np.asarray(img, dtype=np.complex128)
    if img.ndim != 2:
        raise ValueError("dft2: input must be 2D")
    dft_rows = np.apply_along_axis(dft, axis=1, arr=img)
    dft_2d = np.apply_along_axis(dft, axis=0, arr=dft_rows)
    return np.fft.fftshift(dft_2d)

def idft2(F):
    """
    Compute a 2D inverse DFT from a centered 2D spectrum.
    Should reference _idft_matrix().
    
    Args:
        F (array_like): 2D centered DFT array.

    Returns:
        complex128 ndarray: Reconstructed 2D spatial array.

    Raises:
        ValueError: If F is not 2D.
    """
    F = np.asarray(F, dtype=np.complex128)
    if F.ndim != 2:
        raise ValueError("idft2: input must be 2D")
    F_unshifted = np.fft.ifftshift(F)
    idft_rows = np.apply_along_axis(idft, axis=1, arr=F_unshifted)
    idft_2d = np.apply_along_axis(idft, axis=0, arr=idft_rows)
    return idft_2d

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
        k = int(np.floor(total_coeffs * threshold_percentage))
        k = max(k, 1)

    compressed_F = np.zeros_like(img, dtype=np.complex128)
    F_all = np.zeros_like(img, dtype=np.complex128)

    for ch in range(3):
        F_all[:, :, ch] = dft2(img[:, :, ch])
        F = F_all[:, :, ch]
        if k == 0:
            Fm = np.zeros_like(F)
        elif k >= total_coeffs:
            Fm = F
        else:
            flat = F.ravel()
            mags = np.abs(flat)
            idx_top = np.argpartition(-mags, k - 1)[:k]
            mask = np.zeros_like(mags, dtype=bool)
            mask[idx_top] = True
            Fm = (flat * mask).reshape(H, W)
        compressed_F[:, :, ch] = Fm

    recon = np.zeros_like(img, dtype=np.complex128)
    for ch in range(3):
        recon[:, :, ch] = idft2(compressed_F[:, :, ch])

    img_recon = np.real(recon).astype(np.float32)
    img_compressed = np.clip(img_recon, 0, 255).astype(np.uint8)

    mag_combined = np.mean(np.abs(compressed_F), axis=2)
    spectrum_single = 20.0 * np.log1p(mag_combined).astype(np.float32)

    spectrum = np.repeat(spectrum_single[:, :, np.newaxis], 3, axis=2)

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

    F_shifted = np.zeros_like(img, dtype=np.complex128)
    F_masked = np.zeros_like(img, dtype=np.complex128)
    recon = np.zeros_like(img, dtype=np.complex128)

    for ch in range(3):
        F_shifted[:, :, ch] = dft2(img[:, :, ch])

    mask_3c = mask[:, :, None]
    F_masked = F_shifted * mask_3c

    for ch in range(3):
        recon[:, :, ch] = idft2(F_masked[:, :, ch])

    img_low = np.real(recon).astype(np.float32)
    img_low_pass = np.clip(img_low, 0, 255).astype(np.uint8)

    mag_combined = np.mean(np.abs(F_masked), axis=2)
    spectrum_single = 20.0 * np.log1p(mag_combined).astype(np.float32)
    spectrum = np.repeat(spectrum_single[:, :, np.newaxis], 3, axis=2)

    return img_low_pass, spectrum
