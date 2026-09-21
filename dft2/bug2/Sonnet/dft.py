import numpy as np

def _dft_matrix(n):
    """
    Build an n×n DFT matrix.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("dft: n must be a positive integer")
    
    N = n
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    omega = np.exp(-2j * np.pi * i * j / N)
    return omega

def _idft_matrix(n):
    """
    Build an n×n IDFT matrix (with 1/n scaling).
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("idft: n must be a positive integer")
    
    N = n
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    omega = np.exp(2j * np.pi * i * j / N) / N
    return omega

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
    
    n = x.size
    dft_mat = _dft_matrix(n)
    return np.dot(dft_mat, x)

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
    
    n = X.size
    idft_mat = _idft_matrix(n)
    return np.dot(idft_mat, X)

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
    
    M, N = img.shape
    
    dft_rows = np.zeros((M, N), dtype=np.complex128)
    dft_mat_rows = _dft_matrix(M)
    for i in range(N):
        dft_rows[:, i] = np.dot(dft_mat_rows, img[:, i])
    
    dft_result = np.zeros((M, N), dtype=np.complex128)
    dft_mat_cols = _dft_matrix(N)
    for i in range(M):
        dft_result[i, :] = np.dot(dft_mat_cols, dft_rows[i, :])
    
    return np.fft.fftshift(dft_result)

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
    
    F_shifted = np.fft.ifftshift(F)
    
    M, N = F_shifted.shape
    
    idft_rows = np.zeros((M, N), dtype=np.complex128)
    idft_mat_rows = _idft_matrix(M)
    for i in range(N):
        idft_rows[:, i] = np.dot(idft_mat_rows, F_shifted[:, i])
    
    idft_result = np.zeros((M, N), dtype=np.complex128)
    idft_mat_cols = _idft_matrix(N)
    for i in range(M):
        idft_result[i, :] = np.dot(idft_mat_cols, idft_rows[i, :])
    
    return idft_result

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

    F_all = np.zeros((H, W, 3), dtype=np.complex128)
    for ch in range(3):
        F_all[:, :, ch] = dft2(img[:, :, ch])
    
    compressed_F = np.zeros_like(F_all)

    for ch in range(3):
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

    F_all = np.zeros((H, W, 3), dtype=np.complex128)
    for ch in range(3):
        F_all[:, :, ch] = dft2(img[:, :, ch])

    mask_3c = mask[:, :, None]
    F_masked = F_all * mask_3c

    recon = np.zeros_like(img, dtype=np.complex128)
    for ch in range(3):
        recon[:, :, ch] = idft2(F_masked[:, :, ch])
    
    img_low = np.real(recon).astype(np.float32)
    img_low_pass = np.clip(img_low, 0, 255).astype(np.uint8)

    mag_combined = np.mean(np.abs(F_masked), axis=2)
    spectrum_single = 20.0 * np.log1p(mag_combined).astype(np.float32)
    spectrum = np.repeat(spectrum_single[:, :, np.newaxis], 3, axis=2)

    return img_low_pass, spectrum