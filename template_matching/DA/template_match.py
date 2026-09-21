import cv2
import numpy as np
import numba

HUE_R = 0
HUE_Y = 30
HUE_G = 60
hue_dict = {HUE_R: "red", HUE_Y: "yellow", HUE_G: "green"}

@numba.njit
def _ssd_loop(img, tmpl, out):
    """
    Move template across image and calculate squared difference sum at each location.
    """
    h, w = tmpl.shape
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            ssd = 0.0
            for y in range(h):
                for x in range(w):
                    diff = img[i + y, j + x] - tmpl[y, x]
                    ssd += diff * diff
            out[i, j] = ssd

@numba.njit
def _nssd_loop(img, tmpl, out, tmpl_std):
    """
    Similar to SSD but divides by the standard deviation product of region and template.
    Writes np.inf if either std is zero.
    """
    h, w = tmpl.shape
    num_pixels = h * w
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            ssd = 0.0
            patch_sum = 0.0
            patch_sq_sum = 0.0
            for y in range(h):
                for x in range(w):
                    val = img[i + y, j + x]
                    tmpl_val = tmpl[y, x]
                    diff = val - tmpl_val
                    ssd += diff * diff
                    patch_sum += val
                    patch_sq_sum += val * val
            mean_patch = patch_sum / num_pixels
            variance = (patch_sq_sum / num_pixels) - (mean_patch * mean_patch)
            std_patch = variance ** 0.5 if variance > 0 else 0.0
            if std_patch == 0 or tmpl_std == 0:
                out[i, j] = np.inf
            else:
                out[i, j] = ssd / (std_patch * tmpl_std)

@numba.njit
def _ccorr_loop(img, tmpl, out):
    """
    Calculate cross-correlation between template and each image window, using means.
    """
    h, w = tmpl.shape
    num_pixels = h * w
    # Precompute template mean (done once outside the main loop for efficiency)
    tmpl_sum = 0.0
    for y in range(h):
        for x in range(w):
            tmpl_sum += tmpl[y, x]
    tmpl_mean = tmpl_sum / num_pixels
    
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch_sum = 0.0
            corr = 0.0
            # First pass: compute patch mean
            for y in range(h):
                for x in range(w):
                    val = img[i + y, j + x]
                    patch_sum += val
            patch_mean = patch_sum / num_pixels
            # Second pass: compute correlation
            for y in range(h):
                for x in range(w):
                    val = img[i + y, j + x]
                    corr += (val - patch_mean) * (tmpl[y, x] - tmpl_mean)
            out[i, j] = corr

@numba.njit
def _nccorr_loop(img, tmpl_norm, out):
    """
    Using a normalized template, compute normalized cross-correlation for each sliding window.
    Writes -np.inf when region std is effectively zero.
    """
    h, w = tmpl_norm.shape  # tmpl_norm has the same shape as tmpl
    num_pixels = h * w
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch_sum = 0.0
            patch_sq_sum = 0.0
            ncc = 0.0
            # First pass: compute patch mean and std
            for y in range(h):
                for x in range(w):
                    val = img[i + y, j + x]
                    patch_sum += val
                    patch_sq_sum += val * val
            mean_patch = patch_sum / num_pixels
            variance = (patch_sq_sum / num_pixels) - (mean_patch * mean_patch)
            std_patch = variance ** 0.5 if variance > 0 else 0.0
            if std_patch == 0:
                out[i, j] = -np.inf
            else:
                # Second pass: compute normalized correlation
                for y in range(h):
                    for x in range(w):
                        val = img[i + y, j + x]
                        ncc += ((val - mean_patch) / std_patch) * tmpl_norm[y, x]
                out[i, j] = ncc

def template_match(img, tmpl, method):
    """
    Find the upper-left corner position where template best aligns with image.

    Arguments:
        img (np.ndarray): Single-channel or grayscale image.
        tmpl (np.ndarray): Two-dimensional template of same type.
        method (str): Choose from
            "tm_ssd", "tm_nssd", "tm_ccor", "tm_nccor".

    Returns:
        (x, y) coordinate of optimal match position.

    Specific Exceptions:
        TypeError: If img or tmpl are not NumPy arrays.
        ValueError: For invalid shapes, sizes, empty arrays, or zero template std (for NSSD/NCCORR).
        NotImplementedError: For unrecognized methods.
    """
    # Input validation
    if not isinstance(img, np.ndarray) or not isinstance(tmpl, np.ndarray):
        raise TypeError("img and tmpl must be NumPy arrays")
    if img.ndim != 2 or tmpl.ndim != 2:
        raise ValueError("img and tmpl must be 2D arrays")
    if img.size == 0 or tmpl.size == 0:
        raise ValueError("img and tmpl must be non-empty")
    if tmpl.shape[0] > img.shape[0] or tmpl.shape[1] > img.shape[1]:
        raise ValueError("template must not be larger than image")
    
    # Method validation
    allowed_methods = ["tm_ssd", "tm_nssd", "tm_ccor", "tm_nccor"]
    if method not in allowed_methods:
        raise NotImplementedError(f"Unknown method: {method}")
    
    # Prepare output array
    out_shape = (img.shape[0] - tmpl.shape[0] + 1, img.shape[1] - tmpl.shape[1] + 1)
    out = np.empty(out_shape, dtype=np.float64)
    
    # Dispatch based on method with necessary precomputations
    if method == "tm_ssd":
        _ssd_loop(img, tmpl, out)
        # SSD is minimized
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(out)
        return min_loc
    elif method == "tm_nssd":
        # Precompute template std and check for zero
        tmpl_flat = tmpl.ravel()
        tmpl_std = np.std(tmpl_flat)
        if tmpl_std == 0:
            raise ValueError("Template standard deviation is zero for NSSD")
        _nssd_loop(img, tmpl, out, tmpl_std)
        # NSSD is minimized
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(out)
        return min_loc
    elif method == "tm_ccor":
        _ccorr_loop(img, tmpl, out)
        # CCORR is maximized
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(out)
        return max_loc
    elif method == "tm_nccor":
        # Precompute normalized template and check for zero std
        tmpl_flat = tmpl.ravel()
        tmpl_mean = np.mean(tmpl_flat)
        tmpl_std = np.std(tmpl_flat)
        if tmpl_std == 0:
            raise ValueError("Template standard deviation is zero for NCCORR")
        tmpl_norm = (tmpl - tmpl_mean) / tmpl_std
        _nccorr_loop(img, tmpl_norm, out)
        # NCCORR is maximized
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(out)
        return max_loc