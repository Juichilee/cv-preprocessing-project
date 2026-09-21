import cv2
import numpy as np
import numba

# Hue constants and mapping
HUE_R = 0
HUE_Y = 30
HUE_G = 60
hue_dict = {HUE_R: "red", HUE_Y: "yellow", HUE_G: "green"}


@numba.njit
def _ssd_loop(img, tmpl, out):
    """
    Move template across image and calculate squared difference sum at each location.
    """
    th, tw = tmpl.shape
    rows, cols = out.shape
    for y in range(rows):
        for x in range(cols):
            region = img[y : y + th, x : x + tw]
            out[y, x] = np.sum((region - tmpl) ** 2)


@numba.njit
def _nssd_loop(img, tmpl, out, tmpl_std):
    """
    Similar to SSD but divides by the standard deviation product of region and template.
    """
    th, tw = tmpl.shape
    rows, cols = out.shape
    for y in range(rows):
        for x in range(cols):
            region = img[y : y + th, x : x + tw]
            region_std = np.std(region)
            if region_std != 0 and tmpl_std != 0:
                out[y, x] = np.sum((region - tmpl) ** 2) / (region_std * tmpl_std)
            else:
                out[y, x] = np.inf


@numba.njit
def _ccorr_loop(img, tmpl, out):
    """
    Calculate cross-correlation between template and each image window.
    """
    th, tw = tmpl.shape
    rows, cols = out.shape
    tmpl_mean = np.mean(tmpl)
    for y in range(rows):
        for x in range(cols):
            region = img[y : y + th, x : x + tw]
            region_mean = np.mean(region)
            out[y, x] = np.sum((region - region_mean) * (tmpl - tmpl_mean))


@numba.njit
def _nccorr_loop(img, tmpl_norm, out):
    """
    Using a normalized template, compute normalized cross-correlation for each sliding window.
    """
    th, tw = tmpl_norm.shape
    rows, cols = out.shape
    for y in range(rows):
        for x in range(cols):
            region = img[y : y + th, x : x + tw]
            r_mean = np.mean(region)
            r_std = np.std(region)
            if r_std > 1e-12:
                r_norm = (region - r_mean) / r_std
                out[y, x] = np.sum(r_norm * tmpl_norm)
            else:
                out[y, x] = -np.inf


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
        NotImplementedError: For unrecognized methods.
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("`img` must be a NumPy array.")
    if not isinstance(tmpl, np.ndarray):
        raise TypeError("`tmpl` must be a NumPy array.")
    if img.ndim != 2:
        raise ValueError("`img` must be a 2D (grayscale) array.")
    if tmpl.ndim != 2:
        raise ValueError("`tmpl` must be a 2D array.")
    if img.size == 0:
        raise ValueError("`img` must be non-empty.")
    if tmpl.size == 0:
        raise ValueError("`tmpl` must be non-empty.")

    h0, w0 = img.shape
    ht, wt = tmpl.shape
    if ht > h0 or wt > w0:
        raise ValueError("Template must not be larger than the image.")

    supported = ("tm_ssd", "tm_nssd", "tm_ccor", "tm_nccor")
    if method not in supported:
        raise NotImplementedError(f"Unknown method '{method}'. Supported: {supported}")

    # prepare output buffer
    out = np.zeros((h0 - ht + 1, w0 - wt + 1), dtype=float)

    if method == "tm_ssd":
        _ssd_loop(img, tmpl, out)
        _, _, min_loc, _ = cv2.minMaxLoc(out)
        return min_loc

    elif method == "tm_nssd":
        tmpl_std = np.std(tmpl)
        # catch zero‑variance template early
        if tmpl_std < 1e-12:
            raise ValueError("Template has near-zero variance; cannot perform tm_nssd.")
        _nssd_loop(img, tmpl, out, tmpl_std)
        _, _, min_loc, _ = cv2.minMaxLoc(out)
        return min_loc

    elif method == "tm_ccor":
        _ccorr_loop(img, tmpl, out)
        _, _, _, max_loc = cv2.minMaxLoc(out)
        return max_loc

    elif method == "tm_nccor":
        tmpl_std = np.std(tmpl)

        if tmpl_std < 1e-12:
            raise ValueError("Template has zero variance; cannot normalize.")
        tmpl_norm = (tmpl - np.mean(tmpl)) / tmpl_std
        _nccorr_loop(img, tmpl_norm, out)
        _, _, _, max_loc = cv2.minMaxLoc(out)
        return max_loc