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
    raise NotImplementedError

@numba.njit
def _nssd_loop(img, tmpl, out, tmpl_std):
    """
    Similar to SSD but divides by the standard deviation product of region and template.
    """
    raise NotImplementedError

@numba.njit
def _ccorr_loop(img, tmpl, out):
    """
    Calculate cross-correlation between template and each image window.
    """
    raise NotImplementedError

@numba.njit
def _nccorr_loop(img, tmpl_norm, out):
    """
    Using a normalized template, compute normalized cross-correlation for each sliding window.
    """
    raise NotImplementedError

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
    raise NotImplementedError