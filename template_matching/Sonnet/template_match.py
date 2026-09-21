# template_match.py
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
    img_h, img_w = img.shape
    tmpl_h, tmpl_w = tmpl.shape
    
    for y in range(img_h - tmpl_h + 1):
        for x in range(img_w - tmpl_w + 1):
            ssd = 0.0
            for ty in range(tmpl_h):
                for tx in range(tmpl_w):
                    diff = float(img[y + ty, x + tx]) - float(tmpl[ty, tx])
                    ssd += diff * diff
            out[y, x] = ssd

@numba.njit
def _nssd_loop(img, tmpl, out, tmpl_std):
    """
    Similar to SSD but divides by the standard deviation product of region and template.
    """
    img_h, img_w = img.shape
    tmpl_h, tmpl_w = tmpl.shape
    
    for y in range(img_h - tmpl_h + 1):
        for x in range(img_w - tmpl_w + 1):
            ssd = 0.0
            # Calculate mean of the current window
            window_sum = 0.0
            for ty in range(tmpl_h):
                for tx in range(tmpl_w):
                    window_sum += float(img[y + ty, x + tx])
            window_mean = window_sum / (tmpl_h * tmpl_w)
            
            # Calculate standard deviation of the current window
            window_var_sum = 0.0
            for ty in range(tmpl_h):
                for tx in range(tmpl_w):
                    diff = float(img[y + ty, x + tx]) - window_mean
                    window_var_sum += diff * diff
            window_std = np.sqrt(window_var_sum / (tmpl_h * tmpl_w))
            
            # Calculate normalized SSD
            if window_std < 1e-8 or tmpl_std < 1e-8:
                out[y, x] = np.inf
            else:
                for ty in range(tmpl_h):
                    for tx in range(tmpl_w):
                        diff = float(img[y + ty, x + tx]) - float(tmpl[ty, tx])
                        ssd += diff * diff
                out[y, x] = ssd / (window_std * tmpl_std)

@numba.njit
def _ccorr_loop(img, tmpl, out):
    """
    Calculate cross-correlation between template and each image window.
    """
    img_h, img_w = img.shape
    tmpl_h, tmpl_w = tmpl.shape
    tmpl_size = tmpl_h * tmpl_w
    
    # Calculate template mean
    tmpl_sum = 0.0
    for ty in range(tmpl_h):
        for tx in range(tmpl_w):
            tmpl_sum += float(tmpl[ty, tx])
    tmpl_mean = tmpl_sum / tmpl_size
    
    for y in range(img_h - tmpl_h + 1):
        for x in range(img_w - tmpl_w + 1):
            # Calculate window mean
            window_sum = 0.0
            for ty in range(tmpl_h):
                for tx in range(tmpl_w):
                    window_sum += float(img[y + ty, x + tx])
            window_mean = window_sum / tmpl_size
            
            # Calculate cross-correlation
            ccorr = 0.0
            for ty in range(tmpl_h):
                for tx in range(tmpl_w):
                    ccorr += (float(img[y + ty, x + tx]) - window_mean) * (float(tmpl[ty, tx]) - tmpl_mean)
            out[y, x] = ccorr

@numba.njit
def _nccorr_loop(img, tmpl_norm, out):
    """
    Using a normalized template, compute normalized cross-correlation for each sliding window.
    """
    img_h, img_w = img.shape
    tmpl_h, tmpl_w = tmpl_norm.shape
    tmpl_size = tmpl_h * tmpl_w
    
    for y in range(img_h - tmpl_h + 1):
        for x in range(img_w - tmpl_w + 1):
            # Calculate window mean
            window_sum = 0.0
            for ty in range(tmpl_h):
                for tx in range(tmpl_w):
                    window_sum += float(img[y + ty, x + tx])
            window_mean = window_sum / tmpl_size
            
            # Calculate window standard deviation
            window_var_sum = 0.0
            for ty in range(tmpl_h):
                for tx in range(tmpl_w):
                    diff = float(img[y + ty, x + tx]) - window_mean
                    window_var_sum += diff * diff
            window_std = np.sqrt(window_var_sum / tmpl_size)
            
            # Calculate normalized cross-correlation
            if window_std < 1e-8:
                out[y, x] = -np.inf
            else:
                nccorr = 0.0
                for ty in range(tmpl_h):
                    for tx in range(tmpl_w):
                        nccorr += ((float(img[y + ty, x + tx]) - window_mean) / window_std) * float(tmpl_norm[ty, tx])
                out[y, x] = nccorr

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
        TypeError: For invalid input types.
        ValueError: For invalid input dimensions or sizes.
    """
    # Validate inputs
    if not isinstance(img, np.ndarray) or not isinstance(tmpl, np.ndarray):
        raise TypeError("Both image and template must be numpy arrays")
    
    if len(img.shape) != 2 or len(tmpl.shape) != 2:
        raise ValueError("Both image and template must be 2D arrays")
    
    if img.size == 0 or tmpl.size == 0:
        raise ValueError("Image and template cannot be empty")
    
    if tmpl.shape[0] > img.shape[0] or tmpl.shape[1] > img.shape[1]:
        raise ValueError("Template must not be larger than the image")
    
    # Prepare output buffer
    result_h = img.shape[0] - tmpl.shape[0] + 1
    result_w = img.shape[1] - tmpl.shape[1] + 1
    result = np.zeros((result_h, result_w), dtype=np.float32)
    
    # Dispatch to appropriate method
    if method == "tm_ssd":
        _ssd_loop(img, tmpl, result)
        # For SSD, smaller values are better
        _, _, _, min_loc = cv2.minMaxLoc(result)
        return min_loc
    
    elif method == "tm_nssd":
        # Calculate template standard deviation
        tmpl_mean = np.mean(tmpl)
        tmpl_std = np.std(tmpl)
        
        if tmpl_std < 1e-8:
            raise ValueError("Template has zero standard deviation, cannot use NSSD method")
        
        _nssd_loop(img, tmpl, result, tmpl_std)
        # For NSSD, smaller values are better
        _, _, _, min_loc = cv2.minMaxLoc(result)
        return min_loc
    
    elif method == "tm_ccor":
        _ccorr_loop(img, tmpl, result)
        # For correlation, larger values are better
        _, _, max_loc, _ = cv2.minMaxLoc(result)
        return max_loc
    
    elif method == "tm_nccor":
        # Normalize template
        tmpl_mean = np.mean(tmpl)
        tmpl_std = np.std(tmpl)
        
        if tmpl_std < 1e-8:
            raise ValueError("Template has zero standard deviation, cannot use NCCOR method")
        
        tmpl_norm = (tmpl - tmpl_mean) / tmpl_std
        
        _nccorr_loop(img, tmpl_norm, result)
        # For normalized correlation, larger values are better
        _, _, max_loc, _ = cv2.minMaxLoc(result)
        return max_loc
    
    else:
        raise NotImplementedError(f"Unknown method: {method}. Choose from 'tm_ssd', 'tm_nssd', 'tm_ccor', 'tm_nccor'")