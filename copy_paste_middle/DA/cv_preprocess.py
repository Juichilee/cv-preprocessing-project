import math
import numpy as np
import cv2

def extract_red(image):
    """
    Extract the intensity values stored in the red channel of a three‑channel image.

    This function returns a new two‑dimensional array containing the values that were held
    in the third channel of the input. The input image is copied before extraction to avoid
    altering the original data. If the input is not a numeric array with at least three
    channels, the function raises a ValueError.

    Returns:
        numpy.ndarray: A two‑dimensional array of the same height and width as the input,
                       containing the red‑channel intensity values.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    if image.ndim < 3 or image.shape[2] < 3:
        raise ValueError("Input image must have at least three channels")
    temp = np.copy(image)
    return temp[:, :, 2]


def extract_green(image):
    """
    Extract the intensity values stored in the green channel of a three‑channel image.

    This function returns a new two‑dimensional array containing the values that were held
    in the second channel of the input. The input image is copied before extraction to avoid
    altering the original data. If the input is not a numeric array with at least three
    channels, the function raises a ValueError.

    Returns:
        numpy.ndarray: A two‑dimensional array of the same height and width as the input,
                       containing the green‑channel intensity values.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    if image.ndim < 3 or image.shape[2] < 3:
        raise ValueError("Input image must have at least three channels")
    temp = np.copy(image)
    return temp[:, :, 1]


def extract_blue(image):
    """
    Extract the intensity values stored in the blue channel of a three‑channel image.

    This function returns a new two‑dimensional array containing the values that were held
    in the first channel of the input. The input image is copied before extraction to avoid
    altering the original data. If the input is not a numeric array with at least three
    channels, the function raises a ValueError.

    Returns:
        numpy.ndarray: A two‑dimensional array of the same height and width as the input,
                       containing the blue‑channel intensity values.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    if image.ndim < 3 or image.shape[2] < 3:
        raise ValueError("Input image must have at least three channels")
    temp = np.copy(image)
    return temp[:, :, 0]


def swap_green_blue(image):
    """
    Exchange the green and blue channels in a three‑channel image.

    This function makes a full copy of the input and then swaps the data from the second
    and first channels. The result is returned as a new three‑channel array. If the input
    does not have exactly three channels, a ValueError is raised.

    Returns:
        numpy.ndarray: A new BGR image array of the same shape as the input, with green
                       and blue channels swapped.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have exactly three channels")
    temp = np.copy(image)
    blue = extract_blue(temp)
    green = extract_green(temp)
    temp[:, :, 0] = green
    temp[:, :, 1] = blue
    return temp


def copy_paste_middle(src, dst, shape):
    """
    Copy a centered rectangular block from one two-dimensional array into another.

    The function takes a height and width for the block, extracts the block from the
    center of the source array, and places it at the center of a copied destination
    array. Half of the requested height and width for the center are computed by floor 
    division. The rest of the destination data outside the block remains unchanged. Both
    inputs must be two-dimensional and the requested block size must not exceed either
    array in its corresponding dimension. A ValueError is raised if these conditions
    are violated.

    Returns:
        numpy.ndarray: A new two-dimensional array with the same shape as `dst`, containing
                       the rectangular patch from `src` pasted into its center.
    """
    if not isinstance(src, np.ndarray) or not isinstance(dst, np.ndarray):
        raise ValueError("Both inputs must be NumPy arrays")
    if src.ndim != 2 or dst.ndim != 2:
        raise ValueError("Both inputs must be two-dimensional arrays")
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("Shape must be a tuple of (height, width)")
    block_h, block_w = shape
    if not (isinstance(block_h, int) and isinstance(block_w, int)):
        raise ValueError("Block dimensions must be integers")
    if block_h <= 0 or block_w <= 0:
        raise ValueError("Block dimensions must be positive")
    if block_h > src.shape[0] or block_w > src.shape[1]:
        raise ValueError("Block size exceeds source array dimensions")
    if block_h > dst.shape[0] or block_w > dst.shape[1]:
        raise ValueError("Block size exceeds destination array dimensions")

    result = np.copy(dst)
    src_h, src_w = src.shape
    dst_h, dst_w = dst.shape

    src_start_row = (src_h - block_h) // 2
    src_start_col = (src_w - block_w) // 2
    dst_start_row = (dst_h - block_h) // 2
    dst_start_col = (dst_w - block_w) // 2

    result[dst_start_row:dst_start_row + block_h, dst_start_col:dst_start_col + block_w] = \
        src[src_start_row:src_start_row + block_h, src_start_col:src_start_col + block_w]

    return result


def copy_paste_middle_circle(src, dst, radius):
    """
    Copy a filled circular region from the center of one two-dimensional array into another.

    The function selects all points in the source whose distance from its center does not
    exceed the given radius and writes those values into the center of a copied
    destination array at the corresponding offsets. It locates the source center at row (height minus 1) floor-divided
    by two and the source center column similarly. Both inputs must be two-dimensional
    arrays and the radius must be a non-negative integer. A ValueError is raised for
    invalid inputs.

    Returns:
        numpy.ndarray: A new two-dimensional array with the same shape as `dst`, containing
                       the circular patch from `src` pasted into its center.
    """
    if not isinstance(src, np.ndarray) or not isinstance(dst, np.ndarray):
        raise ValueError("Both inputs must be NumPy arrays")
    if src.ndim != 2 or dst.ndim != 2:
        raise ValueError("Both inputs must be two-dimensional arrays")
    if not isinstance(radius, (int, float)) or radius < 0:
        raise ValueError("Radius must be a non-negative number")

    result = np.copy(dst)
    src_h, src_w = src.shape
    dst_h, dst_w = dst.shape

    src_center_row = (src_h - 1) // 2
    src_center_col = (src_w - 1) // 2
    dst_center_row = (dst_h - 1) // 2
    dst_center_col = (dst_w - 1) // 2

    for i in range(src_h):
        for j in range(src_w):
            dist = np.sqrt((i - src_center_row) ** 2 + (j - src_center_col) ** 2)
            if dist <= radius:
                dst_i = dst_center_row + (i - src_center_row)
                dst_j = dst_center_col + (j - src_center_col)
                if 0 <= dst_i < dst_h and 0 <= dst_j < dst_w:
                    result[dst_i, dst_j] = src[i, j]

    return result


def image_stats(image):
    """
    Compute descriptive statistics for a two-dimensional array of numeric values.

    The function returns four numbers in order: the smallest value present, the largest
    value present, the arithmetic average of all entries, and the measure of spread
    around that average known as the standard deviation. The input must be a
    two-dimensional numeric array or a ValueError is raised.

    Returns:
        tuple of float: (minimum_value, maximum_value, mean_value, standard_deviation)
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    if image.ndim != 2:
        raise ValueError("Input must be a two-dimensional array")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("Input array must contain numeric values")
    if image.size == 0:
        raise ValueError("Input array must not be empty")

    minimum = float(np.min(image))
    maximum = float(np.max(image))
    mean_val = float(np.mean(image))
    std_val = float(np.std(image))

    return (minimum, maximum, mean_val, std_val)


def center_and_normalize(image, scale):
    """
    Adjust a two‑dimensional array so that its average remains the same but its standard
    deviation becomes the specified scale value.

    For each element, the function subtracts the original average, divides by the original
    standard deviation, multiplies by the target standard deviation, and then adds back
    the original average. The input must be a two‑dimensional numeric array with a
    non‑zero spread of values and the scale must be a positive number, otherwise
    a ValueError is raised.

    Returns:
        numpy.ndarray: A two‑dimensional array of the same shape as the input, with values
                       re‑centered and scaled.
    """
    raise NotImplementedError


def shift_image_left(image, shift):
    """
    Create a shifted version of a two‑dimensional array by moving all columns to the left.

    The function moves each element by the given number of columns toward the start of 
    each row. The columns on the right that become vacant are filled by repeating the values 
    that would have fallen off the left edge. The input must be a two‑dimensional array 
    and the shift amount must be a non‑negative integer less than the width of the array. 
    Invalid inputs cause a ValueError.

    Returns:
        numpy.ndarray: A two‑dimensional array of the same shape as input.
    """
    raise NotImplementedError


def difference_image(img1, img2, output_min=0.0, output_max=255.0):
    """
    Subtract the second two‑dimensional array from the first and scale the result to a given range.

    The function computes element‑by‑element difference, then shifts and rescales the values
    so that the smallest difference maps to the specified minimum and the largest difference
    maps to the specified maximum. Both inputs must share the same shape and be two‑dimensional
    arrays, otherwise a ValueError is raised.

    Returns:
        numpy.ndarray: A two‑dimensional array of the same shape.
    """
    raise NotImplementedError

def add_noise(image, channel, sigma, mean=0.0):
    """
    Add random Gaussian perturbations to one channel of a three‑channel image.

    The function generates independent random values with the specified average and standard
    deviation, then adds those values to the designated channel in a copy of the input image.
    The input must be a numeric array with at least three channels, the channel index must be
    zero, one, or two, the noise average must be numeric, and the noise standard deviation must
    be non‑negative. A ValueError is raised for invalid inputs.

    Returns:
        numpy.ndarray: A three‑dimensional float64 array of the same shape as input.
    """
    raise NotImplementedError


def build_hybrid_image(image1, image2, cutoff_frequency):
    """
    Combine the coarse structure of one three‑channel image with the fine detail of another.

    The function applies a smoothing filter based on the cutoff frequency to the first input
    to obtain its low‑frequency content. It applies the same filter to the second input and
    subtracts the result to obtain its high‑frequency content. Those two results are summed
    and then clipped so that all values lie within the valid eight‑bit range. Both inputs must
    have identical dimensions and exactly three channels, and the cutoff frequency must be a
    positive integer. Otherwise, a ValueError is raised.

    Returns:
        numpy.ndarray: A three‑dimensional uint8 array of the same shape as inputs.
    """
    raise NotImplementedError


def vis_hybrid_image(hybrid_image, num_scales=5, scale_factor=0.5, padding=5):
    """
    Create a stacked visualization of a hybrid color image at multiple scales.

    This function takes a three-channel (height, width, 3) image and returns a single
    float32 array where the original and each downscaled version are concatenated
    side by side with vertical padding so that the total height stays constant. The
    number of additional scales, the downscaling factor each step, and the padding
    width between images are all configurable. The input must be a NumPy array with
    exactly three channels and non-zero height, otherwise a ValueError is raised.

    During the resizing process, if the resulting dimensions of the downscaled image
    become invalid (i.e., less than 1 pixel in either dimension), the resizing operation
    is skipped for that scale, and the loop terminates early.

    Returns:
        numpy.ndarray: A float32 array of shape (height, output_width, 3).
    """
    raise NotImplementedError