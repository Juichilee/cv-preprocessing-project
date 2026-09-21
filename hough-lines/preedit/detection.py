import cv2
import numpy as np

HUE_R = 0
HUE_Y = 30
HUE_G = 60
hue_dict = {HUE_R: "red", HUE_Y: "yellow", HUE_G: "green"}

def traffic_light_detection(img_in, radii_range):
    """Identifies the central position and active color state of a traffic light within an image using circular detection and HSV color analysis.
    
    Args:
        img_in (numpy.array): The image containing a traffic light.
        radii_range (range): A range of radii values to search for traffic light circles.
    
    Returns:
        tuple: (center, state) where center is a tuple of (x, y) coordinates and state is one of 'red', 'yellow', or 'green'.
    
    Raises:
        TypeError: If img_in is not a numpy array or radii_range is not a range/iterable.
        ValueError: If image is empty, invalid dimensions, or radii_range is empty/invalid.
    """
    gray   = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    hsv    = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, 1, blur.shape[0] / 24,
        param1=10, param2=11,
        minRadius=min(radii_range), maxRadius=max(radii_range) + 1
    )

    # Implement algorithm to find center + state
    raise NotImplementedError

    return t_center, state

def construction_sign_detection(img_in):
    """Locates the geometric center of a construction sign by analyzing edge patterns and calculating the centroid between extreme boundary points.
    
    Args:
        img_in (numpy.array): The image containing a construction sign.
    
    Returns:
        tuple: The (x, y) coordinates of the sign's center.
    
    Raises:
        TypeError: If img_in is not a numpy array.
        ValueError: If image is empty, has invalid dimensions, or no lines are detected.
    """
    if len(img_in.shape) == 3 and img_in.shape[2] == 3:
        gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    elif len(img_in.shape) == 3 and img_in.shape[2] == 4:
        gray = cv2.cvtColor(img_in, cv2.COLOR_BGRA2GRAY)
    else:
        gray = img_in

    blur   = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    edges  = cv2.Canny(blur, 250, 255)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 0, 0)

    # Implement algorithm to find centroid
    raise NotImplementedError

    return centroid
