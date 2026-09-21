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
    # Input validation
    if not isinstance(img_in, np.ndarray):
        raise TypeError("img_in must be a numpy array")
    if len(img_in.shape) < 2 or img_in.size == 0:
        raise ValueError("img_in must be a non-empty image with at least 2 dimensions (height, width)")
    if len(img_in.shape) == 3 and img_in.shape[2] != 3:
        raise ValueError("If img_in is a color image, it must have 3 channels (BGR)")
    
    if not hasattr(radii_range, '__iter__'):
        raise TypeError("radii_range must be an iterable (e.g., range, list, tuple)")
    try:
        min_radius = min(radii_range)
        max_radius = max(radii_range)
        if min_radius >= max_radius:
            raise ValueError("radii_range must have min < max")
    except ValueError:
        raise ValueError("radii_range is empty or invalid")
    
    gray   = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    hsv    = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, 1, blur.shape[0] / 24,
        param1=10, param2=11,
        minRadius=min(radii_range), maxRadius=max(radii_range) + 1
    )
    
    # Check if circles were detected
    if circles is None or len(circles[0]) == 0:
        raise ValueError("No circles detected in the image")
    
    # Extract circle data and filter for vertical traffic light (3 circles with similar x, sorted by y)
    circles = circles[0]  # HoughCircles returns [1, N, 3] array
    candidate_circles = []
    for circle in circles:
        x, y, r = circle
        candidate_circles.append((int(x), int(y), int(r)))
    
    # Sort by y-coordinate and find 3 vertically aligned circles (similar x)
    candidate_circles.sort(key=lambda c: c[1])  # Sort by y
    selected_circles = []
    for i in range(len(candidate_circles) - 2):
        c1, c2, c3 = candidate_circles[i:i+3]
        # Check if x-coordinates are similar (within radius tolerance) and y increases
        if abs(c1[0] - c2[0]) < max(c1[2], c2[2]) and abs(c2[0] - c3[0]) < max(c2[2], c3[2]) and c1[1] < c2[1] < c3[1]:
            selected_circles = [c1, c2, c3]
            break
    
    if len(selected_circles) != 3:
        raise ValueError("Could not detect exactly 3 vertically aligned circles for the traffic light")
    
    # Determine active light: compute average V (value) for each circle's ROI
    max_v = -1
    active_index = -1
    for idx, (x, y, r) in enumerate(selected_circles):
        # Extract ROI (mask the circle)
        mask = np.zeros_like(hsv[:, :, 2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        v_values = hsv[:, :, 2][mask > 0]
        if len(v_values) == 0:
            continue
        avg_v = np.mean(v_values)
        if avg_v > max_v:
            max_v = avg_v
            active_index = idx
    
    if active_index == -1:
        raise ValueError("Could not determine an active light (no valid V values)")
    
    # Map index to state: 0=red (top), 1=yellow (middle), 2=green (bottom)
    state_map = {0: "red", 1: "yellow", 2: "green"}
    state = state_map[active_index]
    
    # Compute center as centroid of the 3 circles
    center_x = int(np.mean([c[0] for c in selected_circles]))
    center_y = int(np.mean([c[1] for c in selected_circles]))
    t_center = (center_x, center_y)
    
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
    # Input validation
    if not isinstance(img_in, np.ndarray):
        raise TypeError("img_in must be a numpy array")
    if len(img_in.shape) < 2 or img_in.size == 0:
        raise ValueError("img_in must be a non-empty image with at least 2 dimensions (height, width)")
    
    if len(img_in.shape) == 3 and img_in.shape[2] == 3:
        gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    elif len(img_in.shape) == 3 and img_in.shape[2] == 4:
        gray = cv2.cvtColor(img_in, cv2.COLOR_BGRA2GRAY)
    else:
        gray = img_in
    
    blur   = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    edges  = cv2.Canny(blur, 250, 255)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 0, 0)
    
    # Check if lines were detected
    if linesP is None or len(linesP) == 0:
        raise ValueError("No lines detected in the image")
    
    # Collect all unique endpoints from lines
    endpoints = set()
    for line in linesP[0]:
        x1, y1, x2, y2 = line
        endpoints.add((int(x1), int(y1)))
        endpoints.add((int(x2), int(y2)))
    
    if len(endpoints) == 0:
        raise ValueError("No valid endpoints found from detected lines")
    
    # Compute centroid (average x, average y)
    xs, ys = zip(*endpoints)
    centroid_x = int(np.mean(xs))
    centroid_y = int(np.mean(ys))
    centroid = (centroid_x, centroid_y)
    
    return centroid