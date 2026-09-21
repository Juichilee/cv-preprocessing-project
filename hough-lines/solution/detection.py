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
        ValueError: If image is empty, invalid dimensions, radii_range is empty/invalid, or no traffic light is found.
    """
    # Input validation
    if not isinstance(img_in, np.ndarray):
        raise TypeError("Input image must be a numpy array")
    
    if img_in.size == 0:
        raise ValueError("Input image cannot be empty")
    
    if len(img_in.shape) != 3:
        raise ValueError("Input image must be a 3-dimensional color image (height, width, channels)")
    
    if img_in.shape[2] != 3:
        raise ValueError("Input image must have exactly 3 color channels (BGR)")
    
    # Validate radii_range
    try:
        radii_list = list(radii_range)
    except TypeError:
        raise TypeError("Radii range must be iterable (range, list, tuple, etc.)")
    
    if not radii_list:
        raise ValueError("Radii range cannot be empty")
    
    if not all(isinstance(r, (int, float)) and r > 0 for r in radii_list):
        raise ValueError("All radii values must be positive numbers")
    
    if min(radii_list) <= 0:
        raise ValueError("All radii values must be greater than 0")
    
    # Check if image dimensions are sufficient for the radii range
    min_dimension = min(img_in.shape[:2])
    max_radius = max(radii_list)
    if max_radius * 2 > min_dimension:
        raise ValueError(f"Maximum radius ({max_radius}) is too large for image dimensions ({img_in.shape[:2]})")
    
    try:
        gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
        hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        raise ValueError(f"OpenCV error during image processing: {str(e)}")
    
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        1,
        blur.shape[0] / 24,
        param1=10,
        param2=11,
        minRadius=min(radii_range),
        maxRadius=max(radii_range) + 1
    )
    
    if circles is None:
        raise ValueError("No traffic light detected in the image")

    t_center = (0, 0)
    brightest = (0, 0, 0)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            c_center = (i[0], i[1])
            
            # Ensure coordinates are within image bounds
            if 0 <= i[1] < hsv.shape[0] and 0 <= i[0] < hsv.shape[1]:
                c_hsv = hsv[i[1], i[0]]
                
                if c_hsv[0] == HUE_Y:
                    t_center = c_center
                brightest = max(brightest, c_hsv, key=lambda x: x[2])
    
    # Ensure brightest[0] is a valid key in hue_dict
    if brightest[0] not in hue_dict:
        # Find closest valid hue value
        closest_hue = min(hue_dict.keys(), key=lambda x: abs(x - brightest[0]))
        state = hue_dict[closest_hue]
    else:
        state = hue_dict[brightest[0]]
    
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
        raise TypeError("Input image must be a numpy array")
    
    if img_in.size == 0:
        raise ValueError("Input image cannot be empty")
    
    if len(img_in.shape) < 2:
        raise ValueError("Input image must be at least 2-dimensional")
    
    # Handle both grayscale and color images
    if len(img_in.shape) == 3:
        if img_in.shape[2] not in [1, 3, 4]:
            raise ValueError("Color image must have 1, 3, or 4 channels")
    
    try:
        # Convert to grayscale if needed
        if len(img_in.shape) == 3 and img_in.shape[2] == 3:
            gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        elif len(img_in.shape) == 3 and img_in.shape[2] == 4:
            gray = cv2.cvtColor(img_in, cv2.COLOR_BGRA2GRAY)
        else:
            gray = img_in
        
        blur = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
        edges = cv2.Canny(blur, 250, 255)
    except cv2.error as e:
        raise ValueError(f"OpenCV error during image processing: {str(e)}")
    
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 0, 0)
    
    if linesP is None:
        raise ValueError("No lines detected in the image - construction sign may not be present or visible")
    
    if len(linesP) == 0:
        raise ValueError("No valid lines found in the image")
    
    points = []
    for line in linesP:
        if len(line[0]) != 4:
            continue  # Skip invalid line data
        
        x0, y0, x1, y1 = line[0]
        
        # Validate coordinate values
        if not all(isinstance(coord, (int, float, np.integer, np.floating)) for coord in [x0, y0, x1, y1]):
            continue  # Skip lines with invalid coordinates
        
        points.append((x0, y0))
        points.append((x1, y1))
    
    if not points:
        raise ValueError("No valid coordinate points extracted from detected lines")
    
    try:
        points = np.array(points, dtype=np.uint16)
    except (ValueError, TypeError):
        raise ValueError("Failed to convert coordinates to valid numpy array")
    
    if points.size == 0:
        raise ValueError("No valid points available for centroid calculation")
    
    rightmost_idx = np.argmax(points[:, 0])
    rightmost = tuple(points[rightmost_idx])
    
    leftmost_idx = np.argmin(points[:, 0])
    leftmost = tuple(points[leftmost_idx])
    
    # Ensure we have valid coordinates for centroid calculation
    try:
        centroid = ((leftmost[0] + rightmost[0]) // 2, (rightmost[1] + leftmost[1]) // 2)
        # Convert to int to ensure valid coordinate types
        centroid = (int(centroid[0]), int(centroid[1]))
    except (TypeError, ValueError):
        raise ValueError("Failed to calculate valid centroid coordinates")
    
    return centroid
