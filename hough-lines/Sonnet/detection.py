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
        raise TypeError("Input image must be a numpy array")
    
    if img_in.size == 0:
        raise ValueError("Input image cannot be empty")
    
    if len(img_in.shape) != 3 or img_in.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel color image")
    
    if not hasattr(radii_range, '__iter__'):
        raise TypeError("radii_range must be an iterable (like range)")
    
    if len(list(radii_range)) == 0:
        raise ValueError("radii_range cannot be empty")
    
    gray   = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    hsv    = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, 1, blur.shape[0] / 24,
        param1=10, param2=11,
        minRadius=min(radii_range), maxRadius=max(radii_range) + 1
    )
    
    # Check if circles were detected
    if circles is None or len(circles) == 0:
        raise ValueError("No traffic light circles detected in the image")
    
    # Convert to integer coordinates
    circles = np.uint16(np.around(circles))
    
    # Sort circles by y-coordinate (vertical position)
    sorted_circles = sorted(circles[0], key=lambda c: c[1])
    
    # If we have at least 3 circles, assume they are the traffic light components
    # (red at top, yellow in middle, green at bottom)
    if len(sorted_circles) >= 3:
        # Extract the top 3 circles (sorted by y-coordinate)
        traffic_circles = sorted_circles[:3]
        
        # Initialize variables to track the active light
        max_value = -1
        active_circle = None
        
        # Check each circle's color in HSV space
        for circle in traffic_circles:
            x, y, r = circle
            
            # Create a mask for the current circle
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Extract the circle region from the HSV image
            circle_hsv = cv2.mean(hsv, mask=mask)
            h, s, v = circle_hsv[:3]
            
            # If this circle has higher value (brightness) than previous ones
            if v > max_value and s > 50:  # Ensure some saturation to avoid white/gray
                max_value = v
                active_circle = circle
        
        if active_circle is None:
            # If no circle meets criteria, use the middle one as fallback
            active_circle = traffic_circles[1]
        
        # Get the center of the active circle
        x, y, r = active_circle
        t_center = (int(x), int(y))
        
        # Determine the state based on the hue of the active circle
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, t_center, r, 255, -1)
        circle_hsv = cv2.mean(hsv, mask=mask)
        h = circle_hsv[0]
        
        # Map the hue to the closest traffic light color
        hue_diffs = {abs(h - key): value for key, value in hue_dict.items()}
        state = hue_diffs[min(hue_diffs.keys())]
        
        return t_center, state
    else:
        # If we don't have enough circles, use the most prominent one
        if len(sorted_circles) > 0:
            # Sort by radius (largest first) to get the most prominent circle
            prominent_circle = sorted(circles[0], key=lambda c: c[2], reverse=True)[0]
            x, y, r = prominent_circle
            t_center = (int(x), int(y))
            
            # Determine color
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, t_center, r, 255, -1)
            circle_hsv = cv2.mean(hsv, mask=mask)
            h = circle_hsv[0]
            
            # Map the hue to the closest traffic light color
            hue_diffs = {abs(h - key): value for key, value in hue_dict.items()}
            state = hue_diffs[min(hue_diffs.keys())]
            
            return t_center, state
        else:
            raise ValueError("Failed to identify traffic light components")

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
        raise ValueError("Input image must have at least 2 dimensions")
    
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
    
    # Collect all line endpoints
    endpoints = []
    for line in linesP:
        x1, y1, x2, y2 = line[0]
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))
    
    if not endpoints:
        raise ValueError("No valid endpoints found from detected lines")
    
    # Convert to numpy array for easier manipulation
    endpoints = np.array(endpoints)
    
    # Calculate the centroid of all endpoints
    centroid_x = np.mean(endpoints[:, 0])
    centroid_y = np.mean(endpoints[:, 1])
    
    # Return as integer coordinates
    return (int(centroid_x), int(centroid_y))