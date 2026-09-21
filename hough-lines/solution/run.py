import os
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import detection
import logging

OUTPUT_DIR = "output/"

# Text overlay settings
_MARKER_COLOR = (255, 0, 255)
_TEXT_COLOR = (90, 90, 90)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_THICK = 2

def place_text(text, pt, img):
    """Positions text adjacent to a specified point within image boundaries.
    
    Automatically adjusts text placement to prevent overflow beyond image edges,
    positioning text either to the right/left and above/below the reference point
    as needed. Draws a white background rectangle behind the text for visibility.
    
    Args:
        text (str): The text string to render on the image.
        pt (tuple): Reference coordinates (x, y) for text positioning.
        img (numpy.array): Target image array for text placement.
    
    Returns:
        None: Directly modifies the input image array.
    
    Raises:
        ValueError: If text is empty or point coordinates are invalid.
        TypeError: If input parameters have incorrect types.
    """
    try:
        # Input validation
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")
        
        if not isinstance(pt, (tuple, list)) or len(pt) != 2:
            raise ValueError("Point must be a tuple or list with exactly 2 coordinates")
        
        if not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("Image must be a non-empty numpy array")
        
        if len(img.shape) < 2:
            raise ValueError("Image must have at least 2 dimensions")
        
        # Ensure coordinates are integers and within bounds
        x_coord = int(pt[0])
        y_coord = int(pt[1])
        
        if x_coord < 0 or y_coord < 0 or x_coord >= img.shape[1] or y_coord >= img.shape[0]:
            logging.warning(f"Point ({x_coord}, {y_coord}) is outside image boundaries")
            # Clamp coordinates to image bounds
            x_coord = max(0, min(x_coord, img.shape[1] - 1))
            y_coord = max(0, min(y_coord, img.shape[0] - 1))
        
        # Calculate text dimensions
        (w, h), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICK)
        margin = 5
        
        # Determine horizontal position
        x = x_coord + margin if x_coord + w + margin < img.shape[1] else x_coord - w - margin
        
        # Determine vertical position
        y_above = y_coord - margin
        y = y_above if y_above - h >= 0 else y_coord + h + margin
        
        # Ensure final positions are within bounds
        x = max(0, min(x, img.shape[1] - w))
        y = max(h, min(y, img.shape[0]))
        
        # Draw background rectangle and text
        top_left = (x, y - h)
        bottom_right = (x + w, y + baseline)
        cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), cv2.FILLED)
        cv2.putText(img, text, (x, y), _FONT, _FONT_SCALE, _TEXT_COLOR, _THICK)
        
    except (ValueError, TypeError) as e:
        logging.error(f"Error in place_text: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in place_text: {e}")
        raise

def draw_traffic_light_center(image_in, center, state):
    """Renders a marker at traffic light center with state information overlay.
    
    Draws a cross-shaped marker at the specified center coordinates and overlays
    formatted text showing the coordinates and current traffic light state.
    
    Args:
        image_in (numpy.array): Source image containing the traffic light.
        center (tuple): Center coordinates (x, y) of the traffic light.
        state (str): Current traffic light state ('red', 'yellow', or 'green').
    
    Returns:
        numpy.array: Modified image with marker and text overlay.
    
    Raises:
        ValueError: If image is invalid or center coordinates are malformed.
        TypeError: If input parameters have incorrect types.
    """
    try:
        # Input validation
        if not isinstance(image_in, np.ndarray) or image_in.size == 0:
            raise ValueError("Input image must be a non-empty numpy array")
        
        if len(image_in.shape) != 3:
            raise ValueError("Input image must be a 3-dimensional color image")
        
        if not isinstance(center, (tuple, list)) or len(center) != 2:
            raise ValueError("Center must be a tuple or list with exactly 2 coordinates")
        
        if not isinstance(state, str) or not state.strip():
            raise ValueError("State must be a non-empty string")
        
        # Convert coordinates to integers safely
        try:
            center_x = int(float(center[0]))
            center_y = int(float(center[1]))
        except (ValueError, TypeError):
            raise ValueError("Center coordinates must be numeric")
        
        center_int = (center_x, center_y)
        
        # Validate coordinates are within image bounds
        if (center_x < 0 or center_y < 0 or 
            center_x >= image_in.shape[1] or center_y >= image_in.shape[0]):
            logging.warning(f"Center coordinates {center_int} are outside image bounds")
            # Clamp to bounds
            center_x = max(0, min(center_x, image_in.shape[1] - 1))
            center_y = max(0, min(center_y, image_in.shape[0] - 1))
            center_int = (center_x, center_y)
        
        # Create a copy to avoid modifying original
        output = image_in.copy()
        
        # Draw marker
        cv2.drawMarker(output, center_int, _MARKER_COLOR, 
                      markerType=cv2.MARKER_CROSS, markerSize=11, thickness=2)
        
        # Create and place text
        text = "(({}, {}), '{}')".format(center_int[0], center_int[1], state.strip())
        place_text(text, center_int, output)
        
        return output
        
    except (ValueError, TypeError) as e:
        logging.error(f"Error in draw_traffic_light_center: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in draw_traffic_light_center: {e}")
        raise

def mark_traffic_signs(image_in, signs_dict):
    """Annotates traffic signs with markers and coordinate labels.
    
    Places cross markers at each sign's center position and adds text labels
    showing the sign name and coordinates. Signs are processed in left-to-right
    order based on their x-coordinates.
    
    Args:
        image_in (numpy.array): Source image containing traffic signs.
        signs_dict (dict): Mapping of sign names to their center coordinates.
    
    Returns:
        numpy.array: Annotated image with markers and labels for each sign.
    
    Raises:
        ValueError: If image is invalid or signs dictionary is malformed.
        TypeError: If input parameters have incorrect types.
    """
    try:
        # Input validation
        if not isinstance(image_in, np.ndarray) or image_in.size == 0:
            raise ValueError("Input image must be a non-empty numpy array")
        
        if len(image_in.shape) != 3:
            raise ValueError("Input image must be a 3-dimensional color image")
        
        if not isinstance(signs_dict, dict):
            raise ValueError("Signs data must be a dictionary")
        
        if not signs_dict:
            logging.warning("Empty signs dictionary provided")
            return image_in.copy()
        
        # Create output copy
        output = image_in.copy()
        
        # Validate and process sign entries
        valid_items = []
        for k, center in signs_dict.items():
            try:
                if not isinstance(k, str) or not k.strip():
                    logging.warning(f"Skipping invalid sign name: {k}")
                    continue
                
                if not isinstance(center, (tuple, list)) or len(center) != 2:
                    logging.warning(f"Skipping sign '{k}' with invalid coordinates: {center}")
                    continue
                
                # Convert coordinates safely
                center_x = float(center[0])
                center_y = float(center[1])
                
                # Check bounds
                if (center_x < 0 or center_y < 0 or 
                    center_x >= image_in.shape[1] or center_y >= image_in.shape[0]):
                    logging.warning(f"Sign '{k}' coordinates ({center_x}, {center_y}) outside image bounds")
                    # Clamp coordinates
                    center_x = max(0, min(center_x, image_in.shape[1] - 1))
                    center_y = max(0, min(center_y, image_in.shape[0] - 1))
                
                valid_items.append((int(center_x), k.strip(), (center_x, center_y)))
                
            except (ValueError, TypeError) as e:
                logging.warning(f"Skipping sign '{k}' due to coordinate error: {e}")
                continue
        
        if not valid_items:
            logging.warning("No valid signs found to mark")
            return output
        
        # Sort by x-coordinate (left to right)
        valid_items.sort()
        
        # Draw markers and labels
        for _, k, center in valid_items:
            center_int = (int(center[0]), int(center[1]))
            
            cv2.drawMarker(output, center_int, _MARKER_COLOR, 
                          markerType=cv2.MARKER_CROSS, markerSize=11, thickness=2)
            
            text = "{}: ({}, {})".format(k, center_int[0], center_int[1])
            place_text(text, center_int, output)
        
        return output
        
    except (ValueError, TypeError) as e:
        logging.error(f"Error in mark_traffic_signs: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in mark_traffic_signs: {e}")
        raise

def traffic_light_detection(input_images, output_labels):
    """Processes images to detect and annotate traffic lights with their states.
    
    Analyzes each input image to identify traffic light locations and states,
    then generates annotated output images with markers and state labels.
    Uses a predefined radius range for circle detection.
    
    Args:
        input_images (list): File paths of source images to process.
        output_labels (list): Corresponding output filename labels for results.
    
    Returns:
        None: Saves processed images to the designated output directory.
    
    Raises:
        ValueError: If input lists are mismatched or contain invalid entries.
        FileNotFoundError: If input images cannot be found or read.
        OSError: If output directory cannot be created or accessed.
    """
    try:
        # Input validation
        if not isinstance(input_images, list) or not isinstance(output_labels, list):
            raise ValueError("Input images and output labels must be lists")
        
        if len(input_images) != len(output_labels):
            raise ValueError("Input images and output labels lists must have the same length")
        
        if not input_images:
            logging.warning("No input images provided")
            return
        
        # Validate output directory
        if not os.path.exists(OUTPUT_DIR):
            try:
                os.makedirs(OUTPUT_DIR)
            except OSError as e:
                raise OSError(f"Cannot create output directory '{OUTPUT_DIR}': {e}")
        
        if not os.access(OUTPUT_DIR, os.W_OK):
            raise OSError(f"Output directory '{OUTPUT_DIR}' is not writable")
        
        radii_range = range(10, 30, 1)
        successful_processes = 0
        
        for i, (img_in, label) in enumerate(zip(input_images, output_labels)):
            try:
                # Validate inputs
                if not isinstance(img_in, str) or not img_in.strip():
                    logging.error(f"Invalid image path at index {i}: {img_in}")
                    continue
                
                if not isinstance(label, str) or not label.strip():
                    logging.error(f"Invalid label at index {i}: {label}")
                    continue
                
                # Check if input file exists
                if not os.path.exists(img_in):
                    logging.error(f"Input image not found: {img_in}")
                    continue
                
                # Read image
                tl = cv2.imread(img_in)
                if tl is None:
                    logging.error(f"Failed to read image: {img_in}")
                    continue
                
                if tl.size == 0:
                    logging.error(f"Empty image loaded: {img_in}")
                    continue
                
                # Process image
                coords, state = detection.traffic_light_detection(tl, radii_range)
                img_out = draw_traffic_light_center(tl, coords, state)
                
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(img_in))[0]
                output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{label.strip()}.png")
                
                # Save output
                success = cv2.imwrite(output_path, img_out)
                if not success:
                    logging.error(f"Failed to save output image: {output_path}")
                    continue
                
                successful_processes += 1
                logging.info(f"Successfully processed: {img_in} -> {output_path}")
                
            except Exception as e:
                logging.error(f"Error processing image {img_in}: {e}")
                continue
        
        logging.info(f"Traffic light detection completed: {successful_processes}/{len(input_images)} images processed successfully")
        
    except (ValueError, OSError) as e:
        logging.error(f"Error in traffic_light_detection: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in traffic_light_detection: {e}")
        raise

def construction_sign_detection(input_images, output_labels):
    """Processes images to detect and annotate construction signs.
    
    Analyzes each input image to locate construction signs and generates
    annotated output images with markers showing the detected sign centers.
    
    Args:
        input_images (list): File paths of source images to analyze.
        output_labels (list): Corresponding output filename labels for results.
    
    Returns:
        None: Saves processed images to the designated output directory.
    
    Raises:
        ValueError: If input lists are mismatched or contain invalid entries.
        FileNotFoundError: If input images cannot be found or read.
        OSError: If output directory cannot be created or accessed.
    """
    try:
        # Input validation
        if not isinstance(input_images, list) or not isinstance(output_labels, list):
            raise ValueError("Input images and output labels must be lists")
        
        if len(input_images) != len(output_labels):
            raise ValueError("Input images and output labels lists must have the same length")
        
        if not input_images:
            logging.warning("No input images provided")
            return
        
        # Validate output directory
        if not os.path.exists(OUTPUT_DIR):
            try:
                os.makedirs(OUTPUT_DIR)
            except OSError as e:
                raise OSError(f"Cannot create output directory '{OUTPUT_DIR}': {e}")
        
        if not os.access(OUTPUT_DIR, os.W_OK):
            raise OSError(f"Output directory '{OUTPUT_DIR}' is not writable")
        
        sign_fns = [detection.construction_sign_detection]
        sign_labels = ['construction']
        successful_processes = 0
        
        for i, (img_in, label) in enumerate(zip(input_images, output_labels)):
            try:
                # Validate inputs
                if not isinstance(img_in, str) or not img_in.strip():
                    logging.error(f"Invalid image path at index {i}: {img_in}")
                    continue
                
                if not isinstance(label, str) or not label.strip():
                    logging.error(f"Invalid label at index {i}: {label}")
                    continue
                
                # Check if input file exists
                if not os.path.exists(img_in):
                    logging.error(f"Input image not found: {img_in}")
                    continue
                
                # Read image
                sign_img = cv2.imread(img_in)
                if sign_img is None:
                    logging.error(f"Failed to read image: {img_in}")
                    continue
                
                if sign_img.size == 0:
                    logging.error(f"Empty image loaded: {img_in}")
                    continue
                
                # Process each detection function
                for fn, name in zip(sign_fns, sign_labels):
                    coords = fn(sign_img)
                    temp_dict = {name: coords}
                    img_out = mark_traffic_signs(sign_img, temp_dict)
                    
                    # Generate output filename
                    base_name = os.path.splitext(os.path.basename(img_in))[0]
                    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{label.strip()}.png")
                    
                    # Save output
                    success = cv2.imwrite(output_path, img_out)
                    if not success:
                        logging.error(f"Failed to save output image: {output_path}")
                        continue
                
                successful_processes += 1
                logging.info(f"Successfully processed: {img_in} -> {output_path}")
                
            except Exception as e:
                logging.error(f"Error processing image {img_in}: {e}")
                continue
        
        logging.info(f"Construction sign detection completed: {successful_processes}/{len(input_images)} images processed successfully")
        
    except (ValueError, OSError) as e:
        logging.error(f"Error in construction_sign_detection: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in construction_sign_detection: {e}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        parser = argparse.ArgumentParser(description="Traffic light and sign detection.")
        parser.add_argument("scene_tl", help="Input filename for traffic light image.")
        parser.add_argument("scene_constr", help="Input filename for construction sign image.")
        
        args = parser.parse_args()
        
        # Validate command line arguments
        if not args.scene_tl or not args.scene_constr:
            raise ValueError("Both scene_tl and scene_constr arguments are required")
        
        # Create Output directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Process images
        traffic_light_detection([args.scene_tl], ["out"])
        construction_sign_detection([args.scene_constr], ["out"])
        
        logging.info("Image processing completed successfully")
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        exit(1)